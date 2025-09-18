# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from fpdf import FPDF
from datetime import datetime, timedelta
import os
from scipy.fft import fft, fftfreq
import tempfile
import unicodedata
import io
from io import BytesIO, StringIO
from ahrs.filters import Mahony
import os
import glob
import streamlit as st
import joblib



# Inicializar una variable en el estado de sesión para controlar el reinicio
if "reiniciar" not in st.session_state:
    st.session_state.reiniciar = False

st.markdown("""
    <style>
    /* Oculta el texto 'Limit 200MB per file • CSV' */
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    div[data-testid="stFileUploader"] button[kind="secondary"] {
        visibility: hidden;
    }
    div[data-testid="stFileUploader"] button[kind="secondary"]::before {
        float: right;
        margin-right: 0;
        content: "Cargar archivos";
        visibility: visible;
        display: inline-block;
        background-color: #FF5722;
        color: white;
        padding: 0.5em 1em;
        border-radius: 6px;
        border: 2px solid white;
        cursor: pointer;
    }
    /* Alinea todo a la derecha */
    div[data-testid="stFileUploader"] > div:first-child {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    div[data-testid="stFileUploader"] > div {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Configuración global de la duración de la ventana ---
ventana_duracion_seg = 2

# ---------FUNCIONES COMPARTIDAS ----------
# Función para extraer datos del paciente de un DataFrame
def extraer_datos_paciente(df):
    """
    Extrae datos personales del paciente y parámetros de configuración de DBS
    desde un DataFrame, sin modificar las columnas originales del DataFrame.
    """
    col_map = {col.lower().strip(): col for col in df.columns}

    datos = {
        "sexo": "No especificado", "edad": 0,
        "mano_medida": "No especificada", "dedo_medido": "No especificado",
        "nombre": None, "apellido": None, "diagnostico": None,
        "antecedente": None, "medicacion": None, "tipo": None,
        "dbs": None, "nucleo": None,
        "voltaje_izq": None, "corriente_izq": None, "contacto_izq": None, "frecuencia_izq": None, "ancho_pulso_izq": None,
        "voltaje_dch": None, "corriente_dch": None, "contacto_dch": None, "frecuencia_dch": None, "ancho_pulso_dch": None,
    }

    if not df.empty:
        # Extraer datos personales usando el mapeo
        if "sexo" in col_map and pd.notna(df.at[0, col_map["sexo"]]):
            datos["sexo"] = str(df.at[0, col_map["sexo"]]).strip()
        if "edad" in col_map and pd.notna(df.at[0, col_map["edad"]]):
            try:
                datos["edad"] = int(float(str(df.at[0, col_map["edad"]]).replace(',', '.')))
            except (ValueError, TypeError):
                datos["edad"] = 0
        if "mano" in col_map and pd.notna(df.at[0, col_map["mano"]]):
            datos["mano_medida"] = str(df.at[0, col_map["mano"]]).strip()
        if "dedo" in col_map and pd.notna(df.at[0, col_map["dedo"]]):
            datos["dedo_medido"] = str(df.at[0, col_map["dedo"]]).strip()

        for key in ["nombre", "apellido", "diagnostico", "antecedente", "medicacion", "tipo"]:
            if key in col_map and pd.notna(df.at[0, col_map[key]]):
                datos[key] = str(df.at[0, col_map[key]])

        # Extraer campos de estimulación con los nombres exactos proporcionados
        # General
        if "dbs" in col_map and pd.notna(df.at[0, col_map["dbs"]]):
            datos["dbs"] = str(df.at[0, col_map["dbs"]])
        if "nucleo" in col_map and pd.notna(df.at[0, col_map["nucleo"]]):
            datos["nucleo"] = str(df.at[0, col_map["nucleo"]])

        # Izquierda
        if "voltaje [mv]_izq" in col_map and pd.notna(df.at[0, col_map["voltaje [mv]_izq"]]):
            datos["voltaje_izq"] = str(df.at[0, col_map["voltaje [mv]_izq"]]).replace(',', '.')
        if "corriente [ma]_izq" in col_map and pd.notna(df.at[0, col_map["corriente [ma]_izq"]]):
            datos["corriente_izq"] = str(df.at[0, col_map["corriente [ma]_izq"]]).replace(',', '.')
        if "contacto_izq" in col_map and pd.notna(df.at[0, col_map["contacto_izq"]]):
            datos["contacto_izq"] = str(df.at[0, col_map["contacto_izq"]])
        if "frecuencia [hz]_izq" in col_map and pd.notna(df.at[0, col_map["frecuencia [hz]_izq"]]):
            datos["frecuencia_izq"] = str(df.at[0, col_map["frecuencia [hz]_izq"]]).replace(',', '.')
        if "ancho de pulso [µs]_izq" in col_map and pd.notna(df.at[0, col_map["ancho de pulso [µs]_izq"]]):
            datos["ancho_pulso_izq"] = str(df.at[0, col_map["ancho de pulso [µs]_izq"]]).replace(',', '.')

        # Derecha
        if "voltaje [mv]_dch" in col_map and pd.notna(df.at[0, col_map["voltaje [mv]_dch"]]):
            datos["voltaje_dch"] = str(df.at[0, col_map["voltaje [mv]_dch"]]).replace(',', '.')
        if "corriente [ma]_dch" in col_map and pd.notna(df.at[0, col_map["corriente [ma]_dch"]]):
            datos["corriente_dch"] = str(df.at[0, col_map["corriente [ma]_dch"]]).replace(',', '.')
        if "contacto_dch" in col_map and pd.notna(df.at[0, col_map["contacto_dch"]]):
            datos["contacto_dch"] = str(df.at[0, col_map["contacto_dch"]])
        if "frecuencia [hz]_dch" in col_map and pd.notna(df.at[0, col_map["frecuencia [hz]_dch"]]):
            datos["frecuencia_dch"] = str(df.at[0, col_map["frecuencia [hz]_dch"]]).replace(',', '.')
        if "ancho de pulso [µs]_dch" in col_map and pd.notna(df.at[0, col_map["ancho de pulso [µs]_dch"]]):
            datos["ancho_pulso_dch"] = str(df.at[0, col_map["ancho de pulso [µs]_dch"]]).replace(',', '.')

    return datos


def filtrar_temblor(signal, fs=100):
    b, a = butter(N=4, Wn=[1, 15], btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def q_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),       2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),             1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),             2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

def analizar_temblor_por_ventanas_resultante(df, fs=100, ventana_seg=ventana_duracion_seg):
    required_cols = ['acel_x', 'acel_y', 'acel_z', 'girox', 'giroy', 'giroz']
    
    # Asegurarse de que las columnas del DataFrame estén en minúsculas para el análisis
    df.columns = df.columns.str.lower()
    
    df_senial = df[[col for col in required_cols if col in df.columns]].dropna()

    if len(df_senial.columns) < 6:
        st.error("Error: el archivo CSV no contiene las 6 columnas de datos requeridas (aceleración y giroscopio).")
        return pd.DataFrame(), pd.DataFrame()
    
    acc = df_senial[['acel_x', 'acel_y', 'acel_z']].to_numpy()
    gyr = np.radians(df_senial[['girox', 'giroy', 'giroz']].to_numpy())
    mahony = Mahony(gyr=gyr, acc=acc, frequency=fs)
    Q = mahony.Q
    linear_accelerations_magnitude = []
    g_world_vector = np.array([0.0, 0.0, 9.81])

    for i in range(len(acc)):
        q = Q[i]
        acc_measured = acc[i]
        R_W_B = q_to_matrix(q)
        gravity_in_sensor_frame = R_W_B @ g_world_vector
        linear_acc_sensor_frame = acc_measured - gravity_in_sensor_frame
        linear_accelerations_magnitude.append(np.linalg.norm(linear_acc_sensor_frame))

    movimiento_lineal = np.array(linear_accelerations_magnitude)
    señal_filtrada = filtrar_temblor(movimiento_lineal, fs)

    resultados_por_ventana = []

    tamaño_ventana = int(fs * ventana_seg)
    if len(señal_filtrada) < tamaño_ventana:
        return pd.DataFrame(), pd.DataFrame()

    num_ventanas = len(señal_filtrada) // tamaño_ventana

    for i in range(num_ventanas):
        segmento = señal_filtrada[i*tamaño_ventana:(i+1)*tamaño_ventana]
        segmento = segmento - np.mean(segmento)

        f, Pxx = welch(segmento, fs=fs, nperseg=tamaño_ventana)
        if len(Pxx) > 0:
            freq_dominante = f[np.argmax(Pxx)]
        else:
            freq_dominante = 0.0

        rms = np.sqrt(np.mean(segmento**2))
        amp_g = (np.max(segmento) - np.min(segmento))/2

        if freq_dominante > 1.5:
            amp_cm = ((amp_g * 100) / ((2 * np.pi * freq_dominante) ** 2))*2
        else:
            amp_cm = 0.0

        resultados_por_ventana.append({
           'Ventana': i,
           'Frecuencia Dominante (Hz)': freq_dominante,
           'RMS (m/s2)': rms,
           'Amplitud Temblor (g)': amp_g,
           'Amplitud Temblor (cm)': amp_cm
          })

    df_por_ventana = pd.DataFrame(resultados_por_ventana)

    if not df_por_ventana.empty:
        promedio = df_por_ventana.mean(numeric_only=True).to_dict()
        df_promedio = pd.DataFrame([{
            'Frecuencia Dominante (Hz)': promedio['Frecuencia Dominante (Hz)'],
            'RMS (m/s2)': promedio['RMS (m/s2)'],
            'Amplitud Temblor (cm)': promedio['Amplitud Temblor (cm)']
        }])
    else:
        df_promedio = pd.DataFrame()

    return df_promedio, df_por_ventana

def manejar_reinicio():
    if st.session_state.get("reiniciar", False):
        for file in os.listdir():
            if file.endswith(".csv"):
                try:
                    os.remove(file)
                except Exception as e:
                    st.warning(f"No se pudo borrar {file}: {e}")

        st.session_state.clear()
        st.experimental_rerun()


# ------------------ MODO PRINCIPAL ---------------------------------------

st.title("🧠 Análisis de Temblor")
opcion = st.sidebar.radio("Selecciona una opción:", ["1️⃣ Análisis de una medición", "2️⃣ Comparación de mediciones", "3️⃣ Diagnóstico tentativo"])
if st.sidebar.button("🔄 Nuevo análisis"):
    manejar_reinicio()
    
# ------------------ MÓDULO 1: ANÁLISIS DE UNA MEDICIÓN --------------------

if opcion == "1️⃣ Análisis de una medición":
    st.title("📈​ Análisis de una Medición")

    def generar_pdf(datos_paciente_dict, df, nombre_archivo="informe_temblor.pdf", fig=None):
        fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 10)
        pdf.cell(200, 10, f"Fecha y hora del análisis: {fecha_hora}", ln=True)
        pdf.set_font("Arial", size=8)
        pdf.ln(10)
        # Helper para imprimir campos solo si tienen valor
        def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
            if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
                pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)
    
        # Impresión de Datos Personales
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos del Paciente", ln=True)
        pdf.set_font("Arial", size=12)
        _imprimir_campo_pdf(pdf, "Nombre", datos_paciente_dict.get("nombre"))
        _imprimir_campo_pdf(pdf, "Apellido", datos_paciente_dict.get("apellido"))
        edad_val = datos_paciente_dict.get("edad")
        if isinstance(edad_val, (int, float)):
            _imprimir_campo_pdf(pdf, "Edad", int(edad_val))
        _imprimir_campo_pdf(pdf, "Sexo", datos_paciente_dict.get("sexo"))
        _imprimir_campo_pdf(pdf, "Diagnóstico", datos_paciente_dict.get("diagnostico"))
        _imprimir_campo_pdf(pdf, "Tipo", datos_paciente_dict.get("tipo"))
        _imprimir_campo_pdf(pdf, "Mano", datos_paciente_dict.get("mano_medida"))
        _imprimir_campo_pdf(pdf, "Dedo", datos_paciente_dict.get("dedo_medido"))
        _imprimir_campo_pdf(pdf, "Antecedente", datos_paciente_dict.get("antecedente"))
        _imprimir_campo_pdf(pdf, "Medicacion", datos_paciente_dict.get("medicacion"))
        pdf.ln(5)
    
        # Impresión de Parámetros de Estimulación
        hay_parametros_estimulacion = datos_paciente_dict.get("dbs") is not None or datos_paciente_dict.get("nucleo") is not None
        hay_parametros_izq = datos_paciente_dict.get("voltaje_izq") is not None
        hay_parametros_dch = datos_paciente_dict.get("voltaje_dch") is not None
        
        if hay_parametros_estimulacion or hay_parametros_izq or hay_parametros_dch:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Configuración de Estimulación", ln=True)
            pdf.set_font("Arial", size=12)
            _imprimir_campo_pdf(pdf, "DBS", datos_paciente_dict.get("dbs"))
            _imprimir_campo_pdf(pdf, "Núcleo", datos_paciente_dict.get("nucleo"))
    
            if hay_parametros_izq:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Configuración Izquierda", ln=True)
                pdf.set_font("Arial", size=12)
                _imprimir_campo_pdf(pdf, "Voltaje", datos_paciente_dict.get("voltaje_izq"), " mV")
                _imprimir_campo_pdf(pdf, "Corriente", datos_paciente_dict.get("corriente_izq"), " mA")
                _imprimir_campo_pdf(pdf, "Contacto", datos_paciente_dict.get("contacto_izq"))
                _imprimir_campo_pdf(pdf, "Frecuencia", datos_paciente_dict.get("frecuencia_izq"), " Hz")
                _imprimir_campo_pdf(pdf, "Ancho de pulso", datos_paciente_dict.get("ancho_pulso_izq"), " µS")
                pdf.ln(2)
    
            if hay_parametros_dch:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Configuración Derecha", ln=True)
                pdf.set_font("Arial", size=12)
                _imprimir_campo_pdf(pdf, "Voltaje", datos_paciente_dict.get("voltaje_dch"), " mV")
                _imprimir_campo_pdf(pdf, "Corriente", datos_paciente_dict.get("corriente_dch"), " mA")
                _imprimir_campo_pdf(pdf, "Contacto", datos_paciente_dict.get("contacto_dch"))
                _imprimir_campo_pdf(pdf, "Frecuencia", datos_paciente_dict.get("frecuencia_dch"), " Hz")
                _imprimir_campo_pdf(pdf, "Ancho de pulso", datos_paciente_dict.get("ancho_pulso_dch"), " µS")
            pdf.ln(5)
        
    
        pdf.set_font("Arial", "B", 12)
        pdf.cell(30, 10, "Test", 1)
        pdf.cell(40, 10, "Frecuencia (Hz)", 1)
        pdf.cell(30, 10, "RMS", 1)
        pdf.cell(50, 10, "Amplitud (cm)", 1)
        pdf.ln(10)
    
        pdf.set_font("Arial", "", 12)
        for _, row in df.iterrows():
            pdf.cell(30, 10, row['Test'], 1)
            pdf.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
            pdf.cell(30, 10, f"{row['RMS (m/s2)']:.4f}", 1)
            pdf.cell(50, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
            pdf.ln(10)
    
        def limpiar_texto_para_pdf(texto):
            return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
        
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Interpretación clínica:", ln=True)
        pdf.set_font("Arial", size=10)
        texto_original = """
        Este informe analiza tres tipos de temblores: en reposo, postural y de acción.
    
        Los valores de referencia considerados son:
          Para las frecuencias (Hz):
        - Temblor Parkinsoniano: 3-6 Hz en reposo.
        - Temblor Esencial: 8-10 Hz en acción o postura.
    
          Para las amplitudes:
        - Mayores a 0.5 cm pueden ser clínicamente relevantes.
    
          Para el RMS (m/s2):
        - Normal/sano: menor a 0.5 m/s2.
        - PK leve: entre 0.5 y 1.5 m/s2.
        - TE o PK severo: mayor a 2 m/s2.
    
        Nota clínica: Los valores de referencia presentados a continuación se basan en literatura científica.
    
        """
        texto_limpio = limpiar_texto_para_pdf(texto_original)
        pdf.multi_cell(0, 8, texto_limpio)
        pdf.set_font("Arial", 'B', 12)
    
        if fig is not None:
            # Altura estimada que ocupará el gráfico (ajusta este valor si es necesario).
            altura_grafico = 150 # Altura de la imagen + título
                
            # Comprobar si queda suficiente espacio en la página actual
            # Se verifica si la posición actual + la altura del gráfico excede la altura de la página.
            if (pdf.get_y() + altura_grafico) > (pdf.h - 20):  # Se resta 20mm para un margen de seguridad
                pdf.add_page()
                
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Gráfico Comparativo de Amplitud de Temblor", ln=True, align="C")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name, format='png', bbox_inches='tight')
                pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)
     
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        return pdf_output


    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba en REPOSO</div>', unsafe_allow_html=True)
    reposo_file = st.file_uploader("", type=["csv"], key="reposo")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba POSTURAL</div>', unsafe_allow_html=True)
    postural_file = st.file_uploader("", type=["csv"], key="postural")

    st.markdown('<div class="prueba-titulo">Subir archivo CSV para prueba en ACCIÓN</div>', unsafe_allow_html=True)
    accion_file = st.file_uploader("", type=["csv"], key="accion")

    st.markdown("""
        <style>
        /* Ocultar el texto original de "Drag and drop file here" */
        div[data-testid="stFileUploaderDropzoneInstructions"] span {
            display: none !important;
        }

        /* Añadir nuestro propio texto arriba del botón */
        div[data-testid="stFileUploaderDropzoneInstructions"]::before {
            content: "Arrastrar archivo aquí";
            font-weight: bold;
            font-size: 16px;
            color: #444;
            display: block;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)


    if st.button("Iniciar análisis"):
        uploaded_files = {
            "Reposo": reposo_file,
            "Postural": postural_file,
            "Acción": accion_file,
        }

        if not any(uploaded_files.values()):
            st.warning("Por favor, sube al menos un archivo para iniciar el análisis.")
        else:
            resultados_globales = []
            datos_paciente_para_pdf = {} 
            ventanas_para_grafico = []
            min_ventanas_count = float('inf')
            
            primer_df_cargado = None
            for file_object in uploaded_files.values():
                if file_object is not None:
                    file_object.seek(0)
                    df = pd.read_csv(file_object, encoding='latin1', header=0)
                    primer_df_cargado = df
                    file_object.seek(0)
                    break
            
            if primer_df_cargado is not None:
                datos_paciente_para_pdf = extraer_datos_paciente(primer_df_cargado)

                # CONVERSIÓN A MINÚSCULAS PARA EL PDF
                if datos_paciente_para_pdf.get("sexo"):
                    datos_paciente_para_pdf["sexo"] = datos_paciente_para_pdf["sexo"].lower()
                if datos_paciente_para_pdf.get("mano_medida"):
                    datos_paciente_para_pdf["mano_medida"] = datos_paciente_para_pdf["mano_medida"].lower()
                if datos_paciente_para_pdf.get("dedo_medido"):
                    datos_paciente_para_pdf["dedo_medido"] = datos_paciente_para_pdf["dedo_medido"].lower()
                if datos_paciente_para_pdf.get("diagnostico"):
                    datos_paciente_para_pdf["diagnostico"] = datos_paciente_para_pdf["diagnostico"].lower()
                if datos_paciente_para_pdf.get("tipo"):
                    datos_paciente_para_pdf["tipo"] = datos_paciente_para_pdf["tipo"].lower()

            for test, file_object in uploaded_files.items():
                if file_object is not None:
                    file_object.seek(0)
                    df = pd.read_csv(file_object, encoding='latin1', header=0)

                    df_promedio, df_ventanas = analizar_temblor_por_ventanas_resultante(df, fs=100)

                    if not df_promedio.empty:
                        fila = df_promedio.iloc[0].to_dict()
                        fila['Test'] = test
                        resultados_globales.append(fila)

                    if not df_ventanas.empty:
                        df_ventanas_copy = df_ventanas.copy()
                        df_ventanas_copy["Test"] = test
                        ventanas_para_grafico.append(df_ventanas_copy)
                        if len(df_ventanas_copy) < min_ventanas_count:
                            min_ventanas_count = len(df_ventanas_copy)

            if ventanas_para_grafico:
                fig, ax = plt.subplots(figsize=(10, 6))
                for df in ventanas_para_grafico:
                    test_name = df["Test"].iloc[0]
                    if min_ventanas_count != float('inf') and len(df) > min_ventanas_count:
                        df_to_plot = df.iloc[:min_ventanas_count].copy()
                    else:
                        df_to_plot = df.copy()
                    
                    df_to_plot["Tiempo (segundos)"] = df_to_plot["Ventana"] * ventana_duracion_seg
                    ax.plot(df_to_plot["Tiempo (segundos)"], df_to_plot["Amplitud Temblor (cm)"], label=f"{test_name}")

                ax.set_title("Amplitud de Temblor por Ventana de Tiempo (Comparación Visual)")
                ax.set_xlabel("Tiempo (segundos)")
                ax.set_ylabel("Amplitud (cm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.warning("No se generaron datos de ventanas para el gráfico.")

            if resultados_globales:
                df_resultados_final = pd.DataFrame(resultados_globales)
                st.subheader("Resultados del Análisis de Temblor")
                st.dataframe(df_resultados_final.set_index('Test'))

                pdf_output = generar_pdf(
                    datos_paciente_para_pdf,
                    df_resultados_final,
                    nombre_archivo="informe_temblor.pdf",
                    fig=fig
                )

                st.download_button("📄 Descargar informe PDF", pdf_output, file_name="informe_temblor.pdf")
                st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
            else:
                st.warning("No se encontraron datos suficientes para el análisis.")
                
# ------------------ MÓDULO 2: COMPARACIÓN DE MEDICIONES -------------------------------

if opcion == "2️⃣ Comparación de mediciones":
    st.title("📊 Comparación de Mediciones")
    
    def generar_pdf_comparativo(datos_paciente_dict, df_comparativo, graficos, nombre_archivo="informe_comparativo.pdf"):
        fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Informe Comparativo de Temblor", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
            if valor is not None and str(valor).strip() != "" and pd.notna(valor) and str(valor).lower() != "no especificado":
                pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos del Paciente", ln=True)
        pdf.set_font("Arial", size=12)
        _imprimir_campo_pdf(pdf, "Nombre", datos_paciente_dict.get("nombre"))
        _imprimir_campo_pdf(pdf, "Apellido", datos_paciente_dict.get("apellido"))
        _imprimir_campo_pdf(pdf, "Edad", datos_paciente_dict.get("edad"))
        _imprimir_campo_pdf(pdf, "Sexo", datos_paciente_dict.get("sexo"))
        _imprimir_campo_pdf(pdf, "Mano", datos_paciente_dict.get("mano_medida"))
        _imprimir_campo_pdf(pdf, "Dedo", datos_paciente_dict.get("dedo_medido"))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Resultados Comparativos", ln=True)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, "Condicion", 1)
        pdf.cell(40, 10, "Frecuencia (Hz)", 1)
        pdf.cell(40, 10, "Amplitud (cm)", 1)
        pdf.ln(10)
    
        pdf.set_font("Arial", "", 12)
        for _, row in df_comparativo.iterrows():
            pdf.cell(50, 10, str(row['Condicion']), 1)
            pdf.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
            pdf.cell(40, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
            pdf.ln(10)
        
        pdf.ln(10)
        
        # Iterar sobre los gráficos generados para el PDF
        for titulo_grafico, fig in graficos.items():
            pdf.set_font("Arial", 'B', 14)
            
            # Verificar si hay espacio suficiente en la página antes de agregar el gráfico
            altura_grafico = 100 # Altura estimada del gráfico en mm
            if (pdf.get_y() + altura_grafico) > (pdf.h - 20):
                pdf.add_page()
            
            pdf.cell(0, 10, titulo_grafico, ln=True, align="C")
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name, format='png', bbox_inches='tight')
                pdf.image(tmpfile.name, x=25, w=150)
                os.remove(tmpfile.name)
        
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        return pdf_output

    uploaded_files_comparacion = st.file_uploader(
        "Sube los archivos CSV para comparar",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files_comparacion:
        if st.button("Iniciar Comparación"):
            resultados_comparativos = []
            etiquetas = []
            ventanas_para_graficos = {}
            graficos = {}

            primer_df_cargado = None
            if uploaded_files_comparacion:
                for file_object in uploaded_files_comparacion:
                    file_object.seek(0)
                    primer_df_cargado = pd.read_csv(file_object, encoding='latin1', header=0)
                    file_object.seek(0)
                    break

            if primer_df_cargado is not None:
                datos_paciente_para_pdf = extraer_datos_paciente(primer_df_cargado)
                
                # Conversión a minúsculas
                if datos_paciente_para_pdf.get("sexo"):
                    datos_paciente_para_pdf["sexo"] = datos_paciente_para_pdf["sexo"].lower()
                if datos_paciente_para_pdf.get("mano_medida"):
                    datos_paciente_para_pdf["mano_medida"] = datos_paciente_para_pdf["mano_medida"].lower()
                if datos_paciente_para_pdf.get("dedo_medido"):
                    datos_paciente_para_pdf["dedo_medido"] = datos_paciente_para_pdf["dedo_medido"].lower()
                
            for i, file_object in enumerate(uploaded_files_comparacion):
                file_object.seek(0)
                df = pd.read_csv(file_object, encoding='latin1', header=0)
                
                nombre_archivo = file_object.name
                condicion = st.text_input(f"Ingresa el nombre para la medición '{nombre_archivo}'", key=f"condicion_{i}")
                
                if not condicion:
                    condicion = f"Medición {i+1}"
                
                df_promedio, df_ventanas = analizar_temblor_por_ventanas_resultante(df, fs=100)

                if not df_promedio.empty:
                    df_promedio['Condicion'] = condicion
                    resultados_comparativos.append(df_promedio)
                
                if not df_ventanas.empty:
                    df_ventanas_copy = df_ventanas.copy()
                    df_ventanas_copy['Condicion'] = condicion
                    ventanas_para_graficos[condicion] = df_ventanas_copy

            if resultados_comparativos:
                df_final = pd.concat(resultados_comparativos, ignore_index=True)
                st.subheader("Resultados Comparativos")
                st.dataframe(df_final.set_index('Condicion'))

                # Gráfico de barras comparativo de Amplitud
                fig_amplitud, ax_amplitud = plt.subplots()
                amplitudes = df_final['Amplitud Temblor (cm)'].values
                condiciones = df_final['Condicion'].values
                ax_amplitud.bar(condiciones, amplitudes, color=['skyblue', 'salmon', 'lightgreen'])
                ax_amplitud.set_title("Amplitud de Temblor por Condición")
                ax_amplitud.set_ylabel("Amplitud (cm)")
                ax_amplitud.grid(axis='y', linestyle='--')
                st.pyplot(fig_amplitud)
                graficos["Gráfico de Amplitud (cm)"] = fig_amplitud

                # Gráfico de barras comparativo de Frecuencia
                fig_frecuencia, ax_frecuencia = plt.subplots()
                frecuencias = df_final['Frecuencia Dominante (Hz)'].values
                ax_frecuencia.bar(condiciones, frecuencias, color=['skyblue', 'salmon', 'lightgreen'])
                ax_frecuencia.set_title("Frecuencia Dominante por Condición")
                ax_frecuencia.set_ylabel("Frecuencia (Hz)")
                ax_frecuencia.grid(axis='y', linestyle='--')
                st.pyplot(fig_frecuencia)
                graficos["Gráfico de Frecuencia (Hz)"] = fig_frecuencia

                pdf_output = generar_pdf_comparativo(datos_paciente_para_pdf, df_final, graficos)

                st.download_button("📄 Descargar Informe Comparativo", pdf_output, file_name="informe_comparativo.pdf")
                st.info("El archivo se descargará en tu carpeta de descargas.")
            else:
                st.warning("No se encontraron datos suficientes para el análisis comparativo.")

    st.markdown('<div class="prueba-titulo">Sube uno o más archivos para comparar.</div>', unsafe_allow_html=True)
            
# ------------------ MÓDULO 3: DIAGNÓSTICO TENTATIVO --------------------
elif opcion == "3️⃣ Diagnóstico tentativo":
    st.title("🩺 Diagnóstico Tentativo")
    st.markdown("### Cargar archivos CSV para el Diagnóstico")

    # Definiciones de funciones para el PDF, fuera del botón
    def extraer_datos_paciente(df_csv):
        datos_paciente = {
            "Nombre": df_csv.loc[0, 'Nombre'] if 'Nombre' in df_csv.columns else 'No especificado',
            "Apellido": df_csv.loc[0, 'Apellido'] if 'Apellido' in df_csv.columns else 'No especificado',
            "Edad": df_csv.loc[0, 'Edad'] if 'Edad' in df_csv.columns else 'No especificada',
            "Sexo": df_csv.loc[0, 'Sexo'] if 'Sexo' in df_csv.columns else 'No especificado',
            "Diagnostico": df_csv.loc[0, 'Diagnostico'] if 'Diagnostico' in df_csv.columns else 'No especificado',
            "Tipo": df_csv.loc[0, 'Tipo'] if 'Tipo' in df_csv.columns else 'No especificado',
            "Antecedente": df_csv.loc[0, 'Antecedente'] if 'Antecedente' in df_csv.columns else 'No especificado',
            "Medicacion": df_csv.loc[0, 'Medicacion'] if 'Medicacion' in df_csv.columns else 'No especificado',
        }
        return datos_paciente

    def extraer_datos_estimulacion(df_csv):
        metadata_dict = {}
        column_map = {
            # Solo se extrae mano y dedo, ya que no hay estimulación en este caso
            "Mano": "Mano",
            "Dedo": "Dedo"
        }
        for csv_col, pdf_label in column_map.items():
            if csv_col in df_csv.columns:
                value = df_csv.loc[0, csv_col]
                metadata_dict[pdf_label] = value
        return metadata_dict

    def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
        if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
            pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

    def imprimir_parametros_y_config(pdf_obj, parametros_dict, titulo):
        pdf_obj.set_font("Arial", 'B', 12)
        pdf_obj.cell(0, 10, titulo, ln=True)
        pdf_obj.set_font("Arial", size=10)
        
        # Solo se imprimen la mano y el dedo, el resto de campos no existen en este contexto
        parametros_a_imprimir_con_unidad = {
            "Mano": "", "Dedo": ""
        }
        for param_key, unit in parametros_dict.items():
            value = parametros_dict.get(param_key)
            if value is not None and str(value).strip() != "" and str(value).lower() != "no especificado":
                pdf_obj.cell(200, 10, f"{param_key}: {value}{unit}", ln=True)
        pdf_obj.ln(5)

    def generar_pdf(paciente_data, estimulacion_data, resultados_df, prediccion_texto, graficos_paths):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Informe de Diagnóstico de Temblor", ln=True, align="C")
        pdf.ln(5)
        
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos del Paciente", ln=True)
        pdf.set_font("Arial", size=12)
        _imprimir_campo_pdf(pdf, "Nombre", paciente_data.get("Nombre"))
        _imprimir_campo_pdf(pdf, "Apellido", paciente_data.get("Apellido"))
        _imprimir_campo_pdf(pdf, "Edad", paciente_data.get("Edad"))
        _imprimir_campo_pdf(pdf, "Sexo", paciente_data.get("Sexo"))
        _imprimir_campo_pdf(pdf, "Diagnóstico", paciente_data.get("Diagnostico"))
        _imprimir_campo_pdf(pdf, "Tipo", paciente_data.get("Tipo"))
        _imprimir_campo_pdf(pdf, "Antecedente", paciente_data.get("Antecedente"))
        _imprimir_campo_pdf(pdf, "Medicacion", paciente_data.get("Medicacion"))
        pdf.ln(5)
        
        imprimir_parametros_y_config(pdf, estimulacion_data, "Configuración de la Medición")
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Resultados del Análisis", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(30, 10, "Test", 1)
        pdf.cell(40, 10, "Frecuencia (Hz)", 1)
        pdf.cell(30, 10, "RMS", 1)
        pdf.cell(50, 10, "Amplitud (cm)", 1)
        pdf.ln(10)
        pdf.set_font("Arial", "", 10)
        for _, row in resultados_df.iterrows():
            pdf.cell(30, 10, row['Test'], 1)
            pdf.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
            pdf.cell(30, 10, f"{row['RMS (m/s2)']:.4f}", 1)
            pdf.cell(50, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
            pdf.ln(10)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Diagnóstico (Predicción)", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, prediccion_texto)
        pdf.ln(5)

        for i, img_path in enumerate(graficos_paths):
            if os.path.exists(img_path):
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, f"Gráfico {i+1}", ln=True, align="C")
                pdf.image(img_path, x=15, w=180)
            else:
                pdf.cell(0, 10, f"Error: No se pudo cargar el gráfico {i+1}", ln=True)
        
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        return pdf_output

    prediccion_reposo_file = st.file_uploader("Archivo de REPOSO para Diagnóstico", type="csv", key="prediccion_reposo")
    prediccion_postural_file = st.file_uploader("Archivo de POSTURAL para Diagnóstico", type="csv", key="prediccion_postural")
    prediccion_accion_file = st.file_uploader("Archivo de ACCION para Diagnóstico", type="csv", key="prediccion_accion")

    st.markdown("""
        <style>
        div[data-testid="stFileUploaderDropzoneInstructions"] span {
            display: none !important;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"]::before {
            content: "Arrastrar archivo aquí";
            font-weight: bold;
            font-size: 16px;
            color: #444;
            display: block;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("Realizar Diagnóstico"):
        prediccion_files_correctas = {
            "Reposo": prediccion_reposo_file,
            "Postural": prediccion_postural_file,
            "Acción": prediccion_accion_file
        }

        any_file_uploaded = any(file is not None for file in prediccion_files_correctas.values())

        if not any_file_uploaded:
            st.warning("Por favor, sube al menos un archivo CSV para realizar el diagnóstico.")
        else:
            avg_tremor_metrics = {}
            datos_paciente = {}
            datos_estimulacion = {}
            first_file_processed = False

            for test_type, uploaded_file in prediccion_files_correctas.items():
                if uploaded_file is not None:
                    uploaded_file.seek(0)
                    df_current_test = pd.read_csv(uploaded_file, encoding='latin1')

                    if not first_file_processed:
                        datos_paciente = extraer_datos_paciente(df_current_test)
                        datos_estimulacion = extraer_datos_estimulacion(df_current_test)
                        first_file_processed = True

                    df_promedio, _ = analizar_temblor_por_ventanas_resultante(df_current_test, fs=100)
                    
                    if not df_promedio.empty:
                        avg_tremor_metrics[test_type] = df_promedio.iloc[0].to_dict()
                    else:
                        st.warning(f"No se pudieron calcular métricas de temblor para {test_type}. Se usarán NaN.")
                        avg_tremor_metrics[test_type] = {
                            'Frecuencia Dominante (Hz)': np.nan,
                            'RMS (m/s2)': np.nan,
                            'Amplitud Temblor (cm)': np.nan
                        }

            if not avg_tremor_metrics:
                st.error("No se pudo procesar ningún archivo cargado para el diagnóstico. Asegúrate de que los archivos contengan datos válidos.")
            else:
                st.subheader("Datos de Temblor Calculados para el Diagnóstico:")
                # Se crea el DataFrame y se convierte el índice en una columna 'Test'
                df_metrics_display = pd.DataFrame.from_dict(avg_tremor_metrics, orient='index').reset_index()
                df_metrics_display = df_metrics_display.rename(columns={'index': 'Test'})
                st.dataframe(df_metrics_display)

                data_for_model = {}
                edad_val = datos_paciente.get('Edad', np.nan)
                try:
                    data_for_model['edad'] = int(float(edad_val)) if pd.notna(edad_val) else np.nan
                except (ValueError, TypeError):
                    data_for_model['edad'] = np.nan

                data_for_model['sexo'] = datos_paciente.get('Sexo', 'no especificado').lower()
                data_for_model['mano_medida'] = datos_estimulacion.get('Mano', 'no especificada').lower()
                data_for_model['dedo_medido'] = datos_estimulacion.get('Dedo', 'no especificado').lower()

                feature_name_map = {
                    "Reposo": "Reposo",
                    "Postural": "Postural",
                    "Acción": "Accion"
                }

                for original_test_type, model_feature_prefix in feature_name_map.items():
                    metrics = avg_tremor_metrics.get(original_test_type, {})
                    data_for_model[f'Frec_{model_feature_prefix}'] = metrics.get('Frecuencia Dominante (Hz)', np.nan)
                    data_for_model[f'RMS_{model_feature_prefix}'] = metrics.get('RMS (m/s2)', np.nan)
                    data_for_model[f'Amp_{model_feature_prefix}'] = metrics.get('Amplitud Temblor (cm)', np.nan)

                expected_features_for_model = [
                    'edad',
                    'Frec_Reposo', 'RMS_Reposo', 'Amp_Reposo',
                    'Frec_Postural', 'RMS_Postural', 'Amp_Postural',
                    'Frec_Accion', 'RMS_Accion', 'Amp_Accion',
                    'sexo', 'mano_medida', 'dedo_medido'
                ]
                df_for_prediction = pd.DataFrame([data_for_model])[expected_features_for_model]

                st.subheader("DataFrame preparado para el Modelo de Predicción:")
                st.dataframe(df_for_prediction)

                model_filename = 'tremor_prediction_model_V2.joblib'
                prediccion_final = "No se pudo realizar el diagnóstico."
                
                try:
                    modelo_cargado = joblib.load(model_filename)
                    prediction = modelo_cargado.predict(df_for_prediction)
                    prediccion_final = prediction[0]

                    st.subheader("Resultados del Diagnóstico:")
                    st.success(f"El diagnóstico tentativo es: **{prediccion_final}**")

                    if hasattr(modelo_cargado, 'predict_proba'):
                        probabilities = modelo_cargado.predict_proba(df_for_prediction)
                        st.write("Probabilidades por clase:")
                        if hasattr(modelo_cargado, 'classes_'):
                            for i, class_label in enumerate(modelo_cargado.classes_):
                                st.write(f"- **{class_label}**: {probabilities[0][i]*100:.2f}%")
                        else:
                            st.info("El modelo no tiene el atributo 'classes_'. No se pueden mostrar las etiquetas de clase.")
                except FileNotFoundError:
                    st.error(f"Error: El archivo del modelo '{model_filename}' no se encontró.")
                    st.error("Asegúrate de que esté en la misma carpeta que este script.")
                except Exception as e:
                    st.error(f"Ocurrió un error al usar el modelo: {e}")
                    st.error("Verifica que el DataFrame `df_for_prediction` coincida con lo que espera el modelo.")

                # Generación de gráficos y guardado en archivos temporales
                all_ventanas_for_plot = []
                current_min_ventanas = float('inf')
                graficos_paths = []

                for test_type, uploaded_file in prediccion_files_correctas.items():
                    if uploaded_file is not None:
                        uploaded_file.seek(0)
                        df_temp = pd.read_csv(uploaded_file, encoding='latin1')
                        _, df_ventanas_temp = analizar_temblor_por_ventanas_resultante(df_temp, fs=100)

                        if not df_ventanas_temp.empty:
                            df_ventanas_temp_copy = df_ventanas_temp.copy()
                            df_ventanas_temp_copy["Test"] = test_type
                            all_ventanas_for_plot.append(df_ventanas_temp_copy)

                            if len(df_ventanas_temp_copy) < current_min_ventanas:
                                current_min_ventanas = len(df_ventanas_temp_copy)

                if all_ventanas_for_plot:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for df_plot in all_ventanas_for_plot:
                        test_name = df_plot["Test"].iloc[0]
                        if current_min_ventanas != float('inf') and len(df_plot) > current_min_ventanas:
                            df_to_plot = df_plot.iloc[:current_min_ventanas].copy()
                        else:
                            df_to_plot = df_plot.copy()

                        df_to_plot["Tiempo (segundos)"] = df_to_plot["Ventana"] * ventana_duracion_seg
                        ax.plot(df_to_plot["Tiempo (segundos)"], df_to_plot["Amplitud Temblor (cm)"], label=f"{test_name}")
                    
                    ax.set_title("Amplitud de Temblor por Ventana de Tiempo")
                    ax.set_xlabel("Tiempo (segundos)")
                    ax.set_ylabel("Amplitud (cm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # Guardar el gráfico en un archivo temporal
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                        graficos_paths.append(tmp_img.name)
                    plt.close(fig)
                else:
                    st.warning("No hay suficientes datos de ventanas para graficar los archivos de predicción.")

                # Generar el PDF y mostrar el botón de descarga
                pdf_output = generar_pdf(
                    datos_paciente, 
                    datos_estimulacion, 
                    df_metrics_display, 
                    f"El diagnóstico tentativo es: {prediccion_final}",
                    graficos_paths
                )

                st.download_button(
                    label="Descargar Informe PDF",
                    data=pdf_output.getvalue(),
                    file_name="informe_diagnostico_temblor.pdf",
                    mime="application/pdf"
                )
                
                # Limpiar los archivos temporales después de la descarga
                for path in graficos_paths:
                    if os.path.exists(path):
                        os.remove(path)
