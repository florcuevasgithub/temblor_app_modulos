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

# --------- Funciones compartidas ----------
# Función para extraer datos del paciente de un DataFrame
def extraer_datos_paciente(df):
    """
    Extrae datos personales del paciente desde un DataFrame,
    sin modificar las columnas originales del DataFrame.
    """
    if df is None or df.empty:
        return {}
    
    # El primer registro contiene todos los metadatos
    df_metadata = df.iloc[0]

    datos = {
        "sexo": str(df_metadata.get('Sexo', 'No especificado')).strip(),
        "edad": int(float(str(df_metadata.get('Edad', 0)).replace(',', '.'))) if pd.notna(df_metadata.get('Edad')) else 0,
        "mano_medida": str(df_metadata.get('Mano', 'No especificada')).strip(),
        "dedo_medido": str(df_metadata.get('Dedo', 'No especificado')).strip(),
        "Nombre": str(df_metadata.get('Nombre', '')).strip(),
        "Apellido": str(df_metadata.get('Apellido', '')).strip(),
        "Diagnostico": str(df_metadata.get('Diagnostico', '')).strip(),
        "Antecedente": str(df_metadata.get('Antecedente', '')).strip(),
        "Medicacion": str(df_metadata.get('Medicacion', '')).strip(),
        "Tipo": str(df_metadata.get('Tipo', '')).strip(),
        "ECP": str(df_metadata.get('ECP', '')).strip(),
        "GPI": str(df_metadata.get('GPI', '')).strip(),
        "NST": str(df_metadata.get('NST', '')).strip(),
        "Polaridad": str(df_metadata.get('Polaridad', '')).strip(),
        "Duracion": float(str(df_metadata.get('Duracion [ms]', 0)).replace(',', '.')) if pd.notna(df_metadata.get('Duracion [ms]')) else None,
        "Pulso": float(str(df_metadata.get('Pulso [µS]', 0)).replace(',', '.')) if pd.notna(df_metadata.get('Pulso [µS]')) else None,
        "Corriente": float(str(df_metadata.get('Corriente [mA]', 0)).replace(',', '.')) if pd.notna(df_metadata.get('Corriente [mA]')) else None,
        "Voltaje": float(str(df_metadata.get('Voltaje [mV]', 0)).replace(',', '.')) if pd.notna(df_metadata.get('Voltaje [mV]')) else None,
        "Frecuencia": float(str(df_metadata.get('Frecuencia [Hz]', 0)).replace(',', '.')) if pd.notna(df_metadata.get('Frecuencia [Hz]')) else None
    }
    
    return datos


def filtrar_temblor(signal, fs=100):
    b, a = butter(N=4, Wn=[1, 15], btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def q_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),         2*(x*y - z*w),           2*(x*z + y*w)],
        [2*(x*y + z*w),                 1 - 2*(x**2 + z**2),       2*(y*z - x*w)],
        [2*(x*z - y*w),                 2*(y*z + x*w),           1 - 2*(x**2 + y**2)]
    ])

def analizar_temblor_por_ventanas_resultante(df, fs=100, ventana_seg=ventana_duracion_seg):
    required_cols = ['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']
    df_senial = df[required_cols].dropna() # Dropping NaNs here might reduce the total length
    acc = df_senial[['Acel_X', 'Acel_Y', 'Acel_Z']].to_numpy()
    gyr = np.radians(df_senial[['GiroX', 'GiroY', 'GiroZ']].to_numpy())
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

        varianza = np.var(segmento)
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


# ------------------ Modo principal --------------------

st.title("🧠 Análisis de Temblor")
opcion = st.sidebar.radio("Selecciona una opción:", ["1️⃣ Análisis de una medición", "2️⃣ Comparación de mediciones", "3️⃣ Diagnóstico tentativo"])
if st.sidebar.button("🔄 Nuevo análisis"):
    manejar_reinicio()
    
#--------------- Módulo 1: Análisis de una medicion-----
if opcion == "1️⃣ Análisis de una medición":
    st.title("📈​ Análisis de una Medición")

    # --------- Funciones auxiliares ----------

    def diagnosticar(df):
        def max_amp(test):
            fila = df[df['Test'] == test]
            return fila['Amplitud Temblor (cm)'].max() if not fila.empty else 0

        def mean_freq(test):
            fila = df[df['Test'] == test]
            return fila['Frecuencia Dominante (Hz)'].mean() if not fila.empty else 0

        if max_amp('Reposo') > 0.3 and 3 <= mean_freq('Reposo') <= 6.5:
            return "Probable Parkinson"
        elif (max_amp('Postural') > 0.3 or max_amp('Acción') > 0.3) and (7.5 <= mean_freq('Postural') <= 12 or 7.5 <= mean_freq('Acción') <= 12):
            return "Probable Temblor Esencial"
        else:
            return "Temblor dentro de parámetros normales"

    # --- Función generar_pdf modificada para aceptar un diccionario de datos del paciente ---
    def generar_pdf(datos_paciente_dict, df, nombre_archivo="informe_temblor.pdf", diagnostico="", fig=None):

        fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        # Helper para imprimir campos solo si tienen valor
        def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
            if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
                pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos del Paciente", ln=True)
        pdf.set_font("Arial", size=12)

        _imprimir_campo_pdf(pdf, "Nombre", datos_paciente_dict.get("Nombre"))
        _imprimir_campo_pdf(pdf, "Apellido", datos_paciente_dict.get("Apellido"))
        
        # Manejo especial para Edad para asegurar que sea un número si es posible
        edad_val = datos_paciente_dict.get("Edad")
        edad_str_to_print = None
        try:
            if edad_val is not None and str(edad_val).strip() != "":
                edad_int = int(float(edad_val))
                edad_str_to_print = str(edad_int)
        except ValueError:
            pass # Si falla la conversión, no se imprimirá

        _imprimir_campo_pdf(pdf, "Edad", edad_str_to_print)
        _imprimir_campo_pdf(pdf, "Sexo", datos_paciente_dict.get("Sexo"))
        _imprimir_campo_pdf(pdf, "Diagnóstico", datos_paciente_dict.get("Diagnostico"))
        _imprimir_campo_pdf(pdf, "Tipo", datos_paciente_dict.get("Tipo")) # Agregado "Tipo"
        _imprimir_campo_pdf(pdf, "Mano", datos_paciente_dict.get("Mano"))
        _imprimir_campo_pdf(pdf, "Dedo", datos_paciente_dict.get("Dedo"))
        _imprimir_campo_pdf(pdf, "Antecedente", datos_paciente_dict.get("Antecedente"))
        _imprimir_campo_pdf(pdf, "Medicacion", datos_paciente_dict.get("Medicacion"))
        
        pdf.ln(5) # Espacio después de los datos del paciente

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos de Configuración", ln=True)
        pdf.set_font("Arial", size=12)

        _imprimir_campo_pdf(pdf, "Tipo", datos_paciente_dict.get("Tipo"))
        _imprimir_campo_pdf(pdf, "ECP", datos_paciente_dict.get("ECP"))
        _imprimir_campo_pdf(pdf, "GPI", datos_paciente_dict.get("GPI"))
        _imprimir_campo_pdf(pdf, "NST", datos_paciente_dict.get("NST"))
        _imprimir_campo_pdf(pdf, "Polaridad", datos_paciente_dict.get("Polaridad"))
        _imprimir_campo_pdf(pdf, "Duracion [ms]", datos_paciente_dict.get("Duracion"))
        _imprimir_campo_pdf(pdf, "Pulso [µS]", datos_paciente_dict.get("Pulso"))
        _imprimir_campo_pdf(pdf, "Corriente  [mA]", datos_paciente_dict.get("Corriente"))
        _imprimir_campo_pdf(pdf, "Voltaje [mV]", datos_paciente_dict.get("Voltaje"))
        _imprimir_campo_pdf(pdf, "Frecuencia [Hz]", datos_paciente_dict.get("Frecuencia"))

        pdf.ln(5) # Espacio después de los datos de Configuración
        

        # --- SECCIÓN: Parámetros de Estimulación (Configuración) ---
        # Definir los parámetros de estimulación y sus unidades
        parametros_estimulacion = {
            "ECP": "", "GPI": "", "NST": "", "Polaridad": "",
            "Duracion": " ms", "Pulso": " µs", "Corriente": " mA",
            "Voltaje": " V", "Frecuencia": " Hz"
        }
        
        # Verificar si hay al menos un parámetro de estimulación presente para imprimir el título
        hay_parametros_estimulacion = False
        for param_key in parametros_estimulacion.keys():
            if datos_paciente_dict.get(param_key) is not None and str(datos_paciente_dict.get(param_key)).strip() != "":
                hay_parametros_estimulacion = True
                break

        if hay_parametros_estimulacion:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Configuración", ln=True) # Título cambiado a "Configuración"
            pdf.set_font("Arial", size=12)
            for param_key, unit in parametros_estimulacion.items():
                _imprimir_campo_pdf(pdf, param_key, datos_paciente_dict.get(param_key), unit)
            pdf.ln(5)
        # --- FIN SECCIÓN ---

        pdf.cell(200, 10, f"Fecha y hora del análisis: {fecha_hora}", ln=True) # Siempre se imprime la fecha/hora

        pdf.ln(10)
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
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name, format='png', bbox_inches='tight')
                pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

        pdf.output(nombre_archivo)


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


    uploaded_files = {
        "Reposo": reposo_file,
        "Postural": postural_file,
        "Acción": accion_file,
    }

    # Inicializa estas variables FUERA del bloque del botón.
    resultados_globales = []
    datos_paciente_para_pdf = {} # Cambiado a diccionario para datos del paciente
    ventanas_para_grafico = []
    min_ventanas_count = float('inf')
    fig = None  

    if st.button("Iniciar análisis"):
        # MODIFICACIÓN: Añadir encoding='latin1' a la lectura del CSV
        mediciones_tests = {}
        for test, file in uploaded_files.items():
            if file is not None:
                file.seek(0) # Reset file pointer for re-reading
                mediciones_tests[test] = pd.read_csv(file, encoding='latin1')

        if not mediciones_tests:
            st.warning("Por favor, sube al menos un archivo para iniciar el análisis.")
        else:
            # Extraer datos del paciente y de estimulación de la primera medición (o de la que se cargue primero)
            # Asegurarse de que solo se extraigan una vez y de un archivo válido
            primer_df_cargado = None
            for test, datos in mediciones_tests.items():
                if datos is not None and not datos.empty:
                    primer_df_cargado = datos
                    break

            if primer_df_cargado is not None:
                # Extraer todos los datos del paciente y configuración usando la función actualizada
                datos_paciente_para_pdf = extraer_datos_paciente(primer_df_cargado)
            
            # Procesar cada test
            for test, datos in mediciones_tests.items():
                df_promedio, df_ventanas = analizar_temblor_por_ventanas_resultante(datos, fs=100)

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
                diagnostico_auto = diagnosticar(df_resultados_final)

                st.subheader("Resultados del Análisis de Temblor")
                st.dataframe(df_resultados_final.set_index('Test'))

                # --- Llamada a generar_pdf con el diccionario de datos del paciente ---
                generar_pdf(
                    datos_paciente_para_pdf, # Ahora pasamos el diccionario
                    df_resultados_final,
                    nombre_archivo="informe_temblor.pdf",
                    diagnostico=diagnostico_auto,
                    fig=fig
                )

                with open("informe_temblor.pdf", "rb") as f:
                    st.download_button("📄 Descargar informe PDF", f, file_name="informe_temblor.pdf")
                    st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
            else:
                st.warning("No se encontraron datos suficientes para el análisis.")
                
# ------------------ MÓDULO 2: COMPARACIÓN DE MEDICIONES --------------------

elif opcion == "2️⃣ Comparación de mediciones":
    st.title("📊 Comparación de Mediciones")

    def extraer_datos_paciente(df_csv):
        # Esta función ahora no extrae la mano ni el dedo
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
        # Mapea los nombres de columna de tu CSV a los nombres que quieres en el PDF
        column_map = {
            "DBS": "DBS", 
            "Nucleo": "Nucleo",
            "Voltaje [mV]_izq": "Voltaje_izq", 
            "Corriente [mA]_izq": "Corriente_izq",
            "Contacto_izq": "Contacto_izq", 
            "Frecuencia [Hz]_izq": "Frecuencia_izq",
            "Ancho de pulso [µS]_izq": "Pulso_izq",
            "Voltaje [mV]_dch": "Voltaje_dch", 
            "Corriente [mA]_dch": "Corriente_dch",
            "Contacto_dch": "Contacto_dch", 
            "Frecuencia [Hz]_dch": "Frecuencia_dch",
            "Ancho de pulso [µS]_dch": "Pulso_dch",
            "Mano": "Mano",
            "Dedo": "Dedo"
        }
        
        for csv_col, pdf_label in column_map.items():
            if csv_col in df_csv.columns:
                value = df_csv.loc[0, csv_col]
                metadata_dict[pdf_label] = value
        return metadata_dict

    st.markdown("### Cargar archivos de la **medición 1**")
    config1_archivos = {
        "Reposo": st.file_uploader("Archivo de REPOSO medición 1", type="csv", key="reposo1"),
        "Postural": st.file_uploader("Archivo de POSTURAL medición 1", type="csv", key="postural1"),
        "Acción": st.file_uploader("Archivo de ACCION medición 1", type="csv", key="accion1")
    }

    st.markdown("### Cargar archivos de la **medición 2**")
    config2_archivos = {
        "Reposo": st.file_uploader("Archivo de REPOSO medición 2", type="csv", key="reposo2"),
        "Postural": st.file_uploader("Archivo de POSTURAL medición 2", type="csv", key="postural2"),
        "Acción": st.file_uploader("Archivo de ACCION medición 2", type="csv", key="accion2")
    }
    
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

    def analizar_configuracion(archivos, fs=100):
        resultados = []
        for test, archivo in archivos.items():
            if archivo is not None:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding='latin1')
                df_promedio, df_ventana = analizar_temblor_por_ventanas_resultante(df, fs=fs)
                if isinstance(df_ventana, pd.DataFrame) and not df_ventana.empty:
                    prom = df_promedio.iloc[0] if not df_promedio.empty else None
                    if prom is not None:
                        freq = prom['Frecuencia Dominante (Hz)']
                        amp = prom['Amplitud Temblor (cm)']
                        rms = prom['RMS (m/s2)']
                        resultados.append({
                            'Test': test,
                            'Frecuencia Dominante (Hz)': round(freq, 2),
                            'RMS (m/s2)': round(rms, 4),
                            'Amplitud Temblor (cm)': round(amp, 2)
                        })
        return pd.DataFrame(resultados)

    if st.button("Comparar Mediciones"):
        archivos_cargados = all([
            config1_archivos[test] is not None and config2_archivos[test] is not None
            for test in ["Reposo", "Postural", "Acción"]
        ])

        if not archivos_cargados:
            st.warning("Por favor, cargue los 3 archivos para ambas mediciones.")
        else:
            df_config1_meta = pd.read_csv(config1_archivos["Reposo"], encoding='latin1')
            df_config2_meta = pd.read_csv(config2_archivos["Reposo"], encoding='latin1')

            datos_paciente = extraer_datos_paciente(df_config1_meta)
            config1_params = extraer_datos_estimulacion(df_config1_meta)
            config2_params = extraer_datos_estimulacion(df_config2_meta)

            df_resultados_config1 = analizar_configuracion(config1_archivos)
            df_resultados_config2 = analizar_configuracion(config2_archivos)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Informe Comparativo de Mediciones", ln=True, align="C")

            pdf.set_font("Arial", size=10)
            pdf.ln(10)
            pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)
            
            def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
                if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
                    pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Datos del Paciente", ln=True)
            pdf.set_font("Arial", size=12)

            _imprimir_campo_pdf(pdf, "Nombre", datos_paciente.get("Nombre"))
            _imprimir_campo_pdf(pdf, "Apellido", datos_paciente.get("Apellido"))
            _imprimir_campo_pdf(pdf, "Edad", datos_paciente.get("Edad"))
            _imprimir_campo_pdf(pdf, "Sexo", datos_paciente.get("Sexo"))
            _imprimir_campo_pdf(pdf, "Diagnóstico", datos_paciente.get("Diagnostico"))
            _imprimir_campo_pdf(pdf, "Tipo", datos_paciente.get("Tipo"))
            _imprimir_campo_pdf(pdf, "Antecedente", datos_paciente.get("Antecedente"))
            _imprimir_campo_pdf(pdf, "Medicacion", datos_paciente.get("Medicacion"))
            pdf.ln(5)

            def imprimir_parametros_y_config(pdf_obj, parametros_dict, titulo):
                pdf_obj.set_font("Arial", 'B', 12)
                pdf_obj.cell(0, 10, titulo, ln=True)
                pdf_obj.set_font("Arial", size=10)
                
                parametros_a_imprimir_con_unidad = {
                    "Mano": "", "Dedo": "",
                    "DBS": "", "Nucleo": "",
                    "Voltaje_izq": " mV", "Corriente_izq": " mA", "Contacto_izq": "",
                    "Frecuencia_izq": " Hz", "Pulso_izq": " µS",
                    "Voltaje_dch": " mV", "Corriente_dch": " mA", "Contacto_dch": "",
                    "Frecuencia_dch": " Hz", "Pulso_dch": " µS"
                }

                for param_key, unit in parametros_a_imprimir_con_unidad.items():
                    value = parametros_dict.get(param_key)
                    if value is not None and str(value).strip() != "" and str(value).lower() != "no especificado":
                        pdf_obj.cell(200, 10, f"{param_key}: {value}{unit}", ln=True)
                pdf_obj.ln(5)

            imprimir_parametros_y_config(pdf, config1_params, "Configuración Medición 1")
            imprimir_parametros_y_config(pdf, config2_params, "Configuración Medición 2")

            def imprimir_resultados(pdf_obj, df_res, titulo):
                pdf_obj.set_font("Arial", 'B', 14)
                pdf_obj.cell(0, 10, titulo, ln=True)
                pdf_obj.set_font("Arial", 'B', 12)
                pdf_obj.cell(30, 10, "Test", 1)
                pdf_obj.cell(40, 10, "Frecuencia (Hz)", 1)
                pdf_obj.cell(30, 10, "RMS", 1)
                pdf_obj.cell(50, 10, "Amplitud (cm)", 1)
                pdf_obj.ln(10)
                pdf_obj.set_font("Arial", "", 10)

                for _, row in df_res.iterrows():
                    pdf_obj.cell(30, 10, row['Test'], 1)
                    pdf_obj.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
                    pdf_obj.cell(30, 10, f"{row['RMS (m/s2)']:.4f}", 1)
                    pdf_obj.cell(50, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
                    pdf_obj.ln(10)
                pdf_obj.ln(5)

            imprimir_resultados(pdf, df_resultados_config1, "Resultados Medición 1")
            imprimir_resultados(pdf, df_resultados_config2, "Resultados Medición 2")
            
            amp_avg_config1 = df_resultados_config1['Amplitud Temblor (cm)'].mean()
            amp_avg_config2 = df_resultados_config2['Amplitud Temblor (cm)'].mean()

            conclusion = ""
            if amp_avg_config1 < amp_avg_config2:
                conclusion = (
                    f"La Medición 1 muestra una amplitud de temblor promedio ({amp_avg_config1:.2f} cm) "
                    f"más baja que la Medición 2 ({amp_avg_config2:.2f} cm), lo que sugiere una mayor reducción del temblor."
                )
            elif amp_avg_config2 < amp_avg_config1:
                conclusion = (
                    f"La Medición 2 muestra una amplitud de temblor promedio ({amp_avg_config2:.2f} cm) "
                    f"más baja que la Medición 1 ({amp_avg_config1:.2f} cm), lo que sugiere una mayor reducción del temblor."
                )
            else:
                conclusion = (
                    f"Ambas mediciones muestran amplitudes de temblor promedio muy similares ({amp_avg_config1:.2f} cm)."
                )

            st.subheader("Resultados Medición 1")
            st.dataframe(df_resultados_config1)

            st.subheader("Resultados Medición 2")
            st.dataframe(df_resultados_config2)

            st.subheader("Comparación Gráfica de Amplitud por Ventana")
            nombres_test = ["Reposo", "Postural", "Acción"]

            for test in nombres_test:
                archivo1 = config1_archivos[test]
                archivo2 = config2_archivos[test]

                if archivo1 is not None and archivo2 is not None:
                    archivo1.seek(0)
                    archivo2.seek(0)
                    df1 = pd.read_csv(archivo1, encoding='latin1')
                    df2 = pd.read_csv(archivo2, encoding='latin1')

                    df1_promedio, df1_ventanas = analizar_temblor_por_ventanas_resultante(df1, fs=100)
                    df2_promedio, df2_ventanas = analizar_temblor_por_ventanas_resultante(df2, fs=100)

                    if not df1_ventanas.empty and not df2_ventanas.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))

                        df1_ventanas["Tiempo (segundos)"] = df1_ventanas["Ventana"] * ventana_duracion_seg
                        df2_ventanas["Tiempo (segundos)"] = df2_ventanas["Ventana"] * ventana_duracion_seg

                        ax.plot(df1_ventanas["Tiempo (segundos)"], df1_ventanas["Amplitud Temblor (cm)"], label="Configuración 1", color="blue")
                        ax.plot(df2_ventanas["Tiempo (segundos)"], df2_ventanas["Amplitud Temblor (cm)"], label="Configuración 2", color="orange")
                        ax.set_title(f"Amplitud por Ventana - {test}")
                        ax.set_xlabel("Tiempo (segundos)")
                        ax.set_ylabel("Amplitud (cm)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                            fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                            image_path_for_pdf = tmp_img.name
                        try:
                            pdf.add_page()
                            pdf.set_font("Arial", 'B', 14)
                            pdf.cell(0, 10, f"Gráfico Comparativo: {test}", ln=True, align="C")
                            pdf.image(image_path_for_pdf, x=15, w=180)
                        finally:
                            os.remove(image_path_for_pdf)
                        plt.close(fig)
                    else:
                        st.warning(f"No hay suficientes datos de ventanas para graficar el test: {test}")
                else:
                    st.warning(f"Faltan archivos para el test {test} en al menos una Medición.")
            
            st.subheader("Conclusión del Análisis Comparativo")
            st.write(conclusion)

            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Conclusión", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 10, conclusion)

            pdf_output = BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0)

            st.download_button(
                label="Descargar Informe PDF",
                data=pdf_output.getvalue(),
                file_name="informe_comparativo_temblor.pdf",
                mime="application/pdf"
            )
            st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
            
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
