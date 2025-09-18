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
def extraer_datos_csv(df):
    """
    Extrae todos los metadatos de un DataFrame de CSV, normalizando
    las claves a minúsculas y limpiando los valores.
    """
    if df.empty or df.shape[0] == 0:
        return {}

    # Mapeo de columnas de CSV a claves normalizadas
    column_map = {
        'nombre': 'nombre', 'apellido': 'apellido', 'edad': 'edad',
        'sexo': 'sexo', 'diagnostico': 'diagnostico', 'tipo': 'tipo',
        'antecedente': 'antecedente', 'medicacion': 'medicacion',
        'mano': 'mano_medida', 'dedo': 'dedo_medido',
        'dbs': 'dbs', 'nucleo': 'nucleo',
        'voltaje [mv]_izq': 'voltaje_izq', 'corriente [ma]_izq': 'corriente_izq',
        'contacto_izq': 'contacto_izq', 'frecuencia [hz]_izq': 'frecuencia_izq',
        'ancho de pulso [µs]_izq': 'ancho_pulso_izq',
        'voltaje [mv]_dch': 'voltaje_dch', 'corriente [ma]_dch': 'corriente_dch',
        'contacto_dch': 'contacto_dch', 'frecuencia [hz]_dch': 'frecuencia_dch',
        'ancho de pulso [µs]_dch': 'ancho_pulso_dch'
    }

    # Inicializa el diccionario de datos con valores por defecto
    datos = {v: 'sin informacion' for v in column_map.values()}

    # Procesa la primera fila del DataFrame (metadatos)
    row = df.iloc[0]
    
    # Crear un diccionario de mapeo de columnas del DataFrame a sus nombres en minúsculas
    df_cols_lower = {col.lower(): col for col in df.columns}

    for csv_col_lower, dict_key in column_map.items():
        if csv_col_lower in df_cols_lower:
            original_col = df_cols_lower[csv_col_lower]
            val = row[original_col]
            
            # Limpiar valores NaN o vacíos
            if pd.notna(val) and str(val).strip().lower() not in ['', 'no especificado', 'sin informacion']:
                if dict_key == 'edad':
                    try:
                        datos[dict_key] = int(float(str(val).replace(',', '.')))
                    except (ValueError, TypeError):
                        datos[dict_key] = 'sin informacion'
                else:
                    datos[dict_key] = str(val).strip()
    return datos

# Función para imprimir campos de forma condicional en el PDF
def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
    if valor not in [None, 'sin informacion']:
        pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)


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
    
# ------------------ MÓDULO 1: ANÁLISIS DE UNA MEDICIÓN --------------------

elif opcion == "1️⃣ Análisis de medición individual":
    st.title("📈 Análisis de Medición Individual")
    
    def generar_pdf_individual(datos_completos, df_resultados, graficos_paths):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Informe de Análisis de Temblor", ln=True, align="C")
        pdf.ln(5)
        
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos del Paciente", ln=True)
        pdf.set_font("Arial", size=12)
        _imprimir_campo_pdf(pdf, "Nombre", datos_completos.get('nombre'))
        _imprimir_campo_pdf(pdf, "Apellido", datos_completos.get('apellido'))
        _imprimir_campo_pdf(pdf, "Edad", datos_completos.get('edad'))
        _imprimir_campo_pdf(pdf, "Sexo", datos_completos.get('sexo'))
        _imprimir_campo_pdf(pdf, "Diagnóstico", datos_completos.get('diagnostico'))
        _imprimir_campo_pdf(pdf, "Tipo", datos_completos.get('tipo'))
        _imprimir_campo_pdf(pdf, "Antecedente", datos_completos.get('antecedente'))
        _imprimir_campo_pdf(pdf, "Medicacion", datos_completos.get('medicacion'))
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Parámetros de la Medición", ln=True)
        pdf.set_font("Arial", size=10)
        _imprimir_campo_pdf(pdf, "Mano", datos_completos.get('mano_medida'))
        _imprimir_campo_pdf(pdf, "Dedo", datos_completos.get('dedo_medido'))
        _imprimir_campo_pdf(pdf, "DBS", datos_completos.get('dbs'))
        _imprimir_campo_pdf(pdf, "Núcleo", datos_completos.get('nucleo'))
        _imprimir_campo_pdf(pdf, "Voltaje (izq)", datos_completos.get('voltaje_izq'), " mV")
        _imprimir_campo_pdf(pdf, "Corriente (izq)", datos_completos.get('corriente_izq'), " mA")
        _imprimir_campo_pdf(pdf, "Contacto (izq)", datos_completos.get('contacto_izq'))
        _imprimir_campo_pdf(pdf, "Frecuencia (izq)", datos_completos.get('frecuencia_izq'), " Hz")
        _imprimir_campo_pdf(pdf, "Ancho de pulso (izq)", datos_completos.get('ancho_pulso_izq'), " µS")
        _imprimir_campo_pdf(pdf, "Voltaje (dch)", datos_completos.get('voltaje_dch'), " mV")
        _imprimir_campo_pdf(pdf, "Corriente (dch)", datos_completos.get('corriente_dch'), " mA")
        _imprimir_campo_pdf(pdf, "Contacto (dch)", datos_completos.get('contacto_dch'))
        _imprimir_campo_pdf(pdf, "Frecuencia (dch)", datos_completos.get('frecuencia_dch'), " Hz")
        _imprimir_campo_pdf(pdf, "Ancho de pulso (dch)", datos_completos.get('ancho_pulso_dch'), " µS")
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Resultados del Análisis", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 10, "Métrica", 1)
        pdf.cell(50, 10, "Valor Promedio", 1)
        pdf.ln(10)
        pdf.set_font("Arial", "", 10)
        
        if not df_resultados.empty:
            for metric, value in df_resultados.iloc[0].items():
                if metric != 'Test':
                    pdf.cell(40, 10, metric.replace(" (cm)", " (cm)").replace(" (m/s2)", " (m/s2)").replace(" (Hz)", " (Hz)"), 1)
                    pdf.cell(50, 10, f"{value:.2f}" if isinstance(value, (int, float)) else str(value), 1)
                    pdf.ln(10)
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

    reposo_file = st.file_uploader("Archivo de REPOSO", type="csv", key="reposo_individual")
    postural_file = st.file_uploader("Archivo de POSTURAL", type="csv", key="postural_individual")
    accion_file = st.file_uploader("Archivo de ACCION", type="csv", key="accion_individual")

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

    if st.button("Generar Análisis"):
        uploaded_files = {
            "Reposo": reposo_file,
            "Postural": postural_file,
            "Acción": accion_file
        }
        
        any_file_uploaded = any(file is not None for file in uploaded_files.values())

        if not any_file_uploaded:
            st.warning("Por favor, sube al menos un archivo CSV para realizar el análisis.")
        else:
            resultados = []
            graficos_paths = []
            datos_completos = {}
            first_file_processed = False

            for test_type, uploaded_file in uploaded_files.items():
                if uploaded_file is not None:
                    uploaded_file.seek(0)
                    df_csv = pd.read_csv(uploaded_file, encoding='latin1')
                    
                    if not first_file_processed:
                        datos_completos = extraer_datos_csv(df_csv)
                        first_file_processed = True

                    df_promedio, df_ventanas = analizar_temblor_por_ventanas_resultante(df_csv, fs=100)
                    
                    if not df_promedio.empty:
                        promedio = df_promedio.iloc[0]
                        resultados.append({
                            'Test': test_type,
                            'Frecuencia Dominante (Hz)': round(promedio['Frecuencia Dominante (Hz)'], 2),
                            'RMS (m/s2)': round(promedio['RMS (m/s2)'], 4),
                            'Amplitud Temblor (cm)': round(promedio['Amplitud Temblor (cm)'], 2)
                        })

                    if not df_ventanas.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_ventanas["Tiempo (segundos)"] = df_ventanas["Ventana"] * 5
                        ax.plot(df_ventanas["Tiempo (segundos)"], df_ventanas["Amplitud Temblor (cm)"], marker='o', linestyle='-')
                        ax.set_title(f"Amplitud por Ventana - {test_type}")
                        ax.set_xlabel("Tiempo (segundos)")
                        ax.set_ylabel("Amplitud (cm)")
                        ax.grid(True)
                        st.pyplot(fig)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                            fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                            graficos_paths.append(tmp_img.name)
                        plt.close(fig)

            if not resultados:
                st.error("No se pudieron calcular métricas de temblor para ningún archivo cargado. Asegúrate de que los archivos contengan datos válidos.")
            else:
                df_resultados = pd.DataFrame(resultados).set_index('Test')
                st.subheader("Resultados del Análisis:")
                st.dataframe(df_resultados)
                
                pdf_output = generar_pdf_individual(datos_completos, df_resultados.reset_index(), graficos_paths)
                
                st.download_button(
                    label="Descargar Informe PDF",
                    data=pdf_output.getvalue(),
                    file_name="informe_analisis_temblor.pdf",
                    mime="application/pdf"
                )
                
                st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
                
                for path in graficos_paths:
                    if os.path.exists(path):
                        os.remove(path)
                
# ------------------ MÓDULO 2: COMPARACIÓN DE MEDICIONES --------------------

elif opcion == "2️⃣ Comparación de mediciones":
    st.title("⚖️ Comparación de Mediciones")
    st.markdown("### Cargar archivos CSV para la Comparación")
    
    def generar_pdf_comparacion(datos_completos, df_resultados, graficos_paths, titulo_grafico):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Informe de Comparación de Mediciones", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos del Paciente", ln=True)
        pdf.set_font("Arial", size=12)
        _imprimir_campo_pdf(pdf, "Nombre", datos_completos.get('nombre'))
        _imprimir_campo_pdf(pdf, "Apellido", datos_completos.get('apellido'))
        _imprimir_campo_pdf(pdf, "Edad", datos_completos.get('edad'))
        _imprimir_campo_pdf(pdf, "Sexo", datos_completos.get('sexo'))
        _imprimir_campo_pdf(pdf, "Diagnóstico", datos_completos.get('diagnostico'))
        _imprimir_campo_pdf(pdf, "Tipo", datos_completos.get('tipo'))
        _imprimir_campo_pdf(pdf, "Antecedente", datos_completos.get('antecedente'))
        _imprimir_campo_pdf(pdf, "Medicacion", datos_completos.get('medicacion'))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Parámetros de la Medición", ln=True)
        pdf.set_font("Arial", size=10)
        _imprimir_campo_pdf(pdf, "Mano", datos_completos.get('mano_medida'))
        _imprimir_campo_pdf(pdf, "Dedo", datos_completos.get('dedo_medido'))
        _imprimir_campo_pdf(pdf, "DBS", datos_completos.get('dbs'))
        _imprimir_campo_pdf(pdf, "Núcleo", datos_completos.get('nucleo'))
        _imprimir_campo_pdf(pdf, "Voltaje (izq)", datos_completos.get('voltaje_izq'), " mV")
        _imprimir_campo_pdf(pdf, "Corriente (izq)", datos_completos.get('corriente_izq'), " mA")
        _imprimir_campo_pdf(pdf, "Contacto (izq)", datos_completos.get('contacto_izq'))
        _imprimir_campo_pdf(pdf, "Frecuencia (izq)", datos_completos.get('frecuencia_izq'), " Hz")
        _imprimir_campo_pdf(pdf, "Ancho de pulso (izq)", datos_completos.get('ancho_pulso_izq'), " µS")
        _imprimir_campo_pdf(pdf, "Voltaje (dch)", datos_completos.get('voltaje_dch'), " mV")
        _imprimir_campo_pdf(pdf, "Corriente (dch)", datos_completos.get('corriente_dch'), " mA")
        _imprimir_campo_pdf(pdf, "Contacto (dch)", datos_completos.get('contacto_dch'))
        _imprimir_campo_pdf(pdf, "Frecuencia (dch)", datos_completos.get('frecuencia_dch'), " Hz")
        _imprimir_campo_pdf(pdf, "Ancho de pulso (dch)", datos_completos.get('ancho_pulso_dch'), " µS")
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Resultados del Análisis", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 12)
        column_widths = [40, 40, 30, 40]
        pdf.cell(column_widths[0], 10, "Test", 1, 0, 'C')
        pdf.cell(column_widths[1], 10, "Frecuencia (Hz)", 1, 0, 'C')
        pdf.cell(column_widths[2], 10, "RMS", 1, 0, 'C')
        pdf.cell(column_widths[3], 10, "Amplitud (cm)", 1, 1, 'C')
        pdf.set_font("Arial", "", 10)

        for _, row in df_resultados.iterrows():
            pdf.cell(column_widths[0], 10, row['Test'], 1, 0)
            pdf.cell(column_widths[1], 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1, 0)
            pdf.cell(column_widths[2], 10, f"{row['RMS (m/s2)']:.4f}", 1, 0)
            pdf.cell(column_widths[3], 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1, 1)
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

    file1 = st.file_uploader("Archivo de la Primera Medición", type="csv", key="file1")
    file2 = st.file_uploader("Archivo de la Segunda Medición", type="csv", key="file2")
    
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
    
    if st.button("Comparar Mediciones"):
        if file1 is None or file2 is None:
            st.warning("Por favor, sube ambos archivos CSV para la comparación.")
        else:
            file1.seek(0)
            file2.seek(0)
            df1 = pd.read_csv(file1, encoding='latin1')
            df2 = pd.read_csv(file2, encoding='latin1')

            datos_completos = extraer_datos_csv(df1)
            
            df_promedio1, df_ventanas1 = analizar_temblor_por_ventanas_resultante(df1, fs=100)
            df_promedio2, df_ventanas2 = analizar_temblor_por_ventanas_resultante(df2, fs=100)
            
            if df_promedio1.empty or df_promedio2.empty:
                st.error("No se pudieron calcular las métricas de temblor para uno o ambos archivos.")
            else:
                df_promedio1['Test'] = 'Medición 1'
                df_promedio2['Test'] = 'Medición 2'
                df_resultados = pd.concat([df_promedio1, df_promedio2], ignore_index=True)
                
                st.subheader("Resultados de la Comparación:")
                st.dataframe(df_resultados.set_index('Test'))
                
                st.subheader("Gráfico Comparativo de Amplitud por Ventana")
                
                min_ventanas = min(len(df_ventanas1), len(df_ventanas2))
                df_ventanas1 = df_ventanas1.iloc[:min_ventanas].copy()
                df_ventanas2 = df_ventanas2.iloc[:min_ventanas].copy()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                df_ventanas1["Tiempo (segundos)"] = df_ventanas1["Ventana"] * 5
                df_ventanas2["Tiempo (segundos)"] = df_ventanas2["Ventana"] * 5

                ax.plot(df_ventanas1["Tiempo (segundos)"], df_ventanas1["Amplitud Temblor (cm)"], label="Medición 1", marker='o')
                ax.plot(df_ventanas2["Tiempo (segundos)"], df_ventanas2["Amplitud Temblor (cm)"], label="Medición 2", marker='x')

                ax.set_title("Comparación de Amplitud de Temblor por Ventana de Tiempo")
                ax.set_xlabel("Tiempo (segundos)")
                ax.set_ylabel("Amplitud (cm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                graficos_paths = []
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                    graficos_paths.append(tmp_img.name)
                plt.close(fig)

                pdf_output = generar_pdf_comparacion(
                    datos_completos,
                    df_resultados,
                    graficos_paths,
                    "Comparación de Amplitud de Temblor por Ventana"
                )
                
                st.download_button(
                    label="Descargar Informe PDF",
                    data=pdf_output.getvalue(),
                    file_name="informe_comparativo_temblor.pdf",
                    mime="application/pdf"
                )
                
                for path in graficos_paths:
                    if os.path.exists(path):
                        os.remove(path)
# ------------------ MÓDULO 3: DIAGNÓSTICO TENTATIVO --------------------
elif opcion == "3️⃣ Diagnóstico tentativo":
    st.title("🩺 Diagnóstico Tentativo")
    st.markdown("### Cargar archivos CSV para el Diagnóstico")
    
    def generar_pdf_diagnostico(datos_completos, resultados_df, prediccion_texto, graficos_paths):
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
        _imprimir_campo_pdf(pdf, "Nombre", datos_completos.get('nombre'))
        _imprimir_campo_pdf(pdf, "Apellido", datos_completos.get('apellido'))
        _imprimir_campo_pdf(pdf, "Edad", datos_completos.get('edad'))
        _imprimir_campo_pdf(pdf, "Sexo", datos_completos.get('sexo'))
        _imprimir_campo_pdf(pdf, "Diagnóstico", datos_completos.get('diagnostico'))
        _imprimir_campo_pdf(pdf, "Tipo", datos_completos.get('tipo'))
        _imprimir_campo_pdf(pdf, "Antecedente", datos_completos.get('antecedente'))
        _imprimir_campo_pdf(pdf, "Medicacion", datos_completos.get('medicacion'))
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Configuración de la Medición", ln=True)
        pdf.set_font("Arial", size=10)
        _imprimir_campo_pdf(pdf, "Mano", datos_completos.get('mano_medida'))
        _imprimir_campo_pdf(pdf, "Dedo", datos_completos.get('dedo_medido'))
        pdf.ln(5)
        
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
            datos_completos = {}
            first_file_processed = False

            for test_type, uploaded_file in prediccion_files_correctas.items():
                if uploaded_file is not None:
                    uploaded_file.seek(0)
                    df_current_test = pd.read_csv(uploaded_file, encoding='latin1')
                    
                    if not first_file_processed:
                        datos_completos = extraer_datos_csv(df_current_test)
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
                df_metrics_display = pd.DataFrame.from_dict(avg_tremor_metrics, orient='index').reset_index()
                df_metrics_display = df_metrics_display.rename(columns={'index': 'Test'})
                st.dataframe(df_metrics_display)

                data_for_model = {}
                data_for_model['edad'] = datos_completos.get('edad', np.nan)
                data_for_model['sexo'] = datos_completos.get('sexo', 'sin informacion').lower()
                data_for_model['mano_medida'] = datos_completos.get('mano_medida', 'sin informacion').lower()
                data_for_model['dedo_medido'] = datos_completos.get('dedo_medido', 'sin informacion').lower()

                feature_name_map = {
                    "Reposo": "Reposo", "Postural": "Postural", "Acción": "Accion"
                }

                for original_test_type, model_feature_prefix in feature_name_map.items():
                    metrics = avg_tremor_metrics.get(original_test_type, {})
                    data_for_model[f'Frec_{model_feature_prefix}'] = metrics.get('Frecuencia Dominante (Hz)', np.nan)
                    data_for_model[f'RMS_{model_feature_prefix}'] = metrics.get('RMS (m/s2)', np.nan)
                    data_for_model[f'Amp_{model_feature_prefix}'] = metrics.get('Amplitud Temblor (cm)', np.nan)

                expected_features_for_model = [
                    'edad', 'Frec_Reposo', 'RMS_Reposo', 'Amp_Reposo',
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
                        df_to_plot["Tiempo (segundos)"] = df_to_plot["Ventana"] * 5
                        ax.plot(df_to_plot["Tiempo (segundos)"], df_to_plot["Amplitud Temblor (cm)"], label=f"{test_name}")
                    
                    ax.set_title("Amplitud de Temblor por Ventana de Tiempo")
                    ax.set_xlabel("Tiempo (segundos)")
                    ax.set_ylabel("Amplitud (cm)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                        graficos_paths.append(tmp_img.name)
                    plt.close(fig)
                else:
                    st.warning("No hay suficientes datos de ventanas para graficar los archivos de predicción.")
                
                pdf_output = generar_pdf_diagnostico(
                    datos_completos,
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
                
                for path in graficos_paths:
                    if os.path.exists(path):
                        os.remove(path)
