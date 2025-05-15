import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from fpdf import FPDF
import base64
from io import BytesIO

# ------------------- FUNCIONES GENERALES -------------------

def filtrar_temblor(data, fs=100, lowcut=3.5, highcut=12):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

def calcular_resultante(acel_x, acel_y, acel_z):
    return np.sqrt(acel_x**2 + acel_y**2 + acel_z**2)

def analizar_resultante_por_ventanas(resultante, fs=100, ventana_tamano=2):
    ventana_muestras = int(ventana_tamano * fs)
    total_muestras = len(resultante)
    num_ventanas = total_muestras // ventana_muestras

    rms_list = []
    varianza_list = []
    frecuencia_dom_list = []
    amplitud_list = []

    for i in range(num_ventanas):
        ventana = resultante[i * ventana_muestras:(i + 1) * ventana_muestras]
        ventana_filtrada = filtrar_temblor(ventana, fs=fs)

        # RMS
        rms = np.sqrt(np.mean(ventana_filtrada ** 2))
        rms_list.append(rms)

        # Varianza
        varianza = np.var(ventana_filtrada)
        varianza_list.append(varianza)

        # Frecuencia dominante
        f, Pxx = welch(ventana_filtrada, fs=fs)
        frecuencia_dominante = f[np.argmax(Pxx)]
        frecuencia_dom_list.append(frecuencia_dominante)

        # Amplitud (media de diferencias máx-mín * 9.81 * 0.5)
        amp = (np.max(ventana_filtrada) - np.min(ventana_filtrada)) * 9.81 * 0.5
        amplitud_list.append(amp)

    return {
        'RMS': np.mean(rms_list),
        'Varianza': np.mean(varianza_list),
        'Frecuencia Dominante': np.mean(frecuencia_dom_list),
        'Amplitud (cm)': np.mean(amplitud_list)
    }

def generar_pdf_resultante(nombre, resultados_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Informe de Análisis de Temblor", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Nombre del paciente: {nombre}", ln=True)
    pdf.ln(5)

    for test, resultados in resultados_dict.items():
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(200, 10, txt=f"Test: {test}", ln=True)
        pdf.set_font("Arial", size=12)
        for clave, valor in resultados.items():
            pdf.cell(200, 10, txt=f"{clave}: {valor:.3f}", ln=True)
        pdf.ln(5)

    buffer = BytesIO()
    pdf.output(buffer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="informe_temblor.pdf">📄 Descargar informe PDF</a>'
    return href

# ------------------- INTERFAZ STREAMLIT -------------------

st.title("🧠 Análisis de Temblor Patológico")
opcion = st.sidebar.selectbox("Selecciona una opción", ["1️⃣ Análisis de una medición", "2️⃣ Comparación entre configuraciones"])

# ------------------- OPCIÓN 1 -------------------

if opcion == "1️⃣ Análisis de una medición":
    st.header("🔍 Análisis de una medición")
    nombre_paciente = st.text_input("Nombre del paciente")

    archivos = {
        "Reposo": st.file_uploader("Cargar CSV de Reposo", type="csv"),
        "Postural": st.file_uploader("Cargar CSV de Postural", type="csv"),
        "Acción": st.file_uploader("Cargar CSV de Acción", type="csv")
    }

    if st.button("Iniciar análisis") and nombre_paciente and all(archivos.values()):
        resultados_totales = {}
        for tipo, archivo in archivos.items():
            df = pd.read_csv(archivo)
            if set(['aceleracion_x', 'aceleracion_y', 'aceleracion_z']).issubset(df.columns):
                resultante = calcular_resultante(df['aceleracion_x'], df['aceleracion_y'], df['aceleracion_z'])
                resultados = analizar_resultante_por_ventanas(resultante)
                resultados_totales[tipo] = resultados

                st.subheader(f"📊 Resultados - {tipo}")
                for k, v in resultados.items():
                    st.write(f"{k}: {v:.3f}")
            else:
                st.error(f"El archivo de {tipo} no contiene las columnas necesarias.")

        st.markdown("---")
        st.markdown("### 📄 Informe PDF")
        enlace_pdf = generar_pdf_resultante(nombre_paciente, resultados_totales)
        st.markdown(enlace_pdf, unsafe_allow_html=True)

# ------------------- OPCIÓN 2 (SIN CAMBIOS) -------------------

elif opcion == "2️⃣ Comparación entre configuraciones":
    st.header("⚖️ Comparación entre configuraciones")

    archivos = {
        "Configuración 1 - Reposo": st.file_uploader("Cargar CSV Config 1 - Reposo", type="csv"),
        "Configuración 1 - Postural": st.file_uploader("Cargar CSV Config 1 - Postural", type="csv"),
        "Configuración 1 - Acción": st.file_uploader("Cargar CSV Config 1 - Acción", type="csv"),
        "Configuración 2 - Reposo": st.file_uploader("Cargar CSV Config 2 - Reposo", type="csv"),
        "Configuración 2 - Postural": st.file_uploader("Cargar CSV Config 2 - Postural", type="csv"),
        "Configuración 2 - Acción": st.file_uploader("Cargar CSV Config 2 - Acción", type="csv"),
    }

    nombre_paciente = st.text_input("Nombre del paciente")

    if st.button("Comparar configuraciones") and nombre_paciente and all(archivos.values()):
        st.write("Procesamiento de comparación entre configuraciones aún no modificado.")
