# -*- coding: utf-8 -*-
# app_temblor.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from fpdf import FPDF
from datetime import datetime, timedelta
import os
import streamlit as st
from scipy.fft import fft, fftfreq
import unicodedata
from fpdf import FPDF
            
        
      
        

# --------- Funciones compartidas ----------
def filtrar_temblor(signal, fs=200):
    b, a = butter(N=4, Wn=[3, 12], btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def analizar_temblor_por_ventanas(signal, eje, fs=200, ventana_seg=2):
    signal = signal.dropna().to_numpy()
    signal_filtrada = filtrar_temblor(signal, fs)
    tamaño_ventana = int(fs * ventana_seg)
    num_ventanas = len(signal_filtrada) // tamaño_ventana
    resultados = []
    for i in range(num_ventanas):
        segmento = signal_filtrada[i*tamaño_ventana:(i+1)*tamaño_ventana]
        if len(segmento) < tamaño_ventana:
            continue
        f, Pxx = welch(segmento, fs=fs, nperseg=tamaño_ventana)
        freq_dominante = f[np.argmax(Pxx)]
        if eje in ['Acel_X', 'Acel_Y', 'Acel_Z']:
            segmento = segmento * 9.81
        varianza = np.var(segmento)
        rms = np.sqrt(np.mean(segmento**2))
        amplitud = np.max(segmento) - np.min(segmento)
        resultados.append({
            'Frecuencia Dominante (Hz)': freq_dominante,
            'Varianza (m2/s4)': varianza,
            'RMS (m/s2)': rms,
            'Amplitud Temblor (g)': amplitud
        })
    return pd.DataFrame(resultados)

def crear_grafico(df, nombre, archivo="grafico_resumen.png"):
    resumen = df.groupby('Condicion')[['Frecuencia Dominante (Hz)', 'Varianza (m2/s4)', 'RMS (m/s2)', 'Amplitud Temblor (cm)']].mean()
    resumen.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Comparación de Medidas - {nombre}")
    plt.ylabel("Valor")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(archivo)
    plt.close()


# ------------------ Modo principal --------------------

st.title("🧠 Análisis de Temblor")
opcion = st.sidebar.radio("Selecciona una opción:", ["1️⃣ Análisis de una medición", "2️⃣ Comparar dos mediciones"])

if opcion == "1️⃣ Análisis de una medición":
        # -*- coding: utf-8 -*-
       
        # --------- Funciones auxiliares ----------
        def filtrar_temblor(senal, fs, lowcut=2, highcut=20, order=4):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, senal)
        
        def analizar_temblor_por_ventanas_resultante(df, fs=50, ventana_seg=2):
            df = df[['Acel_X', 'Acel_Y', 'Acel_Z']].dropna()
            ax = df['Acel_X'].to_numpy()
            ay = df['Acel_Y'].to_numpy()
            az = df['Acel_Z'].to_numpy()
            resultante = np.sqrt(ax**2 + ay**2 + az**2)
        
            señal_filtrada = filtrar_temblor(resultante, fs)
        
            tamaño_ventana = int(fs * ventana_seg)
            num_ventanas = len(señal_filtrada) // tamaño_ventana
            resultados = []
        
            for i in range(num_ventanas):
                segmento = señal_filtrada[i*tamaño_ventana:(i+1)*tamaño_ventana]
                if len(segmento) < tamaño_ventana:
                    continue
                f, Pxx = welch(segmento, fs=fs, nperseg=tamaño_ventana)
                freq_dominante = f[np.argmax(Pxx)]
                varianza = np.var(segmento)
                rms = np.sqrt(np.mean(segmento**2))
                amplitud = np.max(segmento) - np.min(segmento)
                resultados.append({
                    'Frecuencia Dominante (Hz)': freq_dominante,
                    'Varianza (m2/s4)': varianza,
                    'RMS (m/s2)': rms,
                    'Amplitud Temblor (g)': amplitud
                })
        
            return pd.DataFrame(resultados)

        def diagnosticar(df):
            def max_amp(test):
                fila = df[df['Test'] == test]
                return fila['Amplitud Temblor (cm)'].max() if not fila.empty else 0
        
            def mean_freq(test):
                fila = df[df['Test'] == test]
                return fila['Frecuencia Dominante (Hz)'].mean() if not fila.empty else 0
        
            if max_amp('Reposo') > 0.3 and 3 <= mean_freq('Reposo') <= 7:
                return "Probable Parkinson"
            elif (max_amp('Postural') > 0.3 or max_amp('Acción') > 0.3) and (8 <= mean_freq('Postural') <= 10 or 8 <= mean_freq('Acción') <= 10):
                return "Probable Temblor Esencial"
            else:
                return "Temblor dentro de parámetros normales"
            
        def generar_pdf(df, nombre_archivo="informe_temblor.pdf", diagnostico=""):
        
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')
        
            # Tabla de resultados
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(30, 10, "Test", 1)
            pdf.cell(40, 10, "Frecuencia (Hz)", 1)
            pdf.cell(30, 10, "Varianza", 1)
            pdf.cell(30, 10, "RMS", 1)
            pdf.cell(50, 10, "Amplitud (cm)", 1)
            pdf.ln(10)
        
            pdf.set_font("Arial", "", 12)
            for _, row in df.iterrows():
                pdf.cell(30, 10, row['Test'], 1)
                pdf.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
                pdf.cell(30, 10, f"{row['Varianza (m2/s4)']:.4f}", 1)
                pdf.cell(30, 10, f"{row['RMS (m/s2)']:.4f}", 1)
                pdf.cell(50, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
                pdf.ln(10)
            if diagnostico:
                    pdf.ln(10)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Diagnóstico automático:", ln=True)
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, diagnostico)


            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Interpretación clínica:", ln=True)
            pdf.set_font("Arial", size=11)
            texto_original = """
        Este informe analiza tres tipos de temblores: en reposo, postural y de acción.
        
        Los valores de referencia considerados son:
          Para las frecuencias (Hz):
        - Temblor Parkinsoniano: 3-6 Hz en reposo.
        - Temblor Esencial: 8-10 Hz en acción o postura.
        
          Para las amplitudes:
        - Mayores a 0.3 cm pueden ser clínicamente relevantes.
        
          Para la varianza (m2/s4):
        Representa la dispersión de la señal. En el contexto de temblores:
        - Normal/sano: muy baja, puede estar entre 0.001 – 0.1 m2/s4.
        - Temblor leve: entre 0.1 – 0.5 m2/s4.
        - Temblor patológico (PK o TE): suele superar 1.0 m2/s4, llegando hasta 5–10 m2/s4 en casos severos.
        
          Para el RMS (m/s2):
        - Normal/sano: menor a 0.5 m/s2.
        - PK leve: entre 0.5 y 1.5 m/s2.
        - TE o PK severo: puede llegar a 2–3 m/s2 o más.
        
        La clasificación automática es orientativa y debe ser evaluada por un profesional.
        """
        
            texto_limpio = limpiar_texto_para_pdf(texto_original)
            pdf.multi_cell(0, 8, texto_limpio)
            pdf.set_font("Arial", 'B', 12)
            
            pdf.output(nombre_archivo)
        
        
        
        st.title("Análisis de una medicion")
        
        uploaded_files = {
            "Reposo": st.file_uploader("Subir archivo CSV para prueba en reposo", type=["csv"], key="reposo"),
            "Postural": st.file_uploader("Subir archivo CSV para prueba postural", type=["csv"], key="postural"),
            "Acción": st.file_uploader("Subir archivo CSV para prueba en acción", type=["csv"], key="accion"),
        }
        
        if st.button("Iniciar análisis"):
            resultados_globales = []
            mediciones_tests = {test: pd.read_csv(file) for test, file in uploaded_files.items() if file is not None}
        
            for test, datos in mediciones_tests.items():
                df_ventana = analizar_temblor_por_ventanas_resultante(datos, fs=50)
                if not df_ventana.empty:
                    prom = df_ventana.mean(numeric_only=True)
                    freq = prom['Frecuencia Dominante (Hz)']
                    amp_g = prom['Amplitud Temblor (g)']
                    amp_cm = (amp_g * 981) / ((2 * np.pi * freq) ** 2) if freq > 0 else 0.0
        
                    resultados_globales.append({
                        'Test': test,
                        'Frecuencia Dominante (Hz)': round(freq, 2),
                        'Varianza (m2/s4)': round(prom['Varianza (m2/s4)'], 4),
                        'RMS (m/s2)': round(prom['RMS (m/s2)'], 4),
                        'Amplitud Temblor (cm)': round(amp_cm, 2)
                    })
        
            if resultados_globales:
                df_resultados = pd.DataFrame(resultados_globales)
                st.subheader("Resultados Promediados por Test")
                st.dataframe(df_resultados)
                diagnostico = diagnosticar(df_resultados)
                st.subheader("Diagnóstico")
                st.write(diagnostico)
        
                generar_pdf(df_resultados, diagnostico=diagnostico)
                with open("informe_temblor.pdf", "rb") as file:
                    st.download_button(
                        label="📄 Descargar Informe PDF",
                        data=file,
                        file_name="informe_temblor.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No se encontraron datos suficientes para el análisis.")
                    

elif opcion == "2️⃣ Comparar dos mediciones":
    st.title("📊 Comparar dos mediciones")
    st.write("Sube los 6 archivos CSV: 3 de cada configuración del estimulador")

    uploaded_files_conf1 = {}
    uploaded_files_conf2 = {}

    for test in ["Reposo", "Postural", "Acción"]:
        uploaded_files_conf1[test] = st.file_uploader(f"Config 1 - Archivo para test de {test}", type="csv", key=f"c1_{test}")
        uploaded_files_conf2[test] = st.file_uploader(f"Config 2 - Archivo para test de {test}", type="csv", key=f"c2_{test}")

    if all(uploaded_files_conf1.values()) and all(uploaded_files_conf2.values()):
        if st.button("Comparar configuraciones"):
            resultados = []
            config_labels = {1: "Configuración 1", 2: "Configuración 2"}
            datos_personales = None
            config_info = {}

            for i, config_files in enumerate([uploaded_files_conf1, uploaded_files_conf2], start=1):
                for test, file in config_files.items():
                    df = pd.read_csv(file)
                    datos_personales = df.iloc[0].to_frame().T
                    config = f"Config {i}"
                    if config not in config_info:
                        config_info[config] = {k: datos_personales.iloc[0][k] for k in ['Voltaje', 'Frecuencia', 'Corriente'] if k in datos_personales.columns}
                    datos = df.iloc[1:][['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']].apply(pd.to_numeric, errors='coerce')
                    for eje in datos.columns:
                        df_proc = analizar_temblor_por_ventanas(datos[eje], eje, fs=50)
                        if not df_proc.empty:
                            prom = df_proc.mean(numeric_only=True)
                            freq = prom['Frecuencia Dominante (Hz)']
                            amp_g = prom['Amplitud Temblor (g)']
                            amp_cm = (amp_g * 981) / ((2 * np.pi * freq) ** 2) if freq > 0 else 0.0
                            resultados.append({
                                'Condicion': f"{config_labels[i]} - {test}",
                                'Eje': eje,
                                'Frecuencia Dominante (Hz)': round(freq, 2),
                                'Varianza (m2/s4)': round(prom['Varianza (m2/s4)'], 4),
                                'RMS (m/s2)': round(prom['RMS (m/s2)'], 4),
                                'Amplitud Temblor (cm)': round(amp_cm, 2)
                            })

            df_final = pd.DataFrame(resultados)
            st.dataframe(df_final)

            crear_grafico(df_final, datos_personales['Nombre'].values[0], archivo="comparacion_temblor.png")

            # Evaluación simple: menor RMS promedio general
            resumen = df_final.groupby('Condicion')[['RMS (m/s2)', 'Varianza (m2/s4)', 'Amplitud Temblor (cm)']].mean()
            mejor = resumen['RMS (m/s2)'].idxmin()

            st.success(f"La mejor configuración en base a menor RMS promedio es: {mejor}")

            with open("comparacion_temblor.png", "rb") as f:
                st.download_button("📈 Descargar gráfico de comparación", f, file_name="comparacion_temblor.png")
