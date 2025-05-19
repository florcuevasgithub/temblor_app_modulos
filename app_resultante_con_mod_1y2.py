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
from scipy.fft import fft, fftfreq
import unicodedata
import io



# --------- Funciones compartidas ----------
def filtrar_temblor(senal, fs, lowcut=3, highcut=12, order=4):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, senal)

def analizar_temblor_por_ventanas_resultante(df, fs=200, ventana_seg=2):
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
                amp_g = np.max(segmento) - np.min(segmento)
                amp_cm = (amp_g * 981) / ((2 * np.pi * freq_dominante) ** 2) if freq_dominante > 0 else 0.0

                resultados.append({
                    'Frecuencia Dominante (Hz)': freq_dominante,
                    'Varianza (m2/s4)': varianza,
                    'RMS (m/s2)': rms,
                    'Amplitud Temblor (g)': amp_g,
                    'Amplitud Temblor (cm)': amp_cm
                })

            return pd.DataFrame(resultados)


# ------------------ Modo principal --------------------

st.title("🧠 Análisis de Temblor")
opcion = st.sidebar.radio("Selecciona una opción:", ["1️⃣ Análisis de una medición", "2️⃣ Comparar dos configuraciones de estimulación"])

if opcion == "1️⃣ Análisis de una medición":
        # -*- coding: utf-8 -*-

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

        def generar_pdf(nombre_paciente, apellido_paciente, edad, sexo, diag_clinico, mano, dedo,df, nombre_archivo="informe_temblor.pdf", diagnostico=""):

            texto_clinico = diag_clinico if pd.notna(diag_clinico) and str(diag_clinico).strip() != "" else "Sin diagnóstico previo"
            fecha_hora = (datetime.now() - timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Informe de Análisis de Temblor", ln=True, align='C')

            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.cell(200, 10, f"Nombre: {nombre_paciente}", ln=True)
            pdf.cell(200, 10, f"Apellido: {apellido_paciente}", ln=True)
            try:
                edad_int = int(float(edad))  # por si viene como float
                edad_str = str(edad_int)
            except:
                edad_str = "No especificado"
            pdf.cell(200, 10, f"Edad: {edad_str}", ln=True)
            pdf.cell(200, 10, f"Sexo: {sexo}", ln=True)
            pdf.cell(200, 10, f"Diagnóstico clínico: {texto_clinico}", ln=True)
            pdf.cell(200, 10, f"Mano: {mano}", ln=True)
            pdf.cell(200, 10, f"Dedo: {dedo}", ln=True)
            pdf.cell(200, 10, f"Fecha y hora: {fecha_hora}", ln=True)

           # pdf.ln(5)
           # pdf.image("grafico_resumen.png", x=10, w=180)

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

          Para la varianza (m2/s4):
        Representa la dispersión de la señal. En el contexto de temblores:
        - Normal/sano: muy baja, puede estar entre 0.001 – 0.1 m2/s4.
        - Temblor leve: entre 0.1 – 0.5 m2/s4.
        - Temblor patológico (PK o TE): suele superar 1.0 m2/s4, llegando hasta 5–10 m2/s4 en casos severos.

          Para el RMS (m/s2):
        - Normal/sano: menor a 0.5 m/s2.
        - PK leve: entre 0.5 y 1.5 m/s2.
        - TE o PK severo: mayor a 2 m/s2.

        Nota clínica: Los valores de referencia presentados a continuación se basan en literatura científica.

        """

            texto_limpio = limpiar_texto_para_pdf(texto_original)
            pdf.multi_cell(0, 8, texto_limpio)
            pdf.set_font("Arial", 'B', 12)

            if diagnostico:
                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Diagnóstico automático:", ln=True)
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, diagnostico)
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 10, "La clasificacion automatica es orientativa y debe ser evaluada por un profesional.", ln=True)

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
            datos_personales = None

            for test, datos in mediciones_tests.items():
                df_ventana = analizar_temblor_por_ventanas_resultante(datos, fs=200)

                if datos_personales is None:
                    datos_personales = datos.iloc[0].to_frame().T

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

                nombre = datos_personales.iloc[0].get("Nombre", "No especificado")
                apellido = datos_personales.iloc[0].get("Apellido", "No especificado")
                edad = datos_personales.iloc[0].get("Edad", "No especificado")
                sexo = datos_personales.iloc[0].get("Sexo", "No especificado")
                diag_clinico = datos_personales.iloc[0].get("Diagnostico", "No disponible")
                mano = datos_personales.iloc[0].get("Mano", "No disponible")
                dedo = datos_personales.iloc[0].get("Dedo", "No disponible")

                df_resultados = pd.DataFrame(resultados_globales)
                st.subheader("Resultados Promediados por Test")
                st.dataframe(df_resultados)
                diagnostico = diagnosticar(df_resultados)
                st.subheader("Diagnóstico")
                st.write(diagnostico)

                generar_pdf(nombre, apellido, edad, sexo, diag_clinico, mano, dedo, df_resultados, diagnostico=diagnostico)
                with open("informe_temblor.pdf", "rb") as file:
                    st.download_button(
                        label="📄 Descargar Informe PDF",
                        data=file,
                        file_name="informe_temblor.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No se encontraron datos suficientes para el análisis.")




elif opcion == "2️⃣ Comparar dos configuraciones de estimulación":
    st.title("📊 Comparar dos configuraciones de estimulación")

    def analizar_configuracion(archivos):
        resultados = []
        datos_personales = {}
        parametros_estim = {}

        for archivo in archivos:
            df = pd.read_csv(archivo)

            # Extraer info personal y de estimulación desde primera fila
            if not datos_personales:
                datos_personales["Nombre"] = df.iloc[0, 1]
                datos_personales["Edad"] = df.iloc[0, 2]
                datos_personales["Diagnóstico"] = df.iloc[0, 3]
                parametros_estim["ECP"] = df.iloc[0, 4]
                parametros_estim["GPI"] = df.iloc[0, 5]
                parametros_estim["NST"] = df.iloc[0, 6]
                parametros_estim["Polaridad"] = df.iloc[0, 7]
                parametros_estim["Duración"] = df.iloc[0, 8]
                parametros_estim["Pulso"] = df.iloc[0, 9]
                parametros_estim["Corriente"] = df.iloc[0, 10]
                parametros_estim["Voltaje"] = df.iloc[0, 11]
                parametros_estim["Frecuencia"] = df.iloc[0, 12]

            # Limpiar datos y convertir a float
            df = df.iloc[1:].dropna()
            df.columns = ['Time', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
            df = df.astype(float)

            # Aceleración resultante
            df['A_resultante'] = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)

            # Filtrado Butterworth
            fs = 100
            lowcut = 4
            highcut = 12
            b, a = butter(4, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
            señal_filtrada = filtfilt(b, a, df['A_resultante'])

            # Ventanas
            ventana_size = int(2 * fs)
            solape = int(ventana_size * 0.5)
            ventanas = [
                señal_filtrada[i:i + ventana_size]
                for i in range(0, len(señal_filtrada) - ventana_size + 1, solape)
            ]

            frecs, rmss, vars_, amps = [], [], [], []
            for ventana in ventanas:
                N = len(ventana)
                fft_vals = np.fft.fft(ventana)
                fft_freqs = np.fft.fftfreq(N, d=1/fs)
                amplitudes = 2.0 / N * np.abs(fft_vals[:N // 2])
                frecuencias = fft_freqs[:N // 2]
                freq_dominante = frecuencias[np.argmax(amplitudes)]
                frecs.append(freq_dominante)
                rmss.append(np.sqrt(np.mean(ventana**2)))
                vars_.append(np.var(ventana))

                a_max = 9.81 * np.max(ventana)  # m/s²
                f = freq_dominante if freq_dominante > 0 else 5  # evitar división por cero
                amp_cm = (a_max / ((2 * np.pi * f) ** 2)) * 100  # desplazamiento en cm
                amps.append(amp_cm)

            resultados.append({
                "Frecuencia dominante (Hz)": np.mean(frecs),
                "RMS (g)": np.mean(rmss),
                "Varianza": np.mean(vars_),
                "Amplitud (cm)": np.mean(amps)
            })

        df_resultados = pd.DataFrame(resultados)
        promedio = df_resultados.mean().to_frame().T
        return promedio, datos_personales, parametros_estim

    def generar_conclusion_comparativa(res1, res2):
        score1 = res1.values[0]
        score2 = res2.values[0]
        mejor = "Configuración 1" if np.sum(score1) < np.sum(score2) else "Configuración 2"
        return f"La {mejor} presenta un menor promedio en las métricas analizadas, lo que indica una reducción más efectiva del temblor."

    def generar_pdf_comparativo(datos1, params1, res1, datos2, params2, res2, conclusion):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Informe Comparativo de Configuraciones de Estimulación", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        pdf.cell(0, 10, f"Nombre: {datos1['Nombre']}", ln=True)
        pdf.cell(0, 10, f"Edad: {datos1['Edad']}", ln=True)
        pdf.cell(0, 10, f"Diagnóstico: {datos1['Diagnóstico']}", ln=True)

        def agregar_parametros(titulo, params):
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, titulo, ln=True)
            pdf.set_font("Arial", "", 11)
            for k, v in params.items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)

        def agregar_resultados(titulo, res):
            pdf.ln(4)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, titulo, ln=True)
            pdf.set_font("Arial", "", 11)
            for col in res.columns:
                val = res[col].values[0]
                pdf.cell(0, 8, f"{col}: {val:.2f}", ln=True)

        agregar_parametros("Parámetros Configuración 1", params1)
        agregar_resultados("Resultados Configuración 1", res1)

        agregar_parametros("Parámetros Configuración 2", params2)
        agregar_resultados("Resultados Configuración 2", res2)

        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Conclusión:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, conclusion)

        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        return buffer

    st.markdown("### Suba los archivos de la **Configuración 1** (reposo, postural y acción)")
    archivos_config1 = st.file_uploader("Archivos configuración 1", type="csv", accept_multiple_files=True, key="conf1")

    st.markdown("### Suba los archivos de la **Configuración 2** (reposo, postural y acción)")
    archivos_config2 = st.file_uploader("Archivos configuración 2", type="csv", accept_multiple_files=True, key="conf2")

    if len(archivos_config1) == 3 and len(archivos_config2) == 3:
        if st.button("📊 Comparar configuraciones"):
            with st.spinner("Analizando Configuración 1..."):
                res1, datos1, params1 = analizar_configuracion(archivos_config1)

            with st.spinner("Analizando Configuración 2..."):
                res2, datos2, params2 = analizar_configuracion(archivos_config2)

            st.subheader("Resultados promedio de cada configuración")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Configuración 1**")
                st.dataframe(res1.style.format("{:.2f}"))
            with col2:
                st.markdown("**Configuración 2**")
                st.dataframe(res2.style.format("{:.2f}"))

            conclusion = generar_conclusion_comparativa(res1, res2)
            st.markdown("### 🧠 Conclusión automática")
            st.success(conclusion)

            buffer_pdf = generar_pdf_comparativo(datos1, params1, res1, datos2, params2, res2, conclusion)

            st.download_button(
                label="📄 Descargar informe PDF",
                data=buffer_pdf,
                file_name="informe_comparativo.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Debe subir exactamente 3 archivos CSV para cada configuración.")
