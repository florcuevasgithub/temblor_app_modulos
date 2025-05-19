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

    st.markdown("### Cargar archivos de la **Configuración 1**")
    config1_archivos = {
        "Reposo": st.file_uploader("CSV Reposo - Config 1", type="csv", key="reposo1"),
        "Postural": st.file_uploader("CSV Postural - Config 1", type="csv", key="postural1"),
        "Acción": st.file_uploader("CSV Acción - Config 1", type="csv", key="accion1")
    }

    st.markdown("### Cargar archivos de la **Configuración 2**")
    config2_archivos = {
        "Reposo": st.file_uploader("CSV Reposo - Config 2", type="csv", key="reposo2"),
        "Postural": st.file_uploader("CSV Postural - Config 2", type="csv", key="postural2"),
        "Acción": st.file_uploader("CSV Acción - Config 2", type="csv", key="accion2")
    }

    def extraer_datos_personales(df):
        keys = ["Nombre", "Apellido", "Edad", "Sexo"]
        return {k: df.iloc[0][k] if k in df.columns else "No especificado" for k in keys}

    def extraer_parametros_estim(df):
        keys = ["ECP", "GPI", "NST", "Polaridad", "Duración", "Pulso", "Corriente", "Voltaje", "Frecuencia"]
        return {k: df.iloc[0][k] if k in df.columns else "No especificado" for k in keys}

    def analizar_configuracion(archivos, fs=200):
        resultados = []
        for test, archivo in archivos.items():
            if archivo is not None:
                df = pd.read_csv(archivo)
                df_ventana = analizar_temblor_por_ventanas_resultante(df, fs=fs)
                if not df_ventana.empty:
                    prom = df_ventana.mean(numeric_only=True)
                    freq = prom['Frecuencia Dominante (Hz)']
                    amp_g = prom['Amplitud Temblor (g)']
                    amp_cm = (amp_g * 981) / ((2 * np.pi * freq) ** 2) if freq > 0 else 0.0

                    resultados.append({
                        'Test': test,
                        'Frecuencia Dominante (Hz)': round(freq, 2),
                        'Varianza (m2/s4)': round(prom['Varianza (m2/s4)'], 4),
                        'RMS (m/s2)': round(prom['RMS (m/s2)'], 4),
                        'Amplitud Temblor (cm)': round(amp_cm, 2)
                    })
        return pd.DataFrame(resultados)

    if st.button("Comparar configuraciones"):
        # Validar que los 6 archivos estén cargados
        archivos_cargados = all([
            config1_archivos[test] is not None and config2_archivos[test] is not None
            for test in ["Reposo", "Postural", "Acción"]
        ])

        if not archivos_cargados:
            st.warning("Por favor, cargue los 3 archivos CSV para ambas configuraciones.")
        else:
            # Extraer datos personales y parámetros de cada configuración (desde el CSV "Reposo" que es el primero)
            df_config1_reposo = pd.read_csv(config1_archivos["Reposo"])
            df_config2_reposo = pd.read_csv(config2_archivos["Reposo"])

            datos_personales = extraer_datos_personales(df_config1_reposo)
            parametros_config1 = extraer_parametros_estim(df_config1_reposo)
            parametros_config2 = extraer_parametros_estim(df_config2_reposo)

            # Analizar las configuraciones
            df_resultados_config1 = analizar_configuracion(config1_archivos)
            df_resultados_config2 = analizar_configuracion(config2_archivos)

            st.subheader("Resultados Configuración 1")
            st.dataframe(df_resultados_config1)

            st.subheader("Resultados Configuración 2")
            st.dataframe(df_resultados_config2)

            # Comparar promedios totales para decidir cuál es mejor (menor temblor)
            prom_config1 = df_resultados_config1.mean(numeric_only=True)
            prom_config2 = df_resultados_config2.mean(numeric_only=True)

            # Definir una "puntuación" simple como suma de métricas que reflejan temblor (mayor es peor)
            puntaje1 = prom_config1['Frecuencia Dominante (Hz)'] + prom_config1['Varianza (m2/s4)'] + prom_config1['RMS (m/s2)'] + prom_config1['Amplitud Temblor (cm)']
            puntaje2 = prom_config2['Frecuencia Dominante (Hz)'] + prom_config2['Varianza (m2/s4)'] + prom_config2['RMS (m/s2)'] + prom_config2['Amplitud Temblor (cm)']

            if puntaje1 < puntaje2:
                conclusion = "La Configuración 1 muestra una reducción mayor del temblor."
            elif puntaje2 < puntaje1:
                conclusion = "La Configuración 2 muestra una reducción mayor del temblor."
            else:
                conclusion = "Ambas configuraciones muestran resultados similares."

            st.subheader("Conclusión")
            st.write(conclusion)

            # Generar PDF con datos personales, parámetros, resultados y conclusión
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Informe Comparativo de Configuraciones de Estimulación", ln=True, align="C")

            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.cell(0, 10, f"Nombre: {datos_personales.get('Nombre', 'No especificado')}", ln=True)
            pdf.cell(0, 10, f"Apellido: {datos_personales.get('Apellido', 'No especificado')}", ln=True)
            pdf.cell(0, 10, f"Edad: {datos_personales.get('Edad', 'No especificado')}", ln=True)
            pdf.cell(0, 10, f"Sexo: {datos_personales.get('Sexo', 'No especificado')}", ln=True)
            pdf.ln(5)

            def imprimir_parametros(pdf, parametros, titulo):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, titulo, ln=True)
                pdf.set_font("Arial", size=12)
                for key, value in parametros.items():
                    pdf.cell(0, 8, f"{key}: {value}", ln=True)
                pdf.ln(5)

            imprimir_parametros(pdf, parametros_config1, "Parámetros Configuración 1")
            imprimir_parametros(pdf, parametros_config2, "Parámetros Configuración 2")

            def imprimir_resultados(pdf, df, titulo):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, titulo, ln=True)
                pdf.set_font("Arial", 'B', 12)
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
                pdf.ln(5)

            imprimir_resultados(pdf, df_resultados_config1, "Resultados Configuración 1")
            imprimir_resultados(pdf, df_resultados_config2, "Resultados Configuración 2")

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Conclusión", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, conclusion)

            # Guardar PDF en buffer para descargar
            pdf_output = BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)

            st.download_button(
                label="Descargar Informe PDF",
                data=pdf_output,
                file_name="Informe_Comparacion_Configuraciones.pdf",
                mime="application/pdf"
            )
