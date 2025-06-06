# -*- coding: utf-8 -*-

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
from io import BytesIO, StringIO
from ahrs.filters import Mahony
import os
import glob
import streamlit as st

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
# --------- Funciones compartidas ----------
def filtrar_temblor(signal, fs=100):
    b, a = butter(N=4, Wn=[1, 15], btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def q_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),       2*(x*y - z*w),          2*(x*z + y*w)],
        [2*(x*y + z*w),             1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
        [2*(x*z - y*w),             2*(y*z + x*w),          1 - 2*(x**2 + y**2)]
    ])

def analizar_temblor_por_ventanas_resultante(df, fs=100, ventana_seg=2):
    required_cols = ['Acel_X', 'Acel_Y', 'Acel_Z', 'GiroX', 'GiroY', 'GiroZ']
    df = df[required_cols].dropna()
    acc = df[['Acel_X', 'Acel_Y', 'Acel_Z']].to_numpy()
    gyr = np.radians(df[['GiroX', 'GiroY', 'GiroZ']].to_numpy())
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
    num_ventanas = len(señal_filtrada) // tamaño_ventana

    for i in range(num_ventanas):
        segmento = señal_filtrada[i*tamaño_ventana:(i+1)*tamaño_ventana]
        if len(segmento) < tamaño_ventana:
            continue
        segmento = segmento - np.mean(segmento)

        f, Pxx = welch(segmento, fs=fs, nperseg=tamaño_ventana)
        freq_dominante = f[np.argmax(Pxx)]

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
           'Varianza (m2/s4)': varianza,
           'RMS (m/s2)': rms,
           'Amplitud Temblor (g)': amp_g,
           'Amplitud Temblor (cm)': amp_cm
         })

    df_por_ventana = pd.DataFrame(resultados_por_ventana)

    if not df_por_ventana.empty:
        promedio = df_por_ventana.mean(numeric_only=True).to_dict()
        df_promedio = pd.DataFrame([{
            'Frecuencia Dominante (Hz)': promedio['Frecuencia Dominante (Hz)'],
            'Varianza (m2/s4)': promedio['Varianza (m2/s4)'],
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
opcion = st.sidebar.radio("Selecciona una opción:", ["1️⃣ Análisis de una medición", "2️⃣ Comparar dos mediciones"])
if st.sidebar.button("🔄 Nuevo análisis"):
    manejar_reinicio()

if opcion == "1️⃣ Análisis de una medición":
    st.title("📈​ Análisis de una medición")

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

    def generar_pdf(nombre_paciente, apellido_paciente, edad, sexo, diag_clinico, mano, dedo, df, nombre_archivo="informe_temblor.pdf", diagnostico="", fig=None):

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
            edad_int = int(float(edad))
            edad_str = str(edad_int)
        except:
            edad_str = "No especificado"
        pdf.cell(200, 10, f"Edad: {edad_str}", ln=True)
        pdf.cell(200, 10, f"Sexo: {sexo}", ln=True)
        #pdf.cell(200, 10, f"Diagnóstico clínico: {texto_clinico}", ln=True)
        pdf.cell(200, 10, f"Mano: {mano}", ln=True)
        pdf.cell(200, 10, f"Dedo: {dedo}", ln=True)
        pdf.cell(200, 10, f"Fecha y hora: {fecha_hora}", ln=True)

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


    if st.button("Iniciar análisis"):
        resultados_globales = []
        mediciones_tests = {test: pd.read_csv(file) for test, file in uploaded_files.items() if file is not None}
        datos_personales = None
        ventanas_por_test = []

        if not mediciones_tests:
            st.warning("Por favor, sube al menos un archivo para iniciar el análisis.")
        else:
            for test, datos in mediciones_tests.items():
                df_promedio, df_ventanas = analizar_temblor_por_ventanas_resultante(datos, fs=100)
                if datos_personales is None:
                    # Intenta obtener datos personales del primer archivo cargado.
                    # Asegúrate de que las columnas existan antes de intentar acceder.
                    if not datos.empty:
                        datos_personales_temp = {}
                        for col in ["Nombre", "Apellido", "Edad", "Sexo", "Diagnostico", "Mano", "Dedo"]:
                            if col in datos.columns and not pd.isna(datos.iloc[0].get(col)):
                                datos_personales_temp[col] = datos.iloc[0][col]
                        datos_personales = pd.DataFrame([datos_personales_temp])


                if not df_promedio.empty:
                    fila = df_promedio.iloc[0].to_dict()
                    fila['Test'] = test
                    resultados_globales.append(fila)

                if not df_ventanas.empty:
                    df_ventanas["Test"] = test
                    ventanas_por_test.append(df_ventanas)

            fig = None
            if ventanas_por_test:
                fig, ax = plt.subplots(figsize=(10, 6))

                for df in ventanas_por_test:
                    test_name = df["Test"].iloc[0]
                    ax.plot(df["Ventana"], df["Amplitud Temblor (cm)"], label=f"{test_name}")

                ax.set_title("Amplitud de Temblor ")
                ax.set_xlabel("Ventana de tiempo")
                ax.set_ylabel("Amplitud (cm)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)


            if resultados_globales:
                nombre = "No especificado"
                apellido = "No especificado"
                edad = "No especificado"
                sexo = "No especificado"
                diag_clinico = "Sin diagnóstico"
                mano = "No especificado"
                dedo = "No especificado"

                if datos_personales is not None and not datos_personales.empty:
                    nombre = datos_personales.iloc[0].get("Nombre", nombre)
                    apellido = datos_personales.iloc[0].get("Apellido", apellido)
                    edad = datos_personales.iloc[0].get("Edad", edad)
                    sexo = datos_personales.iloc[0].get("Sexo", sexo)
                    diag_clinico = datos_personales.iloc[0].get("Diagnostico", diag_clinico)
                    mano = datos_personales.iloc[0].get("Mano", mano)
                    dedo = datos_personales.iloc[0].get("Dedo", dedo)

                df_resultados_final = pd.DataFrame(resultados_globales)
                diagnostico_auto = diagnosticar(df_resultados_final)

                # --- Mostrar la tabla en Streamlit ---
                st.subheader("Resultados del Análisis de Temblor")
                st.dataframe(df_resultados_final.set_index('Test')) # Display the DataFrame

                generar_pdf(
                    nombre, apellido, edad, sexo,
                    diag_clinico, mano, dedo,
                    df_resultados_final,
                    nombre_archivo="informe_temblor.pdf",
                    diagnostico=diagnostico_auto,
                    fig=fig
                )

                with open("informe_temblor.pdf", "rb") as f:
                    st.download_button("📄 Descargar informe PDF", f, file_name="informe_temblor.pdf")
            else:
                st.warning("No se encontraron datos suficientes para el análisis.")



elif opcion == "2️⃣ Comparar dos mediciones":
    st.title("📊 Comparar dos mediciones")

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
                df = pd.read_csv(archivo)
                df_promedio, df_ventana = analizar_temblor_por_ventanas_resultante(df, fs=fs)
                if isinstance(df_ventana, pd.DataFrame) and not df_ventana.empty:
                    prom = df_promedio.iloc[0] if not df_promedio.empty else None
                    if prom is not None:
                        freq = prom['Frecuencia Dominante (Hz)']
                        amp_cm = prom['Amplitud Temblor (cm)']
                        resultados.append({
                            'Test': test,
                            'Frecuencia Dominante (Hz)': round(freq, 2),
                            #'Varianza (m2/s4)': round(prom['Varianza (m2/s4)'], 4),
                            'RMS (m/s2)': round(prom['RMS (m/s2)'], 4),
                            'Amplitud Temblor (cm)': round(amp_cm, 2)
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
            df_config1_reposo = pd.read_csv(config1_archivos["Reposo"])
            df_config2_reposo = pd.read_csv(config2_archivos["Reposo"])

            def limpiar_campos(df, campos):
                resultado = {}
                for campo in campos:
                    valor = df.iloc[0].get(campo, None)
                    if pd.isna(valor) or str(valor).strip() == "":
                        valor = None
                    resultado[campo] = valor
                return resultado

            campos_personales = ["Nombre", "Apellido", "Edad", "Sexo"]
            campos_estim = ["ECP", "GPI", "NST", "Polaridad", "Duración", "Pulso", "Corriente", "Voltaje", "Frecuencia", "Mano", "Dedo"]

            datos_personales = limpiar_campos(df_config1_reposo, campos_personales)
            parametros_config1 = limpiar_campos(df_config1_reposo, campos_estim)
            parametros_config2 = limpiar_campos(df_config2_reposo, campos_estim)

            try:
                if datos_personales["Edad"] is not None:
                    edad_int = int(float(datos_personales["Edad"]))
                    datos_personales["Edad"] = str(edad_int)
            except Exception:
                datos_personales["Edad"] = None

            df_resultados_config1 = analizar_configuracion(config1_archivos)
            df_resultados_config2 = analizar_configuracion(config2_archivos)

            # Inicializar PDF antes de graficar
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Informe Comparativo de Mediciones", ln=True, align="C")

            pdf.set_font("Arial", size=10)
            pdf.ln(10)
            pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)

            def imprimir_campo_si_valido(pdf, etiqueta, valor):
                if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
                    pdf.cell(0, 10, f"{etiqueta}: {valor}", ln=True)

            def imprimir_parametros(pdf, parametros, titulo):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, titulo, ln=True)
                pdf.set_font("Arial", size=12)
                for key, value in parametros.items():
                    if value is not None and str(value).strip() != "":
                        if key == "Duración":
                             pdf.cell(0, 8, f"{key}: {value} ms", ln=True) # Assuming ms
                        elif key == "Pulso":
                            pdf.cell(0, 8, f"{key}: {value} µs", ln=True) # Assuming µs
                        elif key == "Corriente":
                            pdf.cell(0, 8, f"{key}: {value} mA", ln=True) # Assuming mA
                        elif key == "Voltaje":
                            pdf.cell(0, 8, f"{key}: {value} V", ln=True) # Assuming Volts
                        elif key == "Frecuencia" and key in campos_estim: # This refers to stimulation frequency, not tremor
                            pdf.cell(0, 8, f"{key}: {value} Hz", ln=True)
                        else:
                            pdf.cell(0, 8, f"{key}: {value}", ln=True)
                pdf.ln(5)

            def imprimir_resultados(pdf, df, titulo):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, titulo, ln=True)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(30, 10, "Test", 1)
                pdf.cell(40, 10, "Frecuencia (Hz)", 1)
                #pdf.cell(30, 10, "Varianza", 1)
                pdf.cell(30, 10, "RMS", 1)
                pdf.cell(50, 10, "Amplitud (cm)", 1)
                pdf.ln(10)
                pdf.set_font("Arial", "", 10)

                for _, row in df.iterrows():
                    pdf.cell(30, 10, row['Test'], 1)
                    pdf.cell(40, 10, f"{row['Frecuencia Dominante (Hz)']:.2f}", 1)
                    #pdf.cell(30, 10, f"{row['Varianza (m2/s4)']:.4f}", 1)
                    pdf.cell(30, 10, f"{row['RMS (m/s2)']:.4f}", 1)
                    pdf.cell(50, 10, f"{row['Amplitud Temblor (cm)']:.2f}", 1)
                    pdf.ln(10)
                pdf.ln(5)

            imprimir_campo_si_valido(pdf, "Nombre", datos_personales.get("Nombre"))
            imprimir_campo_si_valido(pdf, "Apellido", datos_personales.get("Apellido"))
            imprimir_campo_si_valido(pdf, "Edad", datos_personales.get("Edad"))
            imprimir_campo_si_valido(pdf, "Sexo", datos_personales.get("Sexo"))

            pdf.ln(5)

            imprimir_parametros(pdf, parametros_config1, "Parámetros Medición 1")
            imprimir_parametros(pdf, parametros_config2, "Parámetros Medición 2")
            imprimir_resultados(pdf, df_resultados_config1, "Resultados Medición 1")
            imprimir_resultados(pdf, df_resultados_config2, "Resultados Medición 2")

            amp_avg_config1 = df_resultados_config1['Amplitud Temblor (cm)'].mean()
            amp_avg_config2 = df_resultados_config2['Amplitud Temblor (cm)'].mean()

            rms_avg_config1 = df_resultados_config1['RMS (m/s2)'].mean()
            rms_avg_config2 = df_resultados_config2['RMS (m/s2)'].mean()

            conclusion = ""
            if amp_avg_config1 < amp_avg_config2:
                conclusion = (
                    f"La Medición 1 muestra una amplitud de temblor promedio ({amp_avg_config1:.2f} cm) "
                    f"más baja que la Medición 2 ({amp_avg_config2:.2f} cm), lo que sugiere una mayor reducción del temblor "
                )
            elif amp_avg_config2 < amp_avg_config1:
                conclusion = (
                    f"La Medición 2 muestra una amplitud de temblor promedio ({amp_avg_config2:.2f} cm) "
                    f"más baja que la Medición 1 ({amp_avg_config1:.2f} cm), lo que sugiere una mayor reducción del temblor "
                )
            else:
                conclusion = (
                    f"Ambas mediciones muestran amplitudes de temblor promedio muy similares ({amp_avg_config1:.2f} cm). "
                )

            st.subheader("Resultados Medición 1")
            st.dataframe(df_resultados_config1)

            st.subheader("Resultados Medición 2")
            st.dataframe(df_resultados_config2)

            st.subheader("Comparación Gráfica de Amplitud por Ventana")
            nombres_test = ["Reposo", "Acción", "Postural"]

            for test in nombres_test:
                archivo1 = config1_archivos[test]
                archivo2 = config2_archivos[test]

                if archivo1 is not None and archivo2 is not None:
                    archivo1.seek(0)
                    archivo2.seek(0)
                    df1 = pd.read_csv(archivo1)
                    df2 = pd.read_csv(archivo2)

                    df1_promedio, df1_ventanas = analizar_temblor_por_ventanas_resultante(df1, fs=100)
                    df2_promedio, df2_ventanas = analizar_temblor_por_ventanas_resultante(df2, fs=100)

                    if not df1_ventanas.empty and not df2_ventanas.empty:
                        fig, ax = plt.subplots(figsize=(10, 5)) # Added figsize for better control
                        ax.plot(df1_ventanas["Ventana"], df1_ventanas["Amplitud Temblor (cm)"], label="Configuración 1", color="blue")
                        ax.plot(df2_ventanas["Ventana"], df2_ventanas["Amplitud Temblor (cm)"], label="Configuración 2", color="orange")
                        ax.set_title(f"Amplitud por Ventana - {test}")
                        ax.set_xlabel("Ventana")
                        ax.set_ylabel("Amplitud (cm)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        # --- Corrected Image Saving for PDF ---
                        # Use tempfile to save the image to a temporary file,
                        # and then pass the path to pdf.image()
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                            fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                            image_path_for_pdf = tmp_img.name

                        try:
                            pdf.image(image_path_for_pdf, x=15, w=180)
                        finally:
                            os.remove(image_path_for_pdf) # Clean up the temporary file

                        pdf.ln(10)
                        plt.close(fig) # Close the figure to free up memory
                    else:
                        st.warning(f"No hay suficientes datos de ventanas para graficar el test: {test}")
                else:
                    st.warning(f"Faltan archivos para el test {test} en al menos una Medición.")

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Conclusión", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 10, conclusion)

            pdf_output = BytesIO()
            # It's better to explicitly output to a BytesIO object directly
            # and then get its value.
            pdf_bytes = pdf.output(dest='S').encode('latin1') # Output to string, then encode
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0) # Rewind the buffer to the beginning

            st.download_button(
                label="Descargar Informe PDF",
                data=pdf_output.getvalue(), # Use .getvalue() when passing BytesIO content to download_button
                file_name="informe_comparativo_temblor.pdf", 
                mime="application/pdf"
            )
            st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
