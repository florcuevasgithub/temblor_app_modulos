import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from fpdf import FPDF
from io import BytesIO
import tempfile
import os
import joblib
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

# Variables Globales
ventana_duracion_seg = 3
fs = 100

# Funciones globales para análisis y PDF
def analizar_temblor_por_ventanas_resultante(df, fs=100, ventana_duracion_seg=3):
    """
    Analiza el temblor resultante por ventanas de tiempo en un DataFrame.
    Calcula la frecuencia dominante, amplitud y RMS para cada ventana.
    Devuelve un DataFrame con las métricas por ventana y un resumen promedio.
    """
    columnas_aceleracion = ['Acc_x', 'Acc_y', 'Acc_z']
    if not all(col in df.columns for col in columnas_aceleracion):
        return pd.DataFrame(), pd.DataFrame()

    num_puntos_ventana = int(fs * ventana_duracion_seg)
    num_ventanas = len(df) // num_puntos_ventana

    resultados_ventanas = []

    for i in range(num_ventanas):
        inicio_idx = i * num_puntos_ventana
        fin_idx = inicio_idx + num_puntos_ventana
        ventana = df.iloc[inicio_idx:fin_idx]
        
        # Calcular el vector resultante de aceleración
        aceleracion_resultante = np.sqrt(ventana['Acc_x']**2 + ventana['Acc_y']**2 + ventana['Acc_z']**2)
        
        # Calcular RMS de la aceleración resultante
        rms_resultante = np.sqrt(np.mean(aceleracion_resultante**2))

        # Calcular PSD y frecuencia dominante
        frecuencias, psd = welch(aceleracion_resultante, fs=fs, nperseg=num_puntos_ventana)
        frecuencia_dominante = frecuencias[np.argmax(psd)]

        # Amplitud del temblor (aproximación)
        # Integración de la aceleración resultante para obtener la velocidad
        velocidad = np.cumsum(aceleracion_resultante) * (1/fs)
        # Integración de la velocidad para obtener el desplazamiento
        desplazamiento = np.cumsum(velocidad) * (1/fs)
        amplitud_temblor = np.ptp(desplazamiento) * 100 # Convertir a cm

        resultados_ventanas.append({
            'Ventana': i + 1,
            'Frecuencia Dominante (Hz)': frecuencia_dominante,
            'RMS (m/s2)': rms_resultante,
            'Amplitud Temblor (cm)': amplitud_temblor
        })

    if not resultados_ventanas:
        return pd.DataFrame(), pd.DataFrame()

    df_ventanas = pd.DataFrame(resultados_ventanas)
    df_promedio = df_ventanas.mean().to_frame().T
    df_promedio['Test'] = 'Promedio'
    df_promedio = df_promedio.drop(columns=['Ventana'])

    return df_promedio, df_ventanas

def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
    if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
        pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

def generar_pdf_analisis(paciente_data, estimulacion_data, resultados_df, graficos_paths):
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
    _imprimir_campo_pdf(pdf, "Nombre", paciente_data.get("Nombre"))
    _imprimir_campo_pdf(pdf, "Apellido", paciente_data.get("Apellido"))
    _imprimir_campo_pdf(pdf, "Edad", paciente_data.get("Edad"))
    _imprimir_campo_pdf(pdf, "Sexo", paciente_data.get("Sexo"))
    _imprimir_campo_pdf(pdf, "Diagnóstico", paciente_data.get("Diagnostico"))
    _imprimir_campo_pdf(pdf, "Tipo", paciente_data.get("Tipo"))
    _imprimir_campo_pdf(pdf, "Mano", paciente_data.get("Mano"))
    _imprimir_campo_pdf(pdf, "Dedo", paciente_data.get("Dedo"))
    _imprimir_campo_pdf(pdf, "Antecedente", paciente_data.get("Antecedente"))
    _imprimir_campo_pdf(pdf, "Medicacion", paciente_data.get("Medicacion"))
    pdf.ln(5)

    def imprimir_estimulacion(pdf_obj, parametros_dict, titulo):
        pdf_obj.set_font("Arial", 'B', 12)
        pdf_obj.cell(0, 10, titulo, ln=True)
        pdf_obj.set_font("Arial", size=10)
        
        parametros_a_imprimir_con_unidad = {
            "DBS": "", "Nucleo": "",
            "Voltaje_izq": " mV", "Corriente_izq": " mA", "Contacto_izq": "",
            "Frecuencia_izq": " Hz", "Pulso_izq": " µS",
            "Voltaje_dch": " mV", "Corriente_dch": " mA", "Contacto_dch": "",
            "Frecuencia_dch": " Hz", "Pulso_dch": " µS"
        }
        for param_key, unit in parametros_a_imprimir_con_unidad.items():
            value = parametros_dict.get(param_key)
            if value is not None and pd.notna(value) and str(value).strip() != "":
                pdf_obj.cell(200, 10, f"{param_key}: {value}{unit}", ln=True)
        pdf_obj.ln(5)

    imprimir_estimulacion(pdf, estimulacion_data, "Configuración de Estimulación")

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
        pdf.cell(40, 10, row['Frecuencia Dominante (Hz)'], 1)
        pdf.cell(30, 10, row['RMS (m/s2)'], 1)
        pdf.cell(50, 10, row['Amplitud Temblor (cm)'], 1)
        pdf.ln(10)
    pdf.ln(5)

    for i, img_path in enumerate(graficos_paths):
        if os.path.exists(img_path):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"Gráfico {i+1}", ln=True, align="C")
            pdf.image(img_path, x=15, w=180)
    
    pdf_output = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

def generar_pdf_comparativo(paciente_data, config1_data, config2_data, df_res1, df_res2, conclusion, graficos_paths):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Informe Comparativo de Mediciones", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
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

    imprimir_parametros_y_config(pdf, config1_data, "Configuración Medición 1")
    imprimir_parametros_y_config(pdf, config2_data, "Configuración Medición 2")

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

    imprimir_resultados(pdf, df_res1, "Resultados Medición 1")
    imprimir_resultados(pdf, df_res2, "Resultados Medición 2")

    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Conclusión", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, conclusion)

    for i, img_path in enumerate(graficos_paths):
        if os.path.exists(img_path):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"Gráfico {i+1}", ln=True, align="C")
            pdf.image(img_path, x=15, w=180)

    pdf_output = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

def generar_pdf_diagnostico(paciente_data, medicion_data, resultados_df, prediccion_texto, graficos_paths):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Informe de Diagnóstico de Temblor", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)

    def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
        if valor is not None and str(valor).strip() != "" and str(valor).lower() != "no especificado":
            pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

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
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Configuración de la Medición", ln=True)
    pdf.set_font("Arial", size=10)
    _imprimir_campo_pdf(pdf, "Mano", medicion_data.get("Mano"))
    _imprimir_campo_pdf(pdf, "Dedo", medicion_data.get("Dedo"))
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


# Sidebar para la navegación de módulos
st.sidebar.title("Menú")
opcion = st.sidebar.radio("Selecciona un módulo:", [
    "1️⃣ Análisis de mediciones",
    "2️⃣ Comparación de mediciones",
    "3️⃣ Diagnóstico tentativo"
])


# ------------------ MÓDULO 1: ANÁLISIS INDIVIDUAL --------------------
if opcion == "1️⃣ Análisis de mediciones":
    st.title("🔬 Análisis de una Medición")
    st.markdown("### Cargar archivos CSV para el análisis")

    def extraer_datos_paciente(df_csv):
        datos_paciente = {
            "Nombre": df_csv.loc[0, 'Nombre'] if 'Nombre' in df_csv.columns else 'No especificado',
            "Apellido": df_csv.loc[0, 'Apellido'] if 'Apellido' in df_csv.columns else 'No especificado',
            "Edad": int(df_csv.loc[0, 'Edad']) if 'Edad' in df_csv.columns and pd.notna(df_csv.loc[0, 'Edad']) else 'No especificada',
            "Sexo": df_csv.loc[0, 'Sexo'] if 'Sexo' in df_csv.columns else 'No especificado',
            "Diagnostico": df_csv.loc[0, 'Diagnostico'] if 'Diagnostico' in df_csv.columns else 'No especificado',
            "Tipo": df_csv.loc[0, 'Tipo'] if 'Tipo' in df_csv.columns else 'No especificado',
            "Mano": df_csv.loc[0, 'Mano'] if 'Mano' in df_csv.columns else 'No especificado',
            "Dedo": df_csv.loc[0, 'Dedo'] if 'Dedo' in df_csv.columns else 'No especificado',
            "Antecedente": df_csv.loc[0, 'Antecedente'] if 'Antecedente' in df_csv.columns else 'No especificado',
            "Medicacion": df_csv.loc[0, 'Medicacion'] if 'Medicacion' in df_csv.columns else 'No especificado',
        }
        return datos_paciente

    def extraer_datos_estimulacion(df_csv):
        metadata_dict = {}
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
            "Ancho de pulso [µS]_dch": "Pulso_dch"
        }
        for csv_col, pdf_label in column_map.items():
            if csv_col in df_csv.columns:
                value = df_csv.loc[0, csv_col]
                metadata_dict[pdf_label] = value
        return metadata_dict

    reposo_file = st.file_uploader("Archivo de REPOSO", type="csv", key="reposo")
    postural_file = st.file_uploader("Archivo de POSTURAL", type="csv", key="postural")
    accion_file = st.file_uploader("Archivo de ACCION", type="csv", key="accion")

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

    if st.button("Realizar Análisis"):
        if reposo_file is None and postural_file is None and accion_file is None:
            st.warning("Por favor, sube al menos un archivo CSV para el análisis.")
        else:
            archivos_cargados = {
                "Reposo": reposo_file,
                "Postural": postural_file,
                "Acción": accion_file
            }
            
            primer_archivo = next((f for f in archivos_cargados.values() if f is not None), None)
            if primer_archivo is not None:
                df_primer_archivo = pd.read_csv(primer_archivo, encoding='latin1')
                datos_paciente = extraer_datos_paciente(df_primer_archivo)
                datos_estimulacion = extraer_datos_estimulacion(df_primer_archivo)

                st.subheader("Datos del Paciente y Configuración de Estimulación")
                st.write(f"**Nombre:** {datos_paciente['Nombre']} {datos_paciente['Apellido']}")
                st.write(f"**Edad:** {datos_paciente['Edad']}")
                st.write(f"**Sexo:** {datos_paciente['Sexo']}")
                st.write(f"**Diagnóstico:** {datos_paciente['Diagnostico']}")
                st.write(f"**Mano de la Medición:** {datos_paciente['Mano']}")
                st.write(f"**Dedo de la Medición:** {datos_paciente['Dedo']}")
                st.write("---")
                st.write("**Parámetros de Estimulación**")
                for key, value in datos_estimulacion.items():
                    if pd.notna(value):
                        st.write(f"**{key.replace('_izq', ' (Izquierda)').replace('_dch', ' (Derecha)')}:** {value}")

                st.subheader("Resultados del Análisis de Temblor por Test")
                resultados_analisis = []
                graficos_paths = []

                for test_name, uploaded_file in archivos_cargados.items():
                    if uploaded_file is not None:
                        uploaded_file.seek(0)
                        df_temp = pd.read_csv(uploaded_file, encoding='latin1')
                        df_promedio, df_ventanas = analizar_temblor_por_ventanas_resultante(df_temp, fs=100)

                        if not df_ventanas.empty:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            df_ventanas["Tiempo (segundos)"] = df_ventanas["Ventana"] * ventana_duracion_seg
                            ax.plot(df_ventanas["Tiempo (segundos)"], df_ventanas["Amplitud Temblor (cm)"])
                            ax.set_title(f"Amplitud de Temblor por Ventana - {test_name}")
                            ax.set_xlabel("Tiempo (segundos)")
                            ax.set_ylabel("Amplitud (cm)")
                            ax.grid(True)
                            st.pyplot(fig)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                                fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                                graficos_paths.append(tmp_img.name)
                            plt.close(fig)

                        if not df_promedio.empty:
                            promedio_data = {
                                "Test": test_name,
                                "Frecuencia Dominante (Hz)": f"{df_promedio.iloc[0]['Frecuencia Dominante (Hz)']:.2f}",
                                "RMS (m/s2)": f"{df_promedio.iloc[0]['RMS (m/s2)']:.4f}",
                                "Amplitud Temblor (cm)": f"{df_promedio.iloc[0]['Amplitud Temblor (cm)']:.2f}",
                            }
                            resultados_analisis.append(promedio_data)

                if resultados_analisis:
                    df_resultados = pd.DataFrame(resultados_analisis)
                    st.dataframe(df_resultados)

                    pdf_output = generar_pdf_analisis(datos_paciente, datos_estimulacion, df_resultados, graficos_paths)
                    st.download_button(
                        label="Descargar Informe PDF",
                        data=pdf_output.getvalue(),
                        file_name="informe_analisis_temblor.pdf",
                        mime="application/pdf"
                    )
                    
                    for path in graficos_paths:
                        if os.path.exists(path):
                            os.remove(path)
                else:
                    st.warning("No se pudieron calcular los resultados para los archivos cargados.")

# ------------------ MÓDULO 2: COMPARACIÓN DE MEDICIONES --------------------

elif opcion == "2️⃣ Comparación de mediciones":
    st.title("📊 Comparación de Mediciones")

    def extraer_datos_paciente(df_csv):
        datos_paciente = {
            "Nombre": df_csv.loc[0, 'Nombre'] if 'Nombre' in df_csv.columns else 'No especificado',
            "Apellido": df_csv.loc[0, 'Apellido'] if 'Apellido' in df_csv.columns else 'No especificado',
            "Edad": int(df_csv.loc[0, 'Edad']) if 'Edad' in df_csv.columns and pd.notna(df_csv.loc[0, 'Edad']) else 'No especificada',
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
                            'Frecuencia Dominante (Hz)': freq,
                            'RMS (m/s2)': rms,
                            'Amplitud Temblor (cm)': amp
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

            st.subheader("Resultados Medición 1")
            st.dataframe(df_resultados_config1)

            st.subheader("Resultados Medición 2")
            st.dataframe(df_resultados_config2)
            
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
            st.subheader("Conclusión del Análisis Comparativo")
            st.write(conclusion)

            st.subheader("Comparación Gráfica de Amplitud por Ventana")
            nombres_test = ["Reposo", "Postural", "Acción"]
            graficos_paths = []

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

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                            fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                            graficos_paths.append(tmp_img.name)
                        plt.close(fig)
                    else:
                        st.warning(f"No hay suficientes datos de ventanas para graficar el test: {test}")
                else:
                    st.warning(f"Faltan archivos para el test {test} en al menos una Medición.")

            pdf_output = generar_pdf_comparativo(
                datos_paciente, config1_params, config2_params, df_resultados_config1, df_resultados_config2, conclusion, graficos_paths
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

    def extraer_datos_paciente(df_csv):
        datos_paciente = {
            "Nombre": df_csv.loc[0, 'Nombre'] if 'Nombre' in df_csv.columns else 'No especificado',
            "Apellido": df_csv.loc[0, 'Apellido'] if 'Apellido' in df_csv.columns else 'No especificado',
            "Edad": int(df_csv.loc[0, 'Edad']) if 'Edad' in df_csv.columns and pd.notna(df_csv.loc[0, 'Edad']) else 'No especificada',
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
            "Mano": "Mano",
            "Dedo": "Dedo"
        }
        for csv_col, pdf_label in column_map.items():
            if csv_col in df_csv.columns:
                value = df_csv.loc[0, csv_col]
                metadata_dict[pdf_label] = value
        return metadata_dict

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

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        fig.savefig(tmp_img.name, format='png', bbox_inches='tight')
                        graficos_paths.append(tmp_img.name)
                    plt.close(fig)
                else:
                    st.warning("No hay suficientes datos de ventanas para graficar los archivos de predicción.")

                pdf_output = generar_pdf_diagnostico(
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
                
                for path in graficos_paths:
                    if os.path.exists(path):
                        os.remove(path)
