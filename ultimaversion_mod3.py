import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import welch, butter, filtfilt
import joblib
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os
import tempfile
import json # Asegúrate de que json también esté importado si lo usas en extraer_datos_paciente
import ahrs # Importar ahrs para la compensación de gravedad


# --- DEFINICIONES DE CONSTANTES GLOBALES ---
ventana_duracion_seg = 2 # Duración fija de la ventana de análisis


# --- DEFINICIONES DE FUNCIONES AUXILIARES (Deben ir aquí, fuera de cualquier if/elif de opción) ---

# Función para extraer datos del paciente de un DataFrame
# Esta versión lee 'sexo', 'edad', 'mano_medida', 'dedo_medido'
def extraer_datos_paciente(df):
    datos = {
        "sexo": "No especificado",
        "edad": 0,
        "mano_medida": "No especificada",
        "dedo_medido": "No especificado"
    }
    if not df.empty:
        # Asegurarse de que las columnas existan y no sean NaN antes de intentar acceder
        if "sexo" in df.columns and pd.notna(df.iloc[0]["sexo"]):
            datos["sexo"] = str(df.iloc[0]["sexo"])
        if "edad" in df.columns and pd.notna(df.iloc[0]["edad"]):
            try:
                datos["edad"] = int(float(str(df.iloc[0]["edad"]).replace(',', '.')))
            except (ValueError, TypeError):
                datos["edad"] = 0
        if "mano_medida" in df.columns and pd.notna(df.iloc[0]["mano_medida"]):
            datos["mano_medida"] = str(df.iloc[0]["mano_medida"])
        if "dedo_medido" in df.columns and pd.notna(df.iloc[0]["dedo_medido"]):
            datos["dedo_medido"] = str(df.iloc[0]["dedo_medido"])
    return datos


# Función de análisis de temblor avanzado (con compensación de gravedad)
# Esta es la versión completa con el uso de 'ahrs' que ya te había compartido
def analizar_temblor_por_ventanas_avanzado(df, fs, ventana_duracion_seg):
    df = df.copy() # Trabajar con una copia para evitar SettingWithCopyWarning

    # Convertir nombres de columnas a minúsculas y reemplazar espacios por guiones bajos
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Renombrar columnas para consistencia si usan "acc_" o "gyr_"
    df = df.rename(columns={
        'acc_x': 'acel_x', 'acc_y': 'acel_y', 'acc_z': 'acel_z',
        'gyr_x': 'gyro_x', 'gyr_y': 'gyro_y', 'gyr_z': 'gyro_z',
        'q_w': 'q_w', 'q_x': 'q_x', 'q_y': 'q_y', 'q_z': 'q_z',
        'qw': 'q_w', 'qx': 'q_x', 'qy': 'q_y', 'qz': 'q_z' # Para manejar qw, qx, qy, qz
    })

    # Asegurarse de que las columnas de cuaterniones y aceleración existan
    required_cols_accel = ['acel_x', 'acel_y', 'acel_z']
    required_cols_quat = ['q_w', 'q_x', 'q_y', 'q_z']
    
    if not all(col in df.columns for col in required_cols_accel + required_cols_quat):
        st.error("Columnas de aceleración o cuaterniones incompletas. Asegúrate de tener 'acel_x, acel_y, acel_z' y 'q_w, q_x, q_y, q_z' (o sus equivalentes 'Acc_X, Acc_Y, Acc_Z, qW, qX, qY, qZ').")
        return pd.DataFrame(), pd.DataFrame()

    # Convertir columnas a formato numérico, manejando comas como decimales
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except ValueError:
                pass # Ignora columnas que no se puedan convertir a numérico

    # --- FILTRADO Y COMPENSACIÓN DE GRAVEDAD ---
    nyquist = 0.5 * fs
    low_cut = 0.5  # Hz, para eliminar deriva (movimientos lentos)
    high_cut = 15.0 # Hz, rango de temblor
    
    b, a = butter(3, [low_cut / nyquist, high_cut / nyquist], btype='band')
    
    # Aplicar filtro a las aceleraciones
    df['acel_x_filt'] = filtfilt(b, a, df['acel_x'])
    df['acel_y_filt'] = filtfilt(b, a, df['acel_y'])
    df['acel_z_filt'] = filtfilt(b, a, df['acel_z'])

    # Compensación de gravedad usando cuaterniones
    q = np.array(df[['q_w', 'q_x', 'q_y', 'q_z']])
    if np.isnan(q).any() or np.isinf(q).any():
        st.error("Error: Cuaterniones contienen valores no numéricos (NaN/Inf). No se puede realizar la compensación de gravedad.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        gravity_vector_global = np.array([0.0, 0.0, 9.81]) 
        acc_global_compensated = np.zeros_like(df[['acel_x_filt', 'acel_y_filt', 'acel_z_filt']].values)

        for i in range(len(df)):
            q_i = q[i]
            gravity_in_sensor_frame = ahrs.common.orientation.q_rotate(q_i, gravity_vector_global)
            compensated_accel_sensor_frame = df[['acel_x_filt', 'acel_y_filt', 'acel_z_filt']].iloc[i].values - gravity_in_sensor_frame
            acc_global_compensated[i] = ahrs.common.orientation.q_rotate(ahrs.common.orientation.q_conjugate(q_i), compensated_accel_sensor_frame)

        df['acel_x_comp'] = acc_global_compensated[:, 0]
        df['acel_y_comp'] = acc_global_compensated[:, 1]
        df['acel_z_comp'] = acc_global_compensated[:, 2]
        df['acel_magnitud_comp'] = np.sqrt(df['acel_x_comp']**2 + df['acel_y_comp']**2 + df['acel_z_comp']**2)

    except Exception as e:
        st.warning(f"Error durante la compensación de gravedad con ahrs: {e}. Usando magnitud de aceleración filtrada sin compensar gravedad.")
        df['acel_magnitud_comp'] = np.sqrt(df['acel_x_filt']**2 + df['acel_y_filt']**2 + df['acel_z_filt']**2)


    # --- ANÁLISIS POR VENTANAS ---
    ventana_puntos = int(ventana_duracion_seg * fs)
    num_ventanas = len(df) // ventana_puntos

    resultados_ventanas = []
    data_para_grafico_primera_ventana = {}

    for i in range(num_ventanas):
        inicio = i * ventana_puntos
        fin = inicio + ventana_puntos
        ventana_df = df.iloc[inicio:fin]

        if len(ventana_df) < ventana_puntos:
            continue

        n_ventana = len(ventana_df['acel_magnitud_comp'])
        frecuencias, psd_values = welch(ventana_df['acel_magnitud_comp'], fs=fs, nperseg=ventana_puntos, scaling='density')

        indices_relevantes = np.where((frecuencias >= 0.5) & (frecuencias <= 15.0))
        if len(indices_relevantes[0]) > 0:
            frecuencias_relevantes = frecuencias[indices_relevantes]
            psd_relevantes = psd_values[indices_relevantes]
            
            if len(psd_relevantes) > 0:
                frecuencia_dominante = frecuencias_relevantes[np.argmax(psd_relevantes)]
            else:
                frecuencia_dominante = 0.0
        else:
            frecuencia_dominante = 0.0

        rms = np.sqrt(np.mean(ventana_df['acel_magnitud_comp']**2))

        if frecuencia_dominante > 0.5:
            amplitude_accel_rms = rms
            amplitude_accel_pico = amplitude_accel_rms * np.sqrt(2)
            amplitude_desplazamiento_m = amplitude_accel_pico / ((2 * np.pi * frecuencia_dominante)**2)
            amplitude_temblor_cm = amplitude_desplazamiento_m * 100
        else:
            amplitude_temblor_cm = 0.0

        resultados_ventanas.append({
            'Ventana': i + 1,
            'Inicio (seg)': inicio / fs,
            'Fin (seg)': fin / fs,
            'Frecuencia Dominante (Hz)': frecuencia_dominante,
            'RMS (m/s2)': rms,
            'Amplitud Temblor (cm)': amplitude_temblor_cm
        })

        if i == 0:
            data_para_grafico_primera_ventana = {
                'frecuencias': frecuencias,
                'psd_values': psd_values,
                'acel_magnitud_comp': ventana_df['acel_magnitud_comp']
            }

    if not resultados_ventanas:
        st.warning("No se pudieron calcular las métricas. Datos insuficientes para una ventana completa.")
        return pd.DataFrame(), pd.DataFrame()

    df_resultados_ventanas = pd.DataFrame(resultados_ventanas)
    
    promedio_resultados = pd.DataFrame([{
        'Frecuencia Dominante (Hz)': df_resultados_ventanas['Frecuencia Dominante (Hz)'].mean(),
        'RMS (m/s2)': df_resultados_ventanas['RMS (m/s2)'].mean(),
        'Amplitud Temblor (cm)': df_resultados_ventanas['Amplitud Temblor (cm)'].mean()
    }])

    return promedio_resultados, data_para_grafico_primera_ventana


# Función para generar PDF (completa y reutilizable)
def generar_pdf(opcion_seleccionada, data, buffer):
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Informe de Análisis de Temblor", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"**Fecha del Informe:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"**Opción Seleccionada:** {opcion_seleccionada}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<h3>Datos del Paciente</h3>", styles['h2']))
    datos_paciente = data.get('datos_paciente', {})
    if datos_paciente:
        story.append(Paragraph(f"**Sexo:** {datos_paciente.get('sexo', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"**Edad:** {datos_paciente.get('edad', 'N/A')}", styles['Normal']))
        if datos_paciente.get('mano_medida') and datos_paciente.get('mano_medida') != "No especificada":
            story.append(Paragraph(f"**Mano Medida:** {datos_paciente.get('mano_medida', 'N/A')}", styles['Normal']))
        if datos_paciente.get('dedo_medido') and datos_paciente.get('dedo_medido') != "No especificado":
            story.append(Paragraph(f"**Dedo Medido:** {datos_paciente.get('dedo_medido', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))
    else:
        story.append(Paragraph("Datos del paciente no disponibles.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    if opcion_seleccionada == "1️⃣ Análisis de una medición":
        story.append(Paragraph("<h3>Análisis de una Medición</h3>", styles['h2']))
        mano = data.get('mano')
        if mano: story.append(Paragraph(f"**Mano analizada:** {mano}", styles['Normal']))
        else: story.append(Paragraph("**Mano analizada:** No especificada (análisis avanzado)", styles['Normal']))
        story.append(Paragraph(f"**Frecuencia de Muestreo (Hz):** {data.get('fs', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"**Duración de Ventana (seg):** {data.get('ventana_duracion_seg', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("<h4>Resultados Promedio del Análisis</h4>", styles['h3']))
        resultados_prom = data.get('resultados_prom', pd.DataFrame())
        if not resultados_prom.empty:
            table_data = [resultados_prom.columns.tolist()] + [[round(val, 3) for val in row] for row in resultados_prom.values.tolist()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.2 * inch))
        
        if data.get('fig_path'):
            try:
                img_path = data['fig_path']
                if os.path.exists(img_path):
                    img = Image(img_path, width=5.5*inch, height=3*inch)
                    story.append(img); story.append(Spacer(1, 0.2 * inch))
                else: story.append(Paragraph("No se encontró la imagen del gráfico.", styles['Italic']))
            except Exception as e: story.append(Paragraph(f"Error al agregar gráfico: {e}", styles['Italic']))


    elif opcion_seleccionada == "2️⃣ Comparar dos mediciones":
        story.append(Paragraph("<h3>Comparación de Dos Mediciones</h3>", styles['h2']))
        for key in ['medicion1', 'medicion2']:
            medicion_data = data.get(key, {})
            if medicion_data:
                story.append(Paragraph(f"<h4>Resultados de {medicion_data.get('nombre', 'Medición')}</h4>", styles['h3']))
                resultados_prom = medicion_data.get('resultados_prom', pd.DataFrame())
                if not resultados_prom.empty:
                    table_data = [resultados_prom.columns.tolist()] + [[round(val, 3) for val in row] for row in resultados_prom.values.tolist()]
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                        ('GRID', (0,0), (-1,-1), 1, colors.black)
                    ]))
                    story.append(table); story.append(Spacer(1, 0.1 * inch))
                
                if medicion_data.get('fig_path'):
                    try:
                        img_path = medicion_data['fig_path']
                        if os.path.exists(img_path):
                            img = Image(img_path, width=5.5*inch, height=3*inch)
                            story.append(img); story.append(Spacer(1, 0.2 * inch))
                        else: story.append(Paragraph(f"No se encontró la imagen del gráfico para {medicion_data.get('nombre', 'Medición')}.", styles['Italic']))
                    except Exception as e: story.append(Paragraph(f"Error al agregar gráfico para {medicion_data.get('nombre', 'Medición')}: {e}", styles['Italic']))
            story.append(Spacer(1, 0.2 * inch))

    elif opcion_seleccionada == "3️⃣ Predicción de Diagnóstico":
        story.append(Paragraph("<h3>Predicción de Diagnóstico</h3>", styles['h2']))
        story.append(Paragraph(f"**Frecuencia de Muestreo (Hz):** {data.get('fs', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"**Duración de Ventana (seg):** {data.get('ventana_duracion_seg', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))
        
        story.append(Paragraph("<h4>Resultados Detallados del Análisis</h4>", styles['h3']))
        resultados_analisis = data.get('resultados_analisis', pd.DataFrame())
        if not resultados_analisis.empty:
            table_data = [resultados_analisis.columns.tolist()] + [[round(val, 3) for val in row] for row in resultados_analisis.values.tolist()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            story.append(table); story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("<h4>Predicción de Diagnóstico Final</h4>", styles['h3']))
        prediccion_final = data.get('prediccion_final', 'N/A')
        story.append(Paragraph(f"**El modelo predice:** {prediccion_final}", styles['Normal']))
        story.append(Paragraph("Nota: El diagnóstico clínico final debe considerar este resultado y el cuadro general del paciente.", styles['Italic']))
        story.append(Spacer(1, 0.2 * inch))
        
        if data.get('fig_path'):
            try:
                img_path = data['fig_path']
                if os.path.exists(img_path):
                    img = Image(img_path, width=5.5*inch, height=3*inch)
                    story.append(img); story.append(Spacer(1, 0.2 * inch))
                else: story.append(Paragraph("No se encontró la imagen del gráfico de métricas para la predicción.", styles['Italic']))
            except Exception as e: story.append(Paragraph(f"Error al agregar gráfico de predicción: {e}", styles['Italic']))

    try:
        doc.build(story)
    except Exception as e:
        st.error(f"Error al construir el PDF: {e}")


# --- CONFIGURACIÓN DE LA PÁGINA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Análisis y Predicción de Temblor")

st.sidebar.title("Menú Principal")
opcion = st.sidebar.radio("Selecciona una opción:", [
    "1️⃣ Análisis de una medición",
    "2️⃣ Comparar dos mediciones",
    "3️⃣ Predicción de Diagnóstico"
])

# --- Lógica de la aplicación según la opción seleccionada ---

# --- Lógica de la Opción 1: Análisis de una medición (AVANZADO) ---
if opcion == "1️⃣ Análisis de una medición":
    st.title("📊 Análisis de una Medición de Temblor (Avanzado)")
    st.markdown("Sube un archivo CSV de una única medición. Este análisis incluye compensación de gravedad.")
    st.info("Para esta opción, se utilizará el análisis de temblor avanzado que espera **datos de cuaterniones (qW, qX, qY, qZ)** para la compensación de gravedad. Las columnas de aceleración deben ser 'Acc_X/Acel_X', 'Acc_Y/Acel_Y', 'Acc_Z/Acel_Z'.")

    uploaded_file_avanzado = st.file_uploader("Sube tu archivo CSV de medición", type=["csv"], key="file_avanzado")

    if uploaded_file_avanzado is not None:
        try:
            df_medicion_avanzado = pd.read_csv(uploaded_file_avanzado, encoding='latin1', decimal=',')
            st.success("Archivo cargado exitosamente.")

            datos_paciente_avanzado = extraer_datos_paciente(df_medicion_avanzado)
            
            st.subheader("Datos del Paciente y Configuración del Análisis")
            st.write("---")
            st.write("**Sexo:**", datos_paciente_avanzado.get('sexo', 'N/A'))
            st.write("**Edad:**", datos_paciente_avanzado.get('edad', 'N/A'))
            if datos_paciente_avanzado.get('mano_medida') and datos_paciente_avanzado.get('mano_medida') != "No especificada":
                st.write("**Mano Medida:**", datos_paciente_avanzado.get('mano_medida'))
            if datos_paciente_avanzado.get('dedo_medido') and datos_paciente_avanzado.get('dedo_medido') != "No especificado":
                st.write("**Dedo Medido:**", datos_paciente_avanzado.get('dedo_medido'))

            fs_advanced = st.slider("Frecuencia de muestreo (Hz)", min_value=50, max_value=200, value=100, step=10, key="fs_advanced")
            st.write(f"**Duración de Ventana (seg):** {ventana_duracion_seg}") # Usar la constante
            st.write("---")

            if st.button("Realizar Análisis Avanzado"):
                if not df_medicion_avanzado.empty:
                    with st.spinner("Realizando análisis avanzado..."):
                        resultados_prom_avanzado, data_grafico_avanzado = analizar_temblor_por_ventanas_avanzado(
                            df_medicion_avanzado, fs_advanced, ventana_duracion_seg
                        )
                    
                    if not resultados_prom_avanzado.empty:
                        st.subheader("Resultados Promedio del Análisis")
                        st.dataframe(resultados_prom_avanzado.round(3))

                        if data_grafico_avanzado:
                            fig_avanzado, axes_avanzado = plt.subplots(1, 2, figsize=(15, 5))
                            axes_avanzado[0].plot(data_grafico_avanzado['frecuencias'], data_grafico_avanzado['psd_values'])
                            axes_avanzado[0].set_title('Densidad Espectral de Potencia (PSD)')
                            axes_avanzado[0].set_xlabel('Frecuencia (Hz)'); axes_avanzado[0].set_ylabel('PSD'); axes_avanzado[0].set_xlim(0, 20); axes_avanzado[0].grid(True)

                            axes_avanzado[1].plot(data_grafico_avanzado['acel_magnitud_comp'])
                            axes_avanzado[1].set_title('Magnitud de Aceleración Compensada (Primera Ventana)')
                            axes_avanzado[1].set_xlabel('Muestras'); axes_avanzado[1].set_ylabel('Aceleración (m/s²)'); axes_avanzado[1].grid(True)

                            plt.tight_layout()
                            st.pyplot(fig_avanzado)
                            plt.close(fig_avanzado)

                            temp_fig_path_avanzado = os.path.join(tempfile.gettempdir(), f"analisis_avanzado_fig.png")
                            fig_avanzado.savefig(temp_fig_path_avanzado, dpi=300, bbox_inches='tight')

                            st.subheader("Generar Informe PDF")
                            pdf_buffer_avanzado = io.BytesIO()
                            generar_pdf(
                                "1️⃣ Análisis de una medición",
                                {
                                    'datos_paciente': datos_paciente_avanzado,
                                    'resultados_prom': resultados_prom_avanzado,
                                    'fs': fs_advanced,
                                    'ventana_duracion_seg': ventana_duracion_seg,
                                    'fig_path': temp_fig_path_avanzado
                                },
                                pdf_buffer_avanzado
                            )

                            st.download_button(
                                label="📄 Descargar Informe PDF",
                                data=pdf_buffer_avanzado.getvalue(),
                                file_name="informe_analisis_avanzado.pdf",
                                mime="application/pdf"
                            )
                            if os.path.exists(temp_fig_path_avanzado):
                                os.remove(temp_fig_path_avanzado)

                    else:
                        st.warning("No se pudo realizar el análisis avanzado. Verifica el formato de tu archivo y las columnas.")
                else:
                    st.warning("El DataFrame de la medición está vacío.")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            st.warning("Asegúrate de que el archivo CSV esté codificado en 'latin1', use coma (',') como separador decimal, y contenga las columnas esperadas (Acc_X/Acel_X, Acc_Y/Acel_Y, Acc_Z/Acel_Z, qW, qX, qY, qZ).")


# --- Lógica de la Opción 2: Comparar dos mediciones ---
elif opcion == "2️⃣ Comparar dos mediciones":
    st.title("⚖️ Comparar Dos Mediciones de Temblor")
    st.markdown("Sube dos archivos CSV para comparar sus métricas de temblor. Ambos análisis incluirán compensación de gravedad.")
    st.info("Ambos análisis utilizarán el método avanzado que espera **datos de cuaterniones**.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Medición 1")
        file1 = st.file_uploader("Archivo CSV de la Medición 1", type=["csv"], key="file1_comp")
        fs1 = st.slider("Frecuencia de muestreo (Hz) Medición 1", min_value=50, max_value=200, value=100, step=10, key="fs1_comp")
        
    with col2:
        st.subheader("Medición 2")
        file2 = st.file_uploader("Archivo CSV de la Medición 2", type=["csv"], key="file2_comp")
        fs2 = st.slider("Frecuencia de muestreo (Hz) Medición 2", min_value=50, max_value=200, value=100, step=10, key="fs2_comp")

    if st.button("Comparar Mediciones"):
        if file1 and file2:
            try:
                df1 = pd.read_csv(file1, encoding='latin1', decimal=',')
                df2 = pd.read_csv(file2, encoding='latin1', decimal=',')

                # Aquí se extraen datos del paciente para cada archivo, pero el PDF solo usa uno para el encabezado general
                datos_paciente1 = extraer_datos_paciente(df1) 
                datos_paciente2 = extraer_datos_paciente(df2) # No se usa en el PDF principal, pero se extrae

                st.subheader("Resultados de la Medición 1")
                resultados_prom1, data_grafico1 = analizar_temblor_por_ventanas_avanzado(df1, fs1, ventana_duracion_seg)
                if not resultados_prom1.empty:
                    st.dataframe(resultados_prom1.round(3))
                    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))
                    axes1[0].plot(data_grafico1['frecuencias'], data_grafico1['psd_values'])
                    axes1[0].set_title('PSD Medición 1'); axes1[0].set_xlabel('Frecuencia (Hz)'); axes1[0].set_ylabel('PSD'); axes1[0].set_xlim(0, 20); axes1[0].grid(True)
                    axes1[1].plot(data_grafico1['acel_magnitud_comp'])
                    axes1[1].set_title('Aceleración Comp. Medición 1'); axes1[1].set_xlabel('Muestras'); axes1[1].set_ylabel('Aceleración (m/s²)'); axes1[1].grid(True)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)
                    temp_fig_path1 = os.path.join(tempfile.gettempdir(), f"comp_fig1.png")
                    fig1.savefig(temp_fig_path1, dpi=300, bbox_inches='tight')
                else:
                    st.warning("No se pudo analizar la Medición 1.")
                    temp_fig_path1 = None

                st.subheader("Resultados de la Medición 2")
                resultados_prom2, data_grafico2 = analizar_temblor_por_ventanas_avanzado(df2, fs2, ventana_duracion_seg)
                if not resultados_prom2.empty:
                    st.dataframe(resultados_prom2.round(3))
                    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
                    axes2[0].plot(data_grafico2['frecuencias'], data_grafico2['psd_values'])
                    axes2[0].set_title('PSD Medición 2'); axes2[0].set_xlabel('Frecuencia (Hz)'); axes2[0].set_ylabel('PSD'); axes2[0].set_xlim(0, 20); axes2[0].grid(True)
                    axes2[1].plot(data_grafico2['acel_magnitud_comp'])
                    axes2[1].set_title('Aceleración Comp. Medición 2'); axes2[1].set_xlabel('Muestras'); axes2[1].set_ylabel('Aceleración (m/s²)'); axes2[1].grid(True)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                    temp_fig_path2 = os.path.join(tempfile.gettempdir(), f"comp_fig2.png")
                    fig2.savefig(temp_fig_path2, dpi=300, bbox_inches='tight')
                else:
                    st.warning("No se pudo analizar la Medición 2.")
                    temp_fig_path2 = None

                st.subheader("Generar Informe de Comparación PDF")
                pdf_buffer_comp = io.BytesIO()
                generar_pdf(
                    "2️⃣ Comparar dos mediciones",
                    {
                        'datos_paciente': datos_paciente1, # Se usa el de la primera medición para el encabezado
                        'medicion1': {'nombre': 'Medición 1', 'resultados_prom': resultados_prom1, 'fig_path': temp_fig_path1},
                        'medicion2': {'nombre': 'Medición 2', 'resultados_prom': resultados_prom2, 'fig_path': temp_fig_path2},
                        'fs1': fs1, 'fs2': fs2,
                        'ventana_duracion_seg': ventana_duracion_seg
                    },
                    pdf_buffer_comp
                )
                st.download_button(
                    label="📄 Descargar Informe de Comparación PDF",
                    data=pdf_buffer_comp.getvalue(),
                    file_name="informe_comparacion_temblor.pdf",
                    mime="application/pdf"
                )
                if temp_fig_path1 and os.path.exists(temp_fig_path1): os.remove(temp_fig_path1)
                if temp_fig_path2 and os.path.exists(temp_fig_path2): os.remove(temp_fig_path2)

            except Exception as e:
                st.error(f"Error al comparar archivos: {e}")
                st.warning("Asegúrate de que ambos archivos CSV estén codificados en 'latin1', usen coma (',') como separador decimal, y contengan las columnas esperadas.")
        else:
            st.warning("Por favor, sube ambos archivos para comparar.")

# --- Lógica de la Opción 3: Predicción de Diagnóstico (USA ANÁLISIS AVANZADO) ---
elif opcion == "3️⃣ Predicción de Diagnóstico":
    st.title("🔮 Predicción de Diagnóstico de Temblor")
    st.markdown("Carga los 3 archivos CSV (Reposo, Postural y Acción) que representan una medición de temblor. El modelo realizará una predicción de diagnóstico basada en este conjunto de datos.")
    st.info("Para esta opción, se utilizará el análisis de temblor avanzado que espera **datos de cuaterniones (qW, qX, qY, qZ)** para la compensación de gravedad.")

    # Cargar el modelo entrenado
    try:
        model = joblib.load('tremor_prediction_model.joblib')
        st.success("Modelo de predicción cargado exitosamente.")
    except FileNotFoundError:
        st.error("Error: El archivo 'tremor_prediction_model.joblib' no se encontró. Asegúrate de que está en la misma carpeta que tu script de Streamlit (o la ruta correcta en Colab/Streamlit Cloud).")
        model = None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que la versión de 'scikit-learn' sea la misma que la usada para guardar el modelo (esperada 1.6.1).")
        model = None

    # Uploaders para los 3 archivos
    st.markdown("##### Sube tus archivos de medición:")
    col1, col2, col3 = st.columns(3)
    with col1:
        reposo_file = st.file_uploader("Archivo de Reposo", type=["csv"], key="reposo_pred")
    with col2:
        postural_file = st.file_uploader("Archivo Postural", type=["csv"], key="postural_pred")
    with col3:
        accion_file = st.file_uploader("Archivo de Acción", type=["csv"], key="accion_pred")

    st.markdown("---")
    fs_advanced_pred = st.slider("Frecuencia de muestreo (Hz) para el análisis avanzado", min_value=50, max_value=200, value=100, step=10, key="fs_advanced_pred")
    st.info(f"La duración de la ventana de análisis es fija en {ventana_duracion_seg} segundos.")


    if st.button("Realizar Predicción"):
        if model:
            prediccion_final = None
            resultados_analisis = pd.DataFrame() # Para guardar el DataFrame de resultados para el PDF
            datos_paciente_prediccion = {} # Se inicializa aquí para que esté disponible

            if reposo_file and postural_file and accion_file:
                try:
                    st.info(f"Procesando datos para la predicción...")
                    df_reposo = pd.read_csv(reposo_file, encoding='latin1', decimal=',')
                    df_postural = pd.read_csv(postural_file, encoding='latin1', decimal=',')
                    df_accion = pd.read_csv(accion_file, encoding='latin1', decimal=',')

                    # --- Paso 1: Extraer datos del paciente (de un solo archivo, por ejemplo, reposo) ---
                    datos_paciente_prediccion = extraer_datos_paciente(df_reposo) 

                    # --- Paso 2: Calcular métricas de temblor para cada tipo de prueba ---
                    res_reposo_prom, _ = analizar_temblor_por_ventanas_avanzado(df_reposo, fs_advanced_pred, ventana_duracion_seg)
                    res_postural_prom, _ = analizar_temblor_por_ventanas_avanzado(df_postural, fs_advanced_pred, ventana_duracion_seg)
                    res_accion_prom, _ = analizar_temblor_por_ventanas_avanzado(df_accion, fs_advanced_pred, ventana_duracion_seg)
                    
                    if not (res_reposo_prom.empty or res_postural_prom.empty or res_accion_prom.empty):
                        # --- Paso 3: Ensamblar el DataFrame de características para la predicción ---
                        # ¡CRUCIAL! Los nombres de las columnas aquí deben coincidir EXACTAMENTE
                        # con los que tu modelo fue entrenado.
                        data_para_prediccion = pd.DataFrame([{
                            # Datos del paciente
                            'sexo': datos_paciente_prediccion.get('sexo', 'No especificado'),
                            'edad': datos_paciente_prediccion.get('edad', 0),
                            'mano_medida': datos_paciente_prediccion.get('mano_medida', 'No especificada'),
                            'dedo_medido': datos_paciente_prediccion.get('dedo_medido', 'No especificado'),
                            
                            # Métricas de Reposo (usando los nombres de tu dataset de entrenamiento)
                            'Frec_Reposo': res_reposo_prom['Frecuencia Dominante (Hz)'].iloc[0],
                            'RMS_Reposo': res_reposo_prom['RMS (m/s2)'].iloc[0],
                            'Amp_Reposo': res_reposo_prom['Amplitud Temblor (cm)'].iloc[0],
                            
                            # Métricas Postural (usando los nombres de tu dataset de entrenamiento)
                            'Frec_Postural': res_postural_prom['Frecuencia Dominante (Hz)'].iloc[0],
                            'RMS_Postural': res_postural_prom['RMS (m/s2)'].iloc[0],
                            'Amp_Postural': res_postural_prom['Amplitud Temblor (cm)'].iloc[0],
                            
                            # Métricas de Acción (usando los nombres de tu dataset de entrenamiento)
                            'Frec_Accion': res_accion_prom['Frecuencia Dominante (Hz)'].iloc[0],
                            'RMS_Accion': res_accion_prom['RMS (m/s2)'].iloc[0],
                            'Amp_Accion': res_accion_prom['Amplitud Temblor (cm)'].iloc[0]
                        }])
                        
                        # Definir explícitamente el orden de las columnas como se usó en el entrenamiento
                        # Esto es una buena práctica para garantizar la consistencia en el orden de las features
                        expected_features_order = [
                            'sexo', 'edad', 'mano_medida', 'dedo_medido',
                            'Frec_Reposo', 'RMS_Reposo', 'Amp_Reposo',
                            'Frec_Postural', 'RMS_Postural', 'Amp_Postural',
                            'Frec_Accion', 'RMS_Accion', 'Amp_Accion'
                        ]
                        
                        # Asegurar que el DataFrame de entrada tenga las columnas en el orden correcto
                        data_para_prediccion = data_para_prediccion[expected_features_order]

                        # --- Paso 4: Realizar la predicción ---
                        prediccion_raw = model.predict(data_para_prediccion)[0]
                        
                        # Mapeo de predicciones a etiquetas legibles
                        label_map = {
                            'PK': 'Temblor por Parkinson',
                            'TE': 'Temblor Esencial',
                            'SANO': 'Sin Temblor Aparente'
                        }
                        prediccion_final = label_map.get(prediccion_raw, prediccion_raw)
                        
                        # Guardar resultados detallados para el PDF
                        resultados_analisis = pd.DataFrame([
                            {'Test': 'Reposo', 'Frecuencia Dominante (Hz)': res_reposo_prom['Frecuencia Dominante (Hz)'].iloc[0], 'RMS (m/s2)': res_reposo_prom['RMS (m/s2)'].iloc[0], 'Amplitud Temblor (cm)': res_reposo_prom['Amplitud Temblor (cm)'].iloc[0]},
                            {'Test': 'Postural', 'Frecuencia Dominante (Hz)': res_postural_prom['Frecuencia Dominante (Hz)'].iloc[0], 'RMS (m/s2)': res_postural_prom['RMS (m/s2)'].iloc[0], 'Amplitud Temblor (cm)': res_postural_prom['Amplitud Temblor (cm)'].iloc[0]},
                            {'Test': 'Acción', 'Frecuencia Dominante (Hz)': res_accion_prom['Frecuencia Dominante (Hz)'].iloc[0], 'RMS (m/s2)': res_accion_prom['RMS (m/s2)'].iloc[0], 'Amplitud Temblor (cm)': res_accion_prom['Amplitud Temblor (cm)'].iloc[0]}
                        ])

                        # Generar gráfico para esta mano
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        metrics_to_plot = ['Frecuencia Dominante (Hz)', 'RMS (m/s2)', 'Amplitud Temblor (cm)']
                        titles = ['Frecuencia Dominante', 'RMS', 'Amplitud Temblor']

                        for i, metric in enumerate(metrics_to_plot):
                            values = [
                                resultados_analisis[resultados_analisis['Test'] == 'Reposo'][metric].iloc[0],
                                resultados_analisis[resultados_analisis['Test'] == 'Postural'][metric].iloc[0],
                                resultados_analisis[resultados_analisis['Test'] == 'Acción'][metric].iloc[0]
                            ]
                            axes[i].bar(['Reposo', 'Postural', 'Acción'], values, color=['skyblue', 'lightcoral', 'lightgreen'])
                            axes[i].set_title(f'{titles[i]} por Test')
                            axes[i].set_ylabel(metric)
                            axes[i].grid(axis='y', linestyle='--', alpha=0.7)

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        temp_fig_path = os.path.join(tempfile.gettempdir(), f"metrica_analisis_prediccion.png")
                        fig.savefig(temp_fig_path, dpi=300, bbox_inches='tight')
                        
                        st.success(f"El modelo predice: **{prediccion_final}**")
                    else:
                        st.warning("No se pudieron obtener datos válidos para la predicción. Asegúrate de que los archivos estén completos y contengan las columnas requeridas (Acc_X/Acel_X, Acc_Y/Acel_Y, Acc_Z/Acel_Z, qW, qX, qY, qZ).")
                except Exception as e:
                    st.error(f"Error al procesar los archivos: {e}")
                    st.warning("Asegúrate de que las columnas en tus archivos CSV de entrada coincidan con las esperadas (Acc_X/Acel_X, Acc_Y/Acel_Y, Acc_Z/Acel_Z, qW, qX, qY, qZ) y que los datos sean válidos (usando coma decimal).")
            else:
                st.warning("Por favor, sube los tres archivos (Reposo, Postural, Acción) para realizar la predicción.")
            
            # --- Mostrar Resumen y Botón de Descarga PDF ---
            if prediccion_final is not None and not resultados_analisis.empty:
                st.subheader("Datos del Paciente y Configuración del Análisis")
                st.write("---")
                st.write("**Sexo:**", datos_paciente_prediccion.get('sexo', 'N/A'))
                st.write("**Edad:**", datos_paciente_prediccion.get('edad', 'N/A'))
                if datos_paciente_prediccion.get('mano_medida') and datos_paciente_prediccion.get('mano_medida') != "No especificada":
                    st.write("**Mano Medida:**", datos_paciente_prediccion.get('mano_medida'))
                if datos_paciente_prediccion.get('dedo_medido') and datos_paciente_prediccion.get('dedo_medido') != "No especificado":
                    st.write("**Dedo Medido:**", datos_paciente_prediccion.get('dedo_medido'))
                
                st.write("**Frecuencia de Muestreo (Hz):**", fs_advanced_pred)
                st.write(f"**Duración de Ventana (seg):** {ventana_duracion_seg}")
                st.write("---")

                st.subheader("Resultados Detallados del Análisis")
                st.dataframe(resultados_analisis.round(3))
                st.write("")

                st.subheader("Predicción de Diagnóstico Final")
                st.write(f"El modelo predice: **{prediccion_final}**")
                st.info("Nota: El diagnóstico clínico final debe considerar este resultado y el cuadro general del paciente.")

                st.write("---")
                st.subheader("Generar Informe PDF")
                pdf_buffer = io.BytesIO()
                
                generar_pdf(
                    "3️⃣ Predicción de Diagnóstico",
                    {
                        'datos_paciente': datos_paciente_prediccion,
                        'resultados_analisis': resultados_analisis,
                        'prediccion_final': prediccion_final,
                        'fig_path': temp_fig_path,
                        'fs': fs_advanced_pred,
                        'ventana_duracion_seg': ventana_duracion_seg
                    },
                    pdf_buffer
                )

                st.download_button(
                    label="📄 Descargar Informe de Predicción PDF",
                    data=pdf_buffer.getvalue(),
                    file_name="informe_prediccion_temblor.pdf",
                    mime="application/pdf"
                )
                if os.path.exists(temp_fig_path):
                    os.remove(temp_fig_path)

        else:
            st.warning("El modelo de predicción no está disponible. Por favor, asegúrate de que el archivo 'tremor_prediction_model.joblib' esté en la carpeta correcta y sin errores.")

