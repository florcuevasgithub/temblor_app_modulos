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

        noverlap_val = int(tamaño_ventana * 0.5)

        f, Pxx = welch(
            segmento, 
            fs=fs, 
            nperseg=tamaño_ventana,
            window='hann',        
            noverlap=noverlap_val # 50% de traslape
        )
       
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
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def normalizar_cadena(cadena):
    """Convierte la cadena a minúsculas y elimina acentos y diacríticos."""
    # 1. Normaliza la cadena a su forma NFD (Descompone acentos)
    cadena_nfd = unicodedata.normalize('NFD', cadena)
    
    # 2. Filtra los caracteres no básicos (acentos)
    cadena_sin_acento = ''.join(c for c in cadena_nfd if unicodedata.category(c) != 'Mn')
    
    # 3. Convierte a minúsculas (para consistencia)
    return cadena_sin_acento.lower()

def parsear_metadatos_del_nombre(nombre_archivo):
    """Extrae metadatos (Identidad, Mano, Dedo, Tipo, DBS, Fecha) del nombre del archivo."""
    
    nombre_upper = nombre_archivo.upper()
    
    # Normalización: Elimina la extensión, dobles guiones y espacios.
    nombre_normalizado = nombre_upper.replace(".CSV", "").replace('__', '_').replace(' ', '')
    tokens = nombre_normalizado.split('_') 
    
    # Listas de tokens válidos
    tokens_mano = ('DERECHA', 'IZQUIERDA')
    
    # Valores de retorno por defecto
    mano = 'MANO NO ENCONTRADA'
    dedo = 'DEDO NO ENCONTRADO'
    tipo_test_en_nombre = 'TIPO NO ENCONTRADO'
    fecha = 'FECHA NO ENCONTRADA'
    tiene_dbs = False 
    nombre_paciente_compuesto = 'IDENTIDAD NO ENCONTRADA'
    
    indice_delimitador = len(tokens)
    
    # 1. Encontrar el índice donde aparece DERECHA o IZQUIERDA (DELIMITADOR CLAVE)
    for i, token in enumerate(tokens):
        if token in tokens_mano:
            indice_delimitador = i
            mano = token 
            break
        
    # 2. La identidad del paciente es la unión de todos los tokens antes del delimitador.
    if indice_delimitador > 0:
        identidad_tokens = tokens[:indice_delimitador]
        # Filtramos tokens vacíos o que parezcan fechas/IDs numéricos sueltos
        identidad_tokens = [t for t in identidad_tokens if t and not (t.isdigit() or '-' in t)]
        
        if identidad_tokens:
            nombre_paciente_compuesto = '_'.join(identidad_tokens)
        else:
            nombre_paciente_compuesto = 'NOMBRE NO ENCONTRADO'
            
    # 3. Extracción de los otros metadatos (a partir del índice de la mano)
    for token in tokens[indice_delimitador:]:
        if token in ('INDICE', 'PULGAR'):
            dedo = token
        elif token in ('REPOSO', 'POSTURAL', 'ACCION'):
            tipo_test_en_nombre = token
        elif '-' in token and any(c.isdigit() for c in token):
            if fecha == 'FECHA NO ENCONTRADA':
                fecha = token
        elif token == 'DBS':
            tiene_dbs = True

    return {
        'Mano': mano.lower(),
        'Dedo': dedo.lower(),
        'Tiene_DBS': tiene_dbs,
        'Tipo_en_Nombre': tipo_test_en_nombre.lower(),
        'Fecha': fecha.lower(),
        'Nombre_Paciente': nombre_paciente_compuesto.lower() # Identidad compuesta
    }
def validar_consistencia_por_nombre_archivo(archivos_dict, nombre_medicion):
    """
    Valida la consistencia de los metadatos de los archivos cargados:
    Identidad, Mano, Dedo, Estado DBS y Fecha deben ser iguales.
    El Tipo de Test debe coincidir con el slot de carga (Reposo, Postural, Acción).
    """
    metadata_list = []
    
    # 1. Extracción de metadatos del nombre del archivo
    for test_carga, archivo in archivos_dict.items():
        if archivo is None:
            continue

        if not hasattr(archivo, 'name'):
            return False, f"Error interno: La entrada para '{test_carga}' no es un objeto de archivo válido."

        try:
            archivo.seek(0)
            meta = parsear_metadatos_del_nombre(archivo.name)
            meta['Test_Carga'] = test_carga 
            metadata_list.append(meta)
            archivo.seek(0) # Rebobinar para el procesamiento posterior

        except Exception as e:
            return False, f"Error al procesar el archivo de {test_carga}: {e}"
            
    if not metadata_list:
        return False, f"Error: No se cargaron archivos para {nombre_medicion}."

    # 2. Establecer referencias (del primer archivo cargado)
    ref = metadata_list[0]
    mano_ref = ref['Mano']
    dedo_ref = ref['Dedo']
    dbs_ref = ref['Tiene_DBS']
    fecha_ref = ref['Fecha']
    nombre_ref = ref['Nombre_Paciente'] # Referencia de la Identidad Compuesta

    # 3. Comprobar la coherencia en todo el conjunto
    for meta in metadata_list:
        
        # A.1. VALIDACIÓN DE IDENTIDAD DEL PACIENTE (CRÍTICO)
        if meta['Nombre_Paciente'] != nombre_ref or 'no encontrado' in meta['Nombre_Paciente']:
            return False, (f"Error de Consistencia de **Identidad de Paciente** en {nombre_medicion} ({meta['Test_Carga']}). "
                           f"Se compara un archivo del paciente '{meta['Nombre_Paciente'].upper()}' con otros del paciente '{nombre_ref.upper()}'. "
                           f"Asegúrese que sea el mismo paciente en todos los archivos.")

        # A.2. Validación de Coherencia (Mano/Dedo)
        if meta['Mano'] != mano_ref or 'no encontrada' in meta['Mano']:
            return False, f"Error de Consistencia de **Mano** en {nombre_medicion} ({meta['Test_Carga']}). Todos los archivos de la misma medición deben ser de la misma mano."
        if meta['Dedo'] != dedo_ref or 'no encontrado' in meta['Dedo']:
            return False, f"Error de Consistencia de **Dedo** en {nombre_medicion} ({meta['Test_Carga']}). Todos los archivos de la misma medición deben ser del mismo dedo."
        
        # B. Validación de Coherencia de DBS
        if meta['Tiene_DBS'] != dbs_ref:
            estado_ref = "con DBS" if dbs_ref else "sin DBS"
            estado_actual = "con DBS" if meta['Tiene_DBS'] else "sin DBS"
            return False, (f"Error de Coherencia DBS en {nombre_medicion} ({meta['Test_Carga']}). "
                           f"Se compara un archivo {estado_actual} con otros {estado_ref}.")

        # C. Validación de Coherencia de FECHA
        if meta['Fecha'] != fecha_ref or 'no encontrada' in meta['Fecha']:
            return False, (f"Error de Consistencia de **Fecha** en {nombre_medicion} ({meta['Test_Carga']}). "
                           f"La medición debe ser del mismo día ({fecha_ref.upper()}).")
                           
        # D. VALIDACIÓN ESTRICTA DE TIPO DE TEST (Maneja Acentos/Mayúsculas)
        test_carga_normalizado = normalizar_cadena(meta['Test_Carga'])
        tipo_en_nombre_lower = meta['Tipo_en_Nombre']

        if tipo_en_nombre_lower == 'tipo no encontrado':
            return False, f"Error de Archivo: No se pudo identificar el tipo de test (REPOSO/POSTURAL/ACCION) en el nombre del archivo."
    
        if tipo_en_nombre_lower != test_carga_normalizado:
            return False, (f"Error de Archivo: El espacio para cargar el archivo de ('{meta['Test_Carga'].upper()}') no coincide con "
                           f"el tipo de archivo cargado ('{tipo_en_nombre_lower.upper()}').")
            
    # FINAL: Si todas las validaciones pasan, devuelve True y un mensaje de éxito.
    return True, "Metadatos, Tipos de Pruebas e Identidad del Paciente consistentes."
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
        pdf.ln(5)

        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Fecha y hora del análisis: {fecha_hora}", ln=True)
        pdf.set_font("Arial", size=12)
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
        
       
        # SECCIÓN: IMAGEN DE REFERENCIA (Ajuste Inteligente)
        # ---------------------------------------------------------------------------------------------------------
        
        RUTA_IMAGEN_REFERENCIA = "cuadro_valores_referencia.jpeg"
        MARGEN_HORIZONTAL = 10  # Margen deseado a cada lado (10 mm)
        ANCHO_PAGINA_TOTAL = 210  # Ancho de página estándar
        
        # ANCHO MÁXIMO SEGURO: 190 mm (deja 10 mm a cada lado)
        ANCHO_MAXIMO_MM = ANCHO_PAGINA_TOTAL - (MARGEN_HORIZONTAL * 2) # 210 - 20 = 190 mm
        
        # Esto garantiza que el salto de página solo ocurra si realmente queda muy poco espacio.
        ALTURA_ESTIMADA_IMAGEN = 50 # REDUCIR ESTE VALOR 
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Cuadro Comparativo de Interpretación Clínica:", ln=True)
        
        # 1. VERIFICACIÓN DE ESPACIO RESTANTE
        # Ahora solo saltará si quedan menos de 50mm, lo cual es mucho menos que tu espacio actual.
        if (pdf.get_y() + ALTURA_ESTIMADA_IMAGEN) > (pdf.h - 20):
            pdf.add_page()
            pdf.ln(5)
        
        try:
            # 2. Insertar la imagen con el ANCHO MÁXIMO (190 mm) y centrado
            POSICION_X = (ANCHO_PAGINA_TOTAL - ANCHO_MAXIMO_MM) / 2 # 20 / 2 = 10 mm
            
            # w=190 mm (máxima legibilidad) y h=0 (mantiene proporción, evita distorsión)
            pdf.image(RUTA_IMAGEN_REFERENCIA, x=POSICION_X, w=ANCHO_MAXIMO_MM, h=0)
            
        except Exception as e:
            pdf.multi_cell(0, 8, f"ADVERTENCIA: No se pudo cargar o procesar el archivo de referencia '{RUTA_IMAGEN_REFERENCIA}'. Error: {e}")

        pdf.ln(2) # Pequeño salto de línea para separarlo de la imagen
        pdf.set_font("Arial", '', 9) # Fuente más pequeña para la nota
        pdf.set_text_color(150, 0, 0) # Opcional: Color rojo o gris oscuro para llamar la atención (R=150, G=0, B=0 es un rojo apagado)
        leyenda_nota = (
            "**NOTA IMPORTANTE:** Los valores de referencia están sacados de diferentes papers científicos como: "
            "“Motion characteristics of subclinical tremors in Parkinson’s disease and normal subjects” y también de la UPDRS.\n\n" # Dos saltos de línea para espacio
            "El diagnóstico y tratamiento final deben ser indicados y validados por el médico especialista.\n" # Un salto de línea
            "Esta herramienta solo provee soporte cuantitativo."
        )
        
        pdf.ln(2) # Pequeño salto de línea para separarlo de la imagen
        pdf.set_font("Arial", 'B', 9) # Fuente en negrita para el título "NOTA IMPORTANTE"
        pdf.set_text_color(150, 0, 0) # Color rojo apagado para la advertencia
        pdf.multi_cell(ANCHO_MAXIMO_MM, 5, 
                       leyenda_nota, 
                       align='C')
        pdf.set_text_color(0, 0, 0) 
        pdf.ln(5) 
        
    
    # ---------------------------------------------------------------------------------------------------------
    
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

        if not all(uploaded_files.values()):
            st.warning("Por favor, sube todos los archivos para iniciar el análisis.")
        else:
            # --- APLICAR VALIDACIÓN  -----------------------------------------------------------------------------
            is_consistent, error_msg = validar_consistencia_por_nombre_archivo(uploaded_files, "Medición Individual")

            if not is_consistent:
                st.error(error_msg)
                st.stop() # Detiene la ejecución si los archivos son inconsistentes
            # -----------------------------------------------------------------------------------------------------
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

elif opcion == "2️⃣ Comparación de mediciones":
    st.title("📊 Comparación de Mediciones")

    def extraer_datos_paciente(df_csv):
        datos_paciente = {
            "Nombre": df_csv.loc[0, 'Nombre'] if 'Nombre' in df_csv.columns else 'No especificado',
            "Apellido": df_csv.loc[0, 'Apellido'] if 'Apellido' in df_csv.columns else 'No especificado',
            "Edad": df_csv.loc[0, 'Edad'] if 'Edad' in df_csv.columns else None,
            "Sexo": df_csv.loc[0, 'Sexo'] if 'Sexo' in df_csv.columns else 'No especificado',
            "Diagnostico": df_csv.loc[0, 'Diagnostico'] if 'Diagnostico' in df_csv.columns else 'No especificado',
            "Tipo": df_csv.loc[0, 'Tipo'] if 'Tipo' in df_csv.columns else 'No especificado',
            "Antecedente": df_csv.loc[0, 'Antecedente'] if 'Antecedente' in df_csv.columns else 'No especificado',
            "Medicacion": df_csv.loc[0, 'Medicacion'] if 'Medicacion' in df_csv.columns else 'No especificado',
        }
        # Convertir a minúsculas
        if datos_paciente["Sexo"] is not None:
            datos_paciente["Sexo"] = str(datos_paciente["Sexo"]).lower()
        if datos_paciente["Diagnostico"] is not None:
            datos_paciente["Diagnostico"] = str(datos_paciente["Diagnostico"]).lower()
            
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
        
        # Convertir a minúsculas
        if "Mano" in metadata_dict and metadata_dict["Mano"] is not None:
            metadata_dict["Mano"] = str(metadata_dict["Mano"]).lower()
        if "Dedo" in metadata_dict and metadata_dict["Dedo"] is not None:
            metadata_dict["Dedo"] = str(metadata_dict["Dedo"]).lower()
            
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
            #----------------------------------------------------------------------------------------------------
           
            # Medicion 1
            is_consistent_1, error_msg_1 = validar_consistencia_por_nombre_archivo(config1_archivos, "Medición 1")

            if not is_consistent_1:
                st.error(error_msg_1)
                st.stop() # Detiene la ejecución si los archivos son inconsistentes
            # Medicion 2
            is_consistent_2, error_msg_2 = validar_consistencia_por_nombre_archivo(config2_archivos, "Medición 2")

            if not is_consistent_2:
                st.error(error_msg_2)
                st.stop() # Detiene la ejecución si los archivos son inconsistentes

            #------------------------------------------------------------------------------------------------------
            df_config1_meta = pd.read_csv(config1_archivos["Reposo"], encoding='latin1')
            df_config2_meta = pd.read_csv(config2_archivos["Reposo"], encoding='latin1')

            datos_paciente = extraer_datos_paciente(df_config1_meta)
            config1_params = extraer_datos_estimulacion(df_config1_meta)
            config2_params = extraer_datos_estimulacion(df_config2_meta)

            df_resultados_config1 = analizar_configuracion(config1_archivos)
            df_resultados_config2 = analizar_configuracion(config2_archivos)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Informe Comparativo de Mediciones", ln=True, align="C")

            pdf.set_font("Arial", size=12)
            pdf.ln(5)
            pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)
            
            def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
                if pd.notna(valor) and valor is not None and str(valor).strip() != "" and str(valor).lower() not in ["no especificado", "nan"]:
                    pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Datos del Paciente", ln=True)
            pdf.set_font("Arial", size=12)

            _imprimir_campo_pdf(pdf, "Nombre", datos_paciente.get("Nombre"))
            _imprimir_campo_pdf(pdf, "Apellido", datos_paciente.get("Apellido"))
            
            edad_val = datos_paciente.get("Edad")
            if pd.notna(edad_val) and edad_val is not None and str(edad_val).strip() != "":
                try:
                    edad_int = int(float(str(edad_val).replace(',', '.')))
                    _imprimir_campo_pdf(pdf, "Edad", edad_int)
                except (ValueError, TypeError):
                    pass
                    
            _imprimir_campo_pdf(pdf, "Sexo", datos_paciente.get("Sexo"))
            _imprimir_campo_pdf(pdf, "Diagnóstico", datos_paciente.get("Diagnostico"))
            _imprimir_campo_pdf(pdf, "Tipo", datos_paciente.get("Tipo"))
            _imprimir_campo_pdf(pdf, "Antecedente", datos_paciente.get("Antecedente"))
            _imprimir_campo_pdf(pdf, "Medicacion", datos_paciente.get("Medicacion"))
            pdf.ln(5)

            def imprimir_parametros_y_config(pdf_obj, parametros_dict, titulo):
                pdf_obj.set_font("Arial", 'B', 12)
                pdf_obj.cell(0, 10, titulo, ln=True)
                pdf_obj.set_font("Arial", size=12)
                
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
                    if pd.notna(value) and value is not None and str(value).strip() != "":
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

                        ax.plot(df1_ventanas["Tiempo (segundos)"], df1_ventanas["Amplitud Temblor (cm)"], label="Medición 1", color="blue")
                        ax.plot(df2_ventanas["Tiempo (segundos)"], df2_ventanas["Amplitud Temblor (cm)"], label="Medición 2", color="orange")
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
                            # Verificar si hay espacio para el gráfico
                            altura_grafico = 100
                            if (pdf.get_y() + altura_grafico) > (pdf.h - 20):
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

            # Eliminar la llamada a pdf.add_page() para evitar el espacio
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Conclusión", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, conclusion)

            pdf_output = BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0)

            pdf.ln(2) # Pequeño salto de línea para separarlo de la imagen
            pdf.set_font("Arial", '', 9) # Fuente más pequeña para la nota
            pdf.set_text_color(150, 0, 0) # Opcional: Color rojo o gris oscuro para llamar la atención (R=150, G=0, B=0 es un rojo apagado)
            leyenda_nota = (
                "**NOTA IMPORTANTE:** Los valores de referencia están sacados de diferentes papers científicos como: "
                "“Motion characteristics of subclinical tremors in Parkinson’s disease and normal subjects” y también de la UPDRS.\n\n" # Dos saltos de línea para espacio
                "El diagnóstico y tratamiento final deben ser indicados y validados por el médico especialista.\n" # Un salto de línea
                "Esta herramienta solo provee soporte cuantitativo."
            )
        
        pdf.ln(2) # Pequeño salto de línea para separarlo de la imagen
        pdf.set_font("Arial", 'B', 9) # Fuente en negrita para el título "NOTA IMPORTANTE"
        pdf.set_text_color(150, 0, 0) # Color rojo apagado para la advertencia
        pdf.multi_cell(ANCHO_MAXIMO_MM, 5, 
                       leyenda_nota, 
                       align='C')
        pdf.set_text_color(0, 0, 0) 
        pdf.ln(5)

            st.download_button(
                label="Descargar Informe PDF",
                data=pdf_output.getvalue(),
                file_name="informe_comparativo_temblor.pdf",
                mime="application/pdf"
            )
            st.info("El archivo se descargará en tu carpeta de descargas predeterminada o el navegador te pedirá la ubicación, dependiendo de tu configuración.")
            
# ------------------ MÓDULO 3: DIAGNÓSTICO TENTATIVO -----------------------------------------------------------------
elif opcion == "3️⃣ Diagnóstico tentativo":
    st.title("🩺 Diagnóstico Tentativo")
    st.markdown("### Cargar archivos CSV para el Diagnóstico")

    def extraer_datos_paciente(df_csv):
        datos_paciente = {
            "Nombre": df_csv.loc[0, 'Nombre'] if 'Nombre' in df_csv.columns else 'No especificado',
            "Apellido": df_csv.loc[0, 'Apellido'] if 'Apellido' in df_csv.columns else 'No especificado',
            "Edad": df_csv.loc[0, 'Edad'] if 'Edad' in df_csv.columns else None,
            "Sexo": df_csv.loc[0, 'Sexo'] if 'Sexo' in df_csv.columns else 'No especificado',
            "Diagnostico": df_csv.loc[0, 'Diagnostico'] if 'Diagnostico' in df_csv.columns else 'No especificado',
            "Tipo": df_csv.loc[0, 'Tipo'] if 'Tipo' in df_csv.columns else 'No especificado',
            "Antecedente": df_csv.loc[0, 'Antecedente'] if 'Antecedente' in df_csv.columns else 'No especificado',
            "Medicacion": df_csv.loc[0, 'Medicacion'] if 'Medicacion' in df_csv.columns else 'No especificado',
        }
        # Convertir a minúsculas
        if datos_paciente["Sexo"] is not None:
            datos_paciente["Sexo"] = str(datos_paciente["Sexo"]).lower()
        if datos_paciente["Diagnostico"] is not None:
            datos_paciente["Diagnostico"] = str(datos_paciente["Diagnostico"]).lower()
            
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
        
        # Convertir a minúsculas
        if "Mano" in metadata_dict and metadata_dict["Mano"] is not None:
            metadata_dict["Mano"] = str(metadata_dict["Mano"]).lower()
        if "Dedo" in metadata_dict and metadata_dict["Dedo"] is not None:
            metadata_dict["Dedo"] = str(metadata_dict["Dedo"]).lower()
            
        return metadata_dict

    def _imprimir_campo_pdf(pdf_obj, etiqueta, valor, unidad=""):
        if pd.notna(valor) and valor is not None and str(valor).strip() != "" and str(valor).lower() not in ["no especificado", "nan"]:
            pdf_obj.cell(200, 10, f"{etiqueta}: {valor}{unidad}", ln=True)

    def imprimir_parametros_y_config(pdf_obj, parametros_dict, titulo):
        pdf_obj.set_font("Arial", 'B', 12)
        pdf_obj.cell(0, 10, titulo, ln=True)
        pdf_obj.set_font("Arial", size=12)
        
        parametros_a_imprimir_con_unidad = {
            "Mano": "", "Dedo": ""
        }
        for param_key, unit in parametros_a_imprimir_con_unidad.items():
            value = parametros_dict.get(param_key)
            if pd.notna(value) and value is not None and str(value).strip() != "" and str(value).lower() not in ["no especificado", "nan"]:
                pdf_obj.cell(200, 10, f"{param_key}: {value}{unit}", ln=True)
        pdf_obj.ln(5)

    def generar_pdf(paciente_data, estimulacion_data, resultados_df, prediccion_texto, probabilidades_texto, graficos_paths):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Informe de Diagnóstico de Temblor", ln=True, align="C")
        pdf.ln(5)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Fecha y hora del análisis: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=True)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Datos del Paciente", ln=True)
        pdf.set_font("Arial", size=12)

        _imprimir_campo_pdf(pdf, "Nombre", paciente_data.get("Nombre"))
        _imprimir_campo_pdf(pdf, "Apellido", paciente_data.get("Apellido"))
        
        edad_val = paciente_data.get("Edad")
        if pd.notna(edad_val) and edad_val is not None and str(edad_val).strip() != "":
            try:
                edad_int = int(float(str(edad_val).replace(',', '.')))
                _imprimir_campo_pdf(pdf, "Edad", edad_int)
            except (ValueError, TypeError):
                pass
                
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

        # Imprimir el diagnóstico antes de los gráficos
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Diagnóstico (Predicción)", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, prediccion_texto)
        pdf.ln(5)
        if probabilidades_texto:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Probabilidades por clase:", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, probabilidades_texto)
            pdf.ln(5)
        
        # Generar gráficos
        for i, img_path in enumerate(graficos_paths):
            if os.path.exists(img_path):
                altura_grafico = 100
                if (pdf.get_y() + altura_grafico) > (pdf.h - 20):
                    pdf.add_page()
                    
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, f"Gráfico {i+1}", ln=True, align="C")
                pdf.image(img_path, x=15, w=180)
            else:
                pdf.cell(0, 10, f"Error: No se pudo cargar el gráfico {i+1}", ln=True)
            pdf.ln(5) # Añadir un espacio entre gráficos

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

        all_files_uploaded = all(file is not None for file in prediccion_files_correctas.values())

        if not all_files_uploaded:
            st.warning("Por favor, sube los 3 archivos para realizar el diagnóstico.")
        else:
            # ---------------------------------------VALIDACIÓN INTERNA--------------------------------------------------
            is_consistent, error_msg = validar_consistencia_por_nombre_archivo(prediccion_files_correctas, "Diagnóstico")

            if not is_consistent:
                st.error(error_msg)
                st.stop() # Detiene la ejecución si los archivos son inconsistentes
            # -----------------------------------------------------------------------------------------------------------
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

                model_filename = 'tremor_prediction_model_con_sanos_OPTIMO_POR_F1.joblib'
                prediccion_final = "No se pudo realizar el diagnóstico."
                probabilidades_texto = ""
                
                try:
                    modelo_cargado = joblib.load(model_filename)
                    prediction = modelo_cargado.predict(df_for_prediction)
                    prediccion_final = prediction[0]

                    st.subheader("Resultados del Diagnóstico:")
                    st.success(f"El diagnóstico tentativo es: **{prediccion_final}**")

                    if hasattr(modelo_cargado, 'predict_proba'):
                        probabilities = modelo_cargado.predict_proba(df_for_prediction)
                        probabilidades_list = []
                        if hasattr(modelo_cargado, 'classes_'):
                            for i, class_label in enumerate(modelo_cargado.classes_):
                                st.write(f"- **{class_label}**: {probabilities[0][i]*100:.2f}%")
                                probabilidades_list.append(f"- {class_label}: {probabilities[0][i]*100:.2f}%")
                            probabilidades_texto = "\n".join(probabilidades_list)
                        else:
                            st.info("El modelo no tiene el atributo 'classes_'. No se pueden mostrar las etiquetas de clase.")
                    st.warning(
                        """
                        **IMPORTANTE (Modo Tentativo):** La predicción del sistema es solo un **soporte cuantitativo**
                        basado en Inteligencia Artificial. El diagnóstico y el tratamiento final deben ser
                        determinados y validados **exclusivamente por el médico especialista** en el contexto clínico completo del paciente.
                        """
                    )
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

                pdf_output = generar_pdf(
                    datos_paciente, 
                    datos_estimulacion, 
                    df_metrics_display, 
                    f"El diagnóstico tentativo es: {prediccion_final}",
                    probabilidades_texto,
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
