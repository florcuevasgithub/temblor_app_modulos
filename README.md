## Aplicación para Análisis Cuantitativo del Temblor
Esta aplicación es una herramienta de soporte al diagnóstico y monitoreo terapéutico, desarrollada para la tesis de Ingenieria Biomedica. 
Utiliza el procesamiento avanzado de señales y técnicas de Machine Learning para cuantificar y clasificar el temblor en pacientes con Parkinson (PK) y Temblor Esencial (TE).

## Tecnologías Clave
La aplicación está construida usando el entorno de datos de Python y se despliega como una aplicación web interactiva.

- Framework Principal: Streamlit (para la interfaz de usuario web).

- Procesamiento de Datos: Pandas y NumPy.

- Análisis de Señales: SciPy (para FFT y análisis de frecuencia).

- Generación de Informes: FPDF (para crear el informe clínico en PDF).

- Visualización: Matplotlib y Plotly (para gráficos interactivos).

- Inteligencia Artificial: Scikit-learn (para el entrenamiento y predicción del modelo).

## Objetivos y Funcionalidades Clave
# Módulo 1: Perfil Cinemático Individual
Función: Procesar mediciones individuales de acelerometría y extraer los biomarcadores cinemáticos esenciales del temblor (Frecuencia Dominante, Amplitud y RMS) bajo las tres condiciones de activación (Reposo, Postural y Acción).

Salida: Informe en PDF con visualizaciones y una interpretación clínica cuantificable.

# Módulo 2: Monitoreo y Validación Terapéutica
Función: Comparar dos o más perfiles completos de temblor del mismo paciente para evaluar la respuesta a diferentes tratamientos o la evolución de la enfermedad.

Monitoreo Flexible: Permite comparar el estado Basal vs. Seguimiento, el Efecto de Medicación (OFF vs. ON), o el impacto de la Estimulación Cerebral Profunda (DBS).

# Módulo 3: Predicción de Tipo de Temblor (IA)
Función: Utilizar un modelo de Machine Learning entrenado con el dataset de biomarcadores para predecir el tipo de temblor (Parkinson, Temblor Esencial, o Fisiológico).

Aporte a la Tesis: Validación de la capacidad de la IA para asistir en el diagnóstico diferencial del temblor con un índice de confianza.
