# Evaluación de Riesgo Crediticio

Este proyecto implementa un modelo de aprendizaje automático para predecir el riesgo de incumplimiento crediticio. Incluye el entrenamiento del modelo, creación de un pipeline, desarrollo de una API REST con Flask y una interfaz de usuario con Streamlit para facilitar la interacción.
## Contenido

    train.py: Script para entrenar el modelo con pipeline, preprocesamiento y guardar el pipeline.

    main.py: Entrenamiento alternativo con MLflow y guardado de codificadores.

    app.py: API REST en Flask para predicciones, carga modelo y codificadores.

    streamlit_app.py: Interfaz web interactiva para ingreso de datos y predicción.

    data/credit_risk_dataset.csv: Dataset con datos de préstamos y clientes.

    Codificadores (*_encoder.pkl) para variables categóricas.

    Modelo serializado pipeline_credit_risk.pkl.

## Requisitos

    Python 3.8+

    Paquetes: pandas, scikit-learn, joblib, mlflow, flask, streamlit

Puedes instalar las dependencias con:

pip install -r requirements.txt

## Uso
### Entrenamiento

Entrena el modelo y guarda el pipeline con:

python train.py

El modelo y los encoders se guardan para su posterior uso.
API Flask

Para ejecutar la API REST que recibe peticiones POST para predecir:

python app.py

La API correrá en http://localhost:5000.

### Ejemplo de endpoint:

    GET / — Estado de la API.

    POST /predict — Recibe JSON con datos y retorna predicción y probabilidad.

## Interfaz con Streamlit

Para ejecutar la app web interactiva:

streamlit run streamlit_app.py

Podrás ingresar los datos del préstamo y obtener predicción inmediata de riesgo.
Estructura de Datos

## El modelo usa las siguientes variables:
    Variable	Tipo	Descripción
    person_emp_length	Numérica	Años de empleo
    person_home_ownership	Categórica	Tipo de propiedad de vivienda
    loan_intent	Categórica	Intención del préstamo
    loan_grade	Categórica	Calificación del préstamo
    cb_person_default_on_file	Categórica	Registro de incumplimiento (Y/N)
    loan_amnt	Numérica	Monto del préstamo
    loan_int_rate	Numérica	Tasa de interés (%)