import streamlit as st
import joblib
import pandas as pd

st.title("Evaluación de Riesgo Crediticio")

# Carga el pipeline entrenado
pipeline = joblib.load('pipeline_credit_risk.pkl')

# Inputs del usuario
person_emp_length = st.slider("Años de empleo", 0, 50, 5)

person_home_ownership = st.selectbox(
    "Propiedad de la vivienda",
    ["OWN", "RENT", "MORTGAGE", "OTHER", "NONE"]
)

loan_intent = st.selectbox(
    "Intención del préstamo",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "OTHER"]
)

loan_grade = st.selectbox(
    "Calificación del préstamo",
    ["A", "B", "C", "D", "E", "F", "G"]
)

cb_person_default_on_file = st.selectbox(
    "Registro de incumplimiento en archivo crediticio",
    ["Y", "N"]
)

loan_amnt = st.number_input("Monto del préstamo", min_value=1000, max_value=50000, value=10000, step=500)

loan_int_rate = st.number_input("Tasa de interés (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

# Crear dataframe de entrada con las columnas en el orden esperado por el pipeline
input_dict = {
    'person_home_ownership': [person_home_ownership],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'cb_person_default_on_file': [cb_person_default_on_file],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
}

input_df = pd.DataFrame(input_dict)

if st.button("Predecir riesgo de incumplimiento"):
    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"Riesgo alto de incumplimiento. Probabilidad: {proba:.2f}")
    else:
        st.success(f"Riesgo bajo de incumplimiento. Probabilidad: {proba:.2f}")
