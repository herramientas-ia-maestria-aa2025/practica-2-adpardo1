import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Cargar dataset
df = pd.read_csv('data/credit_risk_dataset.csv')

# Variables categóricas a codificar
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns=['loan_status'])  
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

mlflow.set_experiment("credit_risk_experiment")
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")
    print(f"Run ID: {run.info.run_id}")

for col, le in encoders.items():
    with open(f'{col}_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

print("Modelo y encoders guardados correctamente.")
