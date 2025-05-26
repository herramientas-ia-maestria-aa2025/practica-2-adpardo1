from flask import Flask, request, jsonify
import mlflow.sklearn
import pickle
import pandas as pd

app = Flask(__name__)

RUN_ID = "f1a82ad91b0c4666b621a5ca77b6932e"  
MODEL_PATH = f"runs:/{RUN_ID}/model"

model = mlflow.sklearn.load_model(MODEL_PATH)

# Cargar encoders
encoders = {}
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_cols:
    with open(f"{col}_encoder.pkl", "rb") as f:
        encoders[col] = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "API de predicción de riesgo crediticio está activa."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        input_df = pd.DataFrame([data])

        # Aplicar encoders a variables categóricas
        for col, le in encoders.items():
            if col not in input_df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400
            input_df[col] = le.transform(input_df[col])

        # Predecir clase (0/1)
        prediction = model.predict(input_df)[0]
        # Predecir probabilidad de clase 1 (riesgo)
        proba = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(proba)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
