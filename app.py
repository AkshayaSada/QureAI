from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
quantum_model = joblib.load("backend/quantum_model.pkl")
quantum_scaler = joblib.load("backend/quantum_scaler.pkl")
quantum_selector = joblib.load("backend/quantum_selector.pkl")
quantum_features = joblib.load("backend/quantum_features.pkl")

# Create FastAPI app
app = FastAPI(title="QureAI API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and supporting files
model = joblib.load("backend/qureai_model.pkl")
label_encoder = joblib.load("backend/label_encoder.pkl")
symptoms = joblib.load("backend/symptoms.pkl")
symptoms = [s for s in symptoms if not str(s).startswith("Unnamed")]

# Data received from frontend
class SymptomRequest(BaseModel):
    symptoms: list[str]


@app.get("/")
def home():
    return {
        "message": "QureAI backend is running!",
        "status": "online"
    }
@app.get("/symptoms")
def get_symptoms():
    return {
        "symptoms": symptoms
    }


@app.post("/predict")
def predict(request: SymptomRequest):

    # -------------------------------
    # Classical Random Forest
    # -------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=symptoms
    )

    for symptom in request.symptoms:
        if symptom in input_data.columns:
            input_data.loc[0, symptom] = 1

    classical_prediction = model.predict(input_data)[0]

    classical_probabilities = model.predict_proba(input_data)[0]

    classical_probability = float(
        classical_probabilities[classical_prediction]
    )

    classical_disease = label_encoder.inverse_transform(
        [classical_prediction]
    )[0]


    # -------------------------------
    # Quantum ML prediction
    # -------------------------------

    quantum_input = pd.DataFrame(
        0,
        index=[0],
        columns=quantum_features
    )

    for symptom in request.symptoms:
        if symptom in quantum_input.columns:
            quantum_input.loc[0, symptom] = 1

    quantum_input_scaled = quantum_scaler.transform(
        quantum_input
    )

    quantum_prediction = quantum_model.predict(
        quantum_input_scaled
    )[0]

    quantum_disease = str(quantum_prediction)


    # -------------------------------
    # Hybrid Decision Engine
    # -------------------------------

    models_agree = (
        classical_disease.lower()
        == quantum_disease.lower()
    )

    if models_agree:
        hybrid_result = classical_disease
        decision_status = "Classical and Quantum models agree"
    else:
        hybrid_result = classical_disease
        decision_status = "Classical and Quantum models produced different results"


    # -------------------------------
    # Explainable AI
    # -------------------------------

    feature_importance = []

    for symptom in request.symptoms:

        if symptom in input_data.columns:

            index = list(
                input_data.columns
            ).index(symptom)

            importance = float(
                model.feature_importances_[index]
            )

            feature_importance.append({
                "symptom": symptom,
                "importance": round(
                    importance * 100,
                    2
                )
            })

    feature_importance.sort(
        key=lambda x: x["importance"],
        reverse=True
    )


    # -------------------------------
    # Final response
    # -------------------------------

    return {
        "possible_condition": hybrid_result,

        "model_probability": round(
            classical_probability * 100,
            2
        ),

        "classical_prediction": classical_disease,

        "quantum_prediction": quantum_disease,

        "models_agree": models_agree,

        "decision_status": decision_status,

        "selected_symptoms": request.symptoms,

        "quantum_features": quantum_features,

        "feature_importance": feature_importance,

        "message":
            "This is an AI-based risk assessment, "
            "not a medical diagnosis."
    }