from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

app = FastAPI(
    title="Patient Length of Stay Prediction API",
    description="API for predicting hospital patient length of stay.",
    version="1.0.0"
)

# Allow the frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved preprocessing pipeline and trained model
preprocessor = joblib.load("los_preprocessor.pkl")
model = joblib.load("final_xgboost_model.pkl")

# Create a SHAP explainer for the trained XGBoost model
explainer = shap.TreeExplainer(model)

# Define the 27 patient inputs expected by the model
class PatientData(BaseModel):   # Create PatientData (class) to inherit Pydantic's BaseModel validation functionality
    patient_id: str
    
    dialysisrenalendstage: int
    asthma: int
    irondef: int
    pneum: int
    substancedependence: int
    psychologicaldisordermajor: int
    depress: int
    psychother: int
    fibrosisandother: int
    malnutrition: int

    hemo: float
    hematocrit: float
    neutrophils: float
    sodium: float
    glucose: float
    bloodureanitro: float
    creatinine: float
    bmi: float
    pulse: float
    respiration: float

    secondarydiagnosisnonicd9: int
    Weekend_Admission: int

    rcount: str
    gender: str
    facid: str
    Admission_DayOfWeek: str

    Admission_Month: int


@app.get("/")
def home():

    # Return a message confirming that the API is running
    return {
        "message": "Patient Length of Stay Prediction API is running"
    }


@app.post("/predict")
def predict(patient: PatientData):      # patient (object), PatientData (class)  

    # Convert the patient data into a Python dictionary
    patient_dict = patient.model_dump()

    # Separate the patient ID because it is not used by the ML model
    patient_id = patient_dict.pop("patient_id")

    # Convert the 27 model features into a one-row DataFrame
    patient_df = pd.DataFrame([patient_dict])

    # Apply the same preprocessing used during model training
    patient_processed = preprocessor.transform(patient_df)

    # Make the LOS prediction
    prediction = model.predict(patient_processed)[0]

    # Calculate SHAP values for this patient's prediction
    shap_values = explainer.shap_values(patient_processed)[0]

    # Get the names of the 42 transformed features
    feature_names = preprocessor.get_feature_names_out()
    
    # Pair each transformed feature with its SHAP value
    feature_contributions = list(zip(feature_names, shap_values))
    
    # Sort features by the absolute size of their SHAP contribution
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Keep only the 5 strongest contributors
    top_contributions = feature_contributions[:5]

    # Convert technical transformed feature names into readable labels
    feature_labels = {
        "dialysisrenalendstage": "Dialysis / Renal End-Stage Disease",
        "asthma": "Asthma",
        "irondef": "Iron Deficiency",
        "pneum": "Pneumonia",
        "substancedependence": "Substance Dependence",
        "psychologicaldisordermajor": "Major Psychological Disorder",
        "depress": "Depression",
        "psychother": "Psychotherapy",
        "fibrosisandother": "Fibrosis and Other Disorders",
        "malnutrition": "Malnutrition",
        "hemo": "Hemoglobin",
        "hematocrit": "Hematocrit",
        "neutrophils": "Neutrophils",
        "sodium": "Sodium",
        "glucose": "Glucose",
        "bloodureanitro": "Blood Urea Nitrogen",
        "creatinine": "Creatinine",
        "bmi": "BMI",
        "pulse": "Pulse Rate",
        "respiration": "Respiratory Rate",
        "secondarydiagnosisnonicd9": "Secondary Diagnosis Count",
        "Weekend_Admission": "Weekend Admission",
        "rcount": "Readmission Count",
        "gender": "Gender",
        "facid": "Facility",
        "Admission_DayOfWeek": "Admission Day of Week"
    }
    
    explanations = []
    
    for feature, value in top_contributions:
    
        # Remove the ColumnTransformer prefix such as "num__" or "cat__"
        clean_feature = feature.split("__", 1)[-1]
    
        # Convert one-hot encoded features back to their original feature name
        original_feature = clean_feature
    
        for name in feature_labels:
            if clean_feature == name or clean_feature.startswith(name + "_"):
                original_feature = name
                break
    
        explanations.append({
            "feature": feature_labels.get(original_feature, original_feature),
            "shap_value": round(float(value), 3),
            "direction": "increased" if value > 0 else "decreased"
        })

    # Assign the LOS category
    if prediction < 4:
        category = "Short Stay"
    elif prediction <= 6:
        category = "Typical Stay"
    else:
        category = "Extended Stay"

    # Send the result back
    return {
        "patient_id": patient_id,
        "predicted_los": round(float(prediction), 2),
        "los_category": category,
        "explanations": explanations
    }