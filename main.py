from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

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
        "los_category": category
    }