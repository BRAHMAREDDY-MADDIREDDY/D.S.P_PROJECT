from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model and label encoders
model = joblib.load('water_quality_model.pkl')
encoder_source = joblib.load('label_encoder_source.pkl')
encoder_color = joblib.load('label_encoder_color.pkl')
encoder_odor = joblib.load('label_encoder_odor.pkl')
encoder_time = joblib.load('label_encoder_time.pkl')

# Define the input data model using Pydantic
class PredictionInput(BaseModel):
    ph: float
    iron: float
    nitrate: float
    chloride: float
    lead: float
    zinc: float
    color: str
    turbidity: float
    fluoride: float
    copper: float
    odor: str
    sulfate: float
    conductivity: float
    chlorine: float
    manganese: float
    tds: float
    source: str
    water_temp: float
    air_temp: float
    month: int
    day: int
    time_of_day: str

# Initialize the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Encode categorical features
        color_encoded = encoder_color.transform([input_data.color])[0]
        odor_encoded = encoder_odor.transform([input_data.odor])[0]
        source_encoded = encoder_source.transform([input_data.source])[0]
        time_encoded = encoder_time.transform([input_data.time_of_day])[0]

        # Prepare input data for prediction
        input_array = np.array([
            input_data.ph, input_data.iron, input_data.nitrate, input_data.chloride,
            input_data.lead, input_data.zinc, color_encoded, input_data.turbidity,
            input_data.fluoride, input_data.copper, odor_encoded, input_data.sulfate,
            input_data.conductivity, input_data.chlorine, input_data.manganese,
            input_data.tds, source_encoded, input_data.water_temp, input_data.air_temp,
            input_data.month, input_data.day, time_encoded
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn main:app --reload