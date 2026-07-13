# importing all the required libraries and dependencies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import pickle as pk
import numpy as np

# initializing the core application
app = FastAPI(
    title = "Income Predicting Management API",
    description = "A production-grade API for serving and managing ML configurations",
    version = "1.0.0"
)

# the data engine (pydantic models)
class ModelConfig(BaseModel):
    age: int = Field()
    workclass: str = Field()
    fnlwgt: int = Field()
    education: str = Field()
    educational_num: int = Field()
    marital_status: str = Field()
    occupation: str = Field()
    relationship: str = Field()
    race: str = Field()
    gender: str = Field()
    capital_gain: int = Field()
    capital_loss: int = Field()
    hours_per_week: int = Field()
    native_country: str = Field()

ARTIFACTS_PATH = "predictive_socio-economic_model_artifacts.pkl"
if not os.path.exists(ARTIFACTS_PATH):
    raise FileNotFoundError(f"could not find model artifacts at: {ARTIFACTS_PATH}. please run your training script first")

with open(ARTIFACTS_PATH, "rb") as file:
    artifacts = pk.load(file)

encoder = artifacts["encoder"]
scaler = artifacts["scaler"]
model = artifacts["model"]
accuracy = artifacts["metrices"]["accuracy"]
report = artifacts["metrices"]["report"]

# GET Request: Fetching data from the server
@app.get("/")
def root():
    return{
        "status": "Healthy",
        "message": "Welcome to the Model Serving API. Navigate to /docs for interactive documentation"
    }

# POST Request: Receiving data to process or execute
@app.post("/predict")
def create_model_configuration(config: ModelConfig):
    try:
        cat_features = [[
            config.workclass, config.education, config.marital_status, 
            config.occupation, config.relationship, config.race, 
            config.gender, config.native_country
        ]]
        
        num_features = [[
            config.age, config.fnlwgt, config.educational_num, 
            config.capital_gain, config.capital_loss, config.hours_per_week
        ]]

        # replicate the main.py preprocessing lofic block
        if encoder is not None:
            encoder_cat = encoder.transform(cat_features)
            x_combined = np.hstack((encoder_cat, num_features))
        else:
            x_combined = np.array(num_features)

        # scaling the data using your saved training rules
        scaled_features = scaler.transform(x_combined)

        # Generating hard classification prediction (this returns string labels like '<=50K' or '>50K')
        raw_prediction = model.predict(scaled_features)[0]

        # Generating prediction probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Check how your classes are ordered in the model to assign confidence score accurately
        # model.classes_ usually outputs array(['<=50K', '>50K']) where index 1 is high income
        risk_probabilities = float(probabilities[1])

        # Building response structure by checking the string text directly
        if raw_prediction == ">50K":
            prediction_code = 1
            result = "Income More than $50k USD"
        else:
            prediction_code = 0
            result = "Income less than $50k USD"

        return {
            "prediction_code": prediction_code,
            "prediction_label": result,
            "confidence_score": round(risk_probabilities * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error occured: {str(e)}")
