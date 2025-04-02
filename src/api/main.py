import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_prediction import ChurnPredictor

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn in financial institutions",
    version="1.0.0"
)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Define model paths
model_info_path = os.path.join(MODEL_DIR, "best_model_info.json")
feature_sets_path = os.path.join(MODEL_DIR, "feature_sets.json")

# Load model info
with open(model_info_path, "r") as f:
    model_info = json.load(f)

model_name = model_info["model_name"]
model_path = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")

# Load business insights if available
business_insights_path = os.path.join(DOCS_DIR, "business_insights.json")
if os.path.exists(business_insights_path):
    with open(business_insights_path, "r") as f:
        business_insights = json.load(f)
else:
    business_insights = {"recommendations": []}

# Create predictor instance
predictor = ChurnPredictor(model_path, feature_sets_path, model_info_path)

# Define input data model
class CustomerData(BaseModel):
    """
    Input data model for customer churn prediction
    """
    CreditScore: int = Field(..., description="Credit score of the customer", example=650)
    Age: int = Field(..., description="Age of the customer", example=45)
    Tenure: int = Field(..., description="Number of years the customer has been with the bank", example=5)
    Balance: float = Field(..., description="Account balance", example=125000.0)
    NumOfProducts: int = Field(..., description="Number of bank products the customer uses", example=2)
    HasCrCard: int = Field(..., description="Whether the customer has a credit card (1=Yes, 0=No)", example=1)
    IsActiveMember: int = Field(..., description="Whether the customer is an active member (1=Yes, 0=No)", example=1)
    EstimatedSalary: float = Field(..., description="Estimated salary of the customer", example=100000.0)
    Geography: str = Field(..., description="Customer's geography (France, Germany, Spain)", example="France")
    Gender: str = Field(..., description="Customer's gender (Male, Female)", example="Male")

# Define prediction response model
class PredictionResponse(BaseModel):
    """
    Response model for churn prediction
    """
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    risk_factors: List[str]
    recommendations: List[str]

# Define model information response
class ModelInfoResponse(BaseModel):
    """
    Response model for model information
    """
    model_name: str
    feature_set: str
    metrics: Dict[str, float]
    features: List[str]

# Define health check response
class HealthResponse(BaseModel):
    """
    Response model for health check
    """
    status: str
    model_name: str
    api_version: str

# Define API endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_name": model_name,
        "api_version": "1.0.0"
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the deployed model
    """
    return {
        "model_name": model_name,
        "feature_set": model_info["feature_set"],
        "metrics": model_info["metrics"],
        "features": predictor.features
    }

@app.get("/insights", response_model=Dict[str, Any])
async def get_insights():
    """
    Get business insights and recommendations
    """
    return business_insights

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Predict customer churn probability
    """
    try:
        # Convert input data to dictionary
        customer_dict = customer.dict()
        
        # Make prediction
        prediction = predictor.predict(customer_dict)
        
        return prediction
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict_churn(customers: List[CustomerData]):
    """
    Predict churn for multiple customers
    """
    try:
        # Convert input data to list of dictionaries
        customers_dict = [customer.dict() for customer in customers]
        
        # Make batch prediction
        predictions = predictor.batch_predict(customers_dict)
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
