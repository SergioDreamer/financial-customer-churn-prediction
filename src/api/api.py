from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pickle
import json
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn in financial institutions",
    version="1.0.0"
)

# Load model and related files
MODEL_DIR = "models"
DOCS_DIR = "docs"

# Load the best model
with open(os.path.join(MODEL_DIR, "best_model_info.json"), "r") as f:
    best_model_info = json.load(f)

model_name = best_model_info["model_name"]
model_path = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load feature sets
with open(os.path.join(MODEL_DIR, "feature_sets.json"), "r") as f:
    feature_sets = json.load(f)

# Get the features used by the model
feature_set_name = best_model_info["feature_set"]
features = feature_sets[feature_set_name]

# Load business insights if available
business_insights_path = os.path.join(DOCS_DIR, "business_insights.json")
if os.path.exists(business_insights_path):
    with open(business_insights_path, "r") as f:
        business_insights = json.load(f)
else:
    business_insights = {"recommendations": []}

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

# Helper function to engineer features
def engineer_features(customer_data: dict) -> pd.DataFrame:
    """
    Engineer features from raw customer data
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer_data])
    
    # One-hot encode categorical variables
    if 'Geography' in df.columns:
        df['Geography_Germany'] = (df['Geography'] == 'Germany').astype(int)
        df['Geography_Spain'] = (df['Geography'] == 'Spain').astype(int)
        df.drop('Geography', axis=1, inplace=True)
    
    if 'Gender' in df.columns:
        df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
        df.drop('Gender', axis=1, inplace=True)
    
    # Create age-related features
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['IsYoung'] = (df['Age'] < 30).astype(int)
    df['IsMiddleAged'] = ((df['Age'] >= 30) & (df['Age'] < 50)).astype(int)
    df['IsSenior'] = (df['Age'] >= 50).astype(int)
    df['IsRetirementAge'] = (df['Age'] >= 65).astype(int)
    
    # Create geography-related features
    if 'Geography_Germany' in df.columns:
        df['GermanyXAge'] = df['Geography_Germany'] * df['Age']
        df['GermanyXBalance'] = df['Geography_Germany'] * df['Balance']
        df['GermanyXSenior'] = df['Geography_Germany'] * df['IsSenior']
    
    # Create balance-related features
    df['HasZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    high_balance_threshold = 100000  # This is an approximation, should be based on actual data
    df['HasHighBalance'] = (df['Balance'] > high_balance_threshold).astype(int)
    
    # Create product-related features
    df['HasMultipleProducts'] = (df['NumOfProducts'] > 1).astype(int)
    df['HasManyProducts'] = (df['NumOfProducts'] >= 3).astype(int)
    df['ProductsXAge'] = df['NumOfProducts'] * df['Age']
    df['ProductsXBalance'] = df['NumOfProducts'] * df['Balance']
    df['ProductsXTenure'] = df['NumOfProducts'] * df['Tenure']
    
    # Create tenure-related features
    df['IsNewCustomer'] = (df['Tenure'] <= 1).astype(int)
    df['IsLongTermCustomer'] = (df['Tenure'] >= 8).astype(int)
    df['TenureSquared'] = df['Tenure'] ** 2
    df['CustomerValue'] = df['Tenure'] * df['Balance'] / 1000
    
    # Create engagement-related features
    df['EngagementScore'] = df['IsActiveMember'] * 0.5 + df['HasCrCard'] * 0.3 + (df['NumOfProducts'] / 4) * 0.2
    df['ActiveXTenure'] = df['IsActiveMember'] * df['Tenure']
    df['ActiveXProducts'] = df['IsActiveMember'] * df['NumOfProducts']
    df['ActiveXAge'] = df['IsActiveMember'] * df['Age']
    df['InactiveSenior'] = ((df['IsActiveMember'] == 0) & (df['IsSenior'] == 1)).astype(int)
    
    # Create risk score features
    df['ChurnRiskScore'] = (
        df['IsSenior'] * 0.25 + 
        df.get('Geography_Germany', 0) * 0.20 + 
        (1 - df['IsActiveMember']) * 0.25 + 
        df['HasManyProducts'] * 0.20 + 
        (df.get('Gender_Male', 0) == 0).astype(int) * 0.10
    )
    
    df['DemographicRiskScore'] = (
        df['IsSenior'] * 0.4 + 
        df.get('Geography_Germany', 0) * 0.4 + 
        (df.get('Gender_Male', 0) == 0).astype(int) * 0.2
    )
    
    df['ProductRiskScore'] = (
        df['HasManyProducts'] * 0.5 + 
        (1 - df['IsActiveMember']) * 0.3 + 
        (1 - df['HasCrCard']) * 0.2
    )
    
    # Select only the features needed by the model
    return df[features]

# Helper function to get risk factors
def get_risk_factors(customer_data: dict, churn_probability: float) -> List[str]:
    """
    Get risk factors for a customer based on their data and churn probability
    """
    risk_factors = []
    
    # Age-related risk
    if customer_data.get('Age', 0) > 50:
        risk_factors.append("Customer is over 50 years old")
    
    # Geography-related risk
    if customer_data.get('Geography') == 'Germany':
        risk_factors.append("Customer is from Germany")
    
    # Product-related risk
    if customer_data.get('NumOfProducts', 0) >= 3:
        risk_factors.append("Customer has 3 or more products")
    
    # Activity-related risk
    if customer_data.get('IsActiveMember', 1) == 0:
        risk_factors.append("Customer is not an active member")
    
    # Balance-related risk
    if customer_data.get('Balance', 0) == 0:
        risk_factors.append("Customer has zero balance")
    
    # Tenure-related risk
    if customer_data.get('Tenure', 0) <= 1:
        risk_factors.append("Customer is relatively new (tenure ≤ 1 year)")
    
    # If no specific risk factors are identified but probability is high
    if not risk_factors and churn_probability > 0.5:
        risk_factors.append("Multiple combined factors contribute to churn risk")
    
    return risk_factors

# Helper function to get recommendations
def get_recommendations(risk_factors: List[str]) -> List[str]:
    """
    Get personalized recommendations based on risk factors
    """
    recommendations = []
    
    for factor in risk_factors:
        if "over 50" in factor:
            recommendations.append("Offer senior-specific benefits or loyalty program")
        elif "Germany" in factor:
            recommendations.append("Provide region-specific customer service or products")
        elif "3 or more products" in factor:
            recommendations.append("Review pricing structure for customers with multiple products")
        elif "not an active member" in factor:
            recommendations.append("Implement re-engagement campaign with special offers")
        elif "zero balance" in factor:
            recommendations.append("Offer incentives for maintaining minimum balance")
        elif "new" in factor:
            recommendations.append("Enhance onboarding experience and early relationship building")
    
    # Add general recommendation if no specific ones are available
    if not recommendations:
        recommendations.append("Implement regular check-ins and personalized offers")
    
    return recommendations

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
        "feature_set": feature_set_name,
        "metrics": best_model_info["metrics"],
        "features": features
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
        
        # Engineer features
        X = engineer_features(customer_dict)
        
        # Make prediction
        churn_probability = float(model.predict_proba(X)[0, 1])
        churn_prediction = bool(churn_probability >= 0.5)
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Get risk factors
        risk_factors = get_risk_factors(customer_dict, churn_probability)
        
        # Get recommendations
        recommendations = get_recommendations(risk_factors)
        
        return {
            "churn_probability": churn_probability,
            "churn_prediction": churn_prediction,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict_churn(customers: List[CustomerData]):
    """
    Predict churn for multiple customers
    """
    try:
        results = []
        
        for customer in customers:
            # Convert input data to dictionary
            customer_dict = customer.dict()
            
            # Engineer features
            X = engineer_features(customer_dict)
            
            # Make prediction
            churn_probability = float(model.predict_proba(X)[0, 1])
            churn_prediction = bool(churn_probability >= 0.5)
            
            # Determine risk level
            if churn_probability < 0.3:
                risk_level = "Low"
            elif churn_probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Get risk factors
            risk_factors = get_risk_factors(customer_dict, churn_probability)
            
            # Get recommendations
            recommendations = get_recommendations(risk_factors)
            
            results.append({
                "churn_probability": churn_probability,
                "churn_prediction": churn_prediction,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendations": recommendations
            })
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
