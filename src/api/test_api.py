import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def test_api_endpoints():
    """
    Test all API endpoints
    """
    base_url = "http://localhost:8000"
    
    # Test health check endpoint
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error: {str(e)}")
        print()
    
    # Test model info endpoint
    print("Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error: {str(e)}")
        print()
    
    # Test insights endpoint
    print("Testing insights endpoint...")
    try:
        response = requests.get(f"{base_url}/insights")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error: {str(e)}")
        print()
    
    # Test predict endpoint
    print("Testing predict endpoint...")
    try:
        # Example customer data
        customer_data = {
            "CreditScore": 650,
            "Age": 45,
            "Tenure": 5,
            "Balance": 125000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 100000.0,
            "Geography": "France",
            "Gender": "Male"
        }
        
        response = requests.post(f"{base_url}/predict", json=customer_data)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error: {str(e)}")
        print()
    
    # Test batch predict endpoint
    print("Testing batch predict endpoint...")
    try:
        # Example customer data
        customers_data = [
            {
                "CreditScore": 650,
                "Age": 45,
                "Tenure": 5,
                "Balance": 125000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 100000.0,
                "Geography": "France",
                "Gender": "Male"
            },
            {
                "CreditScore": 700,
                "Age": 55,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 3,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 120000.0,
                "Geography": "Germany",
                "Gender": "Female"
            }
        ]
        
        response = requests.post(f"{base_url}/batch-predict", json=customers_data)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error: {str(e)}")
        print()

def test_model_prediction():
    """
    Test model prediction directly
    """
    print("Testing model prediction directly...")
    
    # Import the ChurnPredictor class
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.model_prediction import ChurnPredictor
    
    # Define paths
    MODEL_DIR = "../models"
    model_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    feature_sets_path = os.path.join(MODEL_DIR, "feature_sets.json")
    model_info_path = os.path.join(MODEL_DIR, "best_model_info.json")
    
    # Create predictor
    predictor = ChurnPredictor(model_path, feature_sets_path, model_info_path)
    
    # Example customer data
    customer_data = {
        "CreditScore": 650,
        "Age": 45,
        "Tenure": 5,
        "Balance": 125000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100000.0,
        "Geography": "France",
        "Gender": "Male"
    }
    
    # Make prediction
    prediction = predictor.predict(customer_data)
    print("Prediction:", prediction)
    print()
    
    # Test high-risk customer
    high_risk_customer = {
        "CreditScore": 600,
        "Age": 58,
        "Tenure": 1,
        "Balance": 0.0,
        "NumOfProducts": 3,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 80000.0,
        "Geography": "Germany",
        "Gender": "Female"
    }
    
    # Make prediction
    prediction = predictor.predict(high_risk_customer)
    print("High-risk customer prediction:", prediction)
    print()

if __name__ == "__main__":
    print("Note: This test script assumes the API is running at http://localhost:8000")
    print("To run the API, execute the run_api.sh script in a separate terminal")
    print()
    
    # Test model prediction directly
    test_model_prediction()
    
    # Uncomment to test API endpoints when the API is running
    # test_api_endpoints()
