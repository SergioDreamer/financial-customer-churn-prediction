import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ChurnPredictor:
    """
    Class for making churn predictions using the trained model
    """
    def __init__(self, model_path: str, feature_sets_path: str, model_info_path: str):
        """
        Initialize the ChurnPredictor
        
        Args:
            model_path: Path to the serialized model file
            feature_sets_path: Path to the feature sets JSON file
            model_info_path: Path to the model info JSON file
        """
        # Load the model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        # Load feature sets
        with open(feature_sets_path, "r") as f:
            self.feature_sets = json.load(f)
        
        # Load model info
        with open(model_info_path, "r") as f:
            self.model_info = json.load(f)
        
        # Get the features used by the model
        self.feature_set_name = self.model_info["feature_set"]
        self.features = self.feature_sets[self.feature_set_name]
        
        print(f"Loaded model: {self.model_info['model_name']}")
        print(f"Feature set: {self.feature_set_name} with {len(self.features)} features")
    
    def engineer_features(self, customer_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Engineer features from raw customer data
        
        Args:
            customer_data: Dictionary containing customer data
            
        Returns:
            DataFrame with engineered features
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
        return df[self.features]
    
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn for a customer
        
        Args:
            customer_data: Dictionary containing customer data
            
        Returns:
            Dictionary with prediction results
        """
        # Engineer features
        X = self.engineer_features(customer_data)
        
        # Make prediction
        churn_probability = float(self.model.predict_proba(X)[0, 1])
        churn_prediction = bool(churn_probability >= 0.5)
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Get risk factors
        risk_factors = self.get_risk_factors(customer_data, churn_probability)
        
        # Get recommendations
        recommendations = self.get_recommendations(risk_factors)
        
        return {
            "churn_probability": churn_probability,
            "churn_prediction": churn_prediction,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }
    
    def batch_predict(self, customers_data: list) -> list:
        """
        Predict churn for multiple customers
        
        Args:
            customers_data: List of dictionaries containing customer data
            
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        
        for customer_data in customers_data:
            results.append(self.predict(customer_data))
        
        return results
    
    def get_risk_factors(self, customer_data: Dict[str, Any], churn_probability: float) -> list:
        """
        Get risk factors for a customer based on their data and churn probability
        
        Args:
            customer_data: Dictionary containing customer data
            churn_probability: Predicted churn probability
            
        Returns:
            List of risk factors
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
    
    def get_recommendations(self, risk_factors: list) -> list:
        """
        Get personalized recommendations based on risk factors
        
        Args:
            risk_factors: List of risk factors
            
        Returns:
            List of recommendations
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

# Example usage
if __name__ == "__main__":
    # Define paths
    MODEL_DIR = "../../models"
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
