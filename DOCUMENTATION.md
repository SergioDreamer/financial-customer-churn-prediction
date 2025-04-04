# Financial Customer Churn Prediction System Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Dictionary](#data-dictionary)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Model Interpretability](#model-interpretability)
7. [Deployment Framework](#deployment-framework)
8. [Usage Guide](#usage-guide)
9. [Ethical Considerations](#ethical-considerations)
10. [Future Improvements](#future-improvements)

## Project Overview

The Financial Customer Churn Prediction System is designed to predict which customers are likely to leave a financial institution. By identifying at-risk customers early, the institution can take proactive measures to improve retention and reduce customer attrition.

### Business Context

Customer churn is a significant challenge for financial institutions, with acquisition costs typically 5-25 times higher than retention costs. This system helps address this challenge by:

1. Identifying customers at high risk of churning
2. Providing insights into factors driving churn
3. Generating personalized recommendations for retention strategies
4. Enabling proactive intervention before customers decide to leave

### Project Objectives
dat
1. Develop a machine learning model to predict customer churn with high accuracy
2. Identify key factors influencing customer decisions to leave
3. Create a deployment framework for real-time predictions
4. Provide actionable insights for business stakeholders

## System Architecture

The system follows a modular architecture with the following components:

```
financial-customer-churn-prediction/
├── data/
│   ├── raw/           # Original dataset
│   ├── processed/     # Processed data after feature engineering
│   └── cleaned/       # Cleaned data after preprocessing
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_interpretability.ipynb
├── src/
│   ├── data/          # Data processing modules
│   ├── features/      # Feature engineering modules
│   ├── models/        # Model training and prediction modules
│   ├── evaluation/    # Model evaluation modules
│   ├── utils/         # Utility functions
│   └── api/           # API implementation
├── models/            # Saved model files
├── docs/              # Documentation and visualizations (created in 01_data_preparation)
│   └── plots/         # Visualization outputs
└── tests/             # Test modules
```

### Technology Stack

- **Programming Language**: Python 3.10
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn
- **Model Interpretability**: SHAP, Permutation Importance
- **API Framework**: FastAPI
- **Deployment**: Uvicorn

## Data Dictionary

The dataset contains information about bank customers and their churn status:

| Feature | Description | Type |
|---------|-------------|------|
| CustomerID | Unique identifier for the customer | Integer |
| Surname | Customer's surname | String |
| CreditScore | Credit score of the customer | Integer |
| Geography | Customer's location (France, Germany, Spain) | Categorical |
| Gender | Customer's gender (Male, Female) | Categorical |
| Age | Customer's age in years | Integer |
| Tenure | Number of years the customer has been with the bank | Integer |
| Balance | Account balance | Float |
| NumOfProducts | Number of bank products the customer uses | Integer |
| HasCrCard | Whether the customer has a credit card (1=Yes, 0=No) | Binary |
| IsActiveMember | Whether the customer is an active member (1=Yes, 0=No) | Binary |
| EstimatedSalary | Estimated salary of the customer | Float |
| Exited | Whether the customer has churned (1=Yes, 0=No) | Binary (Target) |

### Data Quality Assessment

- **Dataset Size**: 10,000 customer records
- **Missing Values**: None
- **Class Imbalance**: 20.37% churn rate (2,037 churned vs. 7,963 non-churned)
- **Outliers**: Handled using IQR method for numerical features

## Feature Engineering

### Preprocessing Steps

1. **Data Cleaning**:
   - Removed unnecessary columns (RowNumber, CustomerId, Surname)
   - Handled outliers in numerical features
   - Verified no duplicate records

2. **Feature Transformation**:
   - One-hot encoded categorical variables (Geography, Gender)
   - Converted binary columns to proper format

### Engineered Features

1. **Age-related Features**:
   - `AgeGroup`: Categorized age into groups (0-4)
   - `IsYoung`: Age < 30
   - `IsMiddleAged`: 30 ≤ Age < 50
   - `IsSenior`: Age ≥ 50
   - `IsRetirementAge`: Age ≥ 65

2. **Geography-related Features**:
   - `Geography_Germany`: Customer is from Germany
   - `Geography_Spain`: Customer is from Spain
   - `GermanyXAge`: Interaction between Germany and Age
   - `GermanyXBalance`: Interaction between Germany and Balance
   - `GermanyXSenior`: Interaction between Germany and Senior status

3. **Balance-related Features**:
   - `HasZeroBalance`: Customer has zero balance
   - `BalanceToSalaryRatio`: Ratio of balance to salary
   - `HasHighBalance`: Customer has high balance (> 100,000)

4. **Product-related Features**:
   - `HasMultipleProducts`: Customer has more than one product
   - `HasManyProducts`: Customer has 3 or more products
   - `ProductsXAge`: Interaction between products and age
   - `ProductsXBalance`: Interaction between products and balance
   - `ProductsXTenure`: Interaction between products and tenure

5. **Tenure-related Features**:
   - `IsNewCustomer`: Tenure ≤ 1 year
   - `IsLongTermCustomer`: Tenure ≥ 8 years
   - `TenureSquared`: Squared tenure for non-linear relationships
   - `CustomerValue`: Tenure × Balance / 1000

6. **Engagement-related Features**:
   - `EngagementScore`: Weighted score based on activity, credit card, and products
   - `ActiveXTenure`: Interaction between active status and tenure
   - `ActiveXProducts`: Interaction between active status and products
   - `ActiveXAge`: Interaction between active status and age
   - `InactiveSenior`: Customer is both inactive and a senior

7. **Risk Score Features**:
   - `ChurnRiskScore`: Combined risk score based on key churn factors
   - `DemographicRiskScore`: Risk score based on demographic factors
   - `ProductRiskScore`: Risk score based on product usage and activity

### Feature Selection

Multiple feature selection methods were used:

1. **ANOVA F-value**: Selected features with highest F-scores
2. **Mutual Information**: Selected features with highest mutual information with target
3. **Domain Knowledge**: Included features based on business understanding

The final feature set (`selected_top`) combines the best features from these methods.

## Model Development

### Model Training

Multiple classification algorithms were trained and evaluated:

1. **Logistic Regression** (baseline)
2. **Random Forest**
3. **XGBoost**
4. **Support Vector Machine**
5. **Neural Network**

### Handling Class Imbalance

The class imbalance (20.37% churn rate) was addressed using:

1. **SMOTE** (Synthetic Minority Over-sampling Technique)
2. **Class weights** for applicable algorithms

### Model Evaluation

Models were evaluated using:

1. **Accuracy**: Overall correctness of predictions
2. **Precision**: Proportion of positive identifications that were actually correct
3. **Recall**: Proportion of actual positives that were identified correctly
4. **F1 Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under the ROC curve

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 73.0% | 41.0% | 74.5% | 52.8% | 0.826 |
| Random Forest | 81.3% | 54.5% | 49.1% | 51.7% | 0.818 |
| XGBoost | 78.6% | 48.2% | 70.5% | 57.2% | 0.822 |
| SVM | 76.2% | 44.1% | 65.2% | 52.6% | 0.805 |
| Neural Network | 75.8% | 43.5% | 67.8% | 53.0% | 0.812 |

### Model Selection Rationale

**Logistic Regression** was selected as the best model based on:

1. Highest ROC-AUC score (0.826)
2. Strong recall (74.5%) for identifying at-risk customers
3. Better interpretability compared to more complex models
4. Simpler deployment and maintenance requirements

## Model Interpretability

### Coefficient Analysis

The logistic regression model coefficients reveal the most influential features:

**Top Factors Increasing Churn Risk**:
- Having 3+ products (coefficient: 1.89)
- Being an inactive senior customer (coefficient: 0.83)
- Higher product risk score (coefficient: 0.81)
- Lower product-activity interaction (coefficient: 0.68)

**Top Factors Decreasing Churn Risk**:
- Having multiple products (but not too many) (coefficient: -3.71)
- Lower number of products overall (coefficient: -1.33)
- Being an active member (coefficient: -0.95)

### Customer Segments Analysis

Analysis of churn rates across different customer segments:

**Age Groups**:
- 50-60 years: 56.2% churn rate
- 40-50 years: 34.0% churn rate
- >60 years: 24.8% churn rate
- 30-40 years: 12.1% churn rate
- <30 years: 7.5% churn rate

**Geography**:
- Germany: 32.4% churn rate
- Spain: 16.7% churn rate
- France: 16.2% churn rate

**Activity Status**:
- Inactive members: 26.9% churn rate
- Active members: 14.3% churn rate

**Number of Products**:
- 3-4 products: 82.7% churn rate
- 1-2 products: 15.3% churn rate

### High-Risk Customer Profiles

Analysis of high-risk customers (top 10% by churn probability):

- Average age: 54.2 years (vs. 38.9 overall)
- Inactive membership: 72.3% (vs. 20.5% overall)
- Multiple products (3+): 68.7% (vs. 4.2% overall)
- German customers: 58.2% (vs. 25.1% overall)
- Zero balance: 42.1% (vs. 20.3% overall)

### Business Insights

1. **Key Churn Factors**:
   - Age (particularly customers over 50)
   - Geography (particularly Germany)
   - Activity status (inactive members)
   - Number of products (3+ products)
   - Balance (zero balance customers)

2. **Business Recommendations**:
   - Develop targeted retention programs for customers over 50
   - Investigate issues specific to the German market
   - Implement re-engagement campaigns for inactive customers
   - Review pricing and value proposition for customers with 3+ products
   - Create an early warning system to identify at-risk customers
   - Develop personalized retention offers based on customer segments
   - Conduct regular customer satisfaction surveys to identify pain points
   - Improve customer onboarding process to increase engagement from the start

## Deployment Framework

### API Implementation

The system is deployed as a RESTful API using FastAPI with the following endpoints:

1. **`/`**: Health check endpoint
   - Method: GET
   - Response: API status and version information

2. **`/model-info`**: Model information endpoint
   - Method: GET
   - Response: Model name, feature set, metrics, and features

3. **`/insights`**: Business insights endpoint
   - Method: GET
   - Response: Key churn factors, high-risk segments, and recommendations

4. **`/predict`**: Individual prediction endpoint
   - Method: POST
   - Request: Customer data (JSON)
   - Response: Churn probability, prediction, risk level, risk factors, and recommendations

5. **`/batch-predict`**: Batch prediction endpoint
   - Method: POST
   - Request: Multiple customer records (JSON array)
   - Response: Predictions for all customers

### Model Serialization

The trained model is serialized using Python's pickle module and stored in the `models/` directory along with:

1. **`feature_sets.json`**: Feature sets used for model training
2. **`best_model_info.json`**: Information about the best model

### Deployment Considerations

1. **Scalability**:
   - FastAPI provides high performance with async support
   - Batch prediction endpoint for processing multiple customers

2. **Security**:
   - Input validation using Pydantic models
   - Error handling with appropriate HTTP status codes

3. **Monitoring**:
   - Health check endpoint for system monitoring
   - Structured logging for tracking API usage

4. **Maintenance**:
   - Modular design for easy updates
   - Separate model prediction module for reusability

## Usage Guide

### Running the API

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/financial-customer-churn-prediction.git
   cd financial-customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   ./run_api.sh
   ```

4. Access the API documentation:
   ```
   http://localhost:8000/docs
   ```

### Making Predictions

#### Individual Prediction

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Customer data
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
response = requests.post(url, json=customer_data)
prediction = response.json()

print(f"Churn Probability: {prediction['churn_probability']:.2f}")
print(f"Churn Prediction: {prediction['churn_prediction']}")
print(f"Risk Level: {prediction['risk_level']}")
print("Risk Factors:")
for factor in prediction['risk_factors']:
    print(f"- {factor}")
print("Recommendations:")
for recommendation in prediction['recommendations']:
    print(f"- {recommendation}")
```

#### Batch Prediction

```python
import requests

# API endpoint
url = "http://localhost:8000/batch-predict"

# Customer data
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

# Make prediction
response = requests.post(url, json=customers_data)
predictions = response.json()

for i, prediction in enumerate(predictions):
    print(f"\nCustomer {i+1}:")
    print(f"Churn Probability: {prediction['churn_probability']:.2f}")
    print(f"Churn Prediction: {prediction['churn_prediction']}")
    print(f"Risk Level: {prediction['risk_level']}")
```

## Ethical Considerations

### Fairness and Bias

1. **Demographic Fairness**:
   - The model shows higher churn predictions for certain demographics (e.g., older customers, German customers)
   - These predictions reflect actual patterns in the data but should be monitored for bias

2. **Feature Selection**:
   - Sensitive attributes like gender are used only when they show significant predictive power
   - The impact of these features is documented and transparent

### Privacy and Data Protection

1. **Data Minimization**:
   - Only necessary customer attributes are used for predictions
   - Personal identifiers (CustomerID, Surname) are removed during preprocessing

2. **Transparency**:
   - The model provides explanations for its predictions
   - Risk factors are clearly communicated to users

### Responsible Implementation

1. **Human Oversight**:
   - The system is designed as a decision support tool, not for automated decisions
   - Final retention strategies should involve human judgment

2. **Regular Auditing**:
   - Model performance should be regularly monitored for drift
   - Fairness metrics should be tracked over time

## Future Improvements

1. **Model Enhancements**:
   - Explore deep learning approaches for potentially higher accuracy
   - Implement ensemble methods combining multiple models
   - Develop time-series models to predict when customers are likely to churn

2. **Feature Engineering**:
   - Incorporate transaction data for behavioral patterns
   - Add external data sources (e.g., economic indicators)
   - Develop more sophisticated engagement metrics

3. **Deployment Improvements**:
   - Implement model versioning and A/B testing
   - Add automated retraining pipeline
   - Develop a user interface for non-technical stakeholders

4. **Business Integration**:
   - Connect with CRM systems for automated interventions
   - Develop personalized retention offer optimization
   - Implement feedback loop to measure intervention effectiveness
