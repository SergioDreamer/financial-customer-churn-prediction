# Customer Churn Prediction System
## Financial Institution Customer Retention Analysis

### Executive Summary

Our Customer Churn Prediction System identifies banking customers at risk of leaving, providing actionable insights to improve retention. Using machine learning and comprehensive data analysis, we've developed a solution that predicts churn with 82.6% accuracy (ROC-AUC) and delivers personalized recommendations for each customer.

---

### Business Challenge

- Customer acquisition costs 5-25x higher than retention
- Need to identify at-risk customers before they leave
- Understand key factors driving customer decisions
- Develop targeted retention strategies

---

### Data Overview

- 10,000 banking customer records
- 14 features including demographics, account details, and engagement metrics
- 20.37% churn rate (class imbalance)
- Key variables: Age, Geography, Balance, Products, Activity status

---

### Key Insights: Who Churns?

#### Age
- Customers 50-60 years old have 56.2% churn rate (7.5x higher than young customers)
- Seniors represent highest risk demographic segment

#### Geography
- German customers have 32.4% churn rate (2x higher than France/Spain)
- Requires region-specific investigation

#### Product Usage
- Customers with 3+ products have 82.7% churn rate (5.4x higher than 1-2 products)
- Suggests pricing or value proposition issues

#### Activity Status
- Inactive members have 26.9% churn rate (1.9x higher than active members)
- Engagement is critical retention factor

---

### Feature Engineering

Created 30+ domain-specific features including:

- **Age-related**: Age groups, senior indicators
- **Geography-related**: Country indicators, interactions
- **Balance-related**: Zero balance, balance-to-salary ratio
- **Product-related**: Multiple products indicators, interactions
- **Engagement-related**: Activity scores, inactive senior flag
- **Risk scores**: Demographic, product, and composite risk metrics

---

### Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 73.0% | 41.0% | 74.5% | 52.8% | 0.826 |
| Random Forest | 81.3% | 54.5% | 49.1% | 51.7% | 0.818 |
| XGBoost | 78.6% | 48.2% | 70.5% | 57.2% | 0.822 |
| SVM | 76.2% | 44.1% | 65.2% | 52.6% | 0.805 |
| Neural Network | 75.8% | 43.5% | 67.8% | 53.0% | 0.812 |

**Selected Model**: Logistic Regression
- Highest ROC-AUC score
- Strong recall for identifying at-risk customers
- Better interpretability
- Simpler deployment requirements

---

### Model Interpretability

**Top Factors Increasing Churn Risk**:
- Having 3+ products (coefficient: 1.89)
- Being an inactive senior customer (coefficient: 0.83)
- Higher product risk score (coefficient: 0.81)
- Lower product-activity interaction (coefficient: 0.68)

**Top Factors Decreasing Churn Risk**:
- Having multiple products (but not too many) (coefficient: -3.71)
- Lower number of products overall (coefficient: -1.33)
- Being an active member (coefficient: -0.95)

---

### High-Risk Customer Profile

Customers most likely to churn:
- 50+ years old
- From Germany
- Have 3+ products
- Inactive members
- Zero balance accounts
- Recent customers (low tenure)

---

### Business Recommendations

1. **Develop targeted retention programs for customers over 50**
   - Senior-specific benefits and loyalty programs
   - Simplified digital banking options

2. **Investigate and address issues in the German market**
   - Region-specific customer research
   - Tailored service improvements

3. **Review pricing structure for customers with 3+ products**
   - Bundle pricing evaluation
   - Value proposition assessment

4. **Implement re-engagement campaigns for inactive customers**
   - Personalized offers based on past behavior
   - Simplified reactivation process

5. **Create early warning system for at-risk customers**
   - Proactive intervention before churn signals
   - Automated monitoring and alerts

---

### Deployment Solution

**API Framework**
- FastAPI implementation with 5 endpoints:
  - Health check
  - Model information
  - Business insights
  - Individual prediction
  - Batch prediction

**Prediction Response**
- Churn probability
- Risk level classification
- Customer-specific risk factors
- Personalized recommendations

---

### Sample API Usage

```python
# Customer data
customer_data = {
    "CreditScore": 650,
    "Age": 55,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 100000.0,
    "Geography": "Germany",
    "Gender": "Female"
}

# Make prediction
response = requests.post("http://localhost:8000/predict", 
                         json=customer_data)
prediction = response.json()

# Results
# Churn Probability: 0.87
# Risk Level: High
# Risk Factors: 
# - Customer is over 50 years old
# - Customer is from Germany
# - Customer has 3 or more products
# - Customer is not an active member
# - Customer has zero balance
```

---

### Ethical Considerations

- **Fairness**: Monitor for demographic bias
- **Transparency**: Explainable predictions
- **Privacy**: Data minimization practices
- **Human oversight**: Decision support, not automation

---

### Future Improvements

1. **Model Enhancements**
   - Deep learning approaches
   - Ensemble methods
   - Time-series prediction

2. **Feature Engineering**
   - Transaction data integration
   - External data sources
   - Advanced engagement metrics

3. **Business Integration**
   - CRM system connection
   - Retention offer optimization
   - Intervention effectiveness tracking

---

### Thank You

**Questions?**

Contact: [Your Contact Information]
