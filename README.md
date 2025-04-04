# Financial Customer Churn Prediction

## Project Overview
This project implements a machine learning system to predict customer churn for financial institutions. It demonstrates a comprehensive approach to developing a churn prediction model, from data preparation to deployment, with a focus on interpretability and business impact.

## Business Context
Customer churn is a critical concern for financial institutions. Identifying customers at risk of leaving allows for proactive retention strategies, ultimately improving customer lifetime value and reducing acquisition costs. This project provides a framework for:
- Identifying customers at high risk of churning
- Understanding key factors that contribute to churn
- Implementing targeted intervention strategies based on data-driven insights

## Technical Details

### Data
The project uses [dataset description will be updated once data is sourced] to train and evaluate churn prediction models. The data includes customer demographics, transaction patterns, product usage, and engagement metrics.

### Methodology
1. **Data Preparation**: Cleaning, preprocessing, and feature engineering
2. **Exploratory Analysis**: Understanding patterns and relationships in the data
3. **Model Development**: Implementation of multiple classification algorithms
4. **Evaluation**: Comprehensive assessment using various metrics
5. **Interpretability**: SHAP/LIME analysis for feature importance
6. **Deployment**: API framework for model serving

### Technologies Used
- Python for data processing and modeling
- Scikit-learn, XGBoost, and other ML libraries
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualization
- Flask/FastAPI for API development

## Getting Started
This section provides instructions on how to set up and run the customer churn prediction project on your local machine.

### Prerequisites

- Python 3.12 or higher (code executed using Python 3.12.8)
- Jupyter Notebook or JupyterLab
- Git

### Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SergioDreamer/financial-customer-churn-prediction.git
   cd financial-customer-churn-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages (under your venv if created):
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset and place it in the `data/raw` directory. Ensure the data is in CSV format.
    - Original dataset that was used to train the model is included under `data/raw` directory.
    - If you want to use your own dataset, please ensure that it has the same file name and the format as the original dataset.
    
5. To run the project workflow, open the Jupyter notebooks in the `notebooks` directory:
    - I recommend to "Clear All Outputs" and "Run All Cells" in the Jupyter Notebook to ensure that all cells are executed in order and that the outputs are generated correctly.

## Project Structure
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
├── docs/              # Documentation and visualizations 
│   └── plots/         # Visualization outputs
└── tests/             # Test modules
```

> [!NOTE]
> Some folders may be empty or not created when you clone the repository. This is expected as they will be populated during the project workflow.


## Ethical Considerations
This project addresses fairness, bias, and privacy concerns in predictive modeling for financial services. Specific measures include:
- Fairness assessment across demographic groups
- Transparency in model explanations
- Privacy-preserving techniques in data handling

## License
Please refer to the [LICENSE](LICENSE) file for license details.
