"""
Model training and prediction module for customer churn prediction.

This module contains functions for training, evaluating, and using machine learning models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import pickle
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> LogisticRegression:
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        Trained logistic regression model
    """
    logger.info("Training logistic regression model")
    model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestClassifier:
    """
    Train a random forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Trained random forest model
    """
    logger.info("Training random forest model")
    model = RandomForestClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> xgb.XGBClassifier:
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for XGBClassifier
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model")
    model = xgb.XGBClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> lgb.LGBMClassifier:
    """
    Train a LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for LGBMClassifier
        
    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBM model")
    model = lgb.LGBMClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> SVC:
    """
    Train a Support Vector Machine model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for SVC
        
    Returns:
        Trained SVM model
    """
    logger.info("Training SVM model")
    model = SVC(random_state=42, probability=True, **kwargs)
    model.fit(X_train, y_train)
    return model

def handle_class_imbalance(X_train: pd.DataFrame, y_train: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance in the training data.
    
    Args:
        X_train: Training features
        y_train: Training target
        method: Method to handle class imbalance ('smote', 'adasyn', 'random_over', 'random_under')
        
    Returns:
        Rebalanced X_train and y_train
    """
    logger.info(f"Handling class imbalance using {method}")
    
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=42)
    elif method == 'adasyn':
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(random_state=42)
    elif method == 'random_over':
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=42)
    elif method == 'random_under':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=42)
    else:
        logger.warning(f"Unknown method {method}, returning original data")
        return X_train, y_train
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)

def save_model(model: Any, model_path: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    logger.info(f"Saving model to {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        X: Features to predict on
        
    Returns:
        Predicted classes
    """
    logger.info("Making predictions")
    return model.predict(X)

def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Get prediction probabilities using a trained model.
    
    Args:
        model: Trained model
        X: Features to predict on
        
    Returns:
        Prediction probabilities
    """
    logger.info("Getting prediction probabilities")
    return model.predict_proba(X)
