"""
Model evaluation module for customer churn prediction.

This module contains functions for evaluating model performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for the positive class
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Calculating evaluation metrics")
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
    """
    logger.info("Plotting confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for the positive class
        output_path: Path to save the plot
    """
    logger.info("Plotting ROC curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str = None) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for the positive class
        output_path: Path to save the plot
    """
    logger.info("Plotting precision-recall curve")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_models(models_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        models_metrics: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with model comparison
    """
    logger.info("Comparing models")
    return pd.DataFrame.from_dict(models_metrics, orient='index')

def calculate_business_impact(y_true: np.ndarray, y_pred: np.ndarray, 
                             cost_false_negative: float, cost_false_positive: float) -> Dict[str, float]:
    """
    Calculate business impact metrics based on cost-benefit analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_false_negative: Cost of a false negative (missed churn)
        cost_false_positive: Cost of a false positive (unnecessary intervention)
        
    Returns:
        Dictionary of business impact metrics
    """
    logger.info("Calculating business impact metrics")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
    cost_per_customer = total_cost / len(y_true)
    
    return {
        'total_cost': total_cost,
        'cost_per_customer': cost_per_customer,
        'false_negatives': fn,
        'false_positives': fp
    }
