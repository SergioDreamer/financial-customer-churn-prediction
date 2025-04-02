"""
Utility functions for the customer churn prediction project.

This module contains various utility functions used across the project.
"""
import os
import logging
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union
import datetime

def setup_logging(log_file: str = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, logs to console only)
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_json(data: Dict[str, Any], output_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(input_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        Dictionary containing loaded data
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data

def create_timestamp() -> str:
    """
    Create a timestamp string.
    
    Returns:
        Timestamp string in the format YYYYMMDD_HHMMSS
    """
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def create_output_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create an output directory for experiment results.
    
    Args:
        base_dir: Base directory
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created directory
    """
    timestamp = create_timestamp()
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_feature_importance(feature_names: List[str], importance_values: np.ndarray, 
                           output_path: str = None, title: str = "Feature Importance") -> None:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        output_path: Path to save the plot
        title: Plot title
    """
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_values = importance_values[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_names)), sorted_values, align='center')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Importance')
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for features.
    
    Args:
        df: DataFrame containing features
        
    Returns:
        DataFrame with VIF values for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return vif_data.sort_values("VIF", ascending=False)
