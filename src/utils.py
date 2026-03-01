"""
Utility functions for Credit Risk Assessment System
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Data saved to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from {filepath}")
    return data


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Setup logging configuration"""
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """Calculate class weights for imbalanced dataset"""
    
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weights = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format float as percentage string"""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float) -> str:
    """Format float as currency string"""
    return f"${value:,.2f}"


def create_directory_structure(base_path: str = "."):
    """Create all necessary directories for the project"""
    
    base = Path(base_path)
    
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "api",
        "frontend",
        "models",
        "outputs/visualizations",
        "outputs/reports",
        "tests",
        "config"
    ]
    
    for directory in directories:
        (base / directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """Print comprehensive information about a DataFrame"""
    
    print(f"\n=== {name} Info ===")
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nNumeric Summary:\n{df.describe()}")


def ensure_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable types"""
    
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    else:
        return obj


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types"""
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test percentage formatting
    print(f"Percentage: {format_percentage(0.2345)}")
    
    # Test currency formatting
    print(f"Currency: {format_currency(15000.50)}")
    
    # Test serialization
    data = {
        'int': np.int64(42),
        'float': np.float64(3.14),
        'array': np.array([1, 2, 3])
    }
    serialized = ensure_serializable(data)
    print(f"Serialized: {json.dumps(serialized)}")
    
    print("Utilities test complete!")
