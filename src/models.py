"""
Optimized machine learning models with faster runtime.
"""
import numpy as np
import pandas as pd
import joblib  # Changed from pickle - much faster
import os
from typing import Dict, Any, Union, List, Tuple, Optional

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a linear regression model with optimizations."""
    model = LinearRegression()
    
    # Convert to numpy for faster processing
    X_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    model.fit(X_np, y_np)
    return model


def train_random_forest(X_train: pd.DataFrame, 
                        y_train: pd.Series, 
                        is_classification: bool = True, 
                        n_estimators: int = 100,
                        max_depth: Optional[int] = None) -> Union[RandomForestClassifier, RandomForestRegressor]:
    """Train a random forest model with parallel processing."""
    # Add parallel processing for significant speedup
    if is_classification:
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
    
    # Convert to numpy for faster processing
    X_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    model.fit(X_np, y_np)
    return model


def evaluate_regression_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a regression model with optimized prediction."""
    # Convert to numpy for faster prediction
    X_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    y_pred = model.predict(X_np)
    
    mse = mean_squared_error(y_np, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_np, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a classification model with optimized prediction."""
    # Convert to numpy for faster prediction
    X_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    y_pred = model.predict(X_np)
    
    accuracy = accuracy_score(y_np, y_pred)
    
    # For binary classification
    if len(np.unique(y_np)) == 2:
        precision = precision_score(y_np, y_pred)
        recall = recall_score(y_np, y_pred)
        f1 = f1_score(y_np, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    else:
        # For multiclass
        return {
            'accuracy': accuracy
        }


def save_model(model: Any, file_path: str) -> None:
    """Save a trained model using joblib for faster I/O."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Use joblib instead of pickle - much faster and better compression
    joblib.dump(model, file_path, compress=3)


def load_model(file_path: str) -> Any:
    """Load a model using joblib for faster I/O."""
    return joblib.load(file_path)


# Additional optimization functions
def predict_fast(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Fast prediction with optimized data handling."""
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Ensure proper data type for speed
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    
    return model.predict(X)


def predict_batch(model: Any, X: Union[pd.DataFrame, np.ndarray], batch_size: int = 10000) -> np.ndarray:
    """Batch prediction for large datasets."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    
    # For small datasets, predict directly
    if len(X) <= batch_size:
        return model.predict(X)
    
    # For large datasets, process in batches to manage memory
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        batch_pred = model.predict(batch)
        predictions.append(batch_pred)
    
    return np.concatenate(predictions)
