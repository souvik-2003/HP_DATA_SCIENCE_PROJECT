"""
Enhanced machine learning models with missing value handling and performance optimization.
"""
import numpy as np
import pandas as pd
import joblib  # Changed from pickle for better performance
import os
import time
from typing import Dict, Any, Union, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def create_robust_pipeline(model_type: str, model_params: Dict = None, imputation_strategy: str = 'median'):
    """
    Create a robust ML pipeline that handles missing values automatically.
    
    Args:
        model_type: Type of model ('linear_regression', 'random_forest_regression', etc.)
        model_params: Model hyperparameters
        imputation_strategy: 'median', 'mean', 'most_frequent', 'knn'
        
    Returns:
        sklearn Pipeline
    """
    if model_params is None:
        model_params = {}
    
    # Create preprocessing steps
    if imputation_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=imputation_strategy)
    
    # Create model with parallel processing
    if model_type == "linear_regression":
        model = LinearRegression(**model_params)
    elif model_type == "logistic_regression":
        model_params['n_jobs'] = model_params.get('n_jobs', -1)
        model = LogisticRegression(random_state=42, max_iter=1000, **model_params)
    elif model_type == "random_forest_regression":
        model_params['n_jobs'] = model_params.get('n_jobs', -1)
        model = RandomForestRegressor(random_state=42, **model_params)
    elif model_type == "random_forest_classification":
        model_params['n_jobs'] = model_params.get('n_jobs', -1)
        model = RandomForestClassifier(random_state=42, **model_params)
    elif model_type == "hist_gradient_boosting_regression":
        # Native missing value support
        model = HistGradientBoostingRegressor(random_state=42, **model_params)
        return model  # No imputation needed
    elif model_type == "hist_gradient_boosting_classification":
        # Native missing value support
        model = HistGradientBoostingClassifier(random_state=42, **model_params)
        return model  # No imputation needed
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    return pipeline


def train_model_with_missing_support(X_train: pd.DataFrame, 
                                    y_train: pd.Series,
                                    model_type: str,
                                    model_params: Dict = None,
                                    handle_missing: str = 'auto') -> Tuple[Any, Dict]:
    """
    Train model with automatic missing value handling.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model to train
        model_params: Model hyperparameters
        handle_missing: 'auto', 'impute', 'native', or 'none'
        
    Returns:
        Tuple of (trained_model, training_info)
    """
    if model_params is None:
        model_params = {}
    
    training_info = {}
    
    # Check for missing values
    has_missing = X_train.isnull().any().any()
    training_info['has_missing_values'] = has_missing
    
    if has_missing:
        missing_percentage = (X_train.isnull().sum().sum() / (X_train.shape[0] * X_train.shape[1])) * 100
        training_info['missing_percentage'] = missing_percentage
        
        # Decide on strategy
        if handle_missing == 'auto':
            # Use native support for gradient boosting, imputation for others
            if 'gradient_boosting' in model_type or 'hist' in model_type:
                handle_missing = 'native'
            else:
                handle_missing = 'impute'
        
        if handle_missing == 'native':
            # Use models that support missing values natively
            original_model_type = model_type
            if model_type in ["linear_regression", "logistic_regression", "random_forest_regression", "random_forest_classification"]:
                # Switch to gradient boosting equivalent
                if "regression" in model_type:
                    model_type = "hist_gradient_boosting_regression"
                    training_info['model_switched'] = True
                    training_info['original_model'] = original_model_type
                else:
                    model_type = "hist_gradient_boosting_classification"
                    training_info['model_switched'] = True
                    training_info['original_model'] = original_model_type
    
    print(f"Training {model_type} with {'missing value support' if has_missing else 'clean data'}...")
    start_time = time.time()
    
    # Train model
    if handle_missing == 'native' or not has_missing:
        # Direct training (no pipeline needed)
        if model_type == "linear_regression":
            model = LinearRegression(**model_params)
        elif model_type == "logistic_regression":
            model_params['n_jobs'] = model_params.get('n_jobs', -1)
            model = LogisticRegression(random_state=42, max_iter=1000, **model_params)
        elif model_type == "random_forest_regression":
            model_params['n_jobs'] = model_params.get('n_jobs', -1)
            model = RandomForestRegressor(random_state=42, **model_params)
        elif model_type == "random_forest_classification":
            model_params['n_jobs'] = model_params.get('n_jobs', -1)
            model = RandomForestClassifier(random_state=42, **model_params)
        elif model_type == "hist_gradient_boosting_regression":
            model = HistGradientBoostingRegressor(random_state=42, **model_params)
        elif model_type == "hist_gradient_boosting_classification":
            model = HistGradientBoostingClassifier(random_state=42, **model_params)
        
        # Convert to numpy for faster processing
        X_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_np = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        model.fit(X_np, y_np)
        
    else:
        # Use pipeline with imputation
        if missing_percentage < 10:
            imputation_strategy = 'median' if X_train.select_dtypes(include=[np.number]).shape[1] > 0 else 'most_frequent'
        else:
            imputation_strategy = 'knn'
        
        model = create_robust_pipeline(model_type, model_params, imputation_strategy)
        model.fit(X_train, y_train)
        training_info['used_pipeline'] = True
        training_info['imputation_strategy'] = imputation_strategy
    
    training_time = time.time() - start_time
    training_info['training_time'] = training_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_info

# Enable parallel processing for all compatible models
def train_optimized_model(model_type, X_train, y_train, **params):
    if model_type == "random_forest":
        params['n_jobs'] = -1  # Use all CPU cores
        model = RandomForestRegressor(**params)
    elif model_type == "logistic_regression":
        params['n_jobs'] = -1
        model = LogisticRegression(**params)
    
    # Use optimized fit
    model.fit(X_train.values, y_train.values)  # Convert to numpy for speed
    return model


def predict_with_missing_support(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Make predictions with automatic missing value handling.
    
    Args:
        model: Trained model (may be a pipeline)
        X: Input features
        
    Returns:
        Predictions
    """
    # Check if it's a pipeline or direct model
    if hasattr(model, 'named_steps'):
        # It's a pipeline, use it directly
        return model.predict(X)
    else:
        # Direct model, check for missing values
        if isinstance(X, pd.DataFrame):
            has_missing = X.isnull().any().any()
        else:
            has_missing = np.isnan(X).any()
        
        if has_missing:
            # Check if model supports missing values natively
            if hasattr(model, '__class__') and 'HistGradientBoosting' in model.__class__.__name__:
                # Native support, use directly
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return model.predict(X)
            else:
                # Handle missing values with simple imputation
                if isinstance(X, pd.DataFrame):
                    X_imputed = X.copy()
                    numeric_cols = X_imputed.select_dtypes(include=[np.number]).columns
                    categorical_cols = X_imputed.select_dtypes(include=['object', 'category']).columns
                    
                    if len(numeric_cols) > 0:
                        imputer = SimpleImputer(strategy='median')
                        X_imputed[numeric_cols] = imputer.fit_transform(X_imputed[numeric_cols])
                    
                    if len(categorical_cols) > 0:
                        imputer = SimpleImputer(strategy='most_frequent')
                        X_imputed[categorical_cols] = imputer.fit_transform(X_imputed[categorical_cols])
                    
                    X = X_imputed.values
                else:
                    # Numpy array
                    imputer = SimpleImputer(strategy='median')
                    X = imputer.fit_transform(X)
        
        elif isinstance(X, pd.DataFrame):
            X = X.values
        
        return model.predict(X)


def predict_batch_with_missing_support(model: Any, 
                                      X: Union[pd.DataFrame, np.ndarray], 
                                      batch_size: int = 5000) -> np.ndarray:
    """
    Batch prediction with missing value support.
    
    Args:
        model: Trained model
        X: Input data
        batch_size: Size of each processing batch
        
    Returns:
        Batch prediction results
    """
    if len(X) <= batch_size:
        return predict_with_missing_support(model, X)
    
    # Process in batches
    predictions = []
    for i in range(0, len(X), batch_size):
        if isinstance(X, pd.DataFrame):
            batch = X.iloc[i:i + batch_size]
        else:
            batch = X[i:i + batch_size]
        
        batch_pred = predict_with_missing_support(model, batch)
        predictions.append(batch_pred)
    
    return np.concatenate(predictions)


def evaluate_regression_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a regression model with optimized prediction."""
    # Use missing value support for predictions
    y_pred = predict_with_missing_support(model, X_test)
    y_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
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
    # Use missing value support for predictions
    y_pred = predict_with_missing_support(model, X_test)
    y_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    accuracy = accuracy_score(y_np, y_pred)
    
    # For binary classification
    if len(np.unique(y_np)) == 2:
        precision = precision_score(y_np, y_pred, average='binary')
        recall = recall_score(y_np, y_pred, average='binary')
        f1 = f1_score(y_np, y_pred, average='binary')
        
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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path, compress=3)


def load_model(file_path: str) -> Any:
    """Load a model using joblib for faster I/O."""
    return joblib.load(file_path)


# Legacy functions for backwards compatibility
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


# Fast prediction functions
def predict_fast(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Fast prediction with optimized data handling."""
    return predict_with_missing_support(model, X)


def predict_batch(model: Any, X: Union[pd.DataFrame, np.ndarray], batch_size: int = 10000) -> np.ndarray:
    """Batch prediction for large datasets."""
    return predict_batch_with_missing_support(model, X, batch_size)
