"""
Optimized data processing utilities with faster runtime.
"""
import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional


def load_data(file_path: str) -> pd.DataFrame:
    """Load data with optimized settings for better performance."""
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        # Optimized CSV reading
        return pd.read_csv(
            file_path,
            low_memory=False,  # Better type inference
            engine='c'  # Faster C engine
        )
    elif ext.lower() in ['.xls', '.xlsx']:
        # Optimized Excel reading
        return pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data with vectorized operations for speed."""
    # Work on a copy to avoid modifying original
    df_cleaned = df.copy()
    
    # Drop rows with all NaN values - vectorized operation
    df_cleaned = df_cleaned.dropna(how='all')
    
    # Vectorized operations for missing value handling
    # Numeric columns - fill with median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Use vectorized fillna with computed medians
        medians = df_cleaned[numeric_cols].median()
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(medians)
    
    # Categorical columns - fill with mode or 'Unknown'
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_values = df_cleaned[col].mode()
        if not mode_values.empty:
            df_cleaned[col] = df_cleaned[col].fillna(mode_values.iloc[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")
    
    return df_cleaned


def split_data(df: pd.DataFrame, 
               target_column: str, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Split data with optimized memory usage."""
    from sklearn.model_selection import train_test_split
    
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Optimize data types for memory efficiency
    # Convert float64 to float32 where possible
    float_cols = X.select_dtypes(include=['float64']).columns
    X[float_cols] = X[float_cols].astype('float32')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def preprocess_features(df: pd.DataFrame, 
                         categorical_cols: Optional[List[str]] = None, 
                         numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Preprocess features with optimized transformations."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Automatic column detection if not specified
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Optimized preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        n_jobs=-1  # Parallel processing
    )
    
    # Apply preprocessing
    preprocessed_array = preprocessor.fit_transform(df)
    
    # Convert back to DataFrame
    preprocessed_df = pd.DataFrame(
        preprocessed_array,
        index=df.index
    )
    
    return preprocessed_df


# Additional optimization functions
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize data types for memory efficiency and speed."""
    df_optimized = df.copy()
    
    # Optimize integer columns
    int_cols = df_optimized.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                df_optimized[col] = df_optimized[col].astype('uint8')
            elif col_max < 65535:
                df_optimized[col] = df_optimized[col].astype('uint16')
            elif col_max < 4294967295:
                df_optimized[col] = df_optimized[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df_optimized[col] = df_optimized[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df_optimized[col] = df_optimized[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df_optimized[col] = df_optimized[col].astype('int32')
    
    # Optimize float columns
    float_cols = df_optimized.select_dtypes(include=['float64']).columns
    df_optimized[float_cols] = df_optimized[float_cols].astype('float32')
    
    return df_optimized
