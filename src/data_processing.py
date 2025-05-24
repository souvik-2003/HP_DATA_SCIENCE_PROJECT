"""
Enhanced data processing utilities with advanced missing value handling.
"""
import os
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive missing data analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with missing data statistics
    """
    missing_stats = {}
    
    # Basic missing value counts
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # Only show columns with missing values
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        missing_stats['has_missing'] = False
        missing_stats['message'] = "No missing values found!"
        return missing_stats
    
    missing_stats['has_missing'] = True
    missing_stats['missing_counts'] = missing_counts.to_dict()
    missing_stats['missing_percentages'] = missing_percentages.to_dict()
    missing_stats['total_missing_cells'] = missing_counts.sum()
    missing_stats['missing_data_percentage'] = (missing_counts.sum() / (len(df) * len(df.columns))) * 100
    
    # Categorize columns by missing data severity
    missing_stats['low_missing'] = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 5)].index.tolist()
    missing_stats['medium_missing'] = missing_percentages[(missing_percentages > 5) & (missing_percentages <= 20)].index.tolist()
    missing_stats['high_missing'] = missing_percentages[missing_percentages > 20].index.tolist()
    
    # Recommend strategy
    if missing_stats['missing_data_percentage'] < 5:
        missing_stats['recommended_strategy'] = "simple_imputation"
    elif missing_stats['missing_data_percentage'] < 15:
        missing_stats['recommended_strategy'] = "advanced_imputation"
    else:
        missing_stats['recommended_strategy'] = "careful_analysis_required"
    
    return missing_stats


def smart_imputation_strategy(df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Apply intelligent imputation strategy based on data characteristics.
    
    Args:
        df: Input DataFrame
        target_column: Target column name (will be excluded from imputation)
        
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    # Exclude target column from imputation
    columns_to_impute = [col for col in df.columns if col != target_column]
    
    # Separate numeric and categorical columns
    numeric_cols = df_imputed[columns_to_impute].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_imputed[columns_to_impute].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle numeric columns
    if numeric_cols:
        for col in numeric_cols:
            missing_pct = (df_imputed[col].isnull().sum() / len(df_imputed)) * 100
            
            if missing_pct > 0:
                if missing_pct <= 5:
                    # Low missing: use median (robust to outliers)
                    imputer = SimpleImputer(strategy='median')
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                elif missing_pct <= 15:
                    # Medium missing: use KNN imputation for better accuracy
                    if len(numeric_cols) > 1:  # Need multiple features for KNN
                        try:
                            knn_imputer = KNNImputer(n_neighbors=min(5, len(df_imputed) // 10))
                            df_imputed[numeric_cols] = knn_imputer.fit_transform(df_imputed[numeric_cols])
                            break  # KNN handles all numeric columns at once
                        except:
                            # Fallback to median
                            imputer = SimpleImputer(strategy='median')
                            df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                    else:
                        # Fallback to median for single column
                        imputer = SimpleImputer(strategy='median')
                        df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                else:
                    # High missing: use median as safest option
                    imputer = SimpleImputer(strategy='median')
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
    
    # Handle categorical columns
    if categorical_cols:
        for col in categorical_cols:
            missing_pct = (df_imputed[col].isnull().sum() / len(df_imputed)) * 100
            
            if missing_pct > 0:
                if missing_pct <= 10:
                    # Use most frequent value
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
                else:
                    # Create a "Missing" category for high missing percentages
                    df_imputed[col] = df_imputed[col].fillna('Missing_Value')
    
    return df_imputed


def advanced_imputation(df: pd.DataFrame, 
                       target_column: Optional[str] = None,
                       method: str = 'iterative') -> pd.DataFrame:
    """
    Apply advanced imputation methods for better accuracy.
    
    Args:
        df: Input DataFrame
        target_column: Target column name (excluded from imputation)
        method: 'iterative', 'knn', or 'hybrid'
        
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    # Exclude target column
    columns_to_impute = [col for col in df.columns if col != target_column]
    numeric_cols = df_imputed[columns_to_impute].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_imputed[columns_to_impute].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle categorical columns first (simple imputation)
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = cat_imputer.fit_transform(df_imputed[categorical_cols])
    
    # Handle numeric columns with advanced methods
    if numeric_cols and len(numeric_cols) > 1:
        try:
            if method == 'iterative':
                # Iterative imputation (similar to MICE)
                iterative_imputer = IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    n_nearest_features=min(len(numeric_cols), 5)
                )
                df_imputed[numeric_cols] = iterative_imputer.fit_transform(df_imputed[numeric_cols])
                
            elif method == 'knn':
                # KNN imputation
                knn_imputer = KNNImputer(
                    n_neighbors=min(5, len(df_imputed) // 10),
                    weights='distance'
                )
                df_imputed[numeric_cols] = knn_imputer.fit_transform(df_imputed[numeric_cols])
                
            elif method == 'hybrid':
                # Use KNN for low missing, iterative for high missing
                low_missing_cols = []
                high_missing_cols = []
                
                for col in numeric_cols:
                    missing_pct = (df_imputed[col].isnull().sum() / len(df_imputed)) * 100
                    if missing_pct <= 10:
                        low_missing_cols.append(col)
                    else:
                        high_missing_cols.append(col)
                
                if low_missing_cols:
                    knn_imputer = KNNImputer(n_neighbors=min(5, len(df_imputed) // 10))
                    df_imputed[low_missing_cols] = knn_imputer.fit_transform(df_imputed[low_missing_cols])
                
                if high_missing_cols:
                    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
                    df_imputed[high_missing_cols] = iterative_imputer.fit_transform(df_imputed[high_missing_cols])
        except Exception as e:
            print(f"Advanced imputation failed, falling back to simple imputation: {e}")
            # Fallback to simple imputation
            simple_imputer = SimpleImputer(strategy='median')
            df_imputed[numeric_cols] = simple_imputer.fit_transform(df_imputed[numeric_cols])
    
    elif numeric_cols:  # Single numeric column
        # Fallback to simple imputation
        simple_imputer = SimpleImputer(strategy='median')
        df_imputed[numeric_cols] = simple_imputer.fit_transform(df_imputed[numeric_cols])
    
    return df_imputed


def load_data(file_path: str) -> pd.DataFrame:
    """Load data with optimized settings and missing value detection."""
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        # Optimized CSV reading with missing value handling
        df = pd.read_csv(
            file_path,
            low_memory=False,
            engine='c',
            na_values=['', ' ', 'null', 'NULL', 'None', 'NaN', 'nan', '#N/A', 'N/A', 'na', 'NA']
        )
    elif ext.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return df


def clean_data(df: pd.DataFrame, target_column: Optional[str] = None, 
               imputation_method: str = 'smart') -> pd.DataFrame:
    """
    Enhanced data cleaning with intelligent missing value handling.
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        imputation_method: 'smart', 'simple', 'advanced', 'knn', 'iterative'
        
    Returns:
        Cleaned DataFrame
    """
    # Analyze missing data first
    missing_analysis = analyze_missing_data(df)
    
    if not missing_analysis['has_missing']:
        return df.copy()  # No missing values, return as is
    
    # Apply imputation based on method
    try:
        if imputation_method == 'smart':
            df_cleaned = smart_imputation_strategy(df, target_column)
        elif imputation_method == 'simple':
            df_cleaned = df.copy()
            # Simple median/mode imputation
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='median')
                df_cleaned[numeric_cols] = numeric_imputer.fit_transform(df_cleaned[numeric_cols])
            
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_cleaned[categorical_cols] = cat_imputer.fit_transform(df_cleaned[categorical_cols])
        else:
            df_cleaned = advanced_imputation(df, target_column, imputation_method)
    except Exception as e:
        print(f"Imputation failed: {e}. Using simple median/mode imputation.")
        # Fallback to simple imputation
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df_cleaned[numeric_cols] = numeric_imputer.fit_transform(df_cleaned[numeric_cols])
        
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[categorical_cols] = cat_imputer.fit_transform(df_cleaned[categorical_cols])
    
    # Drop rows with all NaN values (if any remain)
    df_cleaned = df_cleaned.dropna(how='all')
    
    return df_cleaned


def split_data(df: pd.DataFrame, 
               target_column: str, 
               test_size: float = 0.2, 
               random_state: int = 42,
               handle_missing: bool = True) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Split data with automatic missing value handling.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of the data to include in the test split
        random_state: Random seed for reproducibility
        handle_missing: Whether to handle missing values automatically
        
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    # Handle missing values if requested
    if handle_missing:
        missing_analysis = analyze_missing_data(df)
        if missing_analysis['has_missing']:
            df = smart_imputation_strategy(df, target_column)
    
    # Check if target has missing values
    if df[target_column].isnull().any():
        print("Warning: Target column has missing values. Removing rows with missing targets.")
        df = df.dropna(subset=[target_column])
    
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Optimize data types for memory efficiency
    float_cols = X.select_dtypes(include=['float64']).columns
    X[float_cols] = X[float_cols].astype('float32')
    
    # Split the data
    try:
        # Try stratified split for classification
        if y.dtype == 'object' or y.nunique() < 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    except:
        # Fallback to regular split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


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
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
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
