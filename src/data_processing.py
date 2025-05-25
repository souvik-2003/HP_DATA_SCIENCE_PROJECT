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


def load_data_chunked(file_path, chunksize=50000, max_rows=None):
    chunks = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        chunk = optimize_dtypes(chunk)  # Optimize immediately
        chunks.append(chunk)
        progress_bar.progress(min(i * chunksize / max_rows, 1.0))
    
    return pd.concat(chunks, ignore_index=True)

def load_data_with_size_check(file_path: str, max_size_mb: int = 1000) -> pd.DataFrame:
    """Load data with updated size limits."""
    
    # Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # Updated size check from 200MB to 1000MB
    if file_size_mb > max_size_mb:
        raise ValueError(f"File too large ({file_size_mb:.1f}MB). Max: {max_size_mb}MB")
    
    # Use different loading strategies based on file size
    if file_size_mb < 100:
        # Small files - standard loading
        return pd.read_csv(file_path, low_memory=False)
    elif file_size_mb < 500:
        # Medium files - optimized loading
        return load_medium_file_optimized(file_path)
    else:
        # Large files (500MB-1GB) - chunked loading
        return load_large_file_chunked(file_path)


def load_large_file_chunked(file_path: str, chunksize: int = 50000) -> pd.DataFrame:
    """Load very large files (500MB-1GB) in chunks."""
    chunks = []
    total_rows = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Estimate total rows for progress
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
        
        chunk_reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
        
        for i, chunk in enumerate(chunk_reader):
            # Optimize chunk immediately
            chunk = optimize_dtypes_enhanced(chunk)
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Update progress
            progress = min(total_rows / total_lines, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing chunk {i+1}... Rows loaded: {total_rows:,}")
            
            # Memory management for very large files
            if i % 20 == 0:  # Every 20 chunks
                import gc
                gc.collect()
        
        # Combine chunks
        status_text.text("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True, copy=False)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully loaded {len(df):,} rows from {file_size_mb:.1f}MB file")
        
        return df
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        raise e


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


def optimize_dtypes_aggressive(df):
    original_memory = df.memory_usage(deep=True).sum()
    
    # Downcast integers
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Downcast floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert to categorical for low cardinality strings
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    savings = (1 - new_memory/original_memory) * 100
    st.info(f"💾 Memory reduced by {savings:.1f}%")
    
    return df



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
