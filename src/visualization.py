"""
Optimized visualization utilities with faster rendering.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
import warnings

# Optimize matplotlib for better performance
plt.style.use('fast')
matplotlib_backend = 'Agg'  # Non-interactive backend for speed


def plot_distribution(data: pd.Series, title: str = '', figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot distribution with optimized rendering."""
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Optimize for data type
        if data.dtype.kind in 'ifc':  # Numeric data
            # Use optimized histogram with automatic binning
            if len(data) > 10000:
                # Sample large datasets for faster plotting
                sample_data = data.sample(n=5000, random_state=42)
                sns.histplot(sample_data, kde=True, ax=ax, bins=50)
                ax.set_title(f'{title} (Sampled from {len(data):,} points)' if title else f'Distribution of {data.name} (Sampled)')
            else:
                sns.histplot(data, kde=True, ax=ax, bins='auto')
                ax.set_title(title if title else f'Distribution of {data.name}')
        else:  # Categorical data
            # Limit categories for better performance
            value_counts = data.value_counts()
            if len(value_counts) > 20:
                top_values = value_counts.head(20)
                sns.barplot(y=top_values.index, x=top_values.values, ax=ax)
                ax.set_title(f'{title} (Top 20)' if title else f'Distribution of {data.name} (Top 20)')
            else:
                sns.countplot(y=data, ax=ax, order=value_counts.index)
                ax.set_title(title if title else f'Distribution of {data.name}')
        
        ax.set_ylabel('Frequency' if data.dtype.kind in 'ifc' else data.name)
        ax.set_xlabel(data.name if data.dtype.kind in 'ifc' else 'Count')
        
        plt.tight_layout()
        return fig


def plot_correlation_matrix(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """Plot correlation matrix with performance optimizations."""
    # Select only numeric columns and limit for performance
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Limit columns for better visualization and performance
    if len(numeric_data.columns) > 20:
        # Use correlation with target or select top variance columns
        variance_cols = numeric_data.var().nlargest(20).index
        numeric_data = numeric_data[variance_cols]
    
    # Optimize correlation computation
    if len(numeric_data) > 10000:
        # Sample for large datasets
        sampled_data = numeric_data.sample(n=5000, random_state=42)
        corr = sampled_data.corr()
        title_suffix = f" (Sampled from {len(numeric_data):,} rows)"
    else:
        corr = numeric_data.corr()
        title_suffix = ""
    
    # Create optimized heatmap
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Use optimized colormap and reduce annotation precision
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only lower triangle
    
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        fmt='.2f',  # Reduced precision for speed
        cmap='RdYlBu_r', 
        vmin=-1, 
        vmax=1, 
        ax=ax,
        square=True,
        cbar_kws={"shrink": .8}
    )
    
    ax.set_title(f'Correlation Matrix{title_suffix}')
    plt.tight_layout()
    
    return fig


def plot_feature_importance(model: Any, feature_names: List[str], 
                          figsize: Tuple[int, int] = (12, 8), top_n: int = 20) -> plt.Figure:
    """Plot feature importance with optimization for many features."""
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create importance DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Limit to top_n features for better visualization
    if len(importance_df) > top_n:
        importance_df = importance_df.tail(top_n)
    
    # Create optimized plot
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Use horizontal bar plot for better label readability
    bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                   color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    ax.set_title(f'Feature Importance (Top {len(importance_df)})')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                             figsize: Tuple[int, int] = (8, 8),
                             sample_size: int = 5000) -> plt.Figure:
    """Plot predicted vs actual with sampling for large datasets."""
    # Sample for large datasets to improve performance
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
        title_suffix = f" (Sampled {sample_size:,} from {len(y_true):,} points)"
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        title_suffix = ""
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create scatter plot with transparency for overlapping points
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.6, s=20, color='steelblue')
    
    # Plot perfect prediction line
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='Perfect Prediction')
    
    # Add R² score if possible
    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except:
        pass
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Actual vs Predicted Values{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_learning_curve(train_scores: List[float], val_scores: List[float], 
                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot learning curve with optimized styling."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    epochs = range(1, len(train_scores) + 1)
    
    ax.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
    ax.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_subplots_grid(data: pd.DataFrame, columns: List[str], 
                        plot_type: str = 'hist', figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Create optimized subplot grid for multiple visualizations."""
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='white')
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        if plot_type == 'hist':
            if data[col].dtype.kind in 'ifc':
                # Sample large datasets
                plot_data = data[col].dropna()
                if len(plot_data) > 5000:
                    plot_data = plot_data.sample(n=5000, random_state=42)
                
                ax.hist(plot_data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{col}', fontsize=10)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
        
        elif plot_type == 'box':
            if data[col].dtype.kind in 'ifc':
                plot_data = data[col].dropna()
                if len(plot_data) > 5000:
                    plot_data = plot_data.sample(n=5000, random_state=42)
                
                ax.boxplot(plot_data)
                ax.set_title(f'{col}', fontsize=10)
                ax.set_ylabel(col)
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


# Utility function for faster plotting
def optimize_plot_performance():
    """Configure matplotlib for better performance."""
    plt.rcParams.update({
        'figure.max_open_warning': 50,
        'agg.path.chunksize': 10000,
        'path.simplify': True,
        'path.simplify_threshold': 0.1,
    })


# Initialize optimizations
optimize_plot_performance()
