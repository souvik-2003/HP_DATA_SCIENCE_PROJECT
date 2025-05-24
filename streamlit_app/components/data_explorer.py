"""
Optimized Data Explorer component with faster rendering and caching.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.visualization import plot_distribution, plot_correlation_matrix, create_subplots_grid


@st.cache_data
def get_data_summary(df):
    """Cached data summary computation."""
    return df.describe()


@st.cache_data
def get_missing_values_info(df):
    """Cached missing values computation."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing = missing[missing > 0]
        missing_pct = (missing / len(df)) * 100
        return pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        })
    return None


@st.cache_data
def get_data_types_info(df):
    """Cached data types computation."""
    return df.dtypes.value_counts()


@st.cache_data
def sample_large_dataframe(df, sample_size=10000):
    """Sample large dataframes for faster processing."""
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42), True
    return df, False


def create_data_explorer():
    """Create optimized data explorer section."""
    st.header("📊 High-Speed Data Explorer")
    
    # Check if data is loaded
    if st.session_state.df is None:
        st.warning("Please load data from the sidebar first.")
        return
    
    # Get data
    df = st.session_state.df
    
    # Performance indicator for large datasets
    if len(df) > 10000:
        st.info(f"🚀 Large dataset detected ({len(df):,} rows). Using optimized rendering and sampling for better performance!")
    
    # Show data options
    st.subheader("📋 Data Overview")
    
    # Quick stats in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory", f"{memory_mb:.2f} MB")
    with col4:
        missing_cells = df.isnull().sum().sum()
        st.metric("Missing Cells", f"{missing_cells:,}")
    
    # Select rows to display with intelligent default
    default_rows = min(20, len(df))
    num_rows = st.slider("Number of rows to display", 5, min(100, len(df)), default_rows)
    
    # Display data with optimized rendering
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    # Summary statistics with caching
    with st.expander("📊 Summary Statistics"):
        with st.spinner("Computing summary statistics..."):
            summary_stats = get_data_summary(df)
            st.dataframe(summary_stats, use_container_width=True)
    
    # Missing values analysis
    with st.expander("❓ Missing Values Analysis"):
        missing_df = get_missing_values_info(df)
        if missing_df is not None:
            st.dataframe(missing_df, use_container_width=True)
            
            # Plot missing values if not too many columns
            if len(missing_df) <= 20:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_df.sort_values('Missing Values', ascending=False).plot(
                    kind='bar', ax=ax, ylabel='Count / Percentage'
                )
                ax.set_title('Missing Values by Column')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Too many columns with missing values to visualize effectively.")
        else:
            st.success("✅ No missing values found!")
    
    # Data types information
    with st.expander("🔢 Data Types"):
        col_types = get_data_types_info(df)
        st.write("Column types distribution:")
        for dtype, count in col_types.items():
            st.write(f"• **{dtype}**: {count} columns")
    
    # Data visualization section
    st.subheader("📈 High-Performance Visualizations")
    
    # Get sample for large datasets
    df_viz, is_sampled = sample_large_dataframe(df, sample_size=5000)
    
    if is_sampled:
        st.info(f"📊 Using {len(df_viz):,} sampled rows for faster visualization")
    
    # Variable distribution
    st.markdown("### 📈 Variable Distribution")
    
    # Select column for distribution with better organization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col_type_filter = st.radio("Filter by type:", ["All", "Numeric", "Categorical"])
    
    if col_type_filter == "Numeric":
        available_cols = numeric_cols
    elif col_type_filter == "Categorical":
        available_cols = categorical_cols
    else:
        available_cols = df.columns.tolist()
    
    if available_cols:
        column = st.selectbox("Select column for distribution plot", available_cols)
        
        # Plot distribution with optimization
        with st.spinner("Creating distribution plot..."):
            fig = plot_distribution(df_viz[column], f'Distribution of {column}')
            st.pyplot(fig)
            plt.close(fig)  # Free memory
    else:
        st.warning("No columns available for the selected type.")
    
    # Multiple distributions grid
    if len(numeric_cols) > 1:
        st.markdown("### 📊 Multiple Distributions Grid")
        
        # Select columns for grid plot
        max_cols = min(9, len(numeric_cols))  # Limit for performance
        selected_cols = st.multiselect(
            "Select columns for grid visualization", 
            numeric_cols,
            default=numeric_cols[:min(6, len(numeric_cols))],
            max_selections=max_cols
        )
        
        if selected_cols and st.button("🚀 Generate Grid Plot"):
            with st.spinner("Creating grid visualization..."):
                fig = create_subplots_grid(df_viz, selected_cols, plot_type='hist')
                st.pyplot(fig)
                plt.close(fig)  # Free memory
    
    # Correlation matrix
    if len(numeric_cols) > 1:
        st.markdown("### 🔗 Correlation Analysis")
        
        # Intelligent column selection for correlation
        if len(numeric_cols) > 15:
            st.info("📊 Too many numeric columns. Showing top 15 by variance for better performance.")
            # Select top variance columns
            variance_cols = df[numeric_cols].var().nlargest(15).index.tolist()
            corr_cols = st.multiselect(
                "Adjust correlation columns", 
                numeric_cols,
                default=variance_cols
            )
        else:
            corr_cols = st.multiselect(
                "Select columns for correlation matrix", 
                numeric_cols,
                default=numeric_cols
            )
        
        if len(corr_cols) >= 2:
            if st.button("🚀 Generate Correlation Matrix"):
                with st.spinner("Computing correlations..."):
                    fig = plot_correlation_matrix(df_viz[corr_cols])
                    st.pyplot(fig)
                    plt.close(fig)  # Free memory
        else:
            st.warning("Select at least 2 columns for correlation analysis.")
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    
    # Data quality insights
    st.markdown("### 🔍 Data Quality Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numeric Columns:**")
        if numeric_cols:
            for col in numeric_cols[:10]:  # Limit display
                outlier_count = len(df[col][np.abs(df[col] - df[col].mean()) > 3 * df[col].std()])
                st.write(f"• {col}: {outlier_count} potential outliers")
            if len(numeric_cols) > 10:
                st.write(f"... and {len(numeric_cols) - 10} more")
        else:
            st.write("No numeric columns found")
    
    with col2:
        st.markdown("**Categorical Columns:**")
        if categorical_cols:
            for col in categorical_cols[:10]:  # Limit display
                unique_count = df[col].nunique()
                st.write(f"• {col}: {unique_count} unique values")
            if len(categorical_cols) > 10:
                st.write(f"... and {len(categorical_cols) - 10} more")
        else:
            st.write("No categorical columns found")
    
    # Export options
    st.markdown("### 📤 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Sample Data"):
            sample_df = df.head(1000)  # Download first 1000 rows
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download Sample CSV",
                data=csv,
                file_name="data_sample.csv",
                mime="text/csv",
            )
    
    with col2:
        if st.button("📊 Download Summary Stats"):
            summary = get_data_summary(df)
            csv = summary.to_csv()
            st.download_button(
                label="Download Summary CSV", 
                data=csv,
                file_name="summary_statistics.csv",
                mime="text/csv",
            )
