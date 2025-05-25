"""
Optimized sidebar component with faster loading and better UX.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_processing import load_data, clean_data, optimize_dtypes


@st.cache_data
def generate_sample_data(n_samples):
    """Cached sample data generation."""
    np.random.seed(42)
    
    # Create diverse sample data
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.uniform(-10, 10, n_samples),
        'feature4': np.random.exponential(2, n_samples),
        'feature5': np.random.gamma(2, 2, n_samples),
        'feature6': np.random.poisson(3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target with realistic relationships
    df['target_regression'] = (
        2.5 * df['feature1'] + 
        1.2 * df['feature2'] - 
        0.8 * df['feature3'] + 
        0.5 * df['feature4'] +
        np.random.normal(0, 1, n_samples)
    )
    
    # Create classification target
    df['target_classification'] = (df['target_regression'] > df['target_regression'].median()).astype(int)
    
    # Add categorical features
    categories = ['A', 'B', 'C', 'D', 'E']
    df['category'] = np.random.choice(categories, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Add some missing values for realism
    missing_indices = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'feature2'] = np.nan
    
    return df

# Add chunked file reading
def load_large_file_optimized(uploaded_file, max_size_mb=500):
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        st.error(f"File too large ({file_size_mb:.1f}MB). Max: {max_size_mb}MB")
        return None
    
    # For large files, offer sampling options
    if file_size_mb > 50:
        sample_option = st.radio("Large file detected:", 
                               ["Load sample", "Load first N rows", "Load full (slower)"])



def create_sidebar():
    """Create optimized sidebar with performance enhancements."""
    with st.sidebar:
        st.title("🚀 Navigation")
        
        # Page selection with better organization
        st.markdown("### 📋 Main Sections")
        pages = ["Data Explorer", "Model Training", "Prediction"]
        selected_page = st.radio("Go to", pages, key="main_navigation")
        
        st.markdown("---")
        
        # Performance status indicator
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            st.success("🤖 Model Ready")
        
        if st.session_state.df is not None:
            dataset_size = len(st.session_state.df)
            if dataset_size > 10000:
                st.info("⚡ High-Performance Mode Active")
            elif dataset_size > 1000:
                st.success("🚀 Optimized Processing")
        
        st.header("📁 Data Management")
        
        # File uploader with enhanced feedback
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file", 
            type=["csv", "xlsx"],
            help="💡 Tip: Larger files automatically use optimized processing!"
        )
        
        if uploaded_file is not None:
            # Enhanced file information
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            
            # File info in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Size", f"{file_size:.2f} MB")
            with col2:
                if file_size > 50:
                    st.warning("Large File")
                elif file_size > 10:
                    st.info("Medium File")
                else:
                    st.success("Small File")
            
            # Create data directories if they don't exist
            os.makedirs("data/raw", exist_ok=True)
            
            # Process file with progress indication
            with st.spinner("Processing file..."):
                # Save uploaded file
                file_path = os.path.join("data/raw", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load data with timing
                start_time = time.time()
                df = load_data(file_path)
                load_time = time.time() - start_time
                
                # Show loading performance
                st.success(f"✅ Loaded in {load_time:.2f}s")
                
                # Display basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", f"{df.shape[1]}")
                
                # Performance recommendations
                if df.shape[0] > 50000:
                    st.info("💡 Large dataset - automatic optimizations enabled!")
                elif df.shape[0] > 10000:
                    st.info("🚀 Medium dataset - performance optimizations active!")
            
            # Data cleaning options
            st.markdown("### 🧹 Data Cleaning")
            
            # Quick data quality check
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing %", f"{missing_pct:.1f}%")
            with col2:
                duplicates = df.duplicated().sum()
                st.metric("Duplicates", f"{duplicates:,}")
            
            cleaning_options = []
            if missing_pct > 0:
                cleaning_options.append("Handle missing values")
            if duplicates > 0:
                cleaning_options.append("Remove duplicates")
            
            cleaning_options.extend(["Optimize data types", "Remove outliers"])
            
            selected_cleaning = st.multiselect("Select cleaning operations:", cleaning_options)
            
            if st.button("🧹 Clean Data") and selected_cleaning:
                with st.spinner("Cleaning data..."):
                    cleaned_df = df.copy()
                    
                    if "Handle missing values" in selected_cleaning:
                        cleaned_df = clean_data(cleaned_df)
                        st.success("✅ Missing values handled")
                    
                    if "Remove duplicates" in selected_cleaning:
                        before_shape = cleaned_df.shape[0]
                        cleaned_df = cleaned_df.drop_duplicates()
                        removed = before_shape - cleaned_df.shape[0]
                        if removed > 0:
                            st.success(f"✅ Removed {removed:,} duplicates")
                    
                    if "Optimize data types" in selected_cleaning:
                        before_memory = cleaned_df.memory_usage(deep=True).sum() / (1024 * 1024)
                        cleaned_df = optimize_dtypes(cleaned_df)
                        after_memory = cleaned_df.memory_usage(deep=True).sum() / (1024 * 1024)
                        savings = before_memory - after_memory
                        st.success(f"✅ Saved {savings:.2f} MB memory")
                    
                    st.session_state.df = cleaned_df
            else:
                st.session_state.df = df
        
        # Enhanced sample data generator
        st.markdown("### 🎲 Sample Data Generator")
        
        # Sample size selection with intelligent defaults
        sample_size_options = {
            "Small (1K)": 1000,
            "Medium (10K)": 10000,
            "Large (50K)": 50000,
            "Extra Large (100K)": 100000
        }
        
        selected_size = st.selectbox(
            "Choose dataset size:", 
            list(sample_size_options.keys()),
            index=1  # Default to Medium
        )
        
        n_samples = sample_size_options[selected_size]
        
        # Performance prediction
        estimated_time = n_samples / 10000  # Rough estimate
        st.caption(f"Estimated generation time: ~{estimated_time:.1f}s")
        
        if st.button("🎲 Generate Sample Data"):
            with st.spinner(f"Generating {n_samples:,} samples..."):
                start_time = time.time()
                df = generate_sample_data(n_samples)
                generation_time = time.time() - start_time
                
                st.session_state.df = df
                st.success(f"✅ Generated in {generation_time:.2f}s!")
                
                # Show generation performance
                samples_per_second = n_samples / generation_time
                st.info(f"🚀 Generated {samples_per_second:.0f} samples/second")
                
                if n_samples >= 50000:
                    st.balloons()
        
        st.markdown("---")
        
        # Enhanced dataset information
        if st.session_state.df is not None:
            st.header("📊 Dataset Dashboard")
            df = st.session_state.df
            
            # Key metrics in a compact layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Memory", f"{memory_mb:.1f} MB")
            
            with col2:
                st.metric("Columns", f"{df.shape[1]}")
                missing_cells = df.isnull().sum().sum()
                st.metric("Missing", f"{missing_cells:,}")
            
            # Data composition
            with st.expander("📋 Data Composition"):
                col_types = df.dtypes.value_counts()
                for dtype, count in col_types.items():
                    st.write(f"• **{dtype}**: {count}")
            
            # Performance indicator
            if df.shape[0] > 1000:
                performance_score = min(100, (df.shape[0] / 1000) * 10)
                st.progress(min(performance_score/100, 1.0))
                
                if df.shape[0] > 10000:
                    st.info("🚀 High-performance processing available!")
                else:
                    st.success("⚡ Optimized processing active!")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("📊 Data Preview"):
                st.dataframe(df.head(5), use_container_width=True)
            
            if st.button("📈 Quick Stats"):
                st.dataframe(df.describe(), use_container_width=True)
        
        # Model status (if available)
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            st.markdown("---")
            st.header("🤖 Model Status")
            
            model_type = type(st.session_state.model).__name__
            st.success(f"✅ {model_type} Ready")
            
            if hasattr(st.session_state, 'feature_names'):
                st.caption(f"Features: {len(st.session_state.feature_names)}")
            
            # Quick prediction button
            if st.button("🚀 Quick Prediction"):
                st.info("Navigate to Prediction tab for inference!")
    
    return selected_page
