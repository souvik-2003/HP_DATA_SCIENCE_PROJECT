"""
Optimized main Streamlit application with performance enhancements.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from streamlit_app.components.sidebar import create_sidebar
from streamlit_app.components.data_explorer import create_data_explorer
from streamlit_app.components.model_training import create_model_training
from streamlit_app.components.prediction import create_prediction

# Optimized page configuration
st.set_page_config(
    page_title="High-Performance Data Science App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': 'https://github.com',
        'About': "# High-Performance Data Science App\nBuilt for speed and efficiency!"
    }
)

# Custom CSS for better performance and appearance
st.markdown("""
<style>
    /* Optimize for performance */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Reduce motion for better performance */
    * {
        animation-duration: 0.1s !important;
        animation-delay: 0s !important;
        transition-duration: 0.1s !important;
    }
    
    /* Better metrics styling */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    /* Success boxes */
    .stSuccess {
        border-left: 4px solid #28a745;
    }
    
    /* Info boxes */
    .stInfo {
        border-left: 4px solid #17a2b8;
    }
    
    /* Hide streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Optimize dataframe rendering */
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_app_info():
    """Cached app information."""
    return {
        "version": "2.0.0",
        "features": ["High-Performance Processing", "Optimized Algorithms", "Smart Caching", "Parallel Computing"],
        "supported_formats": ["CSV", "Excel", "Large Datasets"],
        "max_performance": "100K+ rows"
    }


def initialize_session_state():
    """Initialize session state with performance optimizations."""
    defaults = {
        'df': None,
        'model': None,
        'model_type': None,
        'train_test_split': None,
        'feature_names': None,
        'processing_time': 0.0,
        'last_action': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_performance_metrics():
    """Display performance metrics in the header."""
    if st.session_state.df is not None:
        dataset_size = len(st.session_state.df)
        
        # Performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset Size", f"{dataset_size:,} rows")
        
        with col2:
            memory_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")
        
        with col3:
            if dataset_size > 10000:
                performance_mode = "High-Performance"
                delta = "🚀 Optimized"
            elif dataset_size > 1000:
                performance_mode = "Standard"
                delta = "⚡ Fast"
            else:
                performance_mode = "Light"
                delta = "💨 Quick"
            
            st.metric("Mode", performance_mode, delta=delta)
        
        with col4:
            if hasattr(st.session_state, 'processing_time') and st.session_state.processing_time > 0:
                throughput = dataset_size / st.session_state.processing_time
                st.metric("Throughput", f"{throughput:.0f} rows/sec")
            else:
                st.metric("Status", "Ready", delta="✅")


def main():
    """Optimized main application function."""
    # Initialize session state
    initialize_session_state()
    
    # App header with enhanced information
    st.title("🚀 High-Performance Data Science Application")
    
    # Get app info
    app_info = get_app_info()
    
    # Feature highlights
    st.markdown(f"""
    **Version {app_info['version']} - Optimized for Maximum Performance**
    
    ✨ **Key Features:**
    {' • '.join(app_info['features'])}
    
    📊 **Supports:** {' • '.join(app_info['supported_formats'])} up to {app_info['max_performance']}
    """)
    
    # Performance metrics dashboard
    if st.session_state.df is not None:
        with st.container():
            st.markdown("### 📊 Performance Dashboard")
            display_performance_metrics()
        
        st.markdown("---")
    
    # Navigation and main content
    with st.container():
        # Create sidebar and get selected page
        selected_page = create_sidebar()
        
        # Main content area with performance tracking
        if selected_page == "Data Explorer":
            with st.container():
                create_data_explorer()
        
        elif selected_page == "Model Training":
            with st.container():
                create_model_training()
        
        elif selected_page == "Prediction":
            with st.container():
                create_prediction()
    
    # Footer with performance info
    st.markdown("---")
    
    with st.expander("🔧 Performance Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Optimizations Active:**
            - ⚡ Parallel Processing
            - 🧠 Smart Caching  
            - 📊 Optimized Data Types
            - 🎯 Batch Processing
            - 💾 Memory Management
            """)
        
        with col2:
            st.markdown("""
            **Performance Tips:**
            - Large datasets auto-optimize
            - Use sampling for exploration
            - Parallel training enabled
            - Memory usage monitored
            - Fast model serialization
            """)
        
        # System info
        st.markdown(f"""
        **System:** {os.cpu_count()} CPU cores detected | 
        **Optimization Level:** {'High-Performance' if st.session_state.df is not None and len(st.session_state.df) > 10000 else 'Standard'}
        """)


if __name__ == "__main__":
    main()
