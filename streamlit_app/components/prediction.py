"""
Optimized prediction component with faster runtime.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.models import load_model, predict_fast, predict_batch


@st.cache_resource
def load_cached_model(model_path):
    """Cached model loading for faster repeated access."""
    return load_model(model_path)


def create_prediction():
    """Create optimized prediction section."""
    st.header("🔮 High-Speed Predictions")
    
    # Check if model is loaded or trained
    if st.session_state.model is None:
        st.warning("Please train a model first or load a saved model.")
        
        # Option to load a saved model
        st.subheader("Load Saved Model")
        
        # List available models
        models_dir = "models"
        if os.path.exists(models_dir) and os.listdir(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if model_files:
                selected_model = st.selectbox("Select a model", model_files)
                
                if st.button("⚡ Load Model"):
                    model_path = os.path.join(models_dir, selected_model)
                    try:
                        # Use cached loading for better performance
                        model = load_cached_model(model_path)
                        st.session_state.model = model
                        st.success("✅ Model loaded successfully with optimizations!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
            else:
                st.info("No saved models found.")
        else:
            st.info("No saved models found.")
        
        return
    
    # Get model
    model = st.session_state.model
    
    # Display model info
    st.subheader("Model Information")
    st.write(f"Model type: {type(model).__name__}")
    
    # Performance indicator
    st.success("⚡ Model loaded with high-performance optimizations!")
    
    # Input method selection
    input_method = st.radio("Select input method", ["Manual Input", "File Upload"])
    
    if input_method == "Manual Input":
        # If we have feature names in session state
        if 'feature_names' in st.session_state:
            features = st.session_state.feature_names
            
            # Create input fields for each feature
            st.subheader("Enter Feature Values")
            
            input_values = {}
            # Organize inputs in columns for better layout
            cols = st.columns(min(3, len(features)))
            
            for i, feature in enumerate(features):
                with cols[i % len(cols)]:
                    # Check if feature is categorical
                    if (st.session_state.df is not None and 
                        feature in st.session_state.df.columns and 
                        st.session_state.df[feature].dtype == 'object'):
                        # Create a dropdown for categorical features
                        unique_values = st.session_state.df[feature].unique().tolist()
                        input_values[feature] = st.selectbox(f"{feature}", unique_values)
                    else:
                        # Create a number input for numerical features
                        input_values[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
            
            # Make prediction
            if st.button("🚀 Lightning Prediction", type="primary"):
                # Convert input to numpy array for speed
                input_array = np.array([[input_values[col] for col in features]], dtype=np.float32)
                
                # Time the prediction
                start_time = time.perf_counter()
                prediction = predict_fast(model, input_array)
                prediction_time = (time.perf_counter() - start_time) * 1000
                
                # Display prediction
                st.subheader("🎯 Prediction Result")
                if isinstance(prediction[0], (int, float)):
                    st.success(f"**Predicted value:** {prediction[0]:.4f}")
                else:
                    st.success(f"**Predicted class:** {prediction[0]}")
                
                # Show performance
                st.info(f"⚡ Ultra-fast prediction: {prediction_time:.2f} ms")
                
                if prediction_time < 1:
                    st.balloons()
        else:
            st.warning("Feature information not available. Please train a model first.")
    
    else:  # File Upload
        st.subheader("📁 Batch Prediction Upload")
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file for batch prediction", 
            type=["csv", "xlsx"],
            help="For large files, optimized batch processing will be used automatically!"
        )
        
        if uploaded_file is not None:
            try:
                # Show file size
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
                st.info(f"📊 File size: {file_size:.2f} MB")
                
                # Load data with optimization
                if uploaded_file.name.endswith(".csv"):
                    input_df = pd.read_csv(uploaded_file, low_memory=False, engine='c')
                else:
                    input_df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                # Display data
                st.write("Preview of uploaded data:")
                st.dataframe(input_df.head())
                
                # Performance indicator for large datasets
                if len(input_df) > 1000:
                    st.info(f"🚀 Large dataset detected ({len(input_df):,} rows). Batch processing optimizations will be applied!")
                
                # Check if all required features are present
                if 'feature_names' in st.session_state:
                    missing_features = [f for f in st.session_state.feature_names if f not in input_df.columns]
                    
                    if missing_features:
                        st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                    else:
                        # Make prediction
                        if st.button("🚀 Process Batch Predictions", type="primary"):
                            with st.spinner("Processing predictions with optimizations..."):
                                # Extract features
                                X = input_df[st.session_state.feature_names]
                                
                                # Time the batch prediction
                                start_time = time.time()
                                
                                # Use optimized batch prediction
                                if len(X) > 1000:
                                    predictions = predict_batch(model, X, batch_size=5000)
                                    processing_method = "Optimized batch processing"
                                else:
                                    predictions = predict_fast(model, X)
                                    processing_method = "Fast prediction"
                                
                                total_time = time.time() - start_time
                                
                                # Add predictions to DataFrame
                                input_df['Prediction'] = predictions
                                
                                # Display results
                                st.subheader("📊 Batch Prediction Results")
                                st.dataframe(input_df)
                                
                                # Performance metrics
                                avg_time_per_sample = (total_time * 1000) / len(input_df)
                                throughput = len(input_df) / total_time
                                
                                # Performance dashboard
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Samples", f"{len(input_df):,}")
                                with col2:
                                    st.metric("Total Time", f"{total_time:.2f}s")
                                with col3:
                                    st.metric("Avg Time/Sample", f"{avg_time_per_sample:.2f}ms")
                                with col4:
                                    st.metric("Throughput", f"{throughput:.0f} samples/s")
                                
                                # Show processing method
                                st.success(f"✅ {processing_method} used for optimal performance!")
                                
                                # Option to download results
                                csv = input_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions",
                                    data=csv,
                                    file_name="optimized_predictions.csv",
                                    mime="text/csv",
                                )
                                
                                # Performance celebration for large datasets
                                if len(input_df) > 5000:
                                    st.balloons()
                                    st.success(f"🎉 Successfully processed {len(input_df):,} samples with high-performance optimizations!")
                else:
                    st.warning("Feature information not available. Please train a model first.")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
