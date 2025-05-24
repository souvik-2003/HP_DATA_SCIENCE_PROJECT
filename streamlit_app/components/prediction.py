"""
Enhanced prediction component with missing value support.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models import load_model, predict_with_missing_support, predict_batch_with_missing_support


@st.cache_resource
def load_cached_model(model_path):
    """Cached model loading for faster repeated access."""
    return load_model(model_path)


def create_prediction():
    """Create enhanced prediction section with missing value support."""
    st.header("🔮 High-Speed Predictions with Missing Value Support")
    
    # Check if model is loaded or trained
    if st.session_state.model is None:
        st.warning("Please train a model first or load a saved model.")
        
        # Option to load a saved model
        st.subheader("Load Saved Model")
        
        # List available models
        models_dir = "models"
        if os.path.exists(models_dir) and os.listdir(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and not f.endswith('_metadata.pkl')]
            if model_files:
                selected_model = st.selectbox("Select a model", model_files)
                
                if st.button("⚡ Load Model"):
                    model_path = os.path.join(models_dir, selected_model)
                    metadata_path = os.path.join(models_dir, selected_model.replace('.pkl', '_metadata.pkl'))
                    
                    try:
                        # Load model
                        model = load_cached_model(model_path)
                        st.session_state.model = model
                        
                        # Load metadata if available
                        if os.path.exists(metadata_path):
                            metadata = load_model(metadata_path)
                            st.session_state.feature_names = metadata.get('feature_names', [])
                            st.session_state.model_type = metadata.get('model_type', 'unknown')
                            st.session_state.training_info = metadata.get('training_info', {})
                            
                            if metadata.get('has_preprocessing', False):
                                st.info("✅ Model includes preprocessing pipeline for missing values")
                        
                        st.success("✅ Model loaded successfully with missing value support!")
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Model type:** {type(model).__name__}")
    with col2:
        if hasattr(st.session_state, 'model_type'):
            st.write(f"**Task:** {st.session_state.model_type.title()}")
    with col3:
        if hasattr(model, 'named_steps'):
            st.success("✅ Has preprocessing")
        else:
            st.info("ℹ️ Direct model")
    
    # Performance indicator
    st.success("⚡ Model loaded with high-performance optimizations and missing value support!")
    
    # Input method selection
    input_method = st.radio("Select input method", ["Manual Input", "File Upload"])
    
    if input_method == "Manual Input":
        # If we have feature names in session state
        if hasattr(st.session_state, 'feature_names') and st.session_state.feature_names:
            features = st.session_state.feature_names
            
            # Create input fields for each feature
            st.subheader("Enter Feature Values")
            st.info("💡 You can leave fields empty for missing values - the model will handle them automatically!")
            
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
                        unique_values = ['[Missing Value]'] + list(st.session_state.df[feature].dropna().unique())
                        selected_value = st.selectbox(f"{feature}", unique_values)
                        input_values[feature] = None if selected_value == '[Missing Value]' else selected_value
                    else:
                        # Create a number input for numerical features
                        use_missing = st.checkbox(f"Missing value for {feature}", key=f"missing_{feature}")
                        if use_missing:
                            input_values[feature] = np.nan
                            st.write(f"**{feature}:** Missing Value")
                        else:
                            input_values[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
            
            # Make prediction
            if st.button("🚀 Lightning Prediction", type="primary"):
                # Convert input to DataFrame
                input_df = pd.DataFrame([input_values])
                
                # Time the prediction
                start_time = time.perf_counter()
                prediction = predict_with_missing_support(model, input_df)
                prediction_time = (time.perf_counter() - start_time) * 1000
                
                # Display prediction
                st.subheader("🎯 Prediction Result")
                if isinstance(prediction[0], (int, float)):
                    st.success(f"**Predicted value:** {prediction[0]:.4f}")
                else:
                    st.success(f"**Predicted class:** {prediction[0]}")
                
                # Show performance and missing value info
                has_missing = input_df.isnull().any().any()
                if has_missing:
                    missing_count = input_df.isnull().sum().sum()
                    st.info(f"✅ Handled {missing_count} missing value(s) automatically")
                
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
            help="Files can contain missing values - they will be handled automatically!"
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
                
                # Display data preview
                st.write("Preview of uploaded data:")
                st.dataframe(input_df.head())
                
                # Check for missing values in uploaded data
                missing_analysis = input_df.isnull().sum()
                missing_cols = missing_analysis[missing_analysis > 0]
                
                if len(missing_cols) > 0:
                    st.warning(f"⚠️ Found missing values in {len(missing_cols)} columns - will be handled automatically")
                    with st.expander("Missing Value Details"):
                        for col, count in missing_cols.items():
                            pct = (count / len(input_df)) * 100
                            st.write(f"• {col}: {count:,} missing ({pct:.1f}%)")
                else:
                    st.success("✅ No missing values found in uploaded data")
                
                # Performance indicator for large datasets
                if len(input_df) > 1000:
                    st.info(f"🚀 Large dataset detected ({len(input_df):,} rows). Batch processing optimizations will be applied!")
                
                # Check if all required features are present
                if hasattr(st.session_state, 'feature_names') and st.session_state.feature_names:
                    missing_features = [f for f in st.session_state.feature_names if f not in input_df.columns]
                    
                    if missing_features:
                        st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                    else:
                        # Make prediction
                        if st.button("🚀 Process Batch Predictions", type="primary"):
                            with st.spinner("Processing predictions with missing value support..."):
                                # Extract features
                                X = input_df[st.session_state.feature_names]
                                
                                # Time the batch prediction
                                start_time = time.time()
                                
                                # Use optimized batch prediction with missing value support
                                if len(X) > 1000:
                                    predictions = predict_batch_with_missing_support(model, X, batch_size=5000)
                                    processing_method = "Optimized batch processing with missing value support"
                                else:
                                    predictions = predict_with_missing_support(model, X)
                                    processing_method = "Fast prediction with missing value support"
                                
                                total_time = time.time() - start_time
                                
                                # Add predictions to DataFrame
                                input_df['Prediction'] = predictions
                                
                                # Calculate missing value statistics
                                total_missing = X.isnull().sum().sum()
                                missing_percentage = (total_missing / (X.shape[0] * X.shape[1])) * 100
                                
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
                                
                                # Missing value processing info
                                if total_missing > 0:
                                    st.success(f"✅ Successfully handled {total_missing:,} missing values ({missing_percentage:.2f}% of data)")
                                
                                # Show processing method
                                st.success(f"✅ {processing_method} completed successfully!")
                                
                                # Option to download results
                                csv = input_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions",
                                    data=csv,
                                    file_name="enhanced_predictions.csv",
                                    mime="text/csv",
                                )
                                
                                # Performance celebration for large datasets
                                if len(input_df) > 5000:
                                    st.balloons()
                                    st.success(f"🎉 Successfully processed {len(input_df):,} samples with missing value support!")
                else:
                    st.warning("Feature information not available. Please train a model first.")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                
                # Provide helpful suggestions for missing value errors
                if "missing" in str(e).lower() or "nan" in str(e).lower():
                    st.markdown("""
                    **💡 Suggestions for missing value errors:**
                    1. Check that your uploaded file has the same column names as the training data
                    2. Ensure the model was trained with missing value support
                    3. Try re-training the model with enhanced missing value handling
                    """)
