"""
Enhanced model training component with missing value handling options.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_processing import split_data, analyze_missing_data, clean_data
from src.models import train_model_with_missing_support, predict_with_missing_support
from src.visualization import plot_feature_importance, plot_prediction_vs_actual


def create_model_training():
    """Create enhanced model training section with missing value handling."""
    st.header("🚀 High-Performance Model Training with Missing Value Support")
    
    # Check if data is loaded
    if st.session_state.df is None:
        st.warning("Please load data from the sidebar first.")
        return
    
    df = st.session_state.df
    
    # Analyze missing data
    missing_analysis = analyze_missing_data(df)
    
    # Display missing data information
    if missing_analysis['has_missing']:
        st.warning(f"⚠️ Missing data detected: {missing_analysis['total_missing_cells']:,} missing values ({missing_analysis['missing_data_percentage']:.2f}% of total data)")
        
        with st.expander("📊 Missing Data Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Missing Value Counts:**")
                for col, count in missing_analysis['missing_counts'].items():
                    if count > 0:
                        pct = missing_analysis['missing_percentages'][col]
                        st.write(f"• {col}: {count:,} ({pct:.1f}%)")
            
            with col2:
                st.markdown("**Severity Classification:**")
                if missing_analysis['low_missing']:
                    st.success(f"✅ Low missing (≤5%): {len(missing_analysis['low_missing'])} columns")
                if missing_analysis['medium_missing']:
                    st.warning(f"⚠️ Medium missing (5-20%): {len(missing_analysis['medium_missing'])} columns")
                if missing_analysis['high_missing']:
                    st.error(f"🚨 High missing (>20%): {len(missing_analysis['high_missing'])} columns")
                
                st.info(f"💡 Recommended: {missing_analysis['recommended_strategy'].replace('_', ' ').title()}")
    else:
        st.success("✅ No missing values detected!")
    
    # Data preprocessing options
    st.subheader("🧹 Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing value handling strategy
        if missing_analysis['has_missing']:
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ["auto", "smart", "simple", "advanced", "knn", "iterative", "keep_missing"],
                index=0,
                help="Auto: Automatically choose best strategy based on data characteristics"
            )
        else:
            missing_strategy = "none"
            st.info("No missing value handling needed")
    
    with col2:
        # Data cleaning options
        apply_cleaning = st.checkbox("Apply data cleaning", value=True)
        optimize_dtypes = st.checkbox("Optimize data types", value=True)
    
    # Model configuration
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Task type
        task_type = st.radio("Select task type", ["Regression", "Classification"])
        
        # Target column
        target_column = st.selectbox("Select target column", df.columns)
        
        # Check target for missing values
        target_missing = df[target_column].isnull().sum()
        if target_missing > 0:
            st.warning(f"⚠️ Target column has {target_missing} missing values. These will be removed.")
    
    with col2:
        # Model type selection with missing value support info
        if task_type == "Regression":
            model_options = {
                "Linear Regression": "linear_regression",
                "Random Forest": "random_forest_regression",
                "Gradient Boosting (Missing Value Support)": "hist_gradient_boosting_regression"
            }
        else:
            model_options = {
                "Logistic Regression": "logistic_regression", 
                "Random Forest": "random_forest_classification",
                "Gradient Boosting (Missing Value Support)": "hist_gradient_boosting_classification"
            }
        
        model_display_name = st.selectbox("Select model", list(model_options.keys()))
        model_type = model_options[model_display_name]
        
        # Show info about missing value support
        if "gradient_boosting" in model_type:
            st.info("✅ This model natively supports missing values!")
        elif missing_analysis['has_missing'] and missing_strategy == "keep_missing":
            st.warning("⚠️ This model requires missing value preprocessing")
    
    # Feature selection
    feature_columns = st.multiselect(
        "Select feature columns",
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column][:min(10, len(df.columns)-1)]
    )
    
    if not feature_columns:
        st.error("Please select at least one feature column.")
        return
    
    # Hyperparameters
    with st.expander("🔧 Model Hyperparameters"):
        model_params = {}
        
        if "random_forest" in model_type:
            model_params['n_estimators'] = st.slider("Number of estimators", 10, 500, 100, 10)
            model_params['max_depth'] = st.slider("Max depth", 1, 50, 10)
            st.info("🔥 Using parallel processing with all CPU cores!")
        
        elif "gradient_boosting" in model_type:
            model_params['max_iter'] = st.slider("Max iterations", 50, 500, 100, 25)
            model_params['learning_rate'] = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
            model_params['max_depth'] = st.slider("Max depth", 1, 10, 3)
    
    # Train/test split
    with st.expander("📊 Train/Test Split"):
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 1000, 42, 1)
    
    # Train model button
    if st.button("🏃‍♂️ Train Enhanced Model", type="primary"):
        with st.spinner("Training model with enhanced missing value handling..."):
            try:
                # Prepare data with selected preprocessing
                df_processed = df.copy()
                
                if apply_cleaning and missing_analysis['has_missing'] and missing_strategy != "keep_missing":
                    df_processed = clean_data(df_processed, target_column, missing_strategy)
                    st.success(f"✅ Applied {missing_strategy} missing value handling")
                
                # Split data
                splits = split_data(
                    df_processed, 
                    target_column, 
                    test_size=test_size, 
                    random_state=random_state,
                    handle_missing=(missing_strategy == "keep_missing")
                )
                X_train, X_test = splits['X_train'], splits['X_test']
                y_train, y_test = splits['y_train'], splits['y_test']
                
                # Select features
                X_train = X_train[feature_columns]
                X_test = X_test[feature_columns]
                
                # Save split to session state
                st.session_state.train_test_split = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test
                }
                
                # Train model with enhanced missing value support
                missing_handling = "keep_missing" if missing_strategy == "keep_missing" else "auto"
                
                model, training_info = train_model_with_missing_support(
                    X_train, y_train, model_type, model_params, missing_handling
                )
                
                # Save model to session state
                st.session_state.model = model
                st.session_state.model_type = task_type.lower()
                st.session_state.feature_names = feature_columns
                st.session_state.training_info = training_info
                
                # Display training results
                st.success(f"✅ Model trained successfully in {training_info['training_time']:.2f} seconds!")
                
                # Show training information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training Time", f"{training_info['training_time']:.2f}s")
                
                with col2:
                    if training_info.get('has_missing_values', False):
                        st.metric("Missing Data", f"{training_info.get('missing_percentage', 0):.1f}%")
                    else:
                        st.metric("Data Quality", "Clean")
                
                with col3:
                    if training_info.get('model_switched', False):
                        st.metric("Model", "Auto-switched")
                    else:
                        st.metric("Model", "As selected")
                
                # Additional training info
                if training_info.get('used_pipeline', False):
                    st.info(f"🔧 Used preprocessing pipeline with {training_info['imputation_strategy']} imputation")
                
                if training_info.get('model_switched', False):
                    st.info(f"🔄 Auto-switched to gradient boosting for native missing value support")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                
                # Provide helpful suggestions
                if "NaN" in str(e) or "missing" in str(e).lower():
                    st.markdown("""
                    **💡 Suggestions to fix missing value errors:**
                    1. Try selecting a different missing value strategy
                    2. Use 'Gradient Boosting' model for native missing value support
                    3. Enable data cleaning options
                    4. Check your data for unexpected missing value patterns
                    """)
    
    # Model evaluation (enhanced)
    if st.session_state.model is not None:
        st.subheader("📊 Model Performance Evaluation")
        
        # Get model and split data
        model = st.session_state.model
        splits = st.session_state.train_test_split
        training_info = st.session_state.training_info
        
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        # Make predictions with missing value support
        start_time = time.time()
        y_pred = predict_with_missing_support(model, X_test)
        prediction_time = time.time() - start_time
        
        # Show prediction performance
        samples_per_second = len(X_test) / prediction_time if prediction_time > 0 else float('inf')
        st.info(f"⚡ Predicted {len(X_test):,} samples in {prediction_time:.3f}s (~{samples_per_second:.0f} samples/sec)")
        
        # Model evaluation metrics
        if st.session_state.model_type == "regression":
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("MAE", f"{mae:.4f}")
            with col4:
                st.metric("MSE", f"{mse:.4f}")
            
            # Plot predictions vs actual
            st.subheader("📈 Predictions vs Actual")
            fig = plot_prediction_vs_actual(y_test.values, y_pred)
            st.pyplot(fig)
        
        else:  # Classification
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Prediction Speed", f"{samples_per_second:.0f} samples/sec")
            
            # Classification report
            with st.expander("📋 Detailed Classification Report"):
                report = classification_report(y_test, y_pred)
                st.text(report)
            
            # Confusion matrix
            st.subheader("🔍 Confusion Matrix")
            from sklearn.metrics import ConfusionMatrixDisplay
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_') or (hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', model), 'feature_importances_')):
            st.subheader("🎯 Feature Importance")
            
            # Extract the actual model from pipeline if needed
            actual_model = model.named_steps.get('model', model) if hasattr(model, 'named_steps') else model
            
            if hasattr(actual_model, 'feature_importances_'):
                fig = plot_feature_importance(actual_model, st.session_state.feature_names)
                st.pyplot(fig)
        
        # Save model option
        st.subheader("💾 Save Model")
        
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("Model filename", "enhanced_model")
        with col2:
            include_preprocessing = st.checkbox("Include preprocessing pipeline", value=True)
        
        if st.button("💾 Save Enhanced Model"):
            os.makedirs("models", exist_ok=True)
            
            model_path = f"models/{model_name}.pkl"
            
            # Save model with metadata
            from src.models import save_model
            save_model(model, model_path)
            
            # Save additional metadata
            metadata = {
                'feature_names': st.session_state.feature_names,
                'model_type': st.session_state.model_type,
                'training_info': training_info,
                'missing_strategy': missing_strategy,
                'has_preprocessing': hasattr(model, 'named_steps')
            }
            
            metadata_path = f"models/{model_name}_metadata.pkl"
            save_model(metadata, metadata_path)
            
            st.success(f"✅ Model saved to {model_path} with enhanced missing value support!")
            st.info(f"📋 Metadata saved to {metadata_path}")
