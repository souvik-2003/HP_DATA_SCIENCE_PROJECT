"""
Tests for models module.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import module to test
from src.models import (
    train_linear_regression, evaluate_regression_model, 
    train_random_forest, save_model, load_model
)


class TestModels(unittest.TestCase):
    """Test cases for model functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 100)
        
        self.X_train = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        self.y_train = pd.Series(y, name='target')
    
    def test_train_linear_regression(self):
        """Test the train_linear_regression function."""
        # Train model
        model = train_linear_regression(self.X_train, self.y_train)
        
        # Check that model is returned
        self.assertIsNotNone(model)
        
        # Check that model has expected attributes
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
    
    def test_train_random_forest(self):
        """Test the train_random_forest function."""
        # Train regression model
        reg_model = train_random_forest(
            self.X_train, self.y_train, 
            is_classification=False, 
            n_estimators=10, 
            max_depth=5
        )
        
        # Check that model is returned
        self.assertIsNotNone(reg_model)
        
        # Check that model has expected attributes
        self.assertTrue(hasattr(reg_model, 'feature_importances_'))
    
    def test_save_and_load_model(self):
        """Test the save_model and load_model functions."""
        import tempfile
        
        # Train model
        model = train_linear_regression(self.X_train, self.y_train)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            # Save model
            save_model(model, tmp.name)
            
            # Load model
            loaded_model = load_model(tmp.name)
            
            # Check that loaded model is the same as original
            self.assertEqual(model.coef_.tolist(), loaded_model.coef_.tolist())
            self.assertEqual(model.intercept_, loaded_model.intercept_)


if __name__ == "__main__":
    unittest.main()
