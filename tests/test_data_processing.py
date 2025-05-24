"""
Tests for data processing module.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import module to test
from src.data_processing import clean_data, split_data


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        self.test_data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': ['a', 'b', 'c', 'd', None],
            'target': [10, 20, 30, 40, 50]
        })
    
    def test_clean_data(self):
        """Test the clean_data function."""
        # Clean data
        cleaned = clean_data(self.test_data)
        
        # Check that there are no missing values
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
        
        # Check that the shape is preserved (no rows dropped)
        self.assertEqual(cleaned.shape, self.test_data.shape)
    
    def test_split_data(self):
        """Test the split_data function."""
        # Split data
        splits = split_data(self.test_data, 'target', test_size=0.2, random_state=42)
        
        # Check that all splits are returned
        self.assertIn('X_train', splits)
        self.assertIn('X_test', splits)
        self.assertIn('y_train', splits)
        self.assertIn('y_test', splits)
        
        # Check that split sizes are correct
        self.assertEqual(len(splits['X_train']) + len(splits['X_test']), len(self.test_data))
        self.assertEqual(len(splits['y_train']) + len(splits['y_test']), len(self.test_data))


if __name__ == "__main__":
    unittest.main()
