import unittest
import os

import pandas as pd
import numpy as np

from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """ Test suite for the DataProcessor class for datat handling"""
    def setUp(self):
        """ Make sure our tests folder exists for saving temp CSV files"""
        os.makedirs("tests", exist_ok=True)

    def test_load_data_with_value_column(self):
        """ Create a simple dataframe with year and value """
        df = pd.DataFrame({'Year': [2000, 2001], 'Value': [1.1, 2.2]})
        df.to_csv("tests/value_data.csv", index=False)
        # DataProcessor should  add a precipitation column based  on target
        proc = DataProcessor("tests/value_data.csv", "precipitation")
        df_loaded = proc.load_data()
        self.assertIn("preciptiation", df_loaded.columns)

    def test_missing_target_column(self):
        """makes DataFrame without the target column temperature """
        df = pd.DataFrame({'year': [2000, 2001], 'month': [1, 2]})
        df.to_csv("tests/missing_target.csv", index=False)
        proc = DataProcessor("tests/missing_target.csv", "temperature")
        proc.load_data()
        proc.clean_data()
        df_norm = proc.normalize_temperature()
        #  'temperature' is not in the file
        self.assertNotIn("temperature", df_norm.columns)

    def test_get_features_and_target_with_month(self):
        """ dateframe with year, month(if there) and temp columns"""
        df = pd.DataFrame({
            'year': [2000, 2001],
            'month': [1, 2],
            'temperature': [10.0, 12.0]
        })
        df.to_csv("tests/with_month.csv", index=False)
        proc = DataProcessor("tests/with_month.csv", "temperature")
        proc.load_data()
        proc.clean_data()
        proc.normalize_temperature()
        X, y = proc.get_features_and_target()
        # We expect 2 features (year/month) and 2 target values
        self.assertEqual(X.shape[1], 2)
        self.assertEqual(len(y), 2)

    def test_get_features_missing_month(self):
        """ DataFrame with only year and anomaly columns"""
        df = pd.DataFrame({
            'year': [2000, 2001],
            'anomaly': [0.1, 0.2]
        })
        df.to_csv("tests/no_month.csv", index=False)
        proc = DataProcessor("tests/no_month.csv", "anomaly")
        proc.load_data()
        proc.clean_data()
        proc.normalize_temperature()
        X, y = proc.get_features_and_target()
        # Since the month column is missing we expect only 1 feature :year
        self.assertEqual(X.shape[1], 1)
        self.assertEqual(len(y), 2)

    def test_missing_year_column(self):
        """test with DataFrame missing the year column"""
        df = pd.DataFrame({'month': [1, 2], 'temperature': [10, 12]})
        df.to_csv("tests/missing_year.csv", index=False)
        proc = DataProcessor("tests/missing_year.csv", "temperature")
        proc.load_data()
        X, y = proc.get_features_and_target()
        # Without the yearcolumn there should be no features 
        self.assertEqual(X.size, 0)
        self.assertEqual(y.size, 0)

    def test_empty_file(self):
        """Create an empty CSV file"""
        pd.DataFrame().to_csv("tests/empty.csv", index=False)
        proc = DataProcessor("tests/empty.csv", "temperature")
        df = proc.load_data()
        # The loaded DataFrame should be empty
        self.assertTrue(df.empty)

    def test_load_invalid_file(self):
        """ Loads a file that doesn't exist and expect an empty DataFrame"""
        proc = DataProcessor("tests/fakefile.csv", "temperature")
        df = proc.load_data()
        self.assertTrue(df.empty)

if __name__ == '__main__':
    """Runs all the tests to see if they pass"""
    unittest.main()
