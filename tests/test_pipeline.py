import unittest
import pandas as pd
from src.data_processor import DataProcessor
from src.algorithms import CustomTemperaturePredictor

class TestFullPipeline(unittest.TestCase):
    """testing the whole pipeline class, calling every test"""
    def test_prediction_pipeline(self):
        """Test suite for the full data processing and prediction"""
        df = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'month': [1, 2, 3],
            'temperature': [0.1, 0.3, 0.5]
        })
        df.to_csv("tests/test_pipeline.csv", index=False)
        proc = DataProcessor("tests/test_pipeline.csv", "temperature")
        proc.load_data()
        proc.clean_data()
        proc.normalize_temperature()
        X, y = proc.get_features_and_target()
        model = CustomTemperaturePredictor()
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_pipeline_no_month_column(self):
        """tests the whole datas if month column missing"""
        df = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'anomaly': [0.1, 0.2, 0.3]
        })
        df.to_csv("tests/test_nomonth.csv", index=False)
        proc = DataProcessor("tests/test_nomonth.csv", "anomaly")
        proc.load_data()
        proc.clean_data()
        proc.normalize_temperature()
        X, y = proc.get_features_and_target()
        model = CustomTemperaturePredictor()
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))
    
    def test_pipeline_missing_target_column(self):
        """Tests whole pipeline when target column missing"""
        df = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'month': [1, 2, 3]
        })
        df.to_csv("tests/test_missing_target_pipeline.csv", index=False)
        proc = DataProcessor("tests/test_missing_target_pipeline.csv", "temperature")
        proc.load_data()
        X, y = proc.get_features_and_target()
        self.assertEqual(X.size, 0)
        self.assertEqual(y.size, 0)

    def test_pipeline_empty_file(self):
        """tests when data file is empty """
        pd.DataFrame().to_csv("tests/test_empty_pipeline.csv", index=False)
        proc = DataProcessor("tests/test_empty_pipeline.csv", "temperature")
        df = proc.load_data()
        self.assertTrue(df.empty)
