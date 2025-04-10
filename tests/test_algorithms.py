import unittest

import numpy as np

from src.algorithms import CustomTemperaturePredictor, detect_anomalies, custom_clustering

class TestAlgorithms(unittest.TestCase):
    """Test for our custom climate algorithm"""
    def setUp(self):
        """ a simple dataset for testing"""
        self.X = np.array([[1, 2], [2, 3], [3, 4]])
        self.y = np.array([3, 5, 7])

    def test_predictor_fit_predict(self):
        '''this one tests our custom temperature prediction model'''
        model = CustomTemperaturePredictor(learning_rate=0.01, n_iterations=1000)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        # Check that we got as many predictions as target value
        self.assertEqual(len(predictions), len(self.y))

    def test_custom_clustering(self):
        '''testing our clustering function using 2 clusters'''
        labels = custom_clustering(self.X, n_clusters=2)
        #  label for every sample
        self.assertEqual(len(labels), len(self.X))

    def test_anomaly_detection(self):
        '''creates a simple time series with anomaly'''
        ts = np.array([1, 1, 1, 10, 1, 1, 1])
        anomalies = detect_anomalies(ts, window_size=3, threshold=2.0)
        # at least one anomely detected
        self.assertIn(True, anomalies)

if __name__ == '__main__':
    '''runs all the tests'''
    unittest.main()
