import unittest

from src.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    """Test suite for visualizer class functions """
    def test_plot_trend_runs(self):
        """this checks if our plot_trend function runs ;no crashing"""
        years = [2000, 2001, 2002]
        actual = [0.5, 0.7, 0.9]
        predicted = [0.4, 0.8, 1.0]
        Visualizer.plot_trend(years, actual, predicted, show=False)

    def test_plot_anomalies_runs(self):
        """ checks to make sure our anomaly plot function can handle inputs"""
        ts = [0.2, 0.3, 2.5, 0.2]
        anomalies = [False, False, True, False]
        Visualizer.plot_anomalies(ts, anomalies, show=False)

    def test_plot_clusters_runs(self):
        """checks if plotting clusters works with simple data"""
        data = [(1, 2), (2, 3), (4, 4)]
        labels = [0, 1, 1]
        Visualizer.plot_clusters(data, labels, show=False)

    def test_plot_trend_handles_empty(self):
        """  test plot and should print a warning and not crash"""
        Visualizer.plot_trend([], [], [], show= False)

    def test_plot_clusters_empty(self):
        """When data is empty the function should raise value error"""
        with self.assertRaises(ValueError):
            Visualizer.plot_clusters([], [], show= False)

    def test_plot_anomalies_empty(self):
        """Same here like testing plot clustors we expect a value error"""
        with self.assertRaises(ValueError):
            Visualizer.plot_anomalies([], [], show= False)

if __name__ == '__main__':
    """this one runs all the tests"""
    unittest.main()
