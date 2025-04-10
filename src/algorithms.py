
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple

class CustomTemperaturePredictor(BaseEstimator, RegressorMixin):
    """A temperature predictor for future treends"""
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """Initializes the predictor with rate and iterations"""
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # will be inited later
        self.bias = None     

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomTemperaturePredictor':
        '''Check for any NaN values on x and y'''
        if np.isnan(X).any() or np.isnan(y).any():
            print("NaN detected in input data so skipping training..")
            return self
    
        n_samples, n_features = X.shape
        # Init weights and bias to zeros 
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update our parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values using the parameters."""
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet so call fit() first.")
        return np.dot(X, self.weights) + self.bias


def custom_clustering(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Simple k-means-like clustering and return the labels."""
    # Randomly select initial centroids from data points
    centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
    for _ in range(10):  
        
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])

    return labels


def detect_anomalies(time_series: np.ndarray, window_size: int = 10, threshold: float = 2.0) -> np.ndarray:
    """Detect anomalies using a moving average and threshold."""
    moving_avg = np.convolve(time_series, np.ones(window_size) / window_size, mode='valid') # Compute the moving average over window size
    padded_avg = np.pad(moving_avg, (window_size - 1, 0), mode='edge')
    # Flag anomalies 
    anomalies = np.abs(time_series - padded_avg) > threshold * np.std(time_series)
    return anomalies

