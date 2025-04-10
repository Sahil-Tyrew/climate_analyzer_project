
# This script accepts CLI to select a data file and predict, cluster or anomalies

import argparse
import os

import numpy as np

from src.data_processor import DataProcessor
from src.algorithms import (
    CustomTemperaturePredictor,
    custom_clustering,
    detect_anomalies
)
from src.visualizer import Visualizer



def main():
    """Main CLI setsup our argument parser and run the script from the command line"""
    parser = argparse.ArgumentParser(description="Climate Change Impact Analyzer CLI")
    parser.add_argument("--folder", required=True, help="Subfolder inside data/ (e.g., 'precipitation' or 'temperature_anomaly')")
    parser.add_argument("--file", required=True, help="CSV filename inside the folder")
    parser.add_argument("--action", required=True, choices=["predict", "cluster", "anomalies"], help="Action to perform")
    parser.add_argument("--target_column", required=True, help="Name of the colomn to predict (e.g., 'temperature', 'precipitation', 'anomaly')")

    args = parser.parse_args()

    # Builds full path to the data file 
    data_path = os.path.join("data", args.folder, args.file)

    if not os.path.exists(data_path):
        print(f" Error file not found at {data_path}")
        return

    # Load / preprocess data using DataProcessor
    processor = DataProcessor(data_path, args.target_column)
    df = processor.load_data()
    df = processor.clean_data()
    df = processor.normalize_temperature()  
    X, y = processor.get_features_and_target()

    # Check if we have enough data to work with
    if X.size == 0 or y.size == 0:
        print(" Skippin: Not enough data or columns missing.")
        return

    # Normalize input features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    if args.action == "predict":
        model = CustomTemperaturePredictor(learning_rate=0.001, n_iterations=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        # Show a few predictions vs actual values
        print("\nPredictions:", predictions[:5])
        print("Actual:     ", y[:5])
        # visualize the trend and animate 
        Visualizer.plot_trend(list(range(len(y))), y, predictions)
        Visualizer.animate_temperature_comparison(y.tolist(), predictions.tolist())

    elif args.action == "cluster":
        # Clustering data 
        labels = custom_clustering(X, n_clusters=3)
        print("\n Cluster Labels:", labels[:10])
        # Plot clusters 
        Visualizer.plot_clusters(X, labels)

    elif args.action == "anomalies":
        # detect anomalies 
        anomalies = detect_anomalies(y, window_size=3, threshold=1.5)
        print("\n Anomalies (1 = anomaly):", anomalies.astype(int))
        # Plot anomalies 
        Visualizer.plot_anomalies(y.tolist(), anomalies.tolist())


if __name__ == "__main__":
    """" Run CLI """
    main()

