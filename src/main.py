import os

import numpy as np


from src.data_processor import DataProcessor
from src.algorithms import (
    CustomTemperaturePredictor,
    custom_clustering,
    detect_anomalies
)
from src.visualizer import Visualizer


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def auto_detect_target_column(filename, df):
    """Will guess the target column from the filename or the DataFrame headers"""
    lower_name = filename.lower()

    if "temp" in lower_name:
        return "temperature"
    elif "precip" in lower_name:
        return "precipitation"
    elif "anom" in lower_name:
        return "anomaly"

    # goes through the DF columns for common name
    for col in df.columns:
        if col in ["temperature", "precipitation", "anomaly"]:
            return col
        if "temp" in col.lower():
            return col
        if "precip" in col.lower():
            return col
        if "anom" in col.lower():
            return col
    return None

def list_data_files():
    """ finds all .csv file in the directory """
    options = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                rel_path = os.path.relpath(os.path.join(root, file), DATA_DIR)
                options.append(rel_path)
    return sorted(options)

def main():
    """Main function to process data /prediction /clustering /anomaly detection"""

    
    print("Availabl data files:")
    files = list_data_files()
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")

    choice = input("\nEnter the number of the file you wanna run: ")
    try:
        file_index = int(choice) - 1
        selected_file = files[file_index]
    except (ValueError, IndexError):
        print(" Invalid selection - Exiting")
        return

    filepath = os.path.join(DATA_DIR, selected_file)
    print(f"\n Selected file: {selected_file}")

    raw_df = DataProcessor(filepath, target_column="dummy").load_data()
    target_column = auto_detect_target_column(selected_file, raw_df)

    if not target_column:
        print("Could not detect target column Please change / edit")
        return

    print(f"Detected target column: '{target_column}'")

    processor = DataProcessor(filepath, target_column)
    df = processor.load_data()
    df = processor.clean_data()
    df = processor.normalize_temperature()  # normalizin the temp values
    X, y = processor.get_features_and_target()

    if X.size == 0 or y.size == 0:
        print(" Not enough valid data to process.")
        return

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

   
    
    
    print("\n Running Prediction.")
    model = CustomTemperaturePredictor(learning_rate=0.001, n_iterations=1000)
    model.fit(X, y)

    if model.weights is None:
        print("Prediction skipped: Training failed due to missing or invalid data.")
    else:
        predictions = model.predict(X)
        print(" First 5 Predictions:", predictions[:5])
        print(" Actual Values:       ", y[:5])
        Visualizer.plot_trend(list(range(len(y))), y, predictions)
        Visualizer.animate_temperature_comparison(y.tolist(), predictions.tolist())



    print("\n Running Clustering.")
    labels = custom_clustering(X, n_clusters=3)
    print("Cluster Labels (first 10):", labels[:10])
    Visualizer.plot_clusters(list(X), labels)


    
    print("\n Detecting Anomalies.")
    anomalies = detect_anomalies(y, window_size=3, threshold=1.5)
    print("Anomalies (1 = anomaly):", anomalies.astype(int))
    Visualizer.plot_anomalies(y.tolist(), anomalies.tolist())

    print("\n Done, all steps completed.")

if __name__ == "__main__":
    """" Runs  """
    main()
