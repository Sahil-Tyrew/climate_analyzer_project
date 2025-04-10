import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from src.data_processor import DataProcessor
from src.algorithms import CustomTemperaturePredictor, custom_clustering, detect_anomalies
from src.visualizer import Visualizer


app = Flask(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

UPLOAD_FOLDER = "web/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def list_data_files():
    """ scan the DATA_DIR and return a sorted list of all CSV file paths """
    options = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                # Get a relative path so not to expose full system path
                rel_path = os.path.relpath(os.path.join(root, file), DATA_DIR)
                options.append(rel_path)
    return sorted(options)

def auto_detect_target_column(filename, df):
    """  Guesses the target column name based on the filename or the DF header  """
    lower_name = filename.lower()

    if "temp" in lower_name:
        return "temperature"
    elif "precip" in lower_name:
        return "precipitation"
    elif "anom" in lower_name:
        return "anomaly"

   
   #checking the column names in the CSV
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

@app.route("/", methods=["GET", "POST"])
def index():
    """ Both GET and POST requests and displays available data files runs what user tells it to   """
    result = None  
    plot_created = False  
    file_options = list_data_files()  

    if request.method == "POST":
        selected_file = request.form.get("selected_file")
        action = request.form.get("action")


        if selected_file and action:
            filepath = os.path.join(DATA_DIR, selected_file)
           # help auto-detect columns
            df_preview = pd.read_csv(filepath, comment="#", engine="python")
            target_column = auto_detect_target_column(selected_file, df_preview)

            if not target_column:
                result = " Could not detect target column. Please rename your file or column to include 'temp', 'precip', or 'anom'."
                return render_template("index.html", result=result, plot=plot_created, files=file_options)

            processor = DataProcessor(filepath, target_column)
            df = processor.load_data()
            df = processor.clean_data()
            df = processor.normalize_temperature()
            X, y = processor.get_features_and_target()

            
            if X.size == 0 or y.size == 0:
                result = " Not enough data to process."
            else:
                X_mean = X.mean(axis=0)
                X_std = X.std(axis=0)
                X = (X - X_mean) / X_std

                if action == "predict":
                    # Create / train our temperature prediction model
                    model = CustomTemperaturePredictor()
                    model.fit(X, y)

                    if model.weights is None:
                        result = " Prediction skipped, training failed due to missing or invalid data"
                    else:
                        predictions = model.predict(X)
                        # Plot the trend of actual vs predicted values
                        Visualizer.plot_trend(range(len(y)), y, predictions)
                        # Save the generated plot to the static folder to displkay
                        plt.savefig("web/static/plot.png")
                        plot_created = True
                        result = f" Prediction complete here are the first 5 predictions: {predictions[:5]}"

                elif action == "cluster":
                    labels = custom_clustering(X, n_clusters=3)
                    Visualizer.plot_clusters(list(X), labels)
                    plt.savefig("web/static/plot.png")
                    plot_created = True
                    result = f" Clustering complete, sample labels: {labels[:5]}"

                elif action == "anomalies":
                    anomalies = detect_anomalies(y, window_size=3, threshold=1.5)
                    Visualizer.plot_anomalies(y.tolist(), anomalies.tolist())
                    plt.savefig("web/static/plot.png")
                    plot_created = True
                    result = " Anomaly detection complete."


    return render_template("index.html", result=result, plot=plot_created, files=file_options)

if __name__ == "__main__":
    """ Runs the Flask app in debug mode and off if in production"""
    app.run(debug=True)
