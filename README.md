<<<<<<< HEAD
# Climate Change Impact Analyzer

This project analyzes climate change data and implements algorithms, detects situations, creates predictions and more to visualize the results.


## Setup:

1. Clone repository:
 
    git clone <https://github.com/Sahil-Tyrew/climate_analyzer_project>
    cd climate_change_analyzer

2. Create virtual environment:
    python3 -m venv venv
   
 3. Activate virtual enviornment 
 source venv/bin/activate

4. Install Dependencies 
    pip install -r requirements.txt


## Usage
Run main: 
Option 1:

python3 -m src.main  #or python 

   You will see available data files with the names of the area and what you want to predict.
    Enter the number of the correlated Area+Option.

    Prediction with printed results and plot
    Clustering (works with both 1D and 2D data)
    Anomaly detection
    Animated actual vs predicted comparison using matplotlib.animation

CLI: 
Option 2:

python3 -m src.cli --folder (FolderName) --file (fileName).csv --action (predict, cluster, anomalies) --target_column (anomaly)

## Web Interface 

cd web
python app.py

    Open the site: http://127.0.0.1:5000

    Dropdown Menu Choose Same Options

## Running Tests 

Bash
coverage run -m unittest discover tests
coverage report



    The tests Include:

    All data processor functions
    Predictor, Clustering, Anomaly
    Full CLI + pipeline
    Visualizer logic



## Structure 

    CLIMATE_CHANGE_ANALYZER/
    │
    ├── data/
    |   ...(all data files)
    │
    ├── src/
    │   ├── __init__.py
    │   ├── data_processor.py 
    │   ├── algorithms.py      
    │   ├── vizualizer.py
    │   ├── cli.py
    │   ├── main.py            
    │
    ├── tests/
    |   ├── __init__.py
    │   ├── test_data_processor.py
    │   ├── test_algorithms.py
    │   ├── test_visualizer.py
    |   ├── test_pipeline.py
    │   └── ...                          # Other test files
    │
    ├── website/
    │   ├── app.py                       # Flask web application
    │   ├── templates/
    │   │   └── index.html               # Frontend HTML
    │   └── uploads/
    │       └── ...                      # Any uploaded files or user data
    |
    |
    ├── requirements.txt                 
    ├── .coverage                        # Coverage report file
    └── README.md                        


## Features:

    Automatically detects the CSV Files
    Linear regression for temperature prediction
    Clustering on climate features
    Anomaly detection + moving average
    Static and animated visuailzations
    CLI to run analysis
    High coverage testing using unittest
    Integration test of pipeline
    Website via Flask to do it on WEB
    Matplotlib plots for all outputs

## Data
   Data is sourced from: https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/tavg/land_ocean/12/1/1850-2025 


=======
# climate_analyzer_project
Climate Change Analyzer Project
>>>>>>> 50f9b3c400e93818e38b3b220da2678758173355
