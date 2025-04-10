import pandas as pd
import numpy as np

from typing import Tuple

class DataProcessor:
    """Process data from a CSV file to clean data and normalize techniques and create a module for data loading  and pre processing """
    def __init__(self, file_path: str, target_column: str):
        """Initializes data processor with the file and the target column in the data"""
        self.file_path = file_path
        self.target_column = target_column
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV,this takes the normal data name and renames the target column -potential"""
        try:
            print(f" Loading data from: {self.file_path}")
            self.df = pd.read_csv(
                self.file_path,
                engine="python",
                comment="#",  
            )

            # Normalize column names
            self.df.columns = [col.strip().lower() for col in self.df.columns]

            # Rename standard data columns to match the wanted target colum
            if "value" in self.df.columns:
                self.df.rename(columns={"value": self.target_column}, inplace=True)
            elif "anomaly" in self.df.columns:
                self.df.rename(columns={"anomaly": self.target_column}, inplace=True)

            print(f" Loaded, Columns: {self.df.columns.tolist()}")
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}")
            self.df = pd.DataFrame()
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Replace error and drop the missing values"""
        if self.df is not None and not self.df.empty:
            self.df.replace(-9999, pd.NA, inplace=True)
            self.df.dropna(inplace=True)
        return self.df

    def normalize_temperature(self) -> pd.DataFrame:
        """Normalize the target column using the data values"""
        if self.target_column in self.df.columns:
            self.df[self.target_column] = (
                self.df[self.target_column] - self.df[self.target_column].mean()
            ) / self.df[self.target_column].std()
        else:
            print(f"'{self.target_column}' column not found ; no normalization")
        return self.df

    def get_features_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features year/ month  and the target variable if missing, return the empty array"""
        if self.df is None or self.df.empty:
            return np.array([]), np.array([])

        if 'year' not in self.df.columns or self.target_column not in self.df.columns:
            print(f"Missing required columns; 'year' and '{self.target_column}'")
            return np.array([]), np.array([])

       
        if 'month' in self.df.columns:
            X = self.df[['year', 'month']].values
        else:
            X = self.df[['year']].values
        y = self.df[self.target_column].values
        return X, y
