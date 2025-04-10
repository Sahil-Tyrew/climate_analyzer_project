import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import List, Tuple

class Visualizer:
    """The class that plots the trends"""
    @staticmethod
    def plot_trend(years, actual, predicted, show: bool = True):
        """Plot actual vs predicted value over time"""
        # Convert to list if they ain't already (avoid NumPy truth value issues)
        years = list(years)
        actual = list(actual)
        predicted = list(predicted)

         # If there's no data quits and tell user
        if not years or not actual or not predicted:
            print("Not enough data for  trend.")
            return


        # Create a figure and plot both actual and predicted data
        plt.figure(figsize=(10, 5))
        plt.plot(years, actual, label="Actual", marker='o')
        plt.plot(years, predicted, label="Predicted", linestyle="--", marker='x')
        plt.xlabel("Year")
        plt.ylabel("Normalized value")
        plt.title("Climate Trend over time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def animate_temperature_comparison(actual, predicted, show: bool = True):
        """Animate the comparision between actual and predicted values"""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np

       
        if len(actual) != len(predicted):
            print("Mismatched lengths so cant animate")
            return

        print("Generating animation")

        #sets uo the figure for animation
        fig, ax = plt.subplots()
        x = list(range(len(actual)))
        ax.set_xlim(0, len(actual))
        ax.set_ylim(min(min(actual), min(predicted)), max(max(actual), max(predicted)))
        ax.set_title("Actual vs Predicted Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        line_actual, = ax.plot([], [], label="Actual", color="blue")
        line_predicted, = ax.plot([], [], label="Predicted", color="orange")
        ax.legend()

    
        def init():
            """initializer for animation """
            line_actual.set_data([], [])
            line_predicted.set_data([], [])
            return line_actual, line_predicted


        
        def update(frame):
            """Updates the function for the frame """
            line_actual.set_data(x[:frame], actual[:frame])
            line_predicted.set_data(x[:frame], predicted[:frame])
            return line_actual, line_predicted


        ani = FuncAnimation(
            fig,
            update,
            frames=len(actual) + 1,
            init_func=init,
            interval=100,
            blit=True,
            repeat=False
        )

        plt.show()


    @staticmethod
    def plot_clusters(data, labels, show: bool = True):
        """Plot clustered data with different color for each."""
        if not data or not labels:
            raise ValueError("Input lists cannot be empty")

        # Check if data is 1D or 2D and set up x and y 
        if len(data[0]) == 1:  # 1D data
            x_data = [d[0] for d in data]
            y_data = [0] * len(x_data)  
        else:  # 2D 
            x_data = [d[0] for d in data]
            y_data = [d[1] for d in data]

        #scatter plot if clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, c=labels, cmap='viridis', label="Clusters")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Clustered data')
        plt.legend()
        plt.show()


    @staticmethod
    def plot_anomalies(time_series: List[float], anomalies: List[bool], show: bool = True) -> None:
        """Plot time series with detected anomalies"""
        if not time_series or not anomalies:
            raise ValueError("Input lists are not allowed to be empty")


        
        plt.figure(figsize=(10, 6))
        plt.plot(time_series, label='Time Series')
        # Mark anomalies (non-anomalies shown as NaN)
        plt.plot(np.where(anomalies, time_series, np.nan),'ro', label='Anomales')
        plt.xlabel('Time')
        plt.ylabel('Normalized Value')
        plt.title('Anomaly Detection in Time Series')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
