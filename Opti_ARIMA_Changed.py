import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool

class Opti_ARIMA:
    def __init__(self, train_data : list, p : int, d : int, q : int):
        self.train_data = train_data
        self.p = p
        self.d = d
        self.q = q
        self.model = self.train()


    def train(self, show_graph = False):
        """
        Fits an ARIMA model to the input training data.
        
        Args:
            train_data (np.ndarray): The time series training data.
            p (int): The autoregressive (AR) order of the ARIMA model.
            d (int): The differencing (I) order of the ARIMA model.
            q (int): The moving average (MA) order of the ARIMA model.
            
        Returns:
            model_fit: The fitted ARIMA model.
        """

        # Fit the ARIMA model on the training data
        model = ARIMA(self.train_data, order=(self.p, self.d, self.q))
        model_fit = model.fit(low_memory=True)

        # # Get the AR and MA parameters from the fitted model
        # ar_params = np.r_[1, -model_fit.arparams]  # Include leading 1 for the AR lag polynomial
        # ma_params = np.r_[1, model_fit.maparams]   # Include leading 1 for the MA lag polynomial

        # # Create an ARMA process to get the impulse response function
        # arma_process = ArmaProcess(ar=ar_params, ma=ma_params)

        # # Get parameters from model_fit
        # params = model_fit.params

        # # Identify which parameters correspond to which model components
        # intercept = params[0] if self.d == 0 else None  # Intercept only if d == 0
        # ar_params = params[1:self.p + 1]  # AR coefficients (next p elements)
        # ma_params = params[self.p + 1:self.p + self.q + 1]  # MA coefficients (next q elements)
        # error_variance = params[-1]  # The error variance

        # print("Intercept:", intercept)
        # print("AR coefficients:", ar_params)
        # print("MA coefficients:", ma_params)
        # print("Error variance:", error_variance)

        # if (show_graph):            
        #     # Calculate impulse response over a number of steps
        #     n_steps = 100  # Adjust the number of steps as desired
        #     impulse_response = arma_process.impulse_response(n_steps)

        #     # Plot the impulse response
        #     plt.figure(figsize=(10, 5))
        #     plt.stem(np.arange(n_steps), impulse_response, basefmt=" ")
        #     plt.title("Impulse Response of ARMA Model")
        #     plt.xlabel("Steps")
        #     plt.ylabel("Response")
        #     plt.grid()
        #     plt.show()

        return model_fit

    def forecaster(self, test_data: np.ndarray, forecast_horizon=10):

        last =np.zeros(len(test_data))
        for i in range(len(test_data)):
            forecast = self.model.forecast(steps=forecast_horizon)
            last[i]=forecast[-1]
            self.model = self.model.append(test_data[i:i + 1], refit=False)
        
        return last
    import numpy as np

    def vectorized_arima_forecast(train_data, ar_params, ma_params, d, forecast_horizon=1):
        """
        Efficient ARIMA forecasting using vectorized operations.

        Args:
            train_data (np.ndarray): The historical training data.
            ar_params (np.ndarray): AR coefficients (length p).
            ma_params (np.ndarray): MA coefficients (length q).
            d (int): Differencing order.
            forecast_horizon (int): Number of steps to forecast.

        Returns:
            np.ndarray: Forecasted values for the specified horizon.
        """
        # Apply differencing if necessary
        if d > 0:
            diff_data = np.diff(train_data, n=d)
        else:
            diff_data = train_data

        # Prepare AR and MA lag histories
        ar_order = len(ar_params)
        ma_order = len(ma_params)
        max_lag = max(ar_order, ma_order)

        # Initialize history with zeros for padding
        history = np.zeros(max_lag + len(diff_data))
        history[max_lag:] = diff_data  # Fill the recent data
        residuals = np.zeros(max_lag)  # Initial residuals

        # Forecasts array
        forecasts = np.zeros(forecast_horizon)

        # Efficiently compute forecasts
        for t in range(forecast_horizon):
            # Vectorized AR and MA components
            ar_component = np.dot(ar_params, history[-ar_order:][::-1])  # Reverse slice for correct order
            ma_component = np.dot(ma_params, residuals[-ma_order:][::-1])

            # Combine AR and MA components
            forecasts[t] = ar_component + ma_component

            # Update history and residuals
            history = np.append(history, forecasts[t])
            residuals = np.append(residuals, 0)  # Assume no residual for forecast

        # Reverse differencing if needed
        if d > 0:
            for _ in range(d):
                forecasts = np.cumsum([train_data[-1]] + list(forecasts))

        return forecasts

