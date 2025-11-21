import pandas as pd
import numpy as np

def load_processed(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def make_supervised(df, input_window=24, horizon=24, features=None):
    """
    Returns X, y as numpy arrays.
    X shape: (samples, input_window, n_features)
    y shape: (samples, horizon)
    """
    if features is None:
        features = ['global_active_power', 'hour', 'dayofweek', 'month', 'lag_1', 'lag_24', 'lag_168', 'rolling_24_mean', 'rolling_24_std']
    arr = df[features].values
    X, Y = [], []
    for i in range(input_window, len(arr) - horizon + 1):
        X.append(arr[i-input_window:i])
        Y.append(arr[i:i+horizon, 0])  # predict first column (global_active_power) for horizon
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
