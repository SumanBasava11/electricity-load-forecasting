"""
Load hourly CSV and create features: lags, rolling mean/std, hour/dayofweek, month.
Saves a processed CSV ready for modeling.
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "data")
IN_HOURLY = os.path.join(DATA_DIR, "household_power_hourly.csv")
OUT_PROCESSED = os.path.join(DATA_DIR, "processed_hourly.csv")

def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    # lags
    for lag in [1, 24, 24*7]:
        df[f'lag_{lag}'] = df['global_active_power'].shift(lag)
    # rolling stats
    df['rolling_24_mean'] = df['global_active_power'].rolling(window=24).mean()
    df['rolling_24_std'] = df['global_active_power'].rolling(window=24).std().fillna(0)
    # fill missing via interpolation
    df = df.interpolate(limit_direction='both')
    return df

def main():
    df = pd.read_csv(IN_HOURLY, index_col=0, parse_dates=True)
    df = create_features(df)
    df.to_csv(OUT_PROCESSED)
    print("Processed data saved to", OUT_PROCESSED)

if __name__ == '__main__':
    main()
