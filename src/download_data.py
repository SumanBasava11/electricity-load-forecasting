"""
Downloads the UCI household power consumption dataset, extracts and saves a CSV.
It then resamples to hourly average load (global_active_power).
"""
import os
import zipfile
import requests
from io import BytesIO
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
OUT_RAW = os.path.join(DATA_DIR, "household_power_consumption.txt")
OUT_HOURLY = os.path.join(DATA_DIR, "household_power_hourly.csv")

def download_and_extract():
    print("Downloading dataset from UCI...")
    r = requests.get(UCI_ZIP_URL, stream=True, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        # dataset file inside zip is household_power_consumption.txt
        z.extractall(DATA_DIR)
    print("Extracted to", DATA_DIR)

def build_hourly():
    print("Reading raw file, this may take a while...")
    df = pd.read_csv(OUT_RAW, sep=';', low_memory=False, na_values=['?'])
    # combine Date & Time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')
    # global_active_power: convert to numeric
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    # Resample to hourly average
    hourly = df['Global_active_power'].resample('H').mean().to_frame(name='global_active_power')
    hourly.to_csv(OUT_HOURLY)
    print("Saved hourly aggregated file to", OUT_HOURLY)

if __name__ == '__main__':
    download_and_extract()
    build_hourly()