"""
Simple inference script that loads best model and predicts next horizon.
Saves results as CSV and prints a snippet.
"""
import os
import torch
import numpy as np
from features import load_processed, make_supervised
from models.lstm_model import LSTMForecast

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_lstm.pth")
PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed_hourly.csv")

def predict_next():
    df = load_processed(PROCESSED)
    X, Y = make_supervised(df, input_window=24, horizon=24)
    last_X = X[-1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecast(input_size=last_X.shape[2], hidden_size=64, num_layers=2, horizon=24)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        pred = model(torch.tensor(last_X).float().to(device)).cpu().numpy().squeeze()
    print("Next 24h prediction (sample):", pred[:8])
    out = {"pred_{}".format(i): float(p) for i,p in enumerate(pred)}
    import pandas as pd
    pd.Series(out).to_csv(os.path.join(os.path.dirname(__file__), "..", "predictions.csv"))
    print("Saved predictions to predictions.csv")

if __name__ == '__main__':
    predict_next()
