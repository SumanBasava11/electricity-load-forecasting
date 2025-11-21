"""
Rolling window backtesting example.
"""
import argparse
import os
import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error
from features import load_processed, make_supervised
from models.lstm_model import LSTMForecast

def evaluate_model(model, X_test, Y_test, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in X_test:
            xb_t = torch.tensor(xb).unsqueeze(0).float().to(device)
            pred = model(xb_t).cpu().numpy().squeeze()
            preds.append(pred)
    preds = np.array(preds)
    # For simplicity compute MAPE on first horizon step averaged
    mape = mean_absolute_percentage_error(Y_test[:,0], preds[:,0])
    return mape

def rolling_backtest(model_path, processed_path, input_window=24, horizon=24, window_size=24*30):
    df = load_processed(processed_path)
    X, Y = make_supervised(df, input_window=input_window, horizon=horizon)
    # rolling windows: train on first part, test on next portion; step by window_size
    mape_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for start in range(0, len(X) - 2*window_size, window_size):
        train_end = start + window_size
        test_end = train_end + window_size
        X_train, Y_train = X[:train_end], Y[:train_end]
        X_test, Y_test = X[train_end:test_end], Y[train_end:test_end]
        model = LSTMForecast(input_size=X.shape[2], hidden_size=64, num_layers=2, horizon=horizon)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        mape = evaluate_model(model, X_test, Y_test, device)
        print(f"Window {start} -> MAPE: {mape:.4f}")
        mape_scores.append(mape)
    print("Average MAPE across windows:", np.mean(mape_scores))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--processed", default=os.path.join(os.path.dirname(__file__), "..", "data", "processed_hourly.csv"))
    args = parser.parse_args()
    rolling_backtest(args.model, args.processed)
