"""
Simple Optuna example to tune LSTM hidden size and learning rate.
"""
import optuna
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from features import load_processed, make_supervised
from models.lstm_model import LSTMForecast

def objective(trial):
    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed_hourly.csv")
    df = load_processed(processed)
    X, Y = make_supervised(df, input_window=24, horizon=24)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).float())
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    hidden = trial.suggest_int("hidden", 32, 128)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    model = LSTMForecast(input_size=X.shape[2], hidden_size=hidden, num_layers=1, horizon=24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    # short training
    for epoch in range(3):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_losses.append(loss_fn(pred, yb).item())
    return float(np.mean(val_losses))

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=12)
    print("Best params:", study.best_params)
