# Similar to train_lstm.py but uses SimpleTransformer
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import mlflow
from models.transformer_model import SimpleTransformer
from features import load_processed, make_supervised

def train(args):
    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed_hourly.csv")
    df = load_processed(processed)
    X, Y = make_supervised(df, input_window=args.input_window, horizon=args.horizon)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(input_size=X.shape[2], d_model=args.d_model, nhead=args.nhead, horizon=args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()

    mlflow.start_run()
    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        mean_val = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{args.epochs} - val_loss: {mean_val:.4f}")
        mlflow.log_metric("val_loss", float(mean_val), step=epoch)
        if mean_val < best_val:
            best_val = mean_val
            os.makedirs(args.save_dir, exist_ok=True)
            best_path = os.path.join(args.save_dir, "best_transformer.pth")
            torch.save(model.state_dict(), best_path)
            mlflow.log_artifact(best_path, artifact_path="models")
    mlflow.end_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--d-model", dest="d_model", type=int, default=64)
    parser.add_argument("--nhead", dest="nhead", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-window", dest="input_window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../models"))
    args = parser.parse_args()
    train(args)
