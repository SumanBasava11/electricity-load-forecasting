# Electricity Load Forecasting — LSTM & Transformer

Time-series forecasting project predicting hourly electricity load.
Includes: LSTM and Transformer models, rolling-window backtesting, Optuna HPO, MLflow logging, Dockerfile, and a simple AWS S3/Lambda demo hook.

**Dataset:** Individual Household Electric Power Consumption (UCI). License: CC BY 4.0. :contentReference[oaicite:1]{index=1}

## Features
- Data download & preprocessing
- Baseline models (ARIMA/Prophet usage suggested)
- Deep learning: LSTM and Transformer (PyTorch)
- Rolling-window backtesting
- Hyperparameter tuning (Optuna)
- Experiment logging (MLflow)
- Dockerized training / inference skeleton
- Simple AWS S3 + Lambda demo instructions (prototype-level)

## Quickstart (local)
1. Clone / create repo and add files.
2. Create venv & install:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```bash
python src/download_data.py
python src/data_preprocessing.py

```bash
python src/train_lstm.py --epochs 5 --save-dir models/

```bash
python src/backtesting.py --model models/best_lstm.pth
