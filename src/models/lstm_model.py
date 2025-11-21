import torch.nn as nn
import torch

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        # take last time step
        last = out[:, -1, :]
        out = self.fc(last)
        return out
