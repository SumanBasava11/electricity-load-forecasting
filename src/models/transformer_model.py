import torch.nn as nn
import torch

class SimpleTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, horizon=24):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, src):
        # src shape: [batch, seq_len, features]
        x = self.input_proj(src)  # [batch, seq, d_model]
        # transformer expects [seq, batch, d_model]
        x = x.permute(1,0,2)
        x = self.pos_encoder(x)
        # use same x as tgt placeholder for simple auto-regression (simplified)
        out = self.transformer(x, x)
        out = out.permute(1,0,2)  # [batch, seq, d_model]
        last = out[:, -1, :]
        return self.fc_out(last)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        seq_len = x.size(0)
        return x + self.pe[:seq_len]
