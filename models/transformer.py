"""
Transformer forecasting model.
"""
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    """PV forecasting Transformer model."""
    def __init__(
        self,
        hist_dim: int,
        fcst_dim: int,
        config: dict
    ):
        super().__init__()
        self.cfg = config
        d_model = config['d_model']

        # Projection layers
        self.hist_proj = nn.Linear(hist_dim, d_model)
        if config.get('use_forecast', False):
            self.fcst_proj = nn.Linear(fcst_dim, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['num_heads'],
            dim_feedforward=d_model * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['n_layers'])

        # Output head
        total_seq = config['past_hours'] + (config['future_hours'] if config.get('use_forecast', False) else 0)
        self.head = nn.Sequential(
            nn.Linear(total_seq * d_model, config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['future_hours']),
            nn.Softplus()
        )

    def forward(
        self,
        hist: torch.Tensor,        # (B, past_hours, hist_dim)
        fcst: torch.Tensor = None   # (B, future_hours, fcst_dim) or None
    ) -> torch.Tensor:
        # Encode history
        h = self.hist_proj(hist)
        h = self.pos_enc(h)
        h_enc = self.encoder(h)

        if self.cfg.get('use_forecast', False) and fcst is not None:
            f = self.fcst_proj(fcst)
            f = self.pos_enc(f)
            f_enc = self.encoder(f)
            seq = torch.cat([h_enc, f_enc], dim=1)  # (B, total_seq, d_model)
        else:
            seq = h_enc

        flat = seq.flatten(start_dim=1)  # (B, total_seq * d_model)
        return self.head(flat)           # (B, future_hours)
