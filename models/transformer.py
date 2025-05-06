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
        self.fcst_proj = nn.Linear(fcst_dim, d_model) if fcst_dim > 0 else None


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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        total_seq = config['past_hours'] + (config['future_hours'] if config.get('use_forecast', False) else 0)
        self.head = nn.Sequential(
            nn.Linear(total_seq * d_model, config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['future_hours']),
            nn.Softplus()
        )

    def forward(
        self,
        hist: torch.Tensor,        # shape: (B, past_hours, hist_dim)
        fcst: torch.Tensor = None  # shape: (B, future_hours, fcst_dim), optional
    ) -> torch.Tensor:
        # Encode historical input
        h = self.hist_proj(hist)              # (B, past_hours, d_model)
        h = self.pos_enc(h)
        h_enc = self.encoder(h)               # (B, past_hours, d_model)

        # Encode forecast input if applicable
        if self.cfg.get('use_forecast', False) and fcst is not None and self.fcst_proj is not None:
            f = self.fcst_proj(fcst)          # (B, future_hours, d_model)
            f = self.pos_enc(f)
            f_enc = self.encoder(f)           # (B, future_hours, d_model)
            seq = torch.cat([h_enc, f_enc], dim=1)  # (B, total_seq, d_model)
        else:
            seq = h_enc                       # (B, past_hours, d_model)

        flat = seq.flatten(start_dim=1)       # (B, total_seq * d_model)
        return self.head(flat)                # (B, future_hours)

