"""
RNN forecasting models: LSTM & GRU.
"""
import torch
import torch.nn as nn

class RNNBase(nn.Module):
    """
    Base class for LSTM/GRU forecasting.
    Uses `use_forecast` flag to include exogenous forecast features.
    """
    def __init__(
        self,
        hist_dim: int,
        fcst_dim: int,
        config: dict,
        rnn_type: str = 'LSTM'
    ):
        super().__init__()
        self.cfg = config
        hidden = config['hidden_dim']
        layers = config['num_layers']

        # Project history features
        self.hist_proj = nn.Linear(hist_dim, hidden)
        # Optionally project forecast features
        if config.get('use_forecast', False):
            self.fcst_proj = nn.Linear(fcst_dim, hidden)

        # Recurrent layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden, hidden, num_layers=layers,
                               batch_first=True, dropout=config['dropout'])
        else:
            self.rnn = nn.GRU(hidden, hidden, num_layers=layers,
                              batch_first=True, dropout=config['dropout'])

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden, config['future_hours']),
            nn.Softplus()
        )

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        # hist: (B, past_hours, hist_dim)
        h = self.hist_proj(hist)
        if self.cfg.get('use_forecast', False) and fcst is not None:
            f = self.fcst_proj(fcst)
            seq = torch.cat([h, f], dim=1)  # (B, past+future, hidden)
        else:
            seq = h
        out, _ = self.rnn(seq)             # (B, seq_len, hidden)
        return self.head(out[:, -1, :])    # (B, future_hours)

class LSTM(RNNBase):
    """LSTM forecasting model."""
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__(hist_dim, fcst_dim, config, rnn_type='LSTM')

class GRU(RNNBase):
    """GRU forecasting model."""
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__(hist_dim, fcst_dim, config, rnn_type='GRU')
