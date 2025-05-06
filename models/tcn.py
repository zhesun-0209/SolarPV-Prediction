"""
Temporal Convolutional Network (TCN) for forecasting.
Supports optional use of forecast features via a projection.
"""
import torch
import torch.nn as nn

class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (TCN) for forecasting.
    Supports optional use of forecast features via a projection.
    """
    def __init__(
        self,
        hist_dim: int,
        fcst_dim: int,
        config: dict
    ):
        super().__init__()
        channels = config['tcn_channels']
        kernel = config['kernel_size']
        self.use_fcst = config.get('use_forecast', False)
        future_hours = config['future_hours']

        # Build TCN encoder on historical sequence
        layers = []
        in_ch = hist_dim
        for out_ch in channels:
            layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                          kernel_size=kernel, padding=kernel//2)
            )
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # Forecast feature projection if enabled
        if self.use_fcst:
            self.fcst_proj = nn.Linear(fcst_dim * future_hours, channels[-1])

        # Final prediction head with Softplus
        self.head = nn.Sequential(
            nn.Linear(channels[-1], future_hours),
            nn.Softplus()
        )

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        # hist: (B, past_hours, hist_dim) -> (B, hist_dim, past_hours)
        x = hist.permute(0, 2, 1)
        out = self.encoder(x)            # (B, channels[-1], past_hours)
        last = out[:, :, -1]             # (B, channels[-1])
        if self.use_fcst and fcst is not None:
            # flatten forecast features and project
            f_flat = fcst.reshape(fcst.size(0), -1)
            f_proj = self.fcst_proj(f_flat)
            last = last + f_proj
        return self.head(last)           # (B, future_hours)
