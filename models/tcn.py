import torch
import torch.nn as nn

class TCNModel(nn.Module):
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__()
        channels = config['tcn_channels']
        kernel = config['kernel_size']
        self.use_fcst = config.get('use_forecast', False)
        future_hours = config['future_hours']

        # Build TCN encoder on historical sequence (only if hist_dim > 0)
        if hist_dim > 0:
            layers = []
            in_ch = hist_dim
            for out_ch in channels:
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2))
                layers.append(nn.ReLU())
                in_ch = out_ch
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = None

        # Forecast feature projection if enabled
        if self.use_fcst and fcst_dim > 0:
            self.fcst_proj = nn.Linear(fcst_dim * future_hours, channels[-1])
        else:
            self.fcst_proj = None

        # Final prediction head with Softplus
        self.head = nn.Sequential(
            nn.Linear(channels[-1], future_hours),
            nn.Softplus()
        )

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        # (B, past_hours, hist_dim) → (B, hist_dim, past_hours)
        if self.encoder is not None and hist.shape[-1] > 0:
            x = hist.permute(0, 2, 1)
            out = self.encoder(x)
            last = out[:, :, -1]
        else:
            last = None

        if self.use_fcst and fcst is not None and self.fcst_proj is not None:
            f_flat = fcst.reshape(fcst.size(0), -1)
            f_proj = self.fcst_proj(f_flat)
            if last is not None:
                last = last + f_proj
            else:
                last = f_proj

        if last is None:
            raise ValueError("Both historical and forecast features are missing or zero-dimensional.")

        return self.head(last)
