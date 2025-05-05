"""
Temporal Convolutional Network (TCN) stub for forecasting.
"""
import torch
import torch.nn as nn

class TCNModel(nn.Module):
    """
    Simple TCN model:
    - stacks 1D dilated conv layers over historical features
    - optionally ignores forecast features
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
        layers = []
        in_ch = hist_dim

        # Build TCN encoder
        for out_ch in channels:
            layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                          kernel_size=kernel, padding=kernel//2)
            )
            layers.append(nn.ReLU())
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], config['future_hours'])

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        # hist: (B, past_hours, hist_dim) -> (B, hist_dim, past_hours)
        x = hist.permute(0, 2, 1)
        out = self.encoder(x)            # (B, channels[-1], past_hours)
        last = out[:, :, -1]             # (B, channels[-1])
        return self.head(last)           # (B, future_hours)
