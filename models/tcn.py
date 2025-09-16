import torch
import torch.nn as nn

class TCNModel(nn.Module):
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__()
        channels = config['tcn_channels']
        kernel = config['kernel_size']
        self.use_fcst = config.get('use_forecast', False)
        future_hours = config['future_hours']

        # Store channels for later use
        self.channels = channels
        
        # Build TCN encoder on historical sequence (only if hist_dim > 0)
        if hist_dim > 0:
            layers = []
            in_ch = hist_dim
            for out_ch in channels:
                # 动态调整kernel_size，确保不超过输入长度
                # 这里使用min(kernel, 3)来避免kernel_size过大的问题
                actual_kernel = min(kernel, 3)
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=actual_kernel, padding=actual_kernel // 2))
                layers.append(nn.ReLU())
                in_ch = out_ch
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = None

        # Forecast feature projection if enabled
        if self.use_fcst and fcst_dim > 0:
            # 简化的预测特征处理，与ML模型保持一致
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
            
            # 检查输入长度，如果太短则使用简单的线性层
            if x.shape[-1] < 5:  # 如果序列长度小于5，使用线性层替代卷积
                # 使用全局平均池化 + 线性层
                x_pooled = x.mean(dim=-1)  # (B, hist_dim)
                # 创建一个简单的线性层来替代卷积，输出维度要与channels[-1]匹配
                if not hasattr(self, 'fallback_linear'):
                    self.fallback_linear = nn.Linear(x.shape[1], self.channels[-1]).to(x.device)
                last = self.fallback_linear(x_pooled)  # (B, channels[-1])
            else:
                out = self.encoder(x)
                last = out[:, :, -1]
        else:
            last = None

        if self.use_fcst and fcst is not None and self.fcst_proj is not None:
            # 简化的预测特征处理，与ML模型保持一致
            f_flat = fcst.reshape(fcst.size(0), -1)
            f_proj = self.fcst_proj(f_flat)
            if last is not None:
                last = last + f_proj  # 简单相加融合
            else:
                last = f_proj

        if last is None:
            raise ValueError("Both historical and forecast features are missing or zero-dimensional.")

        return self.head(last)
