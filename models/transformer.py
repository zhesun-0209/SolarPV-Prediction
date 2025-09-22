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
    """PV forecasting Transformer model - 使用正确的简单Encoder-only架构"""
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

        # 改进：局部位置编码
        self.pos_enc = LocalPositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['num_heads'],
            dim_feedforward=d_model * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # 改进：使用ReLU激活和Sigmoid输出
        self.head = nn.Sequential(
            nn.Linear(d_model, config['hidden_dim']),
            nn.ReLU(),  # 改为ReLU
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['future_hours']),  # 直接输出future_hours维度
            nn.Sigmoid()  # 改为Sigmoid，输出[0,1]
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
            
            # 使用历史编码的最后部分和预测编码
            combined = torch.cat([h_enc, f_enc], dim=1)
        else:
            combined = h_enc

        # 使用最后的时间步进行预测
        last_timestep = combined[:, -1, :]  # (B, d_model)
        result = self.head(last_timestep)   # (B, future_hours)
        
        return result * 100  # 乘以100转换为百分比

class LocalPositionalEncoding(nn.Module):
    """局部位置编码 - 支持更长的序列"""
    def __init__(self, d_model, max_len=200):  # 增加到200以支持168小时lookback
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 如果序列长度超过max_len，动态扩展位置编码
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # 动态创建更长的位置编码
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                               (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        else:
            return x + self.pe[:, :seq_len]

