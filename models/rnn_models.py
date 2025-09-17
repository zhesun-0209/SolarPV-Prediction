"""
RNN forecasting models: LSTM & GRU.
"""
import torch
import torch.nn as nn

class RNNBase(nn.Module):
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict, rnn_type: str = 'LSTM'):
        super().__init__()
        self.cfg = config
        hidden = config['hidden_dim']
        layers = config['num_layers']

        self.hist_proj = nn.Linear(hist_dim, hidden) if hist_dim > 0 else None
        self.fcst_proj = nn.Linear(fcst_dim, hidden) if config.get('use_forecast', False) and fcst_dim > 0 else None

        # 简化的RNN架构，与ML模型保持一致
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden, hidden, num_layers=layers,
                               batch_first=True, dropout=config['dropout'])
        else:
            self.rnn = nn.GRU(hidden, hidden, num_layers=layers,
                              batch_first=True, dropout=config['dropout'])

        self.head = nn.Sequential(
            nn.Linear(hidden, config['future_hours']),
            nn.Softplus()
        )

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        seqs = []

        if self.hist_proj is not None and hist.shape[-1] > 0:
            h_proj = self.hist_proj(hist)
            seqs.append(h_proj)

        if self.fcst_proj is not None and fcst is not None and fcst.shape[-1] > 0:
            f_proj = self.fcst_proj(fcst)
            seqs.append(f_proj)

        if not seqs:
            raise ValueError("Both hist and forecast inputs are missing or zero-dimensional.")

        seq = torch.cat(seqs, dim=1)  # (B, past+future, hidden)
        out, _ = self.rnn(seq)        # (B, seq_len, hidden)
        return self.head(out[:, -1, :])  # (B, future_hours)


class LSTM(nn.Module):
    """改进的LSTM forecasting model - 使用残差连接解决周期性问题"""
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__()
        self.cfg = config
        hidden = config['hidden_dim']
        layers = config['num_layers']

        self.hist_proj = nn.Linear(hist_dim, hidden) if hist_dim > 0 else None
        self.fcst_proj = nn.Linear(fcst_dim, hidden) if config.get('use_forecast', False) and fcst_dim > 0 else None

        # LSTM层保持不变
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers,
                           batch_first=True, dropout=config['dropout'])

        # 改进：添加残差连接和更复杂的输出头
        self.head1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        self.head2 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden // 2, config['future_hours']),
            nn.Sigmoid()
        )

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        seqs = []

        if self.hist_proj is not None and hist.shape[-1] > 0:
            h_proj = self.hist_proj(hist)
            seqs.append(h_proj)

        if self.fcst_proj is not None and fcst is not None and fcst.shape[-1] > 0:
            f_proj = self.fcst_proj(fcst)
            seqs.append(f_proj)

        if not seqs:
            raise ValueError("Both hist and forecast inputs are missing or zero-dimensional.")

        seq = torch.cat(seqs, dim=1)
        out, _ = self.lstm(seq)
        
        last_output = out[:, -1, :]
        
        # 残差连接 - 关键改进
        residual = self.head1(last_output)
        combined = last_output + residual
        
        result = self.head2(combined)
        
        return result * 100

class GRU(nn.Module):
    """改进的GRU forecasting model - 解决周期性问题"""
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__()
        self.cfg = config
        hidden = config['hidden_dim']
        layers = config['num_layers']

        self.hist_proj = nn.Linear(hist_dim, hidden) if hist_dim > 0 else None
        self.fcst_proj = nn.Linear(fcst_dim, hidden) if config.get('use_forecast', False) and fcst_dim > 0 else None

        # GRU层保持不变
        self.gru = nn.GRU(hidden, hidden, num_layers=layers,
                          batch_first=True, dropout=config['dropout'])

        # 改进：使用ReLU + Sigmoid激活函数 (解决周期性问题)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),  # 改为ReLU (更好的梯度流)
            nn.Dropout(config['dropout']),
            nn.Linear(hidden // 2, config['future_hours']),
            nn.Sigmoid()  # 改为Sigmoid，输出[0,1]
        )

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        seqs = []

        if self.hist_proj is not None and hist.shape[-1] > 0:
            h_proj = self.hist_proj(hist)
            seqs.append(h_proj)

        if self.fcst_proj is not None and fcst is not None and fcst.shape[-1] > 0:
            f_proj = self.fcst_proj(fcst)
            seqs.append(f_proj)

        if not seqs:
            raise ValueError("Both hist and forecast inputs are missing or zero-dimensional.")

        seq = torch.cat(seqs, dim=1)  # (B, past+future, hidden)
        out, _ = self.gru(seq)        # (B, seq_len, hidden)
        
        # 使用最后时间步
        last_output = out[:, -1, :]    # (B, hidden)
        result = self.head(last_output) # (B, future_hours)
        
        # 改进：乘以100转换为百分比 (解决周期性问题)
        return result * 100
