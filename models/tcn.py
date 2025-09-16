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
        
        # 添加数值稳定性处理
        self.eps = 1e-8

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        last = None
        
        # 处理历史数据（如果存在）
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

        # 处理预测数据（如果存在）
        if self.use_fcst and fcst is not None and self.fcst_proj is not None:
            # 简化的预测特征处理，与ML模型保持一致
            f_flat = fcst.reshape(fcst.size(0), -1)
            f_proj = self.fcst_proj(f_flat)
            
            # 添加调试信息
            print(f"🔍 TCN调试: fcst形状={fcst.shape}, f_flat形状={f_flat.shape}, f_proj形状={f_proj.shape}")
            print(f"🔍 TCN调试: f_proj统计 - min={f_proj.min().item():.6f}, max={f_proj.max().item():.6f}, mean={f_proj.mean().item():.6f}")
            
            if last is not None:
                last = last + f_proj  # 简单相加融合
                print(f"🔍 TCN调试: 融合后last统计 - min={last.min().item():.6f}, max={last.max().item():.6f}, mean={last.mean().item():.6f}")
            else:
                last = f_proj
                print(f"🔍 TCN调试: 使用预测数据作为last - min={last.min().item():.6f}, max={last.max().item():.6f}, mean={last.mean().item():.6f}")

        # 如果既没有历史数据也没有预测数据，创建零向量
        if last is None:
            # 创建一个零向量作为默认输出
            batch_size = hist.size(0) if hist is not None else fcst.size(0)
            last = torch.zeros(batch_size, self.channels[-1]).to(hist.device if hist is not None else fcst.device)
            print(f"⚠️ 警告: TCN模型没有有效输入，使用零向量作为默认输出")

        # 添加数值稳定性处理
        output = self.head(last)
        
        # 检查并处理NaN值
        if torch.isnan(output).any():
            print(f"⚠️ 警告: TCN模型输出包含NaN值，使用零值替代")
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        # 确保输出为正值（太阳能发电量不能为负）
        output = torch.clamp(output, min=self.eps)
        
        return output
