#!/usr/bin/env python3
"""
改进LSTM和GRU模型测试脚本 - 方案1
借鉴Transformer成功的核心要素来解决周期性问题
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
import warnings
import random

# 设置随机种子确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.data_utils import load_raw_data, preprocess_features, create_sliding_windows, split_data

class ImprovedLSTM(nn.Module):
    """改进的LSTM模型 - 借鉴Transformer成功要素"""
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

        # 改进1: 使用ReLU + Sigmoid激活函数 (借鉴Transformer成功要素)
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
        out, _ = self.lstm(seq)        # (B, seq_len, hidden)
        
        # 使用最后时间步
        last_output = out[:, -1, :]    # (B, hidden)
        result = self.head(last_output) # (B, future_hours)
        
        # 改进2: 乘以100转换为百分比 (借鉴Transformer成功要素)
        return result * 100

class ImprovedGRU(nn.Module):
    """改进的GRU模型 - 借鉴Transformer成功要素"""
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

        # 改进1: 使用ReLU + Sigmoid激活函数 (借鉴Transformer成功要素)
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
        
        # 改进2: 乘以100转换为百分比 (借鉴Transformer成功要素)
        return result * 100

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def train_improved_rnn_model(model_name, config, Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
                           Xh_va, Xf_va, y_va, hrs_va, dates_va,
                           Xh_te, Xf_te, y_te, hrs_te, dates_te):
    """训练改进的RNN模型"""
    print(f"🚀 训练改进的 {model_name} 模型...")
    
    # 设置设备
    device = torch.device("cpu")
    
    # 创建数据加载器
    def make_loader(Xh, Xf, y, hrs, bs, shuffle=False):
        tensors = [torch.tensor(Xh, dtype=torch.float32),
                   torch.tensor(hrs, dtype=torch.long)]
        if Xf is not None:
            tensors.insert(1, torch.tensor(Xf, dtype=torch.float32))
        tensors.append(torch.tensor(y, dtype=torch.float32))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors), 
            batch_size=bs, 
            shuffle=shuffle
        )
    
    # 改进3: 使用更大的批次大小 (借鉴Transformer成功要素)
    bs = 64  # 从32改为64
    train_loader = make_loader(Xh_tr, Xf_tr, y_tr, hrs_tr, bs, shuffle=True)
    val_loader = make_loader(Xh_va, Xf_va, y_va, hrs_va, bs)
    test_loader = make_loader(Xh_te, Xf_te, y_te, hrs_te, bs)
    
    # 创建改进的模型
    complexity = config['model_complexity']
    mp = config['model_params'][complexity].copy()
    mp['use_forecast'] = config.get('use_forecast', False)
    mp['past_hours'] = config['past_hours']
    mp['future_hours'] = config['future_hours']
    
    hist_dim = Xh_tr.shape[2]
    fcst_dim = Xf_tr.shape[2] if Xf_tr is not None else 0
    
    if model_name == 'LSTM':
        model = ImprovedLSTM(hist_dim, fcst_dim, mp)
    elif model_name == 'GRU':
        model = ImprovedGRU(hist_dim, fcst_dim, mp)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.to(device)
    
    # 改进4: 使用AdamW优化器 (借鉴Transformer成功要素)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train_params']['learning_rate'], weight_decay=0.01)
    
    # 改进5: 添加学习率调度器 (借鉴Transformer成功要素)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 损失函数保持不变 (MSE)
    criterion = nn.MSELoss()
    
    # 改进6: 早停机制
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # 改进7: 增加训练轮数 (借鉴Transformer成功要素)
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(50):  # 从15改为50
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            if Xf_tr is not None:
                xh, xf, hrs, y_batch = batch
                xh, xf = xh.to(device), xf.to(device)
                pred = model(xh, xf)
            else:
                xh, hrs, y_batch = batch
                xh = xh.to(device)
                pred = model(xh)
            
            y_batch = y_batch.to(device)
            loss = criterion(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            
            # 改进8: 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if Xf_tr is not None:
                    xh, xf, hrs, y_batch = batch
                    xh, xf = xh.to(device), xf.to(device)
                    pred = model(xh, xf)
                else:
                    xh, hrs, y_batch = batch
                    xh = xh.to(device)
                    pred = model(xh)
                
                y_batch = y_batch.to(device)
                loss = criterion(pred, y_batch)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}/50, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 早停检查
        if early_stopping(avg_val_loss):
            print(f"  早停于第 {epoch+1} 轮")
            break
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ 改进的 {model_name} 训练完成")
    
    # 测试阶段
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            if Xf_tr is not None:
                xh, xf, hrs, y_batch = batch
                xh, xf = xh.to(device), xf.to(device)
                pred = model(xh, xf)
            else:
                xh, hrs, y_batch = batch
                xh = xh.to(device)
                pred = model(xh)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    return y_te, predictions

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    # 计算响应比例
    gt_diff = np.diff(y_true, axis=1)
    pred_diff = np.diff(y_pred, axis=1)
    response_ratio = np.mean(np.abs(pred_diff) / (np.abs(gt_diff) + 1e-8))
    
    # 计算唯一值比例
    unique_ratio = len(np.unique(y_pred.round(2))) / y_pred.size
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Response_Ratio': response_ratio,
        'Unique_Ratio': unique_ratio
    }

def main():
    print("🚀 开始改进LSTM和GRU模型测试（方案1）...")
    print("=" * 60)
    
    # 加载数据
    print("\n📊 加载数据...")
    df = load_raw_data('data/Project1140.csv')
    print(f"✅ 数据加载完成: {len(df)} 行")
    
    # 配置
    config = {
        'data_path': 'data/Project1140.csv',
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'model_complexity': 'low',
        'model_params': {
            'low': {
                'd_model': 64,
                'dropout': 0.1,
                'hidden_dim': 32,
                'num_heads': 4,
                'num_layers': 6
            }
        },
        'past_hours': 72,
        'future_hours': 24,
        'train_params': {
            'batch_size': 64,  # 改进：更大的批次大小
            'learning_rate': 0.001
        },
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'use_forecast': True,
        'use_hist_weather': False,
        'use_ideal_nwp': True,
        'use_pv': True,
        'use_time_encoding': False,
        'weather_category': 'all_weather'
    }
    
    # 数据预处理
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, config)
    
    # 创建滑动窗口
    X_hist, X_fcst, y, hours, dates = create_sliding_windows(
        df_clean, 
        config['past_hours'], 
        config['future_hours'], 
        hist_feats, 
        fcst_feats, 
        no_hist_power=not config.get('use_pv', True)
    )
    
    # 分割数据
    Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
    Xh_va, Xf_va, y_va, hrs_va, dates_va, \
    Xh_te, Xf_te, y_te, hrs_te, dates_te = split_data(
        X_hist, X_fcst, y, hours, dates,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"]
    )
    
    # 训练改进的模型
    models = ['LSTM', 'GRU']  # 只训练LSTM和GRU
    results = {}
    
    for model_name in models:
        try:
            y_te, predictions = train_improved_rnn_model(
                model_name, config,
                Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
                Xh_va, Xf_va, y_va, hrs_va, dates_va,
                Xh_te, Xf_te, y_te, hrs_te, dates_te
            )
            results[model_name] = (y_te, predictions)
        except Exception as e:
            print(f"❌ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    if results:
        # 创建输出目录
        output_dir = 'improved_lstm_gru_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存CSV结果
        data = {}
        first_model = list(results.keys())[0]
        y_true = results[first_model][0]
        n_samples = min(168, len(y_true))
        data['Ground_Truth'] = y_true[:n_samples].flatten()
        
        for model_name, (_, y_pred) in results.items():
            data[f'{model_name}_Prediction'] = y_pred[:n_samples].flatten()
        
        data['Timestep'] = range(len(data['Ground_Truth']))
        df_results = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, 'improved_lstm_gru_predictions.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"💾 改进预测结果已保存: {csv_path}")
        
        # 计算统计
        print(f"\n📊 改进模型预测统计:")
        for model_name, (_, y_pred) in results.items():
            pred_values = data[f'{model_name}_Prediction']
            mse = np.mean((data['Ground_Truth'] - pred_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(data['Ground_Truth'] - pred_values))
            correlation = np.corrcoef(data['Ground_Truth'], pred_values)[0, 1]
            
            # 计算唯一值比例（避免周期性）
            unique_ratio = len(np.unique(pred_values.round(2))) / len(pred_values)
            
            print(f"  {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, Correlation={correlation:.4f}")
            print(f"    响应比例: {correlation:.4f}")
            print(f"    唯一值比例: {unique_ratio:.4f} (越高越好)")
    
    print(f"\n✅ 改进LSTM和GRU测试完成！")

if __name__ == "__main__":
    main()