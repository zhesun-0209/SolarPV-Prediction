#!/usr/bin/env python3
"""
训练改进的LSTM和GRU模型，生成168小时预测对比图
解决周期性问题，展示改进效果
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.rnn_models import LSTM, GRU

def create_improved_config():
    """创建改进的模型配置"""
    config = {
        'model': 'LSTM',  # 或 'GRU'
        'hidden_dim': 64,  # 减小隐藏层维度
        'num_layers': 2,   # 减少层数
        'dropout': 0.1,    # 减少dropout
        'future_hours': 168,
        'past_hours': 168,  # 使用168小时历史数据
        'use_forecast': True,
        'use_hist_weather': False,
        'use_pv': True,
        'use_time_encoding': True,
        'epochs': 50,      # 减少训练轮数
        'batch_size': 16,  # 减小批次大小
        'learning_rate': 0.0001,  # 降低学习率
        'patience': 10,
        'min_delta': 0.001,
        'train_params': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'weight_decay': 1e-4
        }
    }
    return config

def load_sample_data():
    """加载示例数据进行训练"""
    print("📁 加载示例数据...")
    
    # 使用项目1140的数据作为示例
    data_path = "data/Project1140.csv"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ 数据形状: {df.shape}")
    print(f"✅ 列名: {df.columns.tolist()}")
    
    return df

def prepare_training_data(df, config):
    """准备训练数据"""
    print("🔧 准备训练数据...")
    
    # 选择特征列 - 使用实际存在的列
    feature_cols = []
    
    # 主要目标变量
    if 'Electricity Generated' in df.columns:
        feature_cols.append('Electricity Generated')
    elif 'pv_power_mw' in df.columns:
        feature_cols.append('pv_power_mw')
    else:
        # 如果没有找到PV功率列，使用第一个数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            feature_cols.append(numeric_cols[0])
    
    if config.get('use_forecast', False):
        # 添加天气预测特征
        weather_cols = [col for col in df.columns if 'pred' in col.lower()]
        feature_cols.extend(weather_cols[:5])  # 选择前5个预测特征
    else:
        # 添加一些历史天气特征
        weather_cols = [col for col in df.columns if any(x in col.lower() for x in ['temperature', 'humidity', 'pressure', 'wind'])]
        feature_cols.extend(weather_cols[:4])  # 选择前4个天气特征
    
    # 确保所有特征列都存在
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"✅ 可用特征: {available_cols}")
    
    if len(available_cols) == 0:
        print("❌ 没有找到可用的特征列")
        return None, None
    
    # 提取数据
    data = df[available_cols].values.astype(np.float32)
    
    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 准备序列数据
    past_hours = config['past_hours']
    future_hours = config['future_hours']
    
    X, y = [], []
    for i in range(past_hours, len(data_scaled) - future_hours + 1):
        X.append(data_scaled[i-past_hours:i])
        y.append(data_scaled[i:i+future_hours, 0])  # 只预测PV功率
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ 序列数据形状: X={X.shape}, y={y.shape}")
    
    # 分割数据
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"✅ 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), scaler

def train_improved_model(model, X_train, y_train, X_val, y_val, config):
    """训练改进的模型"""
    print(f"🚀 开始训练{config['model']}模型...")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # 分离历史数据和预测数据
            hist_data = batch_X[:, :config['past_hours']]  # 历史数据
            fcst_data = batch_X[:, config['past_hours']:]  # 预测数据
            
            # 前向传播
            outputs = model(hist_data, fcst_data)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            hist_val = X_val_tensor[:, :config['past_hours']]
            fcst_val = X_val_tensor[:, config['past_hours']:]
            val_outputs = model(hist_val, fcst_val)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss - config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss={train_losses[-1]:.6f}, Val Loss={val_loss:.6f}")
        
        if patience_counter >= config['patience']:
            print(f"🛑 早停于第 {epoch} 轮")
            break
    
    print(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}")
    return train_losses, val_losses

def generate_168h_predictions(model, X_test, y_test, config, scaler):
    """生成168小时预测并可视化"""
    print("🎨 生成168小时预测对比图...")
    
    # 选择几个测试样本
    n_samples = min(3, len(X_test))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    model.eval()
    with torch.no_grad():
        predictions = []
        ground_truths = []
        
        for idx in sample_indices:
            # 准备输入数据
            X_sample = torch.FloatTensor(X_test[idx:idx+1])
            hist_data = X_sample[:, :config['past_hours']]
            fcst_data = X_sample[:, config['past_hours']:]
            
            # 生成预测
            pred = model(hist_data, fcst_data)
            pred_np = pred.cpu().numpy()[0]
            
            # 反标准化
            pred_original = scaler.inverse_transform(
                np.column_stack([pred_np, np.zeros((len(pred_np), scaler.n_features_in_-1))])
            )[:, 0]
            
            gt_original = scaler.inverse_transform(
                np.column_stack([y_test[idx], np.zeros((len(y_test[idx]), scaler.n_features_in_-1))])
            )[:, 0]
            
            predictions.append(pred_original)
            ground_truths.append(gt_original)
    
    # 创建对比图
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        hours = range(1, len(pred) + 1)
        
        axes[i].plot(hours, gt, 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
        axes[i].plot(hours, pred, 'r--', label=f'Improved {config["model"]} Prediction', linewidth=2, alpha=0.8)
        
        axes[i].set_xlabel('Hours Ahead')
        axes[i].set_ylabel('PV Power (MW)')
        axes[i].set_title(f'Sample {i+1}: 168-Hour PV Power Prediction Comparison')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # 计算并显示指标
        mse = np.mean((pred - gt) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred - gt))
        
        axes[i].text(0.02, 0.98, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'improved_{config["model"].lower()}_168h_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 预测对比图已保存为: improved_{config['model'].lower()}_168h_prediction.png")
    
    return predictions, ground_truths

def main():
    """主函数"""
    print("🚀 开始训练改进的RNN模型 - 168小时预测")
    print("=" * 60)
    
    # 创建配置
    config = create_improved_config()
    print(f"📋 配置: {config['model']} - {config['hidden_dim']}维 - {config['num_layers']}层")
    
    # 加载数据
    df = load_sample_data()
    if df is None:
        return
    
    # 准备训练数据
    data_splits, scaler = prepare_training_data(df, config)
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits
    
    # 创建模型
    hist_dim = X_train.shape[-1]
    fcst_dim = X_train.shape[-1] if config.get('use_forecast', False) else 0
    
    if config['model'] == 'LSTM':
        model = LSTM(hist_dim, fcst_dim, config)
    else:
        model = GRU(hist_dim, fcst_dim, config)
    
    print(f"✅ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    train_losses, val_losses = train_improved_model(model, X_train, y_train, X_val, y_val, config)
    
    # 生成168小时预测对比图
    predictions, ground_truths = generate_168h_predictions(model, X_test, y_test, config, scaler)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{config["model"]} Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-20:], label='Training Loss (Last 20)', color='blue')
    plt.plot(val_losses[-20:], label='Validation Loss (Last 20)', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (Last 20 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'improved_{config["model"].lower()}_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 训练曲线已保存为: improved_{config['model'].lower()}_training_curve.png")
    
    print("\n🎯 改进效果总结:")
    print("   - 添加了时间注意力机制，让模型关注重要的时间步")
    print("   - 使用残差连接，改善梯度流和训练稳定性")
    print("   - 统一了LSTM和GRU的架构配置")
    print("   - 专门针对168小时长期预测进行了优化")

if __name__ == "__main__":
    main()
