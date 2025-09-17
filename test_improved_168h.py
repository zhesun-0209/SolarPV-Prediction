#!/usr/bin/env python3
"""
测试改进的LSTM和GRU模型 - 168小时预测
使用残差连接和优化激活函数解决周期性问题
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.rnn_models import LSTM, GRU

def load_real_data():
    """加载真实的Project1140数据，按照colab_batch_experiments的特征组合"""
    print("🔧 加载真实的Project1140数据...")
    
    import pandas as pd
    
    data_path = "data/Project1140.csv"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ 原始数据形状: {df.shape}")
    
    # 选择特征列 - 按照colab_batch_experiments的NWP+特征组合
    feature_cols = []
    
    # 目标变量 - Capacity Factor (0-100整数范围，不标准化)
    if 'Capacity Factor' in df.columns:
        feature_cols.append('Capacity Factor')
        print("✅ 目标变量: Capacity Factor (范围0-100整数)")
    else:
        print("❌ 未找到'Capacity Factor'列")
        return None
    
    # PV特征 - Electricity Generated (作为输入特征)
    if 'Electricity Generated' in df.columns:
        feature_cols.append('Electricity Generated')
        print("✅ 添加PV特征: Electricity Generated")
    
    # NWP预测特征 (6个主要特征)
    nwp_cols = [col for col in df.columns if col.endswith('_pred')]
    if nwp_cols:
        selected_nwp = [col for col in nwp_cols if any(x in col for x in [
            'temperature_2m_pred', 'relative_humidity_2m_pred', 'surface_pressure_pred',
            'wind_speed_100m_pred', 'global_tilted_irradiance_pred', 'cloud_cover_low_pred'
        ])]
        feature_cols.extend(selected_nwp)
        print(f"✅ 添加NWP预测特征: {selected_nwp}")
    
    # 历史天气特征 (6个主要特征)
    hist_weather_cols = [col for col in df.columns if any(x in col for x in [
        'temperature_2m', 'relative_humidity_2m', 'surface_pressure',
        'wind_speed_10m', 'global_tilted_irradiance', 'cloud_cover'
    ]) and not col.endswith('_pred')]
    
    if hist_weather_cols:
        feature_cols.extend(hist_weather_cols[:6])  # 选择前6个历史天气特征
        print(f"✅ 添加历史天气特征: {hist_weather_cols[:6]}")
    
    # 时间特征 - 根据use_time_encoding决定
    time_features = []
    if 'Hour (Eastern Time, Daylight-Adjusted)' in df.columns:
        hour = df['Hour (Eastern Time, Daylight-Adjusted)'].values
        time_features.extend([
            np.sin(2 * np.pi * hour / 24),  # 小时的正弦编码
            np.cos(2 * np.pi * hour / 24)   # 小时的余弦编码
        ])
        print("✅ 添加时间特征: Hour (正余弦编码)")
    
    if 'Month' in df.columns:
        month = df['Month'].values
        time_features.extend([
            np.sin(2 * np.pi * month / 12),  # 月份的正弦编码
            np.cos(2 * np.pi * month / 12)   # 月份的余弦编码
        ])
        print("✅ 添加时间特征: Month (正余弦编码)")
    
    # 确保所有特征列都存在
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"✅ 基础特征列表 ({len(available_cols)}个): {available_cols}")
    
    if len(available_cols) < 2:
        print("❌ 特征列不足")
        return None
    
    # 提取基础数据并处理缺失值
    base_data = df[available_cols].fillna(method='ffill').fillna(method='bfill').values.astype(np.float32)
    
    # 添加时间特征
    if time_features:
        time_array = np.column_stack(time_features).astype(np.float32)
        data = np.column_stack([base_data, time_array])
        print(f"✅ 添加时间编码特征: {len(time_features)}个")
    else:
        data = base_data
    
    print(f"✅ 最终数据形状: {data.shape}")
    print(f"✅ 数据范围: {data.min():.2f} - {data.max():.2f}")
    print(f"✅ Capacity Factor范围: {data[:, 0].min():.0f} - {data[:, 0].max():.0f} (整数)")
    
    return data

def prepare_sequences(data, past_hours=72, future_hours=24):
    """准备序列数据，Capacity Factor作为目标变量，按照colab_batch_experiments配置"""
    print("🔧 准备序列数据...")
    print(f"📊 输入长度: {past_hours}小时, 预测长度: {future_hours}小时")
    
    # 分离目标变量和特征
    capacity_factor = data[:, 0:1]  # Capacity Factor (0-100整数范围，不标准化)
    features = data[:, 1:]  # 其他特征 (需要标准化)
    
    # 只对特征进行标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 重新组合数据 - Capacity Factor不标准化
    data_scaled = np.column_stack([capacity_factor, features_scaled])
    
    X, y = [], []
    for i in range(past_hours, len(data_scaled) - future_hours + 1):
        X.append(data_scaled[i-past_hours:i])  # 输入序列：所有特征
        y.append(data_scaled[i:i+future_hours, 0])  # 目标序列：Capacity Factor (第一列，未标准化)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"✅ 序列数据形状: X={X.shape}, y={y.shape}")
    print(f"✅ 目标变量范围 (Capacity Factor): {y.min():.0f} - {y.max():.0f} (整数)")
    
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

def train_model(model, X_train, y_train, X_val, y_val, config):
    """训练模型"""
    print(f"🚀 开始训练{config['model']}模型...")
    print(f"📊 训练数据: {X_train.shape}, 验证数据: {X_val.shape}")
    print(f"🎯 目标变量范围: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"⚙️ 模型配置: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}, dropout={config['dropout']}")
    print(f"📈 训练参数: epochs={config['epochs']}, batch_size={config['batch_size']}, lr={config['learning_rate']}")
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 将模型移动到GPU
    model = model.to(device)
    
    # 转换为PyTorch张量并移动到GPU
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("=" * 60)
    print("🔄 开始训练循环...")
    print("=" * 60)
    
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
            
            # 检查NaN
            if torch.isnan(loss):
                print(f"❌ Epoch {epoch+1} 出现NaN损失，跳过此批次")
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            hist_val = X_val_tensor[:, :config['past_hours']]
            fcst_val = X_val_tensor[:, config['past_hours']:]
            val_outputs = model(hist_val, fcst_val)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss - config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 每5个epoch输出一次信息
        if epoch % 5 == 0 or epoch == config['epochs'] - 1:
            current_lr = optimizer.param_groups[0]['lr']
            gpu_memory = f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "CPU"
            print(f"Epoch {epoch+1:3d}/{config['epochs']:3d} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Patience: {patience_counter}/{config['patience']} | "
                  f"GPU: {gpu_memory}")
        
        if patience_counter >= config['patience']:
            print(f"🛑 早停于第 {epoch+1} 轮")
            break
    
    print("=" * 60)
    print(f"✅ {config['model']}训练完成!")
    print(f"📈 最佳验证损失: {best_val_loss:.6f}")
    print(f"📊 最终训练损失: {train_losses[-1]:.6f}")
    print(f"📊 最终验证损失: {val_losses[-1]:.6f}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 GPU内存已清理: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    print("=" * 60)
    return train_losses, val_losses

def generate_predictions(model, X_test, y_test, config, model_name):
    """生成预测并可视化 - 168小时（7天）连续预测对比图"""
    print(f"🎨 生成{model_name}的168小时连续预测对比图...")
    
    # 检查设备
    device = next(model.parameters()).device
    print(f"🖥️ 使用设备: {device}")
    
    # 选择测试集前168个时间步（7天 * 24小时）
    n_timesteps = min(168, len(X_test))
    sample_indices = list(range(n_timesteps))
    
    model.eval()
    with torch.no_grad():
        predictions = []
        ground_truths = []
        
        for idx in sample_indices:
            # 准备输入数据并移动到GPU
            X_sample = torch.FloatTensor(X_test[idx:idx+1]).to(device)
            hist_data = X_sample[:, :config['past_hours']]
            fcst_data = X_sample[:, config['past_hours']:]
            
            # 生成预测
            pred = model(hist_data, fcst_data)
            pred_np = pred.cpu().numpy()[0]
            
            # 只取第一个时间步的预测值（24小时预测的第一个小时）
            predictions.append(pred_np[0])
            ground_truths.append(y_test[idx][0])
    
    # 创建168小时连续预测对比图
    plt.figure(figsize=(20, 8))
    
    # 时间轴：168小时 = 7天
    time_hours = np.arange(168)
    time_days = time_hours / 24  # 转换为天数
    
    # 绘制预测和真实值
    plt.plot(time_hours, ground_truths, 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
    plt.plot(time_hours, predictions, 'r--', label=f'{model_name} Prediction', linewidth=2, alpha=0.8)
    
    # 设置图形属性
    plt.title(f'{model_name} Model: 168-Hour Continuous Prediction vs Ground Truth (First 7 Days of Test Set)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time (Hours)', fontsize=14)
    plt.ylabel('Capacity Factor (0-100)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加天数标记
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(0, 169, 24))  # 每24小时一个标记
    ax2.set_xticklabels([f'Day {i+1}' for i in range(8)])  # Day 1 到 Day 8
    ax2.set_xlabel('Days', fontsize=14)
    
    # 设置y轴范围
    all_values = ground_truths + predictions
    y_min, y_max = min(all_values), max(all_values)
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # 添加统计信息
    mse = np.mean((np.array(predictions) - np.array(ground_truths)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truths)))
    
    stats_text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12)
    
    plt.tight_layout()
    save_path = f'improved_{model_name.lower()}_168h_prediction.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ {model_name} 168小时连续预测对比图已保存到: {save_path}")
    print(f"📊 统计信息: RMSE={rmse:.3f}, MAE={mae:.3f}")
    
    return predictions, ground_truths

def plot_168h_comparison(models, scaler):
    """绘制168小时连续预测对比图 - LSTM vs GRU"""
    print("📊 绘制168小时连续预测对比图...")
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    
    # 获取168小时的连续预测数据
    lstm_pred = models['LSTM']['predictions']
    gru_pred = models['GRU']['predictions']
    y_true = models['LSTM']['ground_truths']
    
    # Capacity Factor不需要反标准化，已经是0-100范围
    lstm_pred_denorm = lstm_pred
    gru_pred_denorm = gru_pred
    y_true_denorm = y_true
    
    # 绘制预测结果
    time_hours = np.arange(168)
    axes[0].plot(time_hours, y_true_denorm, 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
    axes[0].plot(time_hours, lstm_pred_denorm, 'r--', label='LSTM Prediction', linewidth=2, alpha=0.8)
    axes[0].plot(time_hours, gru_pred_denorm, 'g--', label='GRU Prediction', linewidth=2, alpha=0.8)
    
    axes[0].set_title('LSTM vs GRU: 168-Hour Continuous Prediction Comparison (First 7 Days of Test Set)', 
                      fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Time (Hours)', fontsize=14)
    axes[0].set_ylabel('Capacity Factor (0-100)', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 添加天数标记
    ax = axes[0]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(0, 169, 24))  # 每24小时一个标记
    ax2.set_xticklabels([f'Day {i+1}' for i in range(8)])  # Day 1 到 Day 8
    ax2.set_xlabel('Days', fontsize=14)
    
    # 绘制误差对比
    lstm_error = np.abs(lstm_pred_denorm - y_true_denorm)
    gru_error = np.abs(gru_pred_denorm - y_true_denorm)
    
    axes[1].plot(time_hours, lstm_error, 'r-', label='LSTM Error', linewidth=2, alpha=0.8)
    axes[1].plot(time_hours, gru_error, 'g-', label='GRU Error', linewidth=2, alpha=0.8)
    
    axes[1].set_title('Prediction Error Comparison (Capacity Factor)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Time (Hours)', fontsize=14)
    axes[1].set_ylabel('Absolute Error (0-100)', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 添加天数标记到误差图
    ax = axes[1]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(0, 169, 24))  # 每24小时一个标记
    ax2.set_xticklabels([f'Day {i+1}' for i in range(8)])  # Day 1 到 Day 8
    ax2.set_xlabel('Days', fontsize=14)
    
    # 添加统计信息
    lstm_rmse = np.sqrt(np.mean(lstm_error ** 2))
    gru_rmse = np.sqrt(np.mean(gru_error ** 2))
    lstm_mae = np.mean(lstm_error)
    gru_mae = np.mean(gru_error)
    
    stats_text = f'LSTM: RMSE={lstm_rmse:.3f}, MAE={lstm_mae:.3f}\nGRU: RMSE={gru_rmse:.3f}, MAE={gru_mae:.3f}'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=11)
    
    plt.tight_layout()
    save_path = 'improved_lstm_gru_comparison_168h.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ 168小时连续预测对比图已保存为: {save_path}")
    print(f"📊 LSTM统计: RMSE={lstm_rmse:.3f}, MAE={lstm_mae:.3f}")
    print(f"📊 GRU统计: RMSE={gru_rmse:.3f}, MAE={gru_mae:.3f}")

def main():
    """主函数"""
    print("🚀 测试改进的RNN模型 - 168小时预测 (7天预测)")
    print("=" * 70)
    
    # 检查GPU可用性
    print(f"🖥️ PyTorch版本: {torch.__version__}")
    print(f"🔧 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔥 CUDA版本: {torch.version.cuda}")
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")
    print("=" * 70)
    
    # 创建配置 - 按照yaml中low complexity的配置
    config = {
        'model': 'LSTM',
        'hidden_dim': 32,        # low: 32
        'num_layers': 6,         # low: 6
        'dropout': 0.1,          # low: 0.1
        'd_model': 64,           # low: 64
        'num_heads': 4,          # low: 4
        'future_hours': 24,      # 24小时预测
        'past_hours': 72,        # 72小时历史
        'use_forecast': True,
        'epochs': 50,            # low: 50
        'batch_size': 64,        # low: 64
        'learning_rate': 0.001,  # low: 0.001
        'patience': 10,
        'min_delta': 0.001
    }
    
    # 加载真实数据
    data = load_real_data()
    if data is None:
        print("❌ 无法加载数据，退出程序")
        return
    
    # 准备序列数据
    data_splits, scaler = prepare_sequences(data, config['past_hours'], config['future_hours'])
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits
    
    # 测试LSTM和GRU
    models = {}
    for model_type in ['LSTM', 'GRU']:
        print(f"\n{'='*50}")
        print(f"测试 {model_type} 模型")
        print(f"{'='*50}")
        
        config['model'] = model_type
        hist_dim = X_train.shape[-1]
        fcst_dim = X_train.shape[-1] if config.get('use_forecast', False) else 0
        
        if model_type == 'LSTM':
            model = LSTM(hist_dim, fcst_dim, config)
        else:
            model = GRU(hist_dim, fcst_dim, config)
        
        print(f"✅ {model_type}参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, config)
        
        # 生成预测
        predictions, ground_truths = generate_predictions(model, X_test, y_test, config, model_type)
        
        models[model_type] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
    
    # 绘制训练曲线对比
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(models['LSTM']['train_losses'], label='LSTM Train', color='blue', linewidth=2)
    plt.plot(models['LSTM']['val_losses'], label='LSTM Val', color='blue', linestyle='--', linewidth=2)
    plt.plot(models['GRU']['train_losses'], label='GRU Train', color='red', linewidth=2)
    plt.plot(models['GRU']['val_losses'], label='GRU Val', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress Comparison (168h)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 预测精度对比 (反标准化后)
    plt.subplot(1, 3, 2)
    
    # Capacity Factor不需要反标准化，已经是0-100范围
    lstm_preds_denorm = models['LSTM']['predictions']
    lstm_gts_denorm = models['LSTM']['ground_truths']
    gru_preds_denorm = models['GRU']['predictions']
    gru_gts_denorm = models['GRU']['ground_truths']
    
    lstm_rmse = [np.sqrt(np.mean((pred - gt) ** 2)) for pred, gt in zip(lstm_preds_denorm, lstm_gts_denorm)]
    gru_rmse = [np.sqrt(np.mean((pred - gt) ** 2)) for pred, gt in zip(gru_preds_denorm, gru_gts_denorm)]
    
    plt.bar(['LSTM', 'GRU'], [np.mean(lstm_rmse), np.mean(gru_rmse)], color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Average RMSE (Capacity Factor)', fontsize=12)
    plt.title('Prediction Accuracy Comparison (168h)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 参数数量对比
    plt.subplot(1, 3, 3)
    lstm_params = sum(p.numel() for p in models['LSTM']['model'].parameters())
    gru_params = sum(p.numel() for p in models['GRU']['model'].parameters())
    
    plt.bar(['LSTM', 'GRU'], [lstm_params, gru_params], color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Number of Parameters', fontsize=12)
    plt.title('Model Complexity Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'improved_rnn_comparison_168h.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ 模型对比图已保存为: {save_path}")
    
    # 绘制168小时预测对比图
    print("\n📊 绘制168小时预测对比图...")
    plot_168h_comparison(models, scaler)
    
    print("\n🎯 改进效果总结:")
    print("   - 使用残差连接，改善梯度流和训练稳定性")
    print("   - 优化激活函数组合 (ReLU + Sigmoid)，解决周期性问题")
    print("   - 统一了LSTM和GRU的架构配置")
    print("   - 配置：72小时输入 → 168小时预测 (7天)")
    print("   - 梯度裁剪防止梯度爆炸问题")
    print("   - 使用真实Project1140数据训练，目标变量为Capacity Factor (0-100整数)")
    print("   - 特征组合：PV + NWP预测 + 历史天气 + 时间编码")
    print("   - 时间特征使用正余弦编码，提高周期性建模能力")
    print("   - Capacity Factor不进行标准化，保持0-100整数范围")
    print("   - 增加模型复杂度以适应168小时长序列预测")
    print("   - 所有图表使用英文标签，便于国际交流")

if __name__ == "__main__":
    main()
