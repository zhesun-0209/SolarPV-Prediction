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
from tqdm import tqdm
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
    
    # 目标变量 - Capacity Factor (0-100范围，不标准化)
    if 'Capacity Factor' in df.columns:
        feature_cols.append('Capacity Factor')
        print("✅ 目标变量: Capacity Factor (范围0-100)")
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
    print(f"✅ Capacity Factor范围: {data[:, 0].min():.2f} - {data[:, 0].max():.2f}")
    
    return data

def prepare_sequences(data, past_hours=72, future_hours=24):
    """准备序列数据，Capacity Factor作为目标变量，按照colab_batch_experiments配置"""
    print("🔧 准备序列数据...")
    print(f"📊 输入长度: {past_hours}小时, 预测长度: {future_hours}小时")
    
    # 分离目标变量和特征
    capacity_factor = data[:, 0:1]  # Capacity Factor (0-100范围，不标准化)
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
    print(f"✅ 目标变量范围 (Capacity Factor): {y.min():.2f} - {y.max():.2f}")
    
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
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
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
    
    # 创建epoch进度条
    epoch_pbar = tqdm(range(config['epochs']), desc="训练进度", unit="epoch")
    
    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 创建batch进度条
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", 
                         leave=False, unit="batch")
        
        for batch_X, batch_y in batch_pbar:
            optimizer.zero_grad()
            
            # 分离历史数据和预测数据
            hist_data = batch_X[:, :config['past_hours']]  # 历史数据
            fcst_data = batch_X[:, config['past_hours']:]  # 预测数据
            
            # 前向传播
            outputs = model(hist_data, fcst_data)
            loss = criterion(outputs, batch_y)
            
            # 检查NaN
            if torch.isnan(loss):
                tqdm.write(f"❌ Epoch {epoch+1} 出现NaN损失，跳过此批次")
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item()
            
            # 更新batch进度条
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{train_loss/(batch_pbar.n+1):.6f}'
            })
        
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
        
        # 更新epoch进度条
        current_lr = optimizer.param_groups[0]['lr']
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.6f}',
            'Val Loss': f'{val_loss:.6f}',
            'LR': f'{current_lr:.2e}',
            'Patience': f'{patience_counter}/{config["patience"]}'
        })
        
        # 详细训练进程输出
        if epoch % 5 == 0 or epoch == config['epochs'] - 1:
            tqdm.write(f"Epoch {epoch+1:3d}/{config['epochs']:3d} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Patience: {patience_counter}/{config['patience']}")
        
        if patience_counter >= config['patience']:
            tqdm.write(f"🛑 早停于第 {epoch+1} 轮")
            break
    
    # 关闭进度条
    epoch_pbar.close()
    
    print("=" * 60)
    print(f"✅ {config['model']}训练完成!")
    print(f"📈 最佳验证损失: {best_val_loss:.6f}")
    print(f"📊 最终训练损失: {train_losses[-1]:.6f}")
    print(f"📊 最终验证损失: {val_losses[-1]:.6f}")
    print("=" * 60)
    return train_losses, val_losses

def generate_predictions(model, X_test, y_test, config, model_name):
    """生成预测并可视化"""
    print(f"🎨 生成{model_name}的168小时预测...")
    
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
            
            predictions.append(pred_np)
            ground_truths.append(y_test[idx])
    
    # 创建对比图
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        hours = range(1, len(pred) + 1)
        
        axes[i].plot(hours, gt, 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
        axes[i].plot(hours, pred, 'r--', label=f'Improved {model_name} Prediction', linewidth=2, alpha=0.8)
        
        axes[i].set_xlabel('Hours Ahead')
        axes[i].set_ylabel('PV Power')
        axes[i].set_title(f'Sample {i+1}: 168-Hour PV Power Prediction - {model_name}')
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
    plt.savefig(f'improved_{model_name.lower()}_168h_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ {model_name}预测对比图已保存")
    
    return predictions, ground_truths

def plot_24h_comparison(models, scaler):
    """绘制24小时预测对比图"""
    print("📊 绘制24小时预测对比图...")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 选择第一个测试样本进行可视化
    sample_idx = 0
    lstm_pred = models['LSTM']['predictions'][sample_idx]
    gru_pred = models['GRU']['predictions'][sample_idx]
    y_true = models['LSTM']['ground_truths'][sample_idx]
    
    # Capacity Factor不需要反标准化，已经是0-100范围
    lstm_pred_denorm = lstm_pred
    gru_pred_denorm = gru_pred
    y_true_denorm = y_true
    
    # 绘制预测结果
    time_steps = range(24)
    axes[0].plot(time_steps, y_true_denorm, 'b-', label='真实值 (Capacity Factor)', linewidth=2)
    axes[0].plot(time_steps, lstm_pred_denorm, 'r--', label='LSTM预测', linewidth=2)
    axes[0].plot(time_steps, gru_pred_denorm, 'g--', label='GRU预测', linewidth=2)
    
    axes[0].set_title('LSTM vs GRU 预测对比 (24小时) - Capacity Factor', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('时间 (小时)')
    axes[0].set_ylabel('Capacity Factor (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制误差对比
    lstm_error = np.abs(lstm_pred_denorm - y_true_denorm)
    gru_error = np.abs(gru_pred_denorm - y_true_denorm)
    
    axes[1].plot(time_steps, lstm_error, 'r-', label='LSTM误差', linewidth=2)
    axes[1].plot(time_steps, gru_error, 'g-', label='GRU误差', linewidth=2)
    
    axes[1].set_title('预测误差对比 (Capacity Factor)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('时间 (小时)')
    axes[1].set_ylabel('绝对误差 (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_lstm_gru_comparison_24h.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 24小时预测对比图已保存为: improved_lstm_gru_comparison_24h.png")

def main():
    """主函数"""
    print("🚀 测试改进的RNN模型 - 24小时预测 (按照colab_batch_experiments配置)")
    print("=" * 70)
    
    # 创建配置 - 参考colab_batch_experiments的low复杂度设置
    config = {
        'model': 'LSTM',
        'hidden_dim': 32,        # low: 32
        'num_layers': 6,         # low: 6
        'dropout': 0.1,          # low: 0.1
        'd_model': 64,           # low: 64
        'num_heads': 4,          # low: 4
        'future_hours': 24,      # 24小时预测 (按照colab配置)
        'past_hours': 72,        # 72小时历史 (按照colab配置)
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
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(models['LSTM']['train_losses'], label='LSTM Train', color='blue')
    plt.plot(models['LSTM']['val_losses'], label='LSTM Val', color='blue', linestyle='--')
    plt.plot(models['GRU']['train_losses'], label='GRU Train', color='red')
    plt.plot(models['GRU']['val_losses'], label='GRU Val', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress Comparison')
    plt.legend()
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
    plt.ylabel('Average RMSE (Capacity Factor)')
    plt.title('Prediction Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    
    # 参数数量对比
    plt.subplot(1, 3, 3)
    lstm_params = sum(p.numel() for p in models['LSTM']['model'].parameters())
    gru_params = sum(p.numel() for p in models['GRU']['model'].parameters())
    
    plt.bar(['LSTM', 'GRU'], [lstm_params, gru_params], color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Number of Parameters')
    plt.title('Model Complexity Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_rnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制24小时预测对比图
    print("\n📊 绘制24小时预测对比图...")
    plot_24h_comparison(models, scaler)
    
    print("\n🎯 改进效果总结:")
    print("   - 使用残差连接，改善梯度流和训练稳定性")
    print("   - 优化激活函数组合 (ReLU + Sigmoid)，解决周期性问题")
    print("   - 统一了LSTM和GRU的架构配置")
    print("   - 按照colab_batch_experiments配置：72小时输入 → 24小时预测")
    print("   - 梯度裁剪防止梯度爆炸问题")
    print("   - 使用真实Project1140数据训练，目标变量为Capacity Factor (0-100%)")
    print("   - 特征组合：PV + NWP预测 + 历史天气 + 时间编码")
    print("   - 时间特征使用正余弦编码，提高周期性建模能力")
    print("   - Capacity Factor不进行标准化，保持0-100范围")

if __name__ == "__main__":
    main()
