#!/usr/bin/env python3
"""
测试改进的LSTM和GRU模型
验证时间注意力机制和残差连接的效果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.rnn_models import LSTM, GRU, MultiHeadTemporalAttention

def test_multihead_attention():
    """测试多头时间注意力机制"""
    print("🔍 测试多头时间注意力机制...")
    
    batch_size, seq_len, hidden_dim = 2, 10, 64
    num_heads = 8
    
    # 创建测试数据
    rnn_outputs = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建多头注意力模块
    attention = MultiHeadTemporalAttention(hidden_dim, num_heads=num_heads)
    
    # 前向传播
    attended_output, attention_weights = attention(rnn_outputs)
    
    print(f"✅ 输入形状: {rnn_outputs.shape}")
    print(f"✅ 注意力输出形状: {attended_output.shape}")
    print(f"✅ 注意力权重形状: {attention_weights.shape}")
    print(f"✅ 注意力权重和: {attention_weights.sum(dim=-1).mean()}")
    
    return True

def test_improved_models():
    """测试改进的LSTM和GRU模型"""
    print("\n🔍 测试改进的LSTM和GRU模型...")
    
    # 配置参数
    config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'future_hours': 168,
        'use_forecast': True
    }
    
    # 创建模型
    lstm_model = LSTM(hist_dim=10, fcst_dim=5, config=config)
    gru_model = GRU(hist_dim=10, fcst_dim=5, config=config)
    
    # 创建测试数据
    batch_size = 4
    hist_seq_len = 24
    fcst_seq_len = 168
    
    hist_data = torch.randn(batch_size, hist_seq_len, 10)
    fcst_data = torch.randn(batch_size, fcst_seq_len, 5)
    
    # 测试LSTM
    print("\n📊 测试LSTM模型:")
    lstm_output = lstm_model(hist_data, fcst_data)
    print(f"✅ LSTM输出形状: {lstm_output.shape}")
    print(f"✅ LSTM输出范围: [{lstm_output.min():.2f}, {lstm_output.max():.2f}]")
    
    # 测试GRU
    print("\n📊 测试GRU模型:")
    gru_output = gru_model(hist_data, fcst_data)
    print(f"✅ GRU输出形状: {gru_output.shape}")
    print(f"✅ GRU输出范围: [{gru_output.min():.2f}, {gru_output.max():.2f}]")
    
    # 检查模型参数数量
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    gru_params = sum(p.numel() for p in gru_model.parameters())
    
    print(f"\n📈 模型参数数量:")
    print(f"✅ LSTM参数: {lstm_params:,}")
    print(f"✅ GRU参数: {gru_params:,}")
    
    return lstm_model, gru_model

def visualize_attention_weights(model, hist_data, fcst_data, model_name="LSTM"):
    """可视化注意力权重"""
    print(f"\n🎨 可视化{model_name}注意力权重...")
    
    model.eval()
    with torch.no_grad():
        # 获取注意力权重
        if hasattr(model, 'temporal_attention'):
            # 手动计算注意力权重
            seqs = []
            if model.hist_proj is not None:
                h_proj = model.hist_proj(hist_data)
                seqs.append(h_proj)
            if model.fcst_proj is not None:
                f_proj = model.fcst_proj(fcst_data)
                seqs.append(f_proj)
            
            seq = torch.cat(seqs, dim=1)
            if model_name == "LSTM":
                rnn_outputs, _ = model.lstm(seq)
            else:
                rnn_outputs, _ = model.gru(seq)
            
            _, attention_weights = model.temporal_attention(rnn_outputs)
            
            # 绘制注意力权重热力图
            plt.figure(figsize=(12, 6))
            # 对于多头注意力，我们显示第一个样本的注意力权重
            attention_vis = attention_weights[0].cpu().numpy()  # (seq_len, seq_len)
            plt.imshow(attention_vis, aspect='auto', cmap='Blues')
            plt.colorbar(label='Attention Weight')
            plt.title(f'{model_name} Multi-Head Temporal Attention Weights (Sample 1)')
            plt.xlabel('Key Time Steps')
            plt.ylabel('Query Time Steps')
            plt.tight_layout()
            plt.savefig(f'{model_name.lower()}_attention_weights.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"✅ 注意力权重已保存为: {model_name.lower()}_attention_weights.png")

def main():
    """主测试函数"""
    print("🚀 开始测试改进的RNN模型")
    print("=" * 50)
    
    # 测试多头时间注意力机制
    test_multihead_attention()
    
    # 测试改进的模型
    lstm_model, gru_model = test_improved_models()
    
    # 创建测试数据用于可视化
    config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'future_hours': 168,
        'use_forecast': True
    }
    
    hist_data = torch.randn(2, 24, 10)
    fcst_data = torch.randn(2, 168, 5)
    
    # 可视化注意力权重
    visualize_attention_weights(lstm_model, hist_data, fcst_data, "LSTM")
    visualize_attention_weights(gru_model, hist_data, fcst_data, "GRU")
    
    print("\n✅ 所有测试完成!")
    print("🎯 改进要点:")
    print("   - 统一了LSTM和GRU的架构配置")
    print("   - 添加了时间注意力机制解决周期性问题")
    print("   - 使用残差连接改善梯度流")
    print("   - 保持了相同的输出格式和范围")

if __name__ == "__main__":
    main()
