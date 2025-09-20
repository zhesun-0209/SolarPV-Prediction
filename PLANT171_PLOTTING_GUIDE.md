# 厂171模型对比绘图指南

## 概述
这个脚本用于绘制厂171的8个模型在6种情况下的预测图，支持不同的参数组合。

## 功能特点
- **8个模型**: LSTM, GRU, TCN, Transformer, RF, XGB, LGBM, Linear
- **6种情况**: PV, PV+NWP, PV+NWP+, PV+HW, NWP, NWP+
- **参数组合**: 
  - Lookback: 24h, 72h
  - Time Encoding: True, False
  - Complexity: low, high
- **输出**: 2×4子图布局，每个子图显示1个模型的ground truth + 6种情况预测

## 文件结构
```
config/projects/171/          # 厂171的配置文件
├── LSTM_low_PV_72h_noTE.yaml
├── LSTM_high_PV_plus_NWP_24h_TE.yaml
├── ... (384个配置文件)
└── ...

plot_plant171_models.py       # 主绘制脚本
```

## 使用方法

### 在Colab中运行
1. 上传整个项目到Colab
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行脚本：
   ```bash
   python plot_plant171_models.py
   ```

### 输出结果
- 图片保存在 `plant171_comparison_plots/` 目录
- 文件名格式: `plant_171_models_{lookback}h_{TE/noTE}_{complexity}.png`
- 总共生成8张图片（2×2×2=8种参数组合）

## 图片说明
每张图片包含：
- **2行4列**子图布局
- 每个子图代表一个模型
- 每个子图包含7条线：
  - 1条灰色线：Ground Truth
  - 6条彩色线：6种情况的预测结果
- 图例显示不同情况的颜色对应关系

## 参数组合
| Lookback | Time Encoding | Complexity | 输出文件名 |
|----------|---------------|------------|------------|
| 24h      | noTE         | low        | plant_171_models_24h_noTE_low.png |
| 24h      | noTE         | high       | plant_171_models_24h_noTE_high.png |
| 24h      | TE           | low        | plant_171_models_24h_TE_low.png |
| 24h      | TE           | high       | plant_171_models_24h_TE_high.png |
| 72h      | noTE         | low        | plant_171_models_72h_noTE_low.png |
| 72h      | noTE         | high       | plant_171_models_72h_noTE_high.png |
| 72h      | TE           | low        | plant_171_models_72h_TE_low.png |
| 72h      | TE           | high       | plant_171_models_72h_TE_high.png |

## 注意事项
1. 确保数据文件 `data/Project1140.csv` 存在
2. 确保所有配置文件都已生成
3. 脚本会自动创建输出目录
4. 训练过程可能需要较长时间，请耐心等待
5. 如果某个模型训练失败，会跳过该模型但继续处理其他模型

## 故障排除
- 如果出现内存不足，可以减少batch_size
- 如果某个模型训练失败，检查对应的配置文件是否正确
- 如果图片显示异常，检查matplotlib版本和字体设置
