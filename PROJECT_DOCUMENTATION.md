# SolarPV-Prediction 项目文档

## 📋 项目概述

SolarPV-Prediction 是一个基于机器学习和深度学习的太阳能发电预测系统，支持多种模型架构和特征组合的消融实验。

### 🎯 核心功能
- **多模型支持**: Transformer, LSTM, GRU, TCN (DL) + RF, GBR, XGB, LGBM (ML)
- **特征工程**: 历史天气、预测天气、时间编码特征
- **消融实验**: 支持不同特征组合、时间窗口、模型复杂度的系统测试
- **统一评估**: 提供公平的模型比较框架

## 🏗️ 项目结构

```
SolarPV-Prediction/
├── config/
│   └── default.yaml          # 配置文件
├── data/
│   ├── data_utils.py         # 数据处理工具
│   └── Project1033.csv       # 示例数据
├── models/
│   ├── transformer.py        # Transformer模型
│   ├── rnn_models.py         # LSTM/GRU模型
│   ├── tcn.py               # TCN模型
│   └── ml_models.py         # 机器学习模型
├── train/
│   ├── train_dl.py          # 深度学习训练
│   ├── train_ml.py          # 机器学习训练
│   └── train_utils.py       # 训练工具
├── eval/
│   ├── eval_utils.py        # 评估工具
│   └── plot_utils.py        # 绘图工具
├── main.py                  # 主程序
├── requirements.txt         # 依赖包
└── README.md               # 项目说明
```

## ⚙️ 配置说明

### 特征配置
- `use_hist_weather`: 是否使用历史天气特征 (true/false)
- `use_forecast`: 是否使用预测天气特征 (true/false)
- **注意**: 时间编码特征始终包含，不再设置开关

### 时间窗口配置
- `past_days`: 历史天数 (1, 3, 7)
- `future_hours`: 预测小时数 (默认24)

### 模型复杂度配置
- `model_complexity`: 模型复杂度 (low, medium, high)
- 影响模型参数数量和计算复杂度

## 🚀 使用方法

### 基本使用
```bash
python main.py --config config/default.yaml
```

### 消融实验
```bash
# 模型对比
python main.py --config config/default.yaml --model Transformer --use_hist_weather true --use_forecast true

# 特征消融
python main.py --config config/default.yaml --model Transformer --use_hist_weather false --use_forecast true

# 时间窗口测试
python main.py --config config/default.yaml --model Transformer --past_days 7

# 复杂度测试
python main.py --config config/default.yaml --model Transformer --model_complexity high
```

## 📊 模型架构

### 深度学习模型

#### Transformer
- **架构**: 编码器-解码器结构，支持交叉注意力
- **特征融合**: 历史特征和预测特征通过交叉注意力机制融合
- **优势**: 强大的序列建模能力

#### RNN模型 (LSTM/GRU)
- **架构**: 双分支RNN + 融合层
- **特征融合**: 历史特征和预测特征分别通过RNN处理，然后融合
- **优势**: 适合时序数据，计算效率高

#### TCN模型
- **架构**: 时序卷积网络 + 简单融合
- **特征融合**: 历史特征通过TCN处理，预测特征通过线性投影融合
- **优势**: 并行计算，训练速度快

### 机器学习模型

#### 特征工程
- **历史特征**: 展平为 `(batch_size, past_hours * hist_dim)`
- **预测特征**: 展平为 `(batch_size, future_hours * fcst_dim)`
- **融合方式**: 简单拼接 `np.concatenate([h, f], axis=1)`

#### 支持模型
- **Random Forest**: 随机森林
- **Gradient Boosting**: 梯度提升
- **XGBoost**: 极端梯度提升
- **LightGBM**: 轻量级梯度提升

## 📈 评估指标

### 主要评估指标 (原始尺度 - kWh)
- **MSE**: 均方误差 (kWh²)
- **RMSE**: 均方根误差 (kWh)
- **MAE**: 平均绝对误差 (kWh)

### 辅助评估指标 (标准化尺度 - 0-1)
- **Norm MSE**: 标准化均方误差
- **Norm RMSE**: 标准化均方根误差
- **Norm MAE**: 标准化平均绝对误差

### 性能指标
- **训练时间**: 模型训练耗时 (秒)
- **推理时间**: 模型推理耗时 (秒)
- **参数数量**: 模型参数总数

## 💾 结果保存

### 保存文件
- `summary.csv`: 评估指标总结
- `predictions.csv`: 预测结果详情
- `training_log.csv`: 训练日志 (仅DL模型)
- `training_curve.png`: 训练曲线 (仅DL模型)

### 可选保存
- `model.pth`: 模型文件 (默认不保存)
- `forecast_plots.png`: 预测对比图 (默认不保存)
- `val_loss_over_time.png`: 验证损失图 (默认不保存)

### 目录结构
```
outputs/
├── Project_1033/
│   ├── dl/
│   │   ├── transformer/
│   │   │   └── feat{bool}_fcst{bool}_days{1,3,7}_comp{low,medium,high}/
│   │   │       ├── summary.csv
│   │   │       ├── predictions.csv
│   │   │       ├── training_log.csv
│   │   │       └── training_curve.png
│   │   ├── lstm/
│   │   ├── gru/
│   │   └── tcn/
│   └── ml/
│       ├── rf/
│       ├── gbr/
│       ├── xgb/
│       └── lgbm/
```

## 🔬 消融实验设计

### 配置组合
- **特征组合**: 4种 (hist_weather × forecast)
- **时间窗口**: 3种 (past_days: 1, 3, 7)
- **模型复杂度**: 3种 (low, medium, high)
- **模型类型**: 8种 (4个DL + 4个ML)

### 总组合数
- **完整测试**: 4 × 3 × 3 × 8 = **288种组合**
- **单模型测试**: 4 × 3 = **12种组合**
- **模型对比**: **8种组合**

### 推荐测试策略

#### 阶段1: 模型对比 (8种组合)
- 选择medium复杂度
- 选择全部特征开启
- 选择past_days=3
- 测试所有8个模型

#### 阶段2: 特征消融 (4种组合)
- 选择最佳模型
- 选择medium复杂度
- 选择past_days=3
- 测试所有特征组合

#### 阶段3: 时间窗口优化 (3种组合)
- 选择最佳模型和特征组合
- 选择medium复杂度
- 测试所有时间窗口组合

#### 阶段4: 复杂度优化 (3种组合)
- 选择最佳配置
- 测试low/medium/high复杂度

## 📊 存储需求

### 存储空间估算
- **最小测试**: ~2MB (默认文件) / ~80MB (含模型)
- **中等测试**: ~2.5MB (默认文件) / ~120MB (含模型)
- **完整测试**: ~60MB (默认文件) / ~2.9GB (含模型)

### 文件大小
- `summary.csv`: ~1KB
- `predictions.csv`: ~100KB-1MB
- `training_log.csv`: ~10KB-100KB
- `training_curve.png`: ~50KB-200KB
- `model.pth`: ~1MB-100MB

## 🎯 模型统一设计

### 特征处理统一
- **DL模型**: 保持时序结构，使用简单融合方式
- **ML模型**: 使用简单的特征展平和拼接
- **时间编码**: 始终包含，确保所有模型都能利用时间信息

### 计算时间可比
- **统一时间测量**: 训练时间和推理时间
- **简化架构**: 避免过于复杂的机制影响时间比较
- **合理复杂度**: 适中的参数数量和特征数量

### 公平比较
- **特征一致性**: 所有模型使用相同的输入特征
- **复杂度相当**: 避免过于复杂的机制
- **时间可比**: 统一的时间测量方法

## 🔧 技术细节

### 数据预处理
1. **数据加载**: 支持CSV格式的合并数据
2. **特征工程**: 自动添加时间编码特征
3. **数据清洗**: 处理缺失值和异常值
4. **标准化**: MinMax标准化所有特征

### 模型训练
1. **数据分割**: 训练集(80%) + 验证集(10%) + 测试集(10%)
2. **早停机制**: 防止过拟合
3. **学习率调度**: 自适应学习率调整
4. **批量训练**: 支持批量梯度下降

### 模型评估
1. **多指标评估**: MSE, RMSE, MAE
2. **时间测量**: 训练和推理时间
3. **可视化**: 训练曲线和预测结果
4. **结果保存**: 结构化的结果保存

## 📝 使用示例

### 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行基本预测
python main.py --config config/default.yaml --model Transformer

# 3. 运行消融实验
python main.py --config config/default.yaml --model Transformer --use_hist_weather true --use_forecast true --past_days 3 --model_complexity medium
```

### 配置自定义
```yaml
# config/default.yaml
model: 'Transformer'
past_days: 3
future_hours: 24
use_hist_weather: true
use_forecast: true
model_complexity: 'medium'
```

## 🚀 性能优化

### 计算优化
- **并行处理**: 支持多进程训练
- **内存优化**: 批量数据处理
- **GPU加速**: 支持CUDA加速

### 存储优化
- **选择性保存**: 可配置保存内容
- **压缩存储**: 模型文件压缩
- **定期清理**: 自动清理临时文件

## 📚 依赖包

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
lightgbm>=3.2.0
pyyaml>=5.4.0
```

## 🤝 贡献指南

1. **代码规范**: 遵循PEP 8规范
2. **文档更新**: 及时更新相关文档
3. **测试验证**: 确保代码正确性
4. **性能优化**: 关注计算效率

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues: [GitHub Issues]
- 邮箱: [your-email@example.com]

---

**最后更新**: 2024年9月8日
**版本**: v1.0.0
