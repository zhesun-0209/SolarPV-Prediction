# 🚀 Colab训练指南

在Google Colab上训练SolarPV项目的所有模型

## 📁 文件说明

### 1. `colab_train_all_models.py` - 完整训练脚本
- 训练所有341个模型配置
- 支持GPU加速
- 实时显示训练进度和结果
- 自动统计成功率

### 2. `Colab_Training_Notebook.ipynb` - Jupyter Notebook版本
- 交互式训练界面
- 分步骤执行
- 可以随时查看中间结果
- 支持结果下载

### 3. `colab_quick_test.py` - 快速测试脚本
- 只测试5个关键模型
- 用于验证环境配置
- 快速检查是否有问题

## 🎯 使用方法

### 方法1: 使用Jupyter Notebook（推荐）

1. 打开Google Colab
2. 上传 `Colab_Training_Notebook.ipynb`
3. 按顺序运行每个cell
4. 等待训练完成

### 方法2: 使用Python脚本

1. 上传整个项目到Colab
2. 运行以下命令：

```bash
# 模型类型测试（推荐先运行）
python colab_test_models.py

# 快速测试（可选）
python colab_quick_test.py

# 完整训练
python colab_fixed_training.py
```

## 📊 模型配置

项目包含以下模型类型：

### 深度学习模型
- **LSTM**: 长短期记忆网络
- **GRU**: 门控循环单元
- **Transformer**: 注意力机制模型
- **TCN**: 时间卷积网络

### 机器学习模型
- **RF**: 随机森林
- **XGB**: XGBoost
- **LGBM**: LightGBM
- **LSR**: 线性回归

### 复杂度级别
- **Low**: 简单配置，训练快速
- **High**: 复杂配置，性能更好

## ⚙️ 配置参数

### 树模型参数
- **Low**: `n_estimators=50, max_depth=5`
- **High**: `n_estimators=200, max_depth=12`

### DL模型参数
- **Low**: `d_model=64, num_heads=4, num_layers=6, hidden_dim=32, dropout=0.1`
- **High**: `d_model=256, num_heads=16, num_layers=18, hidden_dim=128, dropout=0.3`

## 📈 预期结果

### 训练时间估算
- 单个模型: 5-15分钟
- 全部模型: 20-40小时
- 建议分批训练

### 输出结果
- 模型性能指标（MSE, RMSE, MAE等）
- 训练日志
- 预测结果
- 模型权重（可选）

## 🔧 故障排除

### 常见问题

1. **GPU内存不足**
   - 减少batch_size
   - 使用CPU训练
   - 选择low复杂度模型

2. **训练超时**
   - 增加超时时间
   - 分批训练
   - 跳过复杂模型

3. **依赖包问题**
   - 重新安装requirements.txt
   - 检查Python版本

### 调试建议

1. 先运行快速测试
2. 检查单个模型训练
3. 查看错误日志
4. 调整配置参数

## 📁 结果文件

训练完成后，结果保存在：
```
temp_results/1140/
├── LSTM_low_PV_24h_TE/
├── Transformer_high_PV_72h_TE/
├── RF_low_PV_24h_TE/
└── ...
```

每个模型目录包含：
- 训练指标
- 预测结果
- 模型权重（如果保存）
- 训练日志

## 🎉 成功标志

训练成功的标志：
- 控制台显示 "✅ 训练成功完成!"
- 输出性能指标（mse=, rmse=, mae=）
- 生成结果文件
- 返回码为0

## 📞 支持

如果遇到问题：
1. 检查Colab GPU配额
2. 查看错误日志
3. 尝试简化配置
4. 联系技术支持
