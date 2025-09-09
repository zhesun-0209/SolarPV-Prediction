# SolarPV-Prediction

基于机器学习和深度学习的太阳能发电预测系统，支持多种模型架构和特征组合的消融实验。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行预测
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

## 📊 支持的模型

### 深度学习模型
- **Transformer**: 编码器-解码器结构，支持交叉注意力
- **LSTM**: 长短期记忆网络
- **GRU**: 门控循环单元
- **TCN**: 时序卷积网络

### 机器学习模型
- **Random Forest**: 随机森林
- **Gradient Boosting**: 梯度提升
- **XGBoost**: 极端梯度提升
- **LightGBM**: 轻量级梯度提升

## ⚙️ 配置选项

### 特征配置
- `use_hist_weather`: 历史天气特征 (true/false)
- `use_forecast`: 预测天气特征 (true/false)
- **时间编码特征始终包含**

### 时间窗口
- `past_days`: 历史天数 (1, 3, 7)
- `future_hours`: 预测小时数 (默认24)

### 模型复杂度
- `model_complexity`: 复杂度级别 (low, medium, high)

## 📈 评估指标

- **MSE**: 均方误差 (kWh²)
- **RMSE**: 均方根误差 (kWh)
- **MAE**: 平均绝对误差 (kWh)
- **训练时间**: 模型训练耗时
- **推理时间**: 模型推理耗时

## 💾 结果保存

结果保存在 `outputs/` 目录下，包含：
- `summary.csv`: 评估指标总结
- `predictions.csv`: 预测结果详情
- `training_log.csv`: 训练日志 (仅DL模型)
- `training_curve.png`: 训练曲线 (仅DL模型)

## 📚 详细文档

更多详细信息请参考 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

## 📄 许可证

MIT License