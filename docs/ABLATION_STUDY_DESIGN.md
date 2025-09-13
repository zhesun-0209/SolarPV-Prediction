# Project1140 消融实验设计 (Controlled Ablation Study)

## 实验目标
对Capacity Factor进行建模预测，严格按照3.7 Controlled ablation study要求设计实验。

## 数据特征识别

### 1. 天气特征集 (11个特征)
选择与预测天气特征(_pred后缀)对应的历史天气特征：
1. `global_tilted_irradiance` - 全球倾斜辐射
2. `vapour_pressure_deficit` - 水汽压差  
3. `relative_humidity_2m` - 相对湿度
4. `temperature_2m` - 温度
5. `wind_gusts_10m` - 10米阵风
6. `cloud_cover_low` - 低云覆盖
7. `wind_speed_100m` - 100米风速
8. `snow_depth` - 雪深
9. `dew_point_2m` - 露点温度
10. `precipitation` - 降水
11. `surface_pressure` - 表面气压

### 2. 天气数据类型
- **Historical Weather (HW)**: 历史天气数据 (无_pred后缀)
- **Numerical Weather Prediction (NWP)**: 数值天气预报 (有_pred后缀)
- **Ideal NWP (NWP+)**: 理想NWP，用目标日HW替代预测值

### 3. 历史PV功率
- **目标变量**: `Capacity Factor`
- **输入特征**: `Capacity Factor` (历史值)

## 实验设计矩阵

### 1. 输入特征类别 (6种)
1. **PV**: 仅历史PV功率 (24/72小时)
2. **PV+NWP**: 历史PV + 目标日NWP
3. **PV+NWP+**: 历史PV + 理想NWP (目标日HW)
4. **PV+HW**: 历史PV + 历史HW
5. **NWP**: 仅目标日NWP
6. **NWP+**: 仅理想NWP (目标日HW)

### 2. 回看窗口 (2种)
- **24小时**: 过去1天数据
- **72小时**: 过去3天数据

### 3. 时间编码 (2种)
- **无TE**: 不添加时间编码
- **有TE**: 添加正弦/余弦时间编码 (小时、月份)

### 4. 模型复杂度 (2种)
- **Low (L)**: 低复杂度配置
- **High (H)**: 高复杂度配置

### 5. 机器学习方法 (8种)
1. **RF**: Random Forest
2. **XGB**: XGBoost
3. **LGBM**: LightGBM
4. **LSTM**: Long Short-Term Memory
5. **GRU**: Gated Recurrent Unit
6. **TCN**: Temporal Convolutional Network
7. **Transformer**: Transformer模型
8. **LSR**: Least Squares Regression (基线)

## 配置总数计算

### 完整配置矩阵
- 输入特征类别: 6种
- 回看窗口: 2种
- 时间编码: 2种
- 模型复杂度: 2种
- 机器学习方法: 8种

**总配置数 = 6 × 2 × 2 × 2 × 8 = 384种配置**

### LSR特殊处理
LSR作为基线模型，不区分复杂度：
- LSR配置数 = 6 × 2 × 2 × 1 × 1 = 24种配置
- 其他模型配置数 = 6 × 2 × 2 × 2 × 7 = 336种配置

**实际总配置数 = 24 + 336 = 360种配置**

## 模型复杂度设置

### Low复杂度 (L)
- **训练轮数**: 15 epochs
- **树模型** (RF, XGB, LGBM):
  - n_estimators=50, max_depth=5, learning_rate=0.1
- **深度学习模型** (LSTM, GRU, TCN, Transformer):
  - d_model=64, num_heads=4, num_layers=6, hidden_dim=32, dropout=0.1

### High复杂度 (H)
- **训练轮数**: 50 epochs
- **树模型** (RF, XGB, LGBM):
  - n_estimators=200, max_depth=12, learning_rate=0.01
- **深度学习模型** (LSTM, GRU, TCN, Transformer):
  - d_model=256, num_heads=16, num_layers=18, hidden_dim=128, dropout=0.3

## 研究问题

1. **PV-only sufficiency**: 历史PV单独预测是否足够准确？
2. **Marginal value of NWP**: NWP在多大程度上提升准确性？
3. **Upper bound of NWP**: 理想NWP的最大收益是多少？
4. **Incremental value of HW**: HW在PV基础上是否增加准确性？
5. **NWP-only sufficiency**: NWP单独预测是否足够准确？
6. **Ideal NWP-only sufficiency**: 理想NWP单独预测是否足够准确？

## 实验控制

### 数据预处理
- 固定训练/验证/测试集分割 (80%/10%/10%)
- 标准化训练调度
- 统一的数据预处理流程

### 评估指标
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (R-squared)
- MAPE (Mean Absolute Percentage Error)

## 配置文件结构

每个实验配置将生成独立的YAML配置文件，包含：
- 输入特征设置
- 回看窗口设置
- 时间编码设置
- 模型参数设置
- 训练参数设置

## 预期输出

- 360个实验配置文件
- 每个配置的详细结果 (指标、预测值、模型性能)
- 统计分析报告
- 消融效应分析图表
