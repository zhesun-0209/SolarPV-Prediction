# Colab使用指南

## 在Google Colab中运行多Plant预测结果保存

### 步骤1：设置环境

在Colab notebook中运行以下命令：

```python
# 安装依赖
!pip install -q pyyaml pandas numpy scikit-learn xgboost lightgbm

# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 克隆仓库
!git clone https://github.com/zhesun-0209/SolarPV-Prediction.git /content/SolarPV-Prediction
```

### 步骤2：生成配置文件

```python
# 生成所有Plant的配置文件
!python /content/SolarPV-Prediction/generate_multi_plant_configs.py
```

### 步骤3：运行预测结果保存

```python
# 运行多Plant预测结果保存脚本
!python /content/SolarPV-Prediction/colab_multi_plant_predictions.py
```

## 预期结果

脚本将：
1. 为Plant 171、172、186训练所有340个模型配置
2. 将结果保存到Google Drive的 `Solar PV electricity/plot/` 目录
3. 每个Plant一个文件夹，包含汇总文件和按场景/模型分组的结果

## 存储结构

```
/content/drive/MyDrive/Solar PV electricity/plot/
├── Plant_171/
│   ├── Plant_171_all_predictions_summary.csv
│   ├── Plant_171_experiment_configs.csv
│   ├── by_scenario/
│   │   ├── PV_predictions.csv
│   │   ├── PV+NWP_predictions.csv
│   │   ├── PV+NWP+_predictions.csv
│   │   ├── PV+HW_predictions.csv
│   │   ├── NWP_predictions.csv
│   │   └── NWP+_predictions.csv
│   └── by_model/
│       ├── LSTM_predictions.csv
│       ├── GRU_predictions.csv
│       ├── TCN_predictions.csv
│       ├── Transformer_predictions.csv
│       ├── RF_predictions.csv
│       ├── XGB_predictions.csv
│       ├── LGBM_predictions.csv
│       └── Linear_predictions.csv
├── Plant_172/
│   └── ... (相同结构)
└── Plant_186/
    └── ... (相同结构)
```

## 注意事项

1. **数据文件**：确保 `data/Project171.csv`, `data/Project172.csv`, `data/Project186.csv` 存在
2. **运行时间**：完整运行可能需要数小时（340个配置 × 3个Plant = 1020个实验）
3. **存储空间**：确保Google Drive有足够空间
4. **内存**：建议使用Colab Pro的高内存实例

## 故障排除

如果遇到问题：
1. 检查数据文件是否存在
2. 确保Google Drive已正确挂载
3. 检查Colab实例是否有足够的内存和存储空间
4. 查看控制台输出了解进度和错误信息
