# 全面修复指南

## 🚨 当前问题分析

从你的调试输出可以看出，主要问题是：

1. **LSR模型不支持**: `Unsupported ML model: LSR`
2. **调试代码参数不匹配**: 仍然使用旧的函数调用方式
3. **配置文件未更新**: Colab中使用的是旧的配置文件

## 🔧 立即修复步骤

### 步骤1: 修复现有配置文件（重要！）

```bash
# 在Colab中运行，修复所有LSR配置为Linear
python fix_existing_configs.py
```

这个脚本会：
- 自动将所有 `model: LSR` 改为 `model: Linear`
- 添加缺失的配置参数（past_hours, future_hours等）
- 验证修复结果

### 步骤2: 运行全面模型测试

```bash
# 测试所有DL和ML模型
python test_all_models.py
```

这个脚本会：
- 测试所有模型类型（Linear, RF, XGB, LGBM, LSTM, GRU, TCN, Transformer）
- 验证结果保存功能
- 测试完整实验流程
- 提供详细的错误诊断

### 步骤3: 验证单个实验

```bash
# 测试修复后的LSR配置
python main.py --config config/projects/1140/LSR_low_PV_24h_noTE.yaml
```

### 步骤4: 检查结果文件

```python
import pandas as pd
from pathlib import Path

# 检查实验结果
exp_dir = Path("temp_results/1140/LSR_low_PV_24h_noTE")
if exp_dir.exists():
    files = list(exp_dir.glob("*"))
    print(f"文件列表: {[f.name for f in files]}")
    
    # 检查Excel文件
    excel_files = list(exp_dir.glob("*.xlsx"))
    if excel_files:
        df = pd.read_excel(excel_files[0])
        print(f"Excel内容:")
        print(df.head())
        print(f"列名: {list(df.columns)}")
        
        # 检查关键指标
        key_metrics = ['mae', 'rmse', 'r2', 'mape', 'train_time_sec']
        for metric in key_metrics:
            if metric in df.columns:
                value = df[metric].iloc[0]
                print(f"{metric}: {value}")
            else:
                print(f"缺少指标: {metric}")
```

## 📊 预期结果

修复后应该看到：

### 1. 配置文件修复结果
```
✅ 修复完成!
   总配置文件: 36000
   修复文件数: 4500  (所有LSR配置)
   无需修复: 31500

📊 验证结果:
   LSR配置: 0
   Linear配置: 4500
✅ 所有LSR配置已成功修复为Linear
```

### 2. 全面模型测试结果
```
📋 模型类型测试结果:
  ML模型:
    Linear: ✅ 通过
    RF: ✅ 通过
    XGB: ✅ 通过
    LGBM: ✅ 通过
  DL模型:
    LSTM: ✅ 通过
    GRU: ✅ 通过
    TCN: ✅ 通过
    Transformer: ✅ 通过

📊 结果保存测试: ✅ 通过
🔄 完整流程测试结果:
  Linear_low: ✅ 通过
  RF_low: ✅ 通过
  LSTM_low: ✅ 通过

📊 总体结果: 12/12 测试通过
🎉 所有测试通过！系统可以正常运行实验
```

### 3. 单个实验结果
```
✅ 实验运行成功
📁 结果文件: ['results.xlsx', 'predictions.csv', 'training_log.csv']
📊 Excel内容: (1, 15), 列: ['mae', 'rmse', 'r2', 'mape', 'train_time_sec', ...]
✅ mae: 0.1234
✅ rmse: 0.2345
✅ r2: 0.5678
✅ mape: 12.34
✅ train_time_sec: 5.67
```

## 🚀 重新运行完整实验

修复完成后，可以重新运行完整实验：

```bash
# 重新运行GPU实验
python scripts/run_gpu_only_experiments.py
```

## 🔍 如果仍有问题

如果修复后仍有问题，请检查：

1. **配置修复是否成功**:
   ```bash
   python -c "
   import yaml
   with open('config/projects/1140/LSR_low_PV_24h_noTE.yaml', 'r') as f:
       config = yaml.safe_load(f)
   print(f'模型: {config.get(\"model\")}')
   print(f'past_hours: {config.get(\"past_hours\")}')
   "
   ```

2. **依赖包是否完整**:
   ```python
   import xgboost, lightgbm, sklearn, torch
   print("所有依赖包可用")
   ```

3. **数据文件是否正常**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/Project1140.csv')
   print(f"数据形状: {df.shape}")
   print(f"目标列: {'Capacity Factor' in df.columns}")
   ```

## 📞 问题反馈

如果遇到问题，请提供：
- `fix_existing_configs.py` 的输出
- `test_all_models.py` 的输出
- 具体的错误信息
- 配置文件示例

## 🎯 成功标志

修复成功的标志：
- ✅ 所有LSR配置修复为Linear
- ✅ 所有模型测试通过
- ✅ 性能指标正确保存
- ✅ 实验结果包含完整信息
- ✅ Drive保存正常工作
