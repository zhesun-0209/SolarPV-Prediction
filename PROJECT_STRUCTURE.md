# Project1140 光伏发电预测消融实验项目结构

## 📁 目录结构

```
SolarPV-Prediction/
├── README.md                           # 项目主要说明文档
├── PROJECT_STRUCTURE.md                # 项目结构说明（本文件）
├── requirements.txt                    # Python依赖包
├── main.py                            # 主程序入口
│
├── config/                            # 配置文件目录
│   └── ablation/                      # 消融实验配置文件
│       ├── config_index.yaml          # 配置索引文件
│       └── [360个配置文件].yaml        # 360个实验配置
│
├── data/                              # 数据目录
│   └── Project1140.csv                # 主要数据文件
│
├── scripts/                           # 脚本目录
│   ├── generate_ablation_configs.py   # 配置生成脚本
│   ├── run_ablation_experiments.py    # 实验运行脚本
│   └── test_ablation_configs.py       # 配置测试脚本
│
├── docs/                              # 文档目录
│   ├── ABLATION_STUDY_DESIGN.md       # 消融实验设计文档
│   └── PROJECT1140_ABLATION_EXPERIMENT_SUMMARY.md # 实验总结文档
│
├── models/                            # 模型定义
│   ├── ml_models.py                   # 机器学习模型
│   ├── rnn_models.py                  # RNN模型
│   ├── tcn.py                         # TCN模型
│   └── transformer.py                 # Transformer模型
│
├── train/                             # 训练模块
│   ├── train_dl.py                    # 深度学习训练
│   ├── train_ml.py                    # 机器学习训练
│   └── train_utils.py                 # 训练工具函数
│
├── eval/                              # 评估模块
│   ├── eval_utils.py                  # 评估工具
│   ├── excel_utils.py                 # Excel处理工具
│   └── metrics_utils.py               # 指标计算工具
│
├── utils/                             # 工具模块
│   └── gpu_utils.py                   # GPU工具
│
└── results/                           # 结果目录（运行时生成）
    └── ablation/                      # 消融实验结果
```

## 🎯 核心组件说明

### 1. 主程序 (`main.py`)
- 项目的主要入口点
- 负责协调整个训练和评估流程
- 支持命令行参数配置

### 2. 配置文件 (`config/ablation/`)
- 包含360个消融实验配置
- 每个配置文件对应一个特定的实验设置
- 配置文件包含模型参数、数据设置、训练参数等

### 3. 数据文件 (`data/Project1140.csv`)
- 主要的数据文件
- 包含2022-2024年的光伏发电和天气数据
- 24,028条记录，56个特征

### 4. 脚本文件 (`scripts/`)
- `generate_ablation_configs.py`: 生成所有实验配置
- `run_ablation_experiments.py`: 运行消融实验
- `test_ablation_configs.py`: 测试配置完整性

### 5. 模型模块 (`models/`)
- 包含所有支持的模型实现
- 支持传统机器学习模型和深度学习模型
- 统一的模型接口

### 6. 训练模块 (`train/`)
- 包含训练逻辑和工具函数
- 支持不同模型的训练流程
- 统一的训练接口

### 7. 评估模块 (`eval/`)
- 包含评估指标计算
- 结果保存和可视化
- Excel报告生成

## 🚀 使用流程

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 生成配置（如果需要）
```bash
python scripts/generate_ablation_configs.py
```

### 3. 测试配置
```bash
python scripts/test_ablation_configs.py
```

### 4. 运行实验
```bash
# 运行单个实验
python main.py --config config/ablation/LSR_baseline_PV_24h_noTE.yaml

# 运行所有消融实验
python scripts/run_ablation_experiments.py
```

## 📊 实验设计

- **总配置数**: 360个
- **输入类别**: 6种 (PV, PV+NWP, PV+NWP+, PV+HW, NWP, NWP+)
- **回看窗口**: 2种 (24小时, 72小时)
- **时间编码**: 2种 (无TE, 有TE)
- **模型复杂度**: 2种 (Low, High)
- **模型类型**: 8种 (RF, XGB, LGBM, LSTM, GRU, TCN, Transformer, LSR)

## 🔧 维护说明

- 配置文件由脚本自动生成，不建议手动修改
- 数据文件路径在配置文件中指定
- 结果文件保存在 `results/` 目录下
- 所有脚本支持命令行参数，详见各脚本的帮助信息
