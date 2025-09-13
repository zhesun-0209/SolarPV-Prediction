#!/bin/bash

# Project1140 消融实验启动脚本

echo "🚀 Project1140 光伏发电预测消融实验"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查依赖包
echo "📦 检查依赖包..."
python -c "import torch, pandas, numpy, sklearn, xgboost, lightgbm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖包，请运行: pip install -r requirements.txt"
    exit 1
fi

# 检查数据文件
if [ ! -f "data/Project1140.csv" ]; then
    echo "❌ 数据文件不存在: data/Project1140.csv"
    exit 1
fi

# 检查配置文件
if [ ! -d "config/ablation" ]; then
    echo "⚠️  配置文件不存在，正在生成..."
    python scripts/generate_ablation_configs.py
fi

# 测试配置
echo "🧪 测试配置..."
python scripts/test_ablation_configs.py

if [ $? -eq 0 ]; then
    echo "✅ 配置测试通过"
    echo ""
    echo "🎯 可用的运行选项:"
    echo "1. 运行单个实验:"
    echo "   python main.py --config config/ablation/LSR_baseline_PV_24h_noTE.yaml"
    echo ""
    echo "2. 运行所有消融实验:"
    echo "   python scripts/run_ablation_experiments.py"
    echo ""
    echo "3. 运行特定模型:"
    echo "   python scripts/run_ablation_experiments.py --model-filter LSR,Transformer"
    echo ""
    echo "4. 测试运行:"
    echo "   python scripts/run_ablation_experiments.py --max-configs 5 --dry-run"
else
    echo "❌ 配置测试失败，请检查配置"
    exit 1
fi
