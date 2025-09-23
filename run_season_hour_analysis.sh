#!/bin/bash
"""
Season and Hour Analysis 运行脚本
在服务器上运行season and hour analysis实验
"""

echo "🌟 SolarPV Season and Hour Analysis 实验"
echo "========================================"

# 检查必要文件
echo "🔍 检查环境..."
if [ ! -f "main.py" ]; then
    echo "❌ 错误: main.py 不存在"
    exit 1
fi

if [ ! -d "data" ]; then
    echo "❌ 错误: data 目录不存在"
    exit 1
fi

if [ ! -f "season_and_hour_analysis/scripts/generate_season_hour_configs.py" ]; then
    echo "❌ 错误: 配置文件生成脚本不存在"
    exit 1
fi

if [ ! -f "season_and_hour_analysis/scripts/run_season_hour_experiments.py" ]; then
    echo "❌ 错误: 实验运行脚本不存在"
    exit 1
fi

echo "✅ 环境检查通过"

# 生成配置文件
echo ""
echo "📝 生成配置文件..."
python season_and_hour_analysis/scripts/generate_season_hour_configs.py

if [ $? -ne 0 ]; then
    echo "❌ 配置文件生成失败"
    exit 1
fi

echo "✅ 配置文件生成完成"

# 运行实验
echo ""
echo "🚀 开始运行实验..."
echo "注意: 实验可能需要很长时间，建议在后台运行"
echo "使用 nohup 命令: nohup ./run_season_hour_analysis.sh > season_hour_analysis.log 2>&1 &"

read -p "是否继续运行实验? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python season_and_hour_analysis/scripts/run_season_hour_experiments.py
    echo "✅ 实验完成"
else
    echo "⏸️ 实验已取消"
fi

echo ""
echo "📁 结果保存在: /content/drive/MyDrive/Solar PV electricity/hour and season analysis/"
echo "🎉 Season and Hour Analysis 完成！"
