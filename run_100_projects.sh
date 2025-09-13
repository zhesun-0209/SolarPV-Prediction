#!/bin/bash
"""
100个Project消融实验批量运行脚本
支持断点续训、实时保存到Google Drive
"""

echo "🚀 Project1140 100个Project消融实验启动脚本"
echo "=================================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 检查Google Drive挂载
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "❌ Google Drive未挂载，请先挂载Google Drive"
    echo "   在Colab中运行: drive.mount('/content/drive')"
    exit 1
fi

# 创建结果目录
DRIVE_RESULTS_DIR="/content/drive/MyDrive/Solar PV electricity/ablation results"
mkdir -p "$DRIVE_RESULTS_DIR"

echo "📁 结果保存目录: $DRIVE_RESULTS_DIR"

# 检查数据文件
DATA_DIR="./data"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 数据目录不存在: $DATA_DIR"
    echo "   请确保100个CSV文件已放置在data/目录下"
    exit 1
fi

# 统计数据文件数量
CSV_COUNT=$(find "$DATA_DIR" -name "Project*.csv" | wc -l)
echo "📊 发现 $CSV_COUNT 个Project数据文件"

if [ $CSV_COUNT -eq 0 ]; then
    echo "❌ 未找到Project数据文件"
    echo "   请确保文件命名为Project001.csv, Project002.csv, ..."
    exit 1
fi

# 检查依赖
echo "🔍 检查Python依赖..."
python -c "import pandas, numpy, sklearn, torch, xgboost, lightgbm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ 缺少依赖包，正在安装..."
    pip install -r requirements.txt
fi

# 生成配置（如果不存在）
CONFIG_DIR="./config/projects"
if [ ! -d "$CONFIG_DIR" ]; then
    echo "📝 生成100个Project的配置..."
    python scripts/generate_multi_project_configs.py
    
    if [ $? -ne 0 ]; then
        echo "❌ 配置生成失败"
        exit 1
    fi
else
    echo "✅ 配置已存在，跳过生成"
fi

# 检查断点续训
echo "🔍 检查断点续训状态..."
python -c "
from utils.checkpoint_manager import CheckpointManager
manager = CheckpointManager()
progress_df = manager.get_all_projects_progress()
if not progress_df.empty:
    completed = len(progress_df[progress_df['is_complete'] == True])
    total = len(progress_df)
    print(f'📊 当前进度: {completed}/{total} 个Project已完成')
    if completed > 0:
        print('🔄 将进行断点续训')
else:
    print('🆕 首次运行，将开始全新实验')
"

# 设置运行参数
MAX_WORKERS=${MAX_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-10}
LOG_FILE="multi_project_experiments_$(date +%Y%m%d_%H%M%S).log"

echo "⚙️ 运行参数:"
echo "   最大并发数: $MAX_WORKERS"
echo "   批次大小: $BATCH_SIZE"
echo "   日志文件: $LOG_FILE"

# 创建日志目录
mkdir -p logs

# 运行实验
echo "🚀 开始运行100个Project的消融实验..."
echo "   开始时间: $(date)"
echo "   日志文件: logs/$LOG_FILE"

# 后台运行实验，同时显示日志
nohup python scripts/run_multi_project_experiments.py \
    --drive-path "$DRIVE_RESULTS_DIR" \
    --max-workers $MAX_WORKERS \
    --batch-size $BATCH_SIZE \
    > "logs/$LOG_FILE" 2>&1 &

EXPERIMENT_PID=$!
echo "📝 实验进程ID: $EXPERIMENT_PID"

# 保存PID到文件，便于后续管理
echo $EXPERIMENT_PID > experiment.pid

# 实时监控脚本
cat > monitor_experiment.sh << 'EOF'
#!/bin/bash
PID_FILE="experiment.pid"
LOG_FILE="logs/multi_project_experiments_*.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "✅ 实验正在运行 (PID: $PID)"
        echo "📊 最新日志:"
        tail -n 10 $LOG_FILE 2>/dev/null || echo "日志文件未找到"
    else
        echo "❌ 实验进程已结束"
        echo "📊 最终日志:"
        tail -n 20 $LOG_FILE 2>/dev/null || echo "日志文件未找到"
    fi
else
    echo "❌ 未找到实验PID文件"
fi

# 显示进度报告
echo ""
echo "📈 进度报告:"
python -c "
from utils.checkpoint_manager import CheckpointManager
try:
    manager = CheckpointManager()
    progress_df = manager.get_all_projects_progress()
    if not progress_df.empty:
        completed = len(progress_df[progress_df['is_complete'] == True])
        total = len(progress_df)
        print(f'   完成进度: {completed}/{total} ({completed/total*100:.1f}%)')
        
        # 显示最近完成的Project
        recent_completed = progress_df[progress_df['is_complete'] == True].tail(3)
        if len(recent_completed) > 0:
            print('   最近完成:')
            for _, row in recent_completed.iterrows():
                print(f'     - {row[\"project_id\"]}')
    else:
        print('   暂无进度信息')
except Exception as e:
    print(f'   进度查询失败: {e}')
"
EOF

chmod +x monitor_experiment.sh

echo ""
echo "🎯 实验已启动!"
echo "📋 管理命令:"
echo "   查看状态: ./monitor_experiment.sh"
echo "   查看日志: tail -f logs/$LOG_FILE"
echo "   停止实验: kill $EXPERIMENT_PID"
echo ""
echo "📊 结果将实时保存到: $DRIVE_RESULTS_DIR"
echo "   每个Project一个CSV文件: Project001.csv, Project002.csv, ..."
echo ""
echo "⏰ 预计总耗时: 根据Project数量和模型复杂度，可能需要数小时到数天"
echo "🔄 支持断点续训: 如果中断，重新运行此脚本即可从断点继续"
echo ""
echo "开始时间: $(date)"
echo "=================================================="
