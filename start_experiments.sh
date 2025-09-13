#!/bin/bash
"""
一键启动100个Project消融实验
自动选择最佳策略并保存到Google Drive
"""

echo "🚀 Project1140 一键启动消融实验"
echo "=================================================="

# 检查是否在Colab环境
if [ -d "/content" ]; then
    echo "✅ 检测到Colab环境"
    IS_COLAB=true
else
    echo "💻 本地环境"
    IS_COLAB=false
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDNN_V8_API_ENABLED=1
export FORCE_GPU=1
export USE_GPU=1

echo "🔧 环境变量已设置"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    echo "🎯 GPU信息: $GPU_INFO"
    
    if echo "$GPU_INFO" | grep -i "A100" > /dev/null; then
        echo "🚀 A100检测到，使用极致性能模式"
        STRATEGY="gpu_only"
        MAX_GPU=32
        BATCH_SIZE=40
    else
        echo "💻 标准GPU，使用高性能模式"
        STRATEGY="high_performance"
        MAX_GPU=16
        MAX_CPU=24
        BATCH_SIZE=25
    fi
else
    echo "⚠️ 未检测到GPU，使用CPU模式"
    STRATEGY="standard"
    MAX_WORKERS=8
    BATCH_SIZE=20
fi

# 检查Google Drive
if [ "$IS_COLAB" = true ]; then
    if [ ! -d "/content/drive/MyDrive" ]; then
        echo "❌ Google Drive未挂载"
        echo "请先运行: drive.mount('/content/drive')"
        exit 1
    fi
    DRIVE_PATH="/content/drive/MyDrive/Solar PV electricity/ablation results"
else
    DRIVE_PATH="./results/ablation"
fi

mkdir -p "$DRIVE_PATH"
echo "📁 结果保存路径: $DRIVE_PATH"

# 检查数据文件
DATA_DIR="./data"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ data目录不存在"
    echo "请确保100个CSV文件已放置在data/目录下"
    exit 1
fi

CSV_COUNT=$(find "$DATA_DIR" -name "Project*.csv" | wc -l)
echo "📊 发现 $CSV_COUNT 个Project数据文件"

if [ $CSV_COUNT -eq 0 ]; then
    echo "❌ 未找到Project数据文件"
    echo "请确保文件命名为Project001.csv, Project002.csv, ..."
    exit 1
fi

# 生成配置
CONFIG_DIR="./config/projects"
if [ ! -d "$CONFIG_DIR" ]; then
    echo "📝 生成配置..."
    python scripts/generate_multi_project_configs.py
    if [ $? -ne 0 ]; then
        echo "❌ 配置生成失败"
        exit 1
    fi
else
    echo "✅ 配置已存在"
fi

# 运行实验
echo "🚀 启动实验 (策略: $STRATEGY)..."

LOG_FILE="experiments_$(date +%Y%m%d_%H%M%S).log"

case $STRATEGY in
    "gpu_only")
        echo "🎯 运行GPU专用版..."
        python scripts/run_gpu_only_experiments.py \
            --drive-path "$DRIVE_PATH" \
            --max-gpu-experiments $MAX_GPU \
            --batch-size $BATCH_SIZE \
            > "logs/$LOG_FILE" 2>&1 &
        ;;
    "high_performance")
        echo "🚀 运行高性能版..."
        python scripts/run_high_performance_experiments.py \
            --drive-path "$DRIVE_PATH" \
            --max-gpu-experiments $MAX_GPU \
            --max-cpu-experiments $MAX_CPU \
            --batch-size $BATCH_SIZE \
            > "logs/$LOG_FILE" 2>&1 &
        ;;
    "standard")
        echo "📊 运行标准版..."
        python scripts/run_multi_project_experiments.py \
            --drive-path "$DRIVE_PATH" \
            --max-workers $MAX_WORKERS \
            --batch-size $BATCH_SIZE \
            > "logs/$LOG_FILE" 2>&1 &
        ;;
esac

EXPERIMENT_PID=$!
echo "📝 实验进程ID: $EXPERIMENT_PID"
echo $EXPERIMENT_PID > experiment.pid

# 创建监控脚本
cat > monitor_experiments.sh << 'EOF'
#!/bin/bash
PID_FILE="experiment.pid"
LOG_FILE="logs/experiments_*.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "✅ 实验正在运行 (PID: $PID)"
        echo "📊 最新日志:"
        tail -n 5 $LOG_FILE 2>/dev/null || echo "日志文件未找到"
        
        # 显示GPU使用情况
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "🎯 GPU使用情况:"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
                echo "   GPU: $line"
            done
        fi
    else
        echo "❌ 实验进程已结束"
        echo "📊 最终日志:"
        tail -n 10 $LOG_FILE 2>/dev/null || echo "日志文件未找到"
    fi
else
    echo "❌ 未找到实验PID文件"
fi

# 显示进度
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
    else:
        print('   暂无进度信息')
except Exception as e:
    print(f'   进度查询失败: {e}')
"
EOF

chmod +x monitor_experiments.sh

echo ""
echo "🎯 实验已启动!"
echo "📋 管理命令:"
echo "   查看状态: ./monitor_experiments.sh"
echo "   查看日志: tail -f logs/$LOG_FILE"
echo "   停止实验: kill $EXPERIMENT_PID"
echo ""
echo "📊 结果保存位置: $DRIVE_PATH"
echo "   每个Project一个CSV文件: Project001.csv, Project002.csv, ..."
echo ""
echo "⏰ 预计完成时间:"
case $STRATEGY in
    "gpu_only")
        echo "   GPU专用版: 25天 (10倍加速)"
        ;;
    "high_performance")
        echo "   高性能版: 50天 (5倍加速)"
        ;;
    "standard")
        echo "   标准版: 250天"
        ;;
esac
echo ""
echo "开始时间: $(date)"
echo "=================================================="
