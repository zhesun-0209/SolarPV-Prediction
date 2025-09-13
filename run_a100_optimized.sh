#!/bin/bash
"""
A100优化版100个Project消融实验启动脚本
针对A100 GPU的高性能并行策略
"""

echo "🚀 Project1140 A100优化版消融实验启动脚本"
echo "=================================================="

# 设置环境变量优化
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# 检查A100 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    echo "🎯 GPU信息: $GPU_INFO"
    
    # 检查是否是A100
    if echo "$GPU_INFO" | grep -i "A100" > /dev/null; then
        echo "✅ A100 GPU检测到，启用高性能模式"
        export A100_OPTIMIZED=1
    else
        echo "⚠️ 非A100 GPU，使用标准高性能模式"
        export A100_OPTIMIZED=0
    fi
else
    echo "❌ NVIDIA驱动未安装，将使用CPU模式"
    export A100_OPTIMIZED=0
fi

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

# 检查PyTorch GPU支持
echo "🔍 检查PyTorch GPU支持..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')" 2>/dev/null

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

# 根据A100优化设置运行参数
if [ "$A100_OPTIMIZED" = "1" ]; then
    # A100优化参数
    MAX_GPU_EXPERIMENTS=16
    MAX_CPU_EXPERIMENTS=32
    BATCH_SIZE=25
    echo "🎯 A100优化参数:"
    echo "   GPU并行数: $MAX_GPU_EXPERIMENTS"
    echo "   CPU并行数: $MAX_CPU_EXPERIMENTS"
    echo "   批次大小: $BATCH_SIZE"
else
    # 标准高性能参数
    MAX_GPU_EXPERIMENTS=8
    MAX_CPU_EXPERIMENTS=16
    BATCH_SIZE=15
    echo "⚙️ 标准高性能参数:"
    echo "   GPU并行数: $MAX_GPU_EXPERIMENTS"
    echo "   CPU并行数: $MAX_CPU_EXPERIMENTS"
    echo "   批次大小: $BATCH_SIZE"
fi

LOG_FILE="high_performance_experiments_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p logs

# 运行高性能实验
echo "🚀 开始运行A100优化版消融实验..."
echo "   开始时间: $(date)"
echo "   日志文件: logs/$LOG_FILE"

# 后台运行实验，同时显示日志
nohup python scripts/run_high_performance_experiments.py \
    --drive-path "$DRIVE_RESULTS_DIR" \
    --max-gpu-experiments $MAX_GPU_EXPERIMENTS \
    --max-cpu-experiments $MAX_CPU_EXPERIMENTS \
    --batch-size $BATCH_SIZE \
    > "logs/$LOG_FILE" 2>&1 &

EXPERIMENT_PID=$!
echo "📝 实验进程ID: $EXPERIMENT_PID"

# 保存PID到文件，便于后续管理
echo $EXPERIMENT_PID > experiment.pid

# 实时监控脚本
cat > monitor_a100_experiment.sh << 'EOF'
#!/bin/bash
PID_FILE="experiment.pid"
LOG_FILE="logs/high_performance_experiments_*.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "✅ A100优化实验正在运行 (PID: $PID)"
        echo "📊 最新日志:"
        tail -n 10 $LOG_FILE 2>/dev/null || echo "日志文件未找到"
        
        # 显示GPU使用情况
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "🎯 GPU使用情况:"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while read line; do
                echo "   GPU: $line"
            done
        fi
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

# 显示性能统计
echo ""
echo "🚀 性能统计:"
if [ -f "$LOG_FILE" ]; then
    # 统计GPU和CPU实验数量
    GPU_COUNT=$(grep "GPU实验完成" "$LOG_FILE" 2>/dev/null | wc -l)
    CPU_COUNT=$(grep "CPU实验完成" "$LOG_FILE" 2>/dev/null | wc -l)
    echo "   GPU实验完成: $GPU_COUNT"
    echo "   CPU实验完成: $CPU_COUNT"
    echo "   总实验完成: $((GPU_COUNT + CPU_COUNT))"
    
    # 计算平均速度
    if [ $((GPU_COUNT + CPU_COUNT)) -gt 0 ]; then
        START_TIME=$(head -n 1 "$LOG_FILE" 2>/dev/null | grep -o '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]' | head -n 1)
        if [ -n "$START_TIME" ]; then
            echo "   启动时间: $START_TIME"
        fi
    fi
fi
EOF

chmod +x monitor_a100_experiment.sh

echo ""
echo "🎯 A100优化实验已启动!"
echo "📋 管理命令:"
echo "   查看状态: ./monitor_a100_experiment.sh"
echo "   查看日志: tail -f logs/$LOG_FILE"
echo "   停止实验: kill $EXPERIMENT_PID"
echo ""
echo "🚀 A100优化特性:"
echo "   - GPU并行数: $MAX_GPU_EXPERIMENTS (深度学习模型)"
echo "   - CPU并行数: $MAX_CPU_EXPERIMENTS (传统ML模型)"
echo "   - 混合精度训练 (AMP)"
echo "   - 优化的批处理大小"
echo "   - 智能资源分配"
echo ""
echo "📊 结果将实时保存到: $DRIVE_RESULTS_DIR"
echo "   每个Project一个CSV文件: Project001.csv, Project002.csv, ..."
echo ""
echo "⏰ 预计性能提升:"
echo "   - 相比标准版本: 3-5倍加速"
echo "   - 预计总耗时: 120-1200小时 (5-50天)"
echo "   - 支持断点续训"
echo ""
echo "开始时间: $(date)"
echo "=================================================="
