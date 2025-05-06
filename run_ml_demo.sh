#!/usr/bin/env bash
# run_ml_experiments_demo.sh
# Run ML models for: (1) all False, (2) all True, (3) each single flag True

set -e

BASE_CONFIG="config/default.yaml"
BASE_SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$BASE_CONFIG'))['save_dir'])")
MODELS=("RF" "GBR" "XGB" "LGBM")

FLAGS=("use_hist_weather" "use_time" "use_forecast" "use_stats")

run_experiment() {
  local MODEL=$1
  local FLAG_VALUES=$2
  local FLAG_TAG=$3

  echo "=============== Running ${MODEL} | ${FLAG_TAG} ================"

  CMD="python main.py --config $BASE_CONFIG --save_dir $BASE_SAVE_DIR --model $MODEL"

  for FLAG_VAL in $FLAG_VALUES; do
    KEY=$(echo $FLAG_VAL | cut -d= -f1)
    VAL=$(echo $FLAG_VAL | cut -d= -f2)
    CMD+=" --$KEY $VAL"
  done

  # Meta is always false for ML
  CMD+=" --use_meta false"

  eval $CMD
  echo ">>> Completed ${MODEL} | ${FLAG_TAG}"
  echo
}

for MODEL in "${MODELS[@]}"; do

  # 1. All False
  FLAG_VALUES="use_hist_weather=false use_time=false use_forecast=false use_stats=false"
  run_experiment "$MODEL" "$FLAG_VALUES" "ALL_FALSE"

  # 2. All True
  FLAG_VALUES="use_hist_weather=true use_time=true use_forecast=true use_stats=true"
  run_experiment "$MODEL" "$FLAG_VALUES" "ALL_TRUE"

  # 3. Each single-flag True (others False)
  for ONE_TRUE in "${FLAGS[@]}"; do
    FLAG_VALUES=""
    FLAG_TAG="only_${ONE_TRUE}"
    for FLAG in "${FLAGS[@]}"; do
      if [[ "$FLAG" == "$ONE_TRUE" ]]; then
        FLAG_VALUES+="${FLAG}=true "
      else
        FLAG_VALUES+="${FLAG}=false "
      fi
    done
    run_experiment "$MODEL" "$FLAG_VALUES" "$FLAG_TAG"
  done
done

echo "✅ Selected ML ablation experiments completed."
