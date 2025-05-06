#!/usr/bin/env bash
# run_dl_experiments_demo.sh
# Run DL models for: (1) all False, (2) all True, (3) each single flag True

set -e

BASE_CONFIG="config/default.yaml"
BASE_SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$BASE_CONFIG'))['save_dir'])")
MODELS=("LSTM" "GRU" "TCN")

# Meta-loss params
ALPHA=3.0
PEAK_START=10
PEAK_END=14
THRESHOLD=0.005

# Define ablation flag names
FLAGS=("use_hist_weather" "use_time" "use_forecast" "use_stats" "use_meta")

# Function to run one experiment
run_experiment() {
  local MODEL=$1
  shift
  local FLAG_TAG=$1
  shift
  local FLAG_VALUES=("$@")  # Remaining args as array

  echo "================ Running ${MODEL} | ${FLAG_TAG} ================"

  CMD="python main.py --config $BASE_CONFIG --save_dir $BASE_SAVE_DIR --model $MODEL"

  for FLAG_VAL in "${FLAG_VALUES[@]}"; do
    KEY=$(echo $FLAG_VAL | cut -d= -f1)
    VAL=$(echo $FLAG_VAL | cut -d= -f2)
    CMD+=" --$KEY $VAL"
  done

  if [[ " ${FLAG_VALUES[*]} " == *"use_meta=true"* ]]; then
    CMD+=" --alpha $ALPHA --peak_start $PEAK_START --peak_end $PEAK_END --threshold $THRESHOLD"
  fi

  eval $CMD
  echo ">>> Completed ${MODEL} | ${FLAG_TAG}"
  echo
}

# ================= Run for all models =================
for MODEL in "${MODELS[@]}"; do

  # 1. All False
  run_experiment "$MODEL" "ALL_FALSE" \
    "use_hist_weather=false" "use_time=false" "use_forecast=false" "use_stats=false" "use_meta=false"

  # 2. All True
  run_experiment "$MODEL" "ALL_TRUE" \
    "use_hist_weather=true" "use_time=true" "use_forecast=true" "use_stats=true" "use_meta=true"

  # 3. Each single-flag True
  for ONE_TRUE in "${FLAGS[@]}"; do
    FLAG_TAG="only_${ONE_TRUE}"
    FLAG_VALUES=()
    for FLAG in "${FLAGS[@]}"; do
      if [[ "$FLAG" == "$ONE_TRUE" ]]; then
        FLAG_VALUES+=("${FLAG}=true")
      else
        FLAG_VALUES+=("${FLAG}=false")
      fi
    done
    run_experiment "$MODEL" "$FLAG_TAG" "${FLAG_VALUES[@]}"
  done
done

echo "✅ Selected DL ablation experiments completed."
