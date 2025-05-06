#!/usr/bin/env bash
# run_ml_experiments.sh
#
# Runs all ML models across combinations of the four ablation flags
# (meta always = false). Projects looped internally.

set -e

# Path to YAML config
BASE_CONFIG="config/default.yaml"

# Use Python to extract save_dir from YAML
BASE_SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$BASE_CONFIG'))['save_dir'])")

MODELS=("RF" "GBR" "XGB" "LGBM")

for MODEL in "${MODELS[@]}"; do
  for USE_FEATURE in true false; do
    for USE_TIME in true false; do
      for USE_FCAST in true false; do
        for USE_STATS in true false; do

          FLAG_TAG="feat${USE_FEATURE}_time${USE_TIME}_fcst${USE_FCAST}_stats${USE_STATS}"
          echo "=============== Running ${MODEL} | ${FLAG_TAG} ==============="

          python main.py \
            --config       "$BASE_CONFIG" \
            --save_dir     "$BASE_SAVE_DIR" \
            --model        "$MODEL" \
            --use_feature  "$USE_FEATURE" \
            --use_time     "$USE_TIME" \
            --use_forecast "$USE_FCAST" \
            --use_stats    "$USE_STATS" \
            --use_meta     false

          echo ">>> Completed ${MODEL} | ${FLAG_TAG}"
          echo
        done
      done
    done
  done
done

echo "All ML experiments done."
