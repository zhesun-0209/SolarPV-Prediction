#!/usr/bin/env bash
# run_ml_experiments.sh
#
# This script runs traditional ML experiments for all combinations of
# ablation flags (except meta, which is fixed to false) on all projects.
# It calls main.py with the appropriate overrides and relies on main.py
# to iterate over ProjectID internally.
#
# Usage:
#   chmod +x run_ml_experiments.sh
#   ./run_ml_experiments.sh

# Path to your default YAML config
BASE_CONFIG="config/default.yaml"
# Base output directory (will be extended per project/model/flags)
BASE_SAVE_DIR="/path/to/outputs"

# List of ML model abbreviations (must match parser --model choices)
MODELS=("RF" "GBR" "XGB" "LGBM")

# Loop over each ML model
for MODEL in "${MODELS[@]}"; do
  # Loop over all 2^4 combinations of the four boolean flags
  for USE_FEATURE in true false; do
    for USE_TIME in true false; do
      for USE_FCAST in true false; do
        for USE_STATS in true false; do

          # Build a human-readable tag for this combination
          FLAG_TAG="feat${USE_FEATURE}_time${USE_TIME}_fcst${USE_FCAST}_stats${USE_STATS}"
          echo "----------------------------------------------------------------"
          echo "Running $MODEL with flags: $FLAG_TAG (meta always false)"
          echo "Saving under: $BASE_SAVE_DIR/<project>/${MODEL,,}/$FLAG_TAG/"
          echo

          # Invoke the main pipeline, forcing use_meta=false
          python main.py \
            --config        "$BASE_CONFIG" \
            --save_dir      "$BASE_SAVE_DIR" \
            --model         "$MODEL" \
            --use_feature   "$USE_FEATURE" \
            --use_time      "$USE_TIME" \
            --use_forecast  "$USE_FCAST" \
            --use_stats     "$USE_STATS" \
            --use_meta      false

          echo "Finished $MODEL | $FLAG_TAG"
          echo "----------------------------------------------------------------"
          echo
        done
      done
    done
  done
done

echo "All machine learning experiments completed."
