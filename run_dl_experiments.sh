#!/usr/bin/env bash
# run_dl_experiments.sh
# 
# This script runs deep learning experiments for all combinations of
# ablation flags on all projects. It calls main.py with the appropriate
# overrides and relies on main.py to iterate over ProjectID internally.
#
# Usage:
#   chmod +x run_dl_experiments.sh
#   ./run_dl_experiments.sh

# Path to your default YAML config
BASE_CONFIG="config/default.yaml"
# Base output directory (will be extended per project/model/flags)
BASE_SAVE_DIR="/path/to/outputs"

# List of DL model names (must match parser --model choices)
MODELS=("Transformer" "LSTM" "GRU" "TCN")

# Loop over each model
for MODEL in "${MODELS[@]}"; do
  # Loop over all 2^5 combinations of the five boolean flags
  for USE_FEATURE in true false; do
    for USE_TIME in true false; do
      for USE_FCAST in true false; do
        for USE_STATS in true false; do
          for USE_META in true false; do

            # Build a human-readable tag for this combination
            FLAG_TAG="feat${USE_FEATURE}_time${USE_TIME}_fcst${USE_FCAST}_stats${USE_STATS}_meta${USE_META}"
            echo "------------------------------------------------------------"
            echo "Running $MODEL with flags: $FLAG_TAG"
            echo "Saving under: $BASE_SAVE_DIR/<project>/${MODEL,,}/$FLAG_TAG/"
            echo

            # Invoke the main pipeline
            python main.py \
              --config        "$BASE_CONFIG" \
              --save_dir      "$BASE_SAVE_DIR" \
              --model         "$MODEL" \
              --use_feature   "$USE_FEATURE" \
              --use_time      "$USE_TIME" \
              --use_forecast  "$USE_FCAST" \
              --use_stats     "$USE_STATS" \
              --use_meta      "$USE_META"

            echo "Finished $MODEL | $FLAG_TAG"
            echo "------------------------------------------------------------"
            echo
          done
        done
      done
    done
  done
done

echo "All deep learning experiments completed."
