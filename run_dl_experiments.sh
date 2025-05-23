#!/usr/bin/env bash
# run_dl_experiments.sh
#
# Runs all deep-learning models across every combination of the
# five ablation flags, for every ProjectID internally within main.py.

set -e

# Path to your base config YAML
BASE_CONFIG="config/default.yaml"

# Use Python to extract save_dir from YAML
BASE_SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$BASE_CONFIG'))['save_dir'])")

# List of deep learning models
MODELS=("Transformer" "LSTM" "GRU" "TCN")

# Default meta-loss hyperparameters (can be modified as needed)
ALPHA=3.0
PEAK_START=10
PEAK_END=14
THRESHOLD=0.005

for MODEL in "${MODELS[@]}"; do
  for USE_FEATURE in true false; do
    for USE_TIME in true false; do
      for USE_FCAST in true false; do
        for USE_STATS in true false; do
          for USE_META in true false; do

            FLAG_TAG="feat${USE_FEATURE}_time${USE_TIME}_fcst${USE_FCAST}_stats${USE_STATS}_meta${USE_META}"
            echo "================ Running ${MODEL} | ${FLAG_TAG} ================"

            CMD="python main.py \
              --config       $BASE_CONFIG \
              --save_dir     $BASE_SAVE_DIR \
              --model        $MODEL \
              --use_feature  $USE_FEATURE \
              --use_time     $USE_TIME \
              --use_forecast $USE_FCAST \
              --use_stats    $USE_STATS \
              --use_meta     $USE_META"

            # If meta-loss is enabled, pass additional parameters
            if [ "$USE_META" == "true" ]; then
              CMD+=" \
                --alpha        $ALPHA \
                --peak_start   $PEAK_START \
                --peak_end     $PEAK_END \
                --threshold    $THRESHOLD"
            fi

            # Run the composed command
            eval $CMD

            echo ">>> Completed ${MODEL} | ${FLAG_TAG}"
            echo
          done
        done
      done
    done
  done
done

echo "All DL experiments done."
