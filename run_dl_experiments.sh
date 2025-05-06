#!/usr/bin/env bash
# run_dl_experiments.sh
#
# Runs all deep‐learning models across every combination of the
# five ablation flags, for every ProjectID internally within main.py.

set -e

BASE_CONFIG="config/default.yaml"
BASE_SAVE_DIR="/path/to/outputs"   # modify to your outputs root

MODELS=("Transformer" "LSTM" "GRU" "TCN")

for MODEL in "${MODELS[@]}"; do
  for USE_FEATURE in true false; do
    for USE_TIME in true false; do
      for USE_FCAST in true false; do
        for USE_STATS in true false; do
          for USE_META in true false; do

            FLAG_TAG="feat${USE_FEATURE}_time${USE_TIME}_fcst${USE_FCAST}_stats${USE_STATS}_meta${USE_META}"
            echo "================ Running ${MODEL} | ${FLAG_TAG} ================"

            python main.py \
              --config       "$BASE_CONFIG" \
              --save_dir     "$BASE_SAVE_DIR" \
              --model        "$MODEL" \
              --use_feature  "$USE_FEATURE" \
              --use_time     "$USE_TIME" \
              --use_forecast "$USE_FCAST" \
              --use_stats    "$USE_STATS" \
              --use_meta     "$USE_META"

            echo ">>> Completed ${MODEL} | ${FLAG_TAG}"
            echo
          done
        done
      done
    done
  done
done

echo "All DL experiments done."
