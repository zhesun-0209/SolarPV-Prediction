#!/usr/bin/env python3
"""
Main script: load config, preprocess data, train & evaluate models per ProjectID
for solar power forecasting.
"""

import os
import time
import argparse
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import (
    load_raw_data,
    preprocess_features,
    create_sliding_windows,
    split_data
)
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model
from eval.eval_utils import save_results


def main():
    parser = argparse.ArgumentParser(
        description="Solar Power Forecasting Pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- Load & preprocess data ---
    df = load_raw_data(config["data_path"])
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
        preprocess_features(df, config)

    # iterate over each ProjectID
    for pid in df_clean['ProjectID'].unique():
        df_proj = df_clean[df_clean['ProjectID'] == pid]
        if df_proj.empty:
            print(f"[WARN] Project {pid} has no valid data, skipping")
            continue

        # --- Build sliding windows & split for this project ---
        X_hist, X_fcst, y, hours, dates = create_sliding_windows(
            df_proj,
            past_hours   = config["past_hours"],
            future_hours = config["future_hours"],
            hist_feats   = hist_feats,
            fcst_feats   = fcst_feats
        )
        (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
         Xh_va, Xf_va, y_va, hrs_va, dates_va,
         Xh_te, Xf_te, y_te, hrs_te, dates_te) = split_data(
            X_hist, X_fcst, y, hours, dates,
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=  config.get("val_ratio",   0.1)
        )

        # prepare save directory for this project
        proj_save_dir = os.path.join(
            config["save_dir"],
            config["model"],
            f"Project_{pid}"
        )
        os.makedirs(proj_save_dir, exist_ok=True)

        cfg = deepcopy(config)
        cfg["save_dir"] = proj_save_dir

        # --- Train model ---
        start_time = time.time()
        if cfg["model"] in ["Transformer", "LSTM", "GRU", "TCN"]:
            model, metrics = train_dl_model(
                cfg,
                (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr),
                (Xh_va, Xf_va, y_va, hrs_va, dates_va),
                (Xh_te, Xf_te, y_te, hrs_te, dates_te),
                (scaler_hist, scaler_fcst, scaler_target)
            )
        else:
            # flatten features for ML
            def flatten(Xh, Xf):
                h_flat = Xh.reshape(Xh.shape[0], -1)
                if Xf is not None:
                    f_flat = Xf.reshape(Xf.shape[0], -1)
                else:
                    f_flat = np.zeros_like(h_flat)
                return np.concatenate([h_flat, f_flat], axis=1)

            X_train_flat = flatten(Xh_tr, Xf_tr)
            y_train_flat = y_tr.reshape(y_tr.shape[0], -1)
            X_test_flat  = flatten(Xh_te, Xf_te)

            model, metrics = train_ml_model(
                cfg,
                X_train_flat, y_train_flat,
                X_test_flat,  y_te
            )

        # record timing and config flags
        metrics["train_time_sec"] = round(time.time() - start_time, 2)
        metrics.update({
            "model":        cfg["model"],
            "use_feature":  cfg.get("use_feature", True),
            "past_hours":   cfg["past_hours"],
            "future_hours": cfg["future_hours"]
        })

        # --- Save all results for this project ---
        save_results(
            model,
            metrics,
            dates_te,
            y_te,
            Xh_te,
            Xf_te,
            cfg
        )

        print(f"[INFO] Project {pid} | {cfg['model']} done in {metrics['train_time_sec']}s, "
              f"Test Loss = {metrics.get('test_loss', np.nan):.4f}")


if __name__ == "__main__":
    main()
