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


def str2bool(v):
    return v.lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Solar Power Forecasting Pipeline")
    parser.add_argument("--config",        type=str,   required=True, help="Path to YAML config file")
    parser.add_argument("--data_path",     type=str,   help="Override data_path")
    parser.add_argument("--save_dir",      type=str,   help="Override save_dir")
    parser.add_argument("--model",         type=str,   help="Override model type")
    parser.add_argument("--past_hours",    type=int,   help="Override past_hours")
    parser.add_argument("--future_hours",  type=int,   help="Override future_hours")
    parser.add_argument("--use_feature",   type=str,   choices=["true","false"], help="Override use_feature")
    parser.add_argument("--use_time",      type=str,   choices=["true","false"], help="Override use_time")
    parser.add_argument("--use_forecast",  type=str,   choices=["true","false"], help="Override use_forecast")
    parser.add_argument("--use_stats",     type=str,   choices=["true","false"], help="Override use_stats")
    parser.add_argument("--use_meta",      type=str,   choices=["true","false"], help="Override use_meta")
    parser.add_argument("--train_ratio",   type=float, help="Override train_ratio")
    parser.add_argument("--val_ratio",     type=float, help="Override val_ratio")
    parser.add_argument("--plot_days",     type=int,   help="Override plot_days")
    parser.add_argument("--batch_size",    type=int,   help="Override train_params.batch_size")
    parser.add_argument("--epochs",        type=int,   help="Override train_params.epochs")
    parser.add_argument("--learning_rate", type=float, help="Override train_params.learning_rate")
    parser.add_argument("--weight_decay",  type=float, help="Override train_params.weight_decay")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- Override config from CLI ---
    if args.data_path:    config["data_path"]      = args.data_path
    if args.save_dir:     config["save_dir"]       = args.save_dir
    if args.model:        config["model"]          = args.model
    if args.past_hours:   config["past_hours"]     = args.past_hours
    if args.future_hours: config["future_hours"]   = args.future_hours
    if args.use_feature:  config["use_feature"]    = str2bool(args.use_feature)
    if args.use_time:     config["use_time"]       = str2bool(args.use_time)
    if args.use_forecast: config["use_forecast"]   = str2bool(args.use_forecast)
    if args.use_stats:    config["use_stats"]      = str2bool(args.use_stats)
    if args.use_meta:     config["use_meta"]       = str2bool(args.use_meta)
    if args.train_ratio:  config["train_ratio"]    = args.train_ratio
    if args.val_ratio:    config["val_ratio"]      = args.val_ratio
    if args.plot_days:    config["plot_days"]      = args.plot_days

    tp = config.setdefault("train_params", {})
    if args.batch_size:    tp["batch_size"]    = args.batch_size
    if args.epochs:        tp["epochs"]        = args.epochs
    if args.learning_rate: tp["learning_rate"] = args.learning_rate
    if args.weight_decay:  tp["weight_decay"]  = args.weight_decay

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
            train_ratio=config["train_ratio"],
            val_ratio=  config["val_ratio"]
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
            "use_feature":  cfg["use_feature"],
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
