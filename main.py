#!/usr/bin/env python3
"""
main.py

Orchestrates the entire Solar Power Forecasting Pipeline:
  1. Loads a YAML config (with optional CLI overrides)
  2. Preprocesses raw data
  3. Splits into sliding windows and train/val/test sets per ProjectID
  4. Trains either a DL or ML model
  5. Saves results under:
       <base_save_dir>/Project_<pid>/<alg_type>/<model_name_lower>/<flag_tag>/
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

def str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")

def main():
    parser = argparse.ArgumentParser(description="Solar Power Forecasting Pipeline")

    # Core config
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_dir", type=str)

    # Ablation flags
    parser.add_argument("--model", type=str)
    parser.add_argument("--past_hours", type=int)
    parser.add_argument("--future_hours", type=int)
    parser.add_argument("--use_feature", type=str, choices=["true", "false"])
    parser.add_argument("--use_time", type=str, choices=["true", "false"])
    parser.add_argument("--use_forecast", type=str, choices=["true", "false"])
    parser.add_argument("--use_stats", type=str, choices=["true", "false"])
    parser.add_argument("--use_meta", type=str, choices=["true", "false"])
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--plot_days", type=int)

    # DL model params
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--tcn_channels", type=str)
    parser.add_argument("--kernel_size", type=int)

    # ML model params
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--ml_learning_rate", type=float)
    parser.add_argument("--random_state", type=int)

    # Training params
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--early_stop_patience", type=int)
    parser.add_argument("--loss_type", type=str)

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config from CLI
    if args.data_path: config["data_path"] = args.data_path
    if args.save_dir: config["save_dir"] = args.save_dir
    if args.model: config["model"] = args.model
    if args.past_hours: config["past_hours"] = args.past_hours
    if args.future_hours: config["future_hours"] = args.future_hours
    if args.use_feature: config["use_feature"] = str2bool(args.use_feature)
    if args.use_time: config["use_time"] = str2bool(args.use_time)
    if args.use_forecast: config["use_forecast"] = str2bool(args.use_forecast)
    if args.use_stats: config["use_stats"] = str2bool(args.use_stats)
    if args.use_meta: config["use_meta"] = str2bool(args.use_meta)
    if args.train_ratio: config["train_ratio"] = args.train_ratio
    if args.val_ratio: config["val_ratio"] = args.val_ratio
    if args.plot_days: config["plot_days"] = args.plot_days

    # Override model_params
    mp = config.setdefault("model_params", {})
    if args.d_model: mp["d_model"] = args.d_model
    if args.num_heads: mp["num_heads"] = args.num_heads
    if args.num_layers: mp["num_layers"] = args.num_layers
    if args.hidden_dim: mp["hidden_dim"] = args.hidden_dim
    if args.dropout: mp["dropout"] = args.dropout
    if args.tcn_channels: mp["tcn_channels"] = eval(args.tcn_channels)
    if args.kernel_size: mp["kernel_size"] = args.kernel_size
    if args.n_estimators: mp["n_estimators"] = args.n_estimators
    if args.max_depth is not None: mp["max_depth"] = args.max_depth
    if args.ml_learning_rate is not None: mp["learning_rate"] = args.ml_learning_rate
    if args.random_state: mp["random_state"] = args.random_state

    # Override train_params
    tp = config.setdefault("train_params", {})
    if args.batch_size: tp["batch_size"] = args.batch_size
    if args.epochs: tp["epochs"] = args.epochs
    if args.learning_rate: tp["learning_rate"] = args.learning_rate
    if args.weight_decay: tp["weight_decay"] = args.weight_decay
    if args.early_stop_patience: tp["early_stop_patience"] = args.early_stop_patience
    if args.loss_type: tp["loss_type"] = args.loss_type

    # Load raw data once
    df = load_raw_data(config["data_path"])

    flag_tag = (
        f"feat{config['use_feature']}_"
        f"time{config['use_time']}_"
        f"fcst{config['use_forecast']}_"
        f"stats{config['use_stats']}_"
        f"meta{config['use_meta']}"
    )

    is_dl = config["model"] in ["Transformer", "LSTM", "GRU", "TCN"]
    alg_type = "dl" if is_dl else "ml"

    for pid in df["ProjectID"].unique():
        df_proj = df[df["ProjectID"] == pid]
        if df_proj.empty:
            print(f"[WARN] Project {pid} has no data, skipping")
            continue

        # 👇 每个项目单独预处理
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
            preprocess_features(df_proj, config)

        Xh, Xf, y, hrs, dates = create_sliding_windows(
            df_clean,
            past_hours=config["past_hours"],
            future_hours=config["future_hours"],
            hist_feats=hist_feats,
            fcst_feats=fcst_feats
        )
        splits = split_data(
            Xh, Xf, y, hrs, dates,
            train_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"]
        )
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
        Xh_va, Xf_va, y_va, hrs_va, dates_va, \
        Xh_te, Xf_te, y_te, hrs_te, dates_te = splits

        proj_dir = os.path.join(
            config["save_dir"],
            f"Project_{pid}",
            alg_type,
            config["model"].lower(),
            flag_tag
        )
        os.makedirs(proj_dir, exist_ok=True)

        cfg = deepcopy(config)
        cfg["save_dir"] = proj_dir

        # === Train ===
        start = time.time()
        if is_dl:
            model, metrics = train_dl_model(
                cfg,
                (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr),
                (Xh_va, Xf_va, y_va, hrs_va, dates_va),
                (Xh_te, Xf_te, y_te, hrs_te, dates_te),
                (scaler_hist, scaler_fcst, scaler_target)
            )
        else:
            model, metrics = train_ml_model(
                cfg,
                Xh_tr, Xf_tr, y_tr,
                Xh_te, Xf_te, y_te,
                scaler_target
            )
        metrics["train_time_sec"] = round(time.time() - start, 2)

        # === Save ===
        cfg["scaler_target"] = scaler_target
        save_results(
            model,
            metrics,
            dates_te,
            y_te,
            Xh_te,
            Xf_te,
            cfg
        )
        print(f"[INFO] Project {pid} | {cfg['model']} done, test_loss={metrics['test_loss']:.4f}")

if __name__ == "__main__":
    main()
