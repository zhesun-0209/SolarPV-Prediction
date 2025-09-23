#!/usr/bin/env python3
"""
main.py

Orchestrates the entire Solar Power Forecasting Pipeline:
  1. Loads a YAML config file (with optional CLI overrides)
  2. Preprocesses raw data
  3. Applies sliding windows and splits data per ProjectID
  4. Trains a DL or ML model
  5. Saves results under:
       <save_dir>/Project_<pid>/<alg_type>/<model_name>/<flag_tag>/
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
from eval.eval_utils import save_results, save_season_hour_results

def str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")

def main():
    parser = argparse.ArgumentParser(description="Solar Power Forecasting Pipeline")

    # === Core settings ===
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--plant_id", type=str, help="Plant ID for result tracking")

    # === Ablation settings ===
    parser.add_argument("--model", type=str)
    parser.add_argument("--past_days", type=int, choices=[1, 3])
    parser.add_argument("--past_hours", type=int)
    parser.add_argument("--future_hours", type=int)
    parser.add_argument("--use_pv", type=str, choices=["true", "false"],
                       help="Use historical PV power data")
    parser.add_argument("--use_hist_weather", type=str, choices=["true", "false"])
    parser.add_argument("--use_forecast", type=str, choices=["true", "false"])
    parser.add_argument("--weather_category", type=str, choices=["none", "all_weather"],
                       help="Weather feature category: none or all_weather")
    parser.add_argument("--use_time_encoding", type=str, choices=["true", "false"],
                       help="Use time encoding features (month/hour sin/cos)")
    parser.add_argument("--use_ideal_nwp", type=str, choices=["true", "false"],
                       help="Use ideal NWP features (without _pred suffix)")
    parser.add_argument("--no_hist_power", type=str, choices=["true", "false"], 
                       help="Only use forecast weather, no historical power data")
    parser.add_argument("--model_complexity", type=str, choices=["low", "high", "level1", "level2", "level3", "level4"])
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--plot_days", type=int)

    # === Deep learning model parameters ===
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--tcn_channels", type=str)
    parser.add_argument("--kernel_size", type=int)

    # === ML model parameters ===
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--ml_learning_rate", type=float)
    parser.add_argument("--random_state", type=int)

    # === Training parameters ===
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--loss_type", type=str)
    
    # === Save options ===
    parser.add_argument("--save_model", type=str, choices=["true", "false"])
    # --save_summary 已移除，不再保存summary.csv
    parser.add_argument("--save_predictions", type=str, choices=["true", "false"])
    parser.add_argument("--save_training_log", type=str, choices=["true", "false"])
    # 绘图功能已移除，默认不保存图片
    parser.add_argument("--save_excel_results", type=str, choices=["true", "false"])


    args = parser.parse_args()

    # === Load base config from YAML ===
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"🔍 调试: 初始配置加载完成")
    print(f"🔍 调试: config['model'] = {config.get('model', 'NOT_FOUND')}")
    print(f"🔍 调试: config['train_params'] = {config.get('train_params', 'NOT_FOUND')}")
    print(f"🔍 调试: config['model_params'] = {config.get('model_params', 'NOT_FOUND')}")
    
    # Validate required config keys
    required_keys = ['data_path', 'save_dir', 'model', 'past_hours', 'future_hours']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # === Override general settings from CLI ===
    if args.data_path: config["data_path"] = args.data_path
    if args.save_dir: config["save_dir"] = args.save_dir
    if args.plant_id: config["plant_id"] = args.plant_id
    if args.model: config["model"] = args.model
    if args.past_days: config["past_days"] = args.past_days
    if args.past_hours: config["past_hours"] = args.past_hours
    if args.future_hours: config["future_hours"] = args.future_hours
    if args.use_pv: config["use_pv"] = str2bool(args.use_pv)
    if args.use_hist_weather: config["use_hist_weather"] = str2bool(args.use_hist_weather)
    if args.use_forecast: config["use_forecast"] = str2bool(args.use_forecast)
    if args.weather_category: config["weather_category"] = args.weather_category
    if args.use_time_encoding: config["use_time_encoding"] = str2bool(args.use_time_encoding)
    if args.use_ideal_nwp: config["use_ideal_nwp"] = str2bool(args.use_ideal_nwp)
    if args.no_hist_power: config["no_hist_power"] = str2bool(args.no_hist_power)
    if args.train_ratio: config["train_ratio"] = args.train_ratio
    if args.val_ratio: config["val_ratio"] = args.val_ratio
    if args.plot_days: config["plot_days"] = args.plot_days
    if args.model_complexity: config["model_complexity"] = args.model_complexity

    # === Override model-specific parameters ===
    # 根据模型复杂度选择参数
    complexity = config.get("model_complexity", "low")
    is_dl = config["model"] in ["Transformer", "LSTM", "GRU", "TCN"]
    
    if is_dl:
        # 深度学习模型参数
        if complexity.startswith("level"):
            # 四档复杂度：level1, level2, level3, level4
            dl_params = config["model_params"].get(complexity, config["model_params"]["level2"])
        else:
            # 兼容旧的两档复杂度：low, high
            dl_params = config["model_params"].get(complexity, config["model_params"]["low"])
        config["model_params"] = dl_params
    else:
        # 机器学习模型参数
        if complexity.startswith("level"):
            # 四档复杂度：ml_level1, ml_level2, ml_level3, ml_level4
            ml_params = config["model_params"].get(f"ml_{complexity}", config["model_params"]["ml_level2"])
        else:
            # 兼容旧的两档复杂度：ml_low, ml_high
            ml_params = config["model_params"].get(f"ml_{complexity}", config["model_params"]["ml_low"])
        config["model_params"] = ml_params
    
    # 仍然允许CLI覆盖
    mp = config["model_params"]
    if args.d_model: mp["d_model"] = args.d_model
    if args.num_heads: mp["num_heads"] = args.num_heads
    if args.num_layers: mp["num_layers"] = args.num_layers
    if args.hidden_dim: mp["hidden_dim"] = args.hidden_dim
    if args.dropout: mp["dropout"] = args.dropout
    if args.tcn_channels: 
        try:
            mp["tcn_channels"] = [int(x.strip()) for x in args.tcn_channels.split(',')]
        except ValueError:
            raise ValueError("tcn_channels must be comma-separated integers")
    if args.kernel_size: mp["kernel_size"] = args.kernel_size
    if args.n_estimators: mp["n_estimators"] = args.n_estimators
    if args.max_depth is not None: mp["max_depth"] = args.max_depth
    if args.ml_learning_rate is not None: mp["learning_rate"] = args.ml_learning_rate
    if args.random_state: mp["random_state"] = args.random_state

    # === Override training-specific parameters ===
    tp = config.setdefault("train_params", {})
    if args.batch_size: tp["batch_size"] = args.batch_size
    if args.epochs: tp["epochs"] = args.epochs
    if args.learning_rate: tp["learning_rate"] = args.learning_rate
    if args.loss_type: tp["loss_type"] = args.loss_type

    # === Override save options ===
    save_opts = config.get("save_options", {})
    if args.save_model: save_opts["save_model"] = str2bool(args.save_model)
    # save_summary 已移除，不再保存summary.csv
    if args.save_predictions: save_opts["save_predictions"] = str2bool(args.save_predictions)
    if args.save_training_log: save_opts["save_training_log"] = str2bool(args.save_training_log)
    # 绘图功能已移除
    if args.save_excel_results: save_opts["save_excel_results"] = str2bool(args.save_excel_results)
    
    # 确保save_excel_results默认为True（如果配置文件中没有设置）
    if "save_excel_results" not in save_opts:
        save_opts["save_excel_results"] = True
    
    config["save_options"] = save_opts

    # === Calculate past_hours from past_days ===
    if "past_days" in config:
        if not config.get("use_pv", True) or config.get("no_hist_power", False):
            # 仅预测天气模式：不需要历史数据
            config["past_hours"] = 0
        else:
            config["past_hours"] = config["past_days"] * 24

    # === Load raw dataset once ===
    if not os.path.exists(config["data_path"]):
        raise FileNotFoundError(f"Data file not found: {config['data_path']}")
    
    try:
        df = load_raw_data(config["data_path"])
        if df.empty:
            raise ValueError("Loaded dataset is empty")
        if 'ProjectID' not in df.columns:
            raise ValueError("Dataset must contain 'ProjectID' column")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}")

    # === Compose flag tag to name subfolders ===
    use_pv = config.get("use_pv", True)
    use_hist_weather = config.get("use_hist_weather", False)
    use_forecast = config.get("use_forecast", False)
    weather_category = config.get("weather_category", "none")
    use_time_encoding = config.get("use_time_encoding", True)
    past_days = config.get("past_days", 3)
    model_complexity = config.get("model_complexity", "low")
    
    if not use_pv:
        # 仅预测天气模式
        flag_tag = (
            f"pv{str(use_pv).lower()}_"
            f"hist{str(use_hist_weather).lower()}_"
            f"fcst{str(use_forecast).lower()}_"
            f"{weather_category}_"
            f"{'time' if use_time_encoding else 'notime'}_"
            f"nohist_"
            f"comp{model_complexity}"
        )
    else:
        # 正常模式
        flag_tag = (
            f"pv{str(use_pv).lower()}_"
            f"hist{str(use_hist_weather).lower()}_"
            f"fcst{str(use_forecast).lower()}_"
            f"{weather_category}_"
            f"{'time' if use_time_encoding else 'notime'}_"
            f"days{past_days}_"
            f"comp{model_complexity}"
        )

    is_dl = config["model"] in ["Transformer", "LSTM", "GRU", "TCN"]
    alg_type = "dl" if is_dl else "ml"

    # === Train for each project independently ===
    for pid in df["ProjectID"].unique():
        df_proj = df[df["ProjectID"] == pid]
        if df_proj.empty:
            print(f"[WARN] Project {pid} has no data, skipping")
            continue

        # Step 1: Preprocess and normalize
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
            preprocess_features(df_proj, config)

        # Step 2: Create sliding windows and split data
        X_hist, X_fcst, y, hours, dates = create_sliding_windows(
            df_clean,
            past_hours=config["past_hours"],
            future_hours=config["future_hours"],
            hist_feats=hist_feats,
            fcst_feats=fcst_feats,
            no_hist_power=not config.get("use_pv", True)
        )
        
        # Step 2.5: Split data into train/validation/test sets
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
        Xh_va, Xf_va, y_va, hrs_va, dates_va, \
        Xh_te, Xf_te, y_te, hrs_te, dates_te = split_data(
            X_hist, X_fcst, y, hours, dates,
            train_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"]
        )

        # Step 3: Use plant-level save directory (no subfolders)
        # 直接使用厂级别的目录，不创建子文件夹
        plant_save_dir = config["save_dir"]  # 直接使用配置中的save_dir
        os.makedirs(plant_save_dir, exist_ok=True)
        cfg = deepcopy(config)
        cfg["save_dir"] = plant_save_dir

        # Step 4: Train model
        print(f"🔍 调试: 准备开始训练，模型类型: {'DL' if is_dl else 'ML'}")
        print(f"🔍 调试: cfg['model'] = {cfg.get('model', 'NOT_FOUND')}")
        print(f"🔍 调试: cfg['train_params'] = {cfg.get('train_params', 'NOT_FOUND')}")
        print(f"🔍 调试: cfg['model_params'] = {cfg.get('model_params', 'NOT_FOUND')}")
        
        start = time.time()
        try:
            if is_dl:
                print(f"🔍 调试: 调用train_dl_model")
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
                    dates_te,            
                    scaler_target
                )
            metrics["train_time_sec"] = round(time.time() - start, 2)
        except Exception as e:
            print(f"[ERROR] Training failed for Project {pid}: {str(e)}")
            continue

        # Step 5: Save metrics and plots
        cfg["scaler_target"] = scaler_target
        cfg["plant_id"] = str(pid)  # 设置plant_id为ProjectID
        print(f"🔍 调试: 准备调用save_results，plant_id={cfg['plant_id']}")
        print(f"🔍 调试: cfg['train_params'] = {cfg.get('train_params', 'NOT_FOUND')}")
        
        # 根据实验类型选择保存函数
        experiment_type = cfg.get('experiment_type', 'default')
        if experiment_type == 'season_hour_analysis':
            print(f"🔍 调试: 使用season and hour analysis保存模式")
            save_season_hour_results(model, metrics, dates_te, y_te, Xh_te, Xf_te, cfg)
        else:
            print(f"🔍 调试: 使用默认保存模式")
            save_results(model, metrics, dates_te, y_te, Xh_te, Xf_te, cfg)
        print(f"🔍 调试: save_results调用完成")
        print(f"[INFO] Project {pid} | {cfg['model']} done, mse={metrics['mse']:.4f}, rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}, r_square={metrics.get('r_square', 0):.4f}")
        print(f"[METRICS] inference_time={metrics.get('inference_time_sec', 0):.4f}, param_count={metrics.get('param_count', 0)}, samples_count={metrics.get('samples_count', 0)}")
        print(f"[METRICS] best_epoch={metrics.get('best_epoch', 0)}, final_lr={metrics.get('final_lr', 0):.6f}")
        print(f"[METRICS] nrmse={metrics.get('nrmse', 0):.4f}, smape={metrics.get('smape', 0):.4f}, gpu_memory_used={metrics.get('gpu_memory_used', 0)}")

if __name__ == "__main__":
    main()
