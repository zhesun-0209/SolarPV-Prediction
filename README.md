# Solar Power Forecasting Pipeline

A unified, end-to-end framework to train, evaluate, and compare multiple machine learning (ML) and deep learning (DL) models for photovoltaic (PV) power forecasting.

**Supported models:**  
- **Deep Learning:** 
  - Transformer
  - LSTM
  - GRU
  - TCN  
- **Machine Learning:** 
  - Random Forest (RF)
  - Gradient Boosting (GBR)
  - XGBoost (XGB)
  - LightGBM (LGBM)

---

## 📁 Project Structure  
    .
    ├── config/
    │   └── default.yaml            # Default configuration file
    ├── data/
    │   └── data_utils.py           # Loading, preprocessing & windowing
    ├── eval/
    │   ├── eval_utils.py           # Save results & aggregate metrics
    │   └── plot_utils.py           # Forecast & training-curve plotting
    ├── models/
    │   ├── transformer.py          # Transformer architecture
    │   ├── rnn_models.py           # LSTM & GRU definitions
    │   ├── tcn.py                  # Temporal Convolutional Network
    │   └── ml_models.py            # RF/GBR/XGB/LGBM training wrappers
    ├── train/
    │   ├── train_utils.py          # Optimizer, scheduler, earlystop, weights
    │   ├── train_dl.py             # DL training & meta-weight loop
    │   └── train_ml.py             # ML training & single-epoch logging
    ├── main.py                     # Orchestrates config → data → train → eval
    ├── requirements.txt            # Python package dependencies
    └── README.md                   # This file

## 🚀 Installation  
1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/solar-forecasting.git  
   cd solar-forecasting
   ``` 

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv  
   source venv/bin/activate
   ``` 

3. **Install dependencies** 
   ```bash
   pip install -r requirements.txt
   ```  

4. **Verify installations**
   ```bash
   python -c "import torch, sklearn, xgboost, lightgbm, yaml, pandas; print('OK')"
   ```

## ⚙️ Configuration  
All pipeline settings live in `config/default.yaml`. Adjust as needed:

    # Paths
    data_path:      '/path/to/data.csv'
    save_dir:       '/path/to/outputs'

    # Model & Ablation
    model:          'Transformer'      # Transformer, LSTM, GRU, TCN, RF, GBR, XGB, LGBM
    past_hours:     72
    future_hours:   24
    use_feature:    true               # include exogenous features
    use_time:       true               # include time encodings
    use_forecast:   true               # include weather forecasts
    use_stats:      true               # include statistical features
    use_meta:       true               # enable dynamic meta-weight

    # Data splits
    train_ratio:    0.8
    val_ratio:      0.1

    # Plotting options
    plot_days:      7

    # Model-specific parameters
    model_params:
      # Deep Learning
      d_model:       64
      n_heads:       4
      n_layers:      2
      hidden_dim:    64
      dropout:       0.1
      tcn_channels:  [64, 64]
      kernel_size:   3

      # Machine Learning
      n_estimators:  100
      max_depth:     null
      learning_rate: 0.1
      random_state:  42

    # Training parameters
    train_params:
      batch_size:           32
      epochs:               50
      learning_rate:        1e-3
      weight_decay:         1e-4
      early_stop_patience:  10
      loss_type:            'mse'
      future_hours:         24

Tip: No code changes required—simply update this file and re-run.
## 🔄 Usage  
Run the entire pipeline:

    python main.py --config config/default.yaml

This will:
- Load & preprocess data
- Construct chronological sliding windows
- Split data into train/validation/test
- Train the specified model
- Save outputs in `<save_dir>/<model>/`:
    - `summary.csv` (aggregated metrics)
    - `predictions.csv` (hourly true vs. predicted)
    - `training_log.csv` (epoch losses)
    - Model weights (`best_model.pth` or `best_model.pkl`)
    - Forecast plot (`forecast_{plot_days}d.png`)
    - Training curve (`training_curve.png`)
    - Meta-weight evolution charts (`hour_weights/*.png`)

## 📂 Data Requirements  
Your input CSV must include:
- Datetime fields
  - Year
  - Month
  - Day
  - Hour
- Historical features:
    - apparent_temperature_min [degF]
    - relative_humidity_min [percent]
    - wind_speed_prop_avg [mile/hr]
    - solar_insolation_total [MJ/m^2]
- Forecast features: 
  - temperature_2m
  - relative_humidity_2m
  - wind_speed_10m
  - direct_radiation
- Statistical features
  - mean_hour_stat
  - var_hour_stat
- Target:
  - Electricity Generated

🔧 Module Overview  
- `data/data_utils.py`: Data loading, feature selection/scaling, sliding-window creation, chronological split.  
- `train/train_utils.py`: Optimizer/scheduler creation, early stopping, dynamic weight computation & plotting.  
- `train/train_dl.py`: Deep learning training loop with optional meta-weight; saves `best_model.pth`.  
- `train/train_ml.py`: Traditional ML training (RF/GBR/XGB/LGBM); saves `best_model.pkl`.  
- `eval/eval_utils.py`: Writes `summary.csv`, `predictions.csv`, `training_log.csv`; calls plotting functions.  
- `eval/plot_utils.py`: Generates continuous forecast vs. true plots and training curves.

## 📊 Outputs  
After running, inspect `<save_dir>/<Model>/`:

    summary.csv
    predictions.csv
    training_log.csv
    best_model.pth   # or best_model.pkl
    forecast_{plot_days}d.png
    training_curve.png
    hour_weights/    # if use_meta=True

## 🛠️ Extending  
- **Add new model:** implement under `models/` and register in `train_dl.py` or `train_ml.py`.  
- **New features:** update `BASE_*_FEATURES` in `data/data_utils.py`.  
- **Alternate loss:** configure via `train_params.loss_type` or modify in `train/train_dl.py`.

## 📜 License  
MIT License  
