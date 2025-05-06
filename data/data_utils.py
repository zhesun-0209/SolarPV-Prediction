# ======== data/data_utils.py ========

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_HIST_FEATURES = [
    'apparent_temperature_min [degF]',
    'relative_humidity_min [percent]',
    'wind_speed_prop_avg [mile/hr]',
    'solar_insolation_total [MJ/m^2]',
    'Month_cos', 'Hour_sin', 'Hour_cos',
]
BASE_STAT_FEATURES = ['mean_hour_stat', 'var_hour_stat']
BASE_FCST_FEATURES = [
    'temperature_2m', 'relative_humidity_2m',
    'wind_speed_10m', 'direct_radiation'
]
TARGET_COL = 'Electricity Generated'

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    return df

def preprocess_features(df: pd.DataFrame, config: dict):
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    # Initialize feature lists
    hist_feats = [TARGET_COL]
    fcst_feats = []

    # Append weather-based historical features if use_feature is True
    if config.get('use_feature', True):
        hist_feats += BASE_HIST_FEATURES

    # Add time features if use_time = True
    if config.get('use_time', False):
        for col in ('Month_cos', 'Hour_sin', 'Hour_cos'):
            if col not in hist_feats:
                hist_feats.append(col)

    # Add statistical features
    if config.get('use_stats', False):
        hist_feats += BASE_STAT_FEATURES

    # Add forecast features
    if config.get('use_forecast', False):
        fcst_feats += BASE_FCST_FEATURES

    # Drop rows with missing required features
    na_check_feats = hist_feats + fcst_feats
    df_clean = df_clean.dropna(subset=na_check_feats).reset_index(drop=True)

    # Normalize features
    scaler_hist = MinMaxScaler()
    df_clean[hist_feats] = scaler_hist.fit_transform(df_clean[hist_feats])

    scaler_fcst = None
    if fcst_feats:
        scaler_fcst = MinMaxScaler()
        df_clean[fcst_feats] = scaler_fcst.fit_transform(df_clean[fcst_feats])

    scaler_target = MinMaxScaler()
    df_clean[[TARGET_COL]] = scaler_target.fit_transform(df_clean[[TARGET_COL]])

    df_clean = df_clean.sort_values('Datetime').reset_index(drop=True)
    return df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target

def create_sliding_windows(df, past_hours, future_hours, hist_feats, fcst_feats):
    X_hist, X_fcst, y, hours, dates = [], [], [], [], []
    n = len(df)

    for start in range(0, n - past_hours - future_hours + 1, future_hours):
        h_end = start + past_hours
        f_end = h_end + future_hours

        hist_win = df.iloc[start:h_end]
        fut_win = df.iloc[h_end:f_end]

        X_hist.append(hist_win[hist_feats].values)

        if fcst_feats:
            X_fcst.append(fut_win[fcst_feats].values)

        y.append(fut_win[TARGET_COL].values)
        hours.append(fut_win['Hour'].values)
        dates.append(fut_win['Datetime'].iloc[-1])

    X_hist = np.stack(X_hist)
    y = np.stack(y)
    hours = np.stack(hours)
    X_fcst = np.stack(X_fcst) if fcst_feats else None

    return X_hist, X_fcst, y, hours, dates

def split_data(X_hist, X_fcst, y, hours, dates, train_ratio=0.8, val_ratio=0.1):
    N = X_hist.shape[0]
    i_tr = int(N * train_ratio)
    i_val = int(N * (train_ratio + val_ratio))
    slice_ = lambda arr: (arr[:i_tr], arr[i_tr:i_val], arr[i_val:])

    Xh_tr, Xh_va, Xh_te = slice_(X_hist)
    y_tr, y_va, y_te = slice_(y)
    hrs_tr, hrs_va, hrs_te = slice_(hours)
    dates_arr = np.array(dates)
    dates_tr, dates_va, dates_te = slice_(dates_arr)

    if X_fcst is not None:
        Xf_tr, Xf_va, Xf_te = slice_(X_fcst)
    else:
        Xf_tr = Xf_va = Xf_te = None

    return (
        Xh_tr, Xf_tr, y_tr, hrs_tr, list(dates_tr),
        Xh_va, Xf_va, y_va, hrs_va, list(dates_va),
        Xh_te, Xf_te, y_te, hrs_te, list(dates_te)
    )

# ======== main.py ========

    for pid in df["ProjectID"].unique():
        df_proj_raw = df[df["ProjectID"] == pid]
        if df_proj_raw.empty:
            print(f"[WARN] Project {pid} has no data, skipping")
            continue

        df_proj, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
            preprocess_features(df_proj_raw, config)

        Xh, Xf, y, hrs, dates = create_sliding_windows(
            df_proj,
            past_hours=config["past_hours"],
            future_hours=config["future_hours"],
            hist_feats=hist_feats,
            fcst_feats=fcst_feats
        )

        splits = split_data(Xh, Xf, y, hrs, dates,
                            train_ratio=config["train_ratio"],
                            val_ratio=config["val_ratio"])

        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
        Xh_va, Xf_va, y_va, hrs_va, dates_va, \
        Xh_te, Xf_te, y_te, hrs_te, dates_te = splits

        proj_dir = os.path.join(
            config["save_dir"], f"Project_{pid}", alg_type,
            config["model"].lower(), flag_tag
        )
        os.makedirs(proj_dir, exist_ok=True)

        cfg = deepcopy(config)
        cfg["save_dir"] = proj_dir

        # Train
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
                cfg, Xh_tr, Xf_tr, y_tr,
                Xh_te, Xf_te, y_te,
                scaler_target
            )
        metrics["train_time_sec"] = round(time.time() - start, 2)

        cfg["scaler_target"] = scaler_target
        save_results(
            model, metrics, dates_te, y_te, Xh_te, Xf_te, cfg
        )
        print(f"[INFO] Project {pid} | {cfg['model']} done, test_loss={metrics['test_loss']:.4f}")
