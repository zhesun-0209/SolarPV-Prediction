"""
data/data_utils.py

Data loading and preprocessing utilities for solar power forecasting.
Ensures:
  - Ablation flags (use_feature, use_time, use_stats, use_forecast) are respected.
  - Chronological order by sorting on 'Datetime'.
  - Day-by-day sliding windows with hour-of-day output for meta-weighting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Default feature lists
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
    """
    Load raw CSV and parse Year/Month/Day/Hour into a single 'Datetime' column.

    Args:
        path: file path to CSV.

    Returns:
        df: DataFrame including parsed 'Datetime'.
    """
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    return df


def preprocess_features(df: pd.DataFrame, config: dict):
    """
    Clean and scale features based on config flags.
    Applies in order:
      1. Drop rows with missing target.
      2. Select features using: use_feature, use_time, use_stats, use_forecast.
      3. Drop any rows with NA in chosen features.
      4. Scale historical, forecast, and target separately via MinMaxScaler.
      5. Sort by 'Datetime' to guarantee chronological order.

    Args:
        df: raw DataFrame with Weather & target columns.
        config: dict containing ablation flags and paths.

    Returns:
        df_clean: cleaned & sorted DataFrame.
        hist_feats: list of column names used as history features.
        fcst_feats: list of column names used as forecast features.
        scaler_hist: fitted MinMaxScaler for hist_feats.
        scaler_fcst: fitted MinMaxScaler for fcst_feats (or None if not used).
        scaler_target: fitted MinMaxScaler for TARGET_COL.
    """
    # 1) Drop rows missing the target
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    # 2) Determine feature sets
    if not config.get('use_feature', True):
        # Only past target
        hist_feats = [TARGET_COL]
        fcst_feats = []
    else:
        # History
        hist_feats = BASE_HIST_FEATURES.copy()
        if not config.get('use_time', True):
            for col in ('Month_cos', 'Hour_sin', 'Hour_cos'):
                hist_feats.remove(col)
        if config.get('use_stats', False):
            hist_feats += BASE_STAT_FEATURES
        hist_feats += [TARGET_COL]
        # Forecast
        fcst_feats = BASE_FCST_FEATURES if config.get('use_forecast', False) else []

    # 3) Drop any remaining NA
    df_clean = df_clean.dropna(subset=hist_feats + fcst_feats).reset_index(drop=True)

    # 4) Scale features
    scaler_hist = MinMaxScaler()
    df_clean[hist_feats] = scaler_hist.fit_transform(df_clean[hist_feats])

    scaler_fcst = None
    if fcst_feats:
        scaler_fcst = MinMaxScaler()
        df_clean[fcst_feats] = scaler_fcst.fit_transform(df_clean[fcst_feats])

    scaler_target = MinMaxScaler()
    df_clean[[TARGET_COL]] = scaler_target.fit_transform(df_clean[[TARGET_COL]])

    # 5) Ensure chronological order
    df_clean = df_clean.sort_values('Datetime').reset_index(drop=True)

    return df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target


def create_sliding_windows(
    df: pd.DataFrame,
    past_hours: int,
    future_hours: int,
    hist_feats: list,
    fcst_feats: list
):
    """
    Build non-overlapping day-by-day sliding windows.

    Args:
        df: DataFrame sorted by 'Datetime'.
        past_hours: number of past hours to include.
        future_hours: forecast horizon (e.g. 24 for next day).
        hist_feats: list of history feature columns.
        fcst_feats: list of forecast feature columns.

    Returns:
        X_hist: np.ndarray of shape (N, past_hours, len(hist_feats)).
        X_fcst: np.ndarray or None of shape (N, future_hours, len(fcst_feats)).
        y:      np.ndarray of shape (N, future_hours).
        hours:  np.ndarray of shape (N, future_hours) with hour-of-day.
        dates:  list of length N with the end datetime of each window.
    """
    X_hist, X_fcst, y, hours, dates = [], [], [], [], []
    n = len(df)

    # step = future_hours ensures one window per day
    for start in range(0, n - past_hours - future_hours + 1, future_hours):
        h_end = start + past_hours
        f_end = h_end + future_hours

        hist_win = df.iloc[start:h_end]
        fut_win  = df.iloc[h_end:f_end]

        X_hist.append(hist_win[hist_feats].values)
        if fcst_feats:
            X_fcst.append(fut_win[fcst_feats].values)
        y.append(fut_win[TARGET_COL].values)
        hours.append(fut_win['Hour'].values)
        dates.append(fut_win['Datetime'].iloc[-1])

    X_hist = np.stack(X_hist)
    y      = np.stack(y)
    hours  = np.stack(hours)
    X_fcst = np.stack(X_fcst) if fcst_feats else None

    return X_hist, X_fcst, y, hours, dates


def split_data(
    X_hist, X_fcst, y, hours, dates,
    train_ratio=0.8, val_ratio=0.1
):
    """
    Chronological split into train/validation/test sets.

    Args:
        X_hist: (N, past_hours, hist_feats)
        X_fcst: (N, future_hours, fcst_feats) or None
        y:      (N, future_hours)
        hours:  (N, future_hours)
        dates:  list of N datetimes
        train_ratio: fraction for training (first slice)
        val_ratio:   fraction for validation (second slice)

    Returns:
        Tuple: (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
                Xh_va, Xf_va, y_va, hrs_va, dates_va,
                Xh_te, Xf_te, y_te, hrs_te, dates_te)
    """
    N = X_hist.shape[0]
    i_tr  = int(N * train_ratio)
    i_val = int(N * (train_ratio + val_ratio))

    # Helper to slice sequences
    slice_ = lambda arr: (arr[:i_tr], arr[i_tr:i_val], arr[i_val:])

    Xh_tr, Xh_va, Xh_te = slice_(X_hist)
    y_tr,  y_va,  y_te  = slice_(y)
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
