# ======== data/data_utils.py ========

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 基于实际数据中的天气特征，简化为两种类别
# 太阳辐射特征 - 最重要的特征
IRRADIANCE_FEATURES = [
    'global_tilted_irradiance',    # 全球倾斜辐射 (最重要的辐射特征)
]

# 全部天气特征 - 包含所有天气变量
ALL_WEATHER_FEATURES = [
    'global_tilted_irradiance',    # 全球倾斜辐射
    'vapour_pressure_deficit',     # 水汽压差
    'relative_humidity_2m',        # 相对湿度
    'temperature_2m',              # 温度
    'wind_gusts_10m',             # 10米阵风
    'cloud_cover_low',            # 低云覆盖
    'wind_speed_100m',            # 100米风速
    'snow_depth',                 # 雪深
    'dew_point_2m',               # 露点温度
    'surface_pressure',           # 表面气压
    'precipitation',              # 降水
]

# 根据天气特征类别选择特征
def get_weather_features_by_category(weather_category):
    """
    根据天气特征类别返回天气特征
    
    Args:
        weather_category: 'irradiance', 'all_weather'
    
    Returns:
        list: 选中的天气特征列表
    """
    if weather_category == 'irradiance':
        return IRRADIANCE_FEATURES
    elif weather_category == 'all_weather':
        return ALL_WEATHER_FEATURES
    else:
        raise ValueError(f"Invalid weather_category: {weather_category}")

# 保持向后兼容性
BASE_HIST_FEATURES = IRRADIANCE_FEATURES
BASE_FCST_FEATURES = IRRADIANCE_FEATURES

# 时间编码特征
TIME_FEATURES = ['month_cos', 'month_sin', 'hour_cos', 'hour_sin']

TARGET_COL = 'Capacity Factor'

# 统计特征函数已移除

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    return df

# def preprocess_features(df: pd.DataFrame, config: dict):
#     df_clean = df.dropna(subset=[TARGET_COL]).copy()

#     hist_feats = []
#     fcst_feats = []

#     if config.get('use_hist_weather', True):
#         hist_feats += BASE_HIST_FEATURES

#     if config.get('use_time', False):
#         for col in ('Month_cos', 'Hour_sin', 'Hour_cos'):
#             if col not in hist_feats:
#                 hist_feats.append(col)

#     if config.get('use_stats', False):
#         hist_feats += BASE_STAT_FEATURES

#     if config.get('use_forecast', False):
#         fcst_feats += BASE_FCST_FEATURES

#     # Drop rows with missing values in all relevant features
#     na_check_feats = hist_feats + fcst_feats + [TARGET_COL]
#     df_clean = df_clean.dropna(subset=na_check_feats).reset_index(drop=True)

#     if hist_feats:
#         scaler_hist = MinMaxScaler()
#         df_clean[hist_feats] = scaler_hist.fit_transform(df_clean[hist_feats])
#     else:
#         scaler_hist = None

#     scaler_target = MinMaxScaler()
#     df_clean[[TARGET_COL]] = scaler_target.fit_transform(df_clean[[TARGET_COL]])

#     if fcst_feats:
#         scaler_fcst = MinMaxScaler()
#         df_clean[fcst_feats] = scaler_fcst.fit_transform(df_clean[fcst_feats])
#     else:
#         scaler_fcst = None

#     df_clean = df_clean.sort_values('Datetime').reset_index(drop=True)

#     return df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target
def preprocess_features(df: pd.DataFrame, config: dict):
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    # 添加时间编码特征（根据开关决定）
    use_time_encoding = config.get('use_time_encoding', True)
    if use_time_encoding:
        df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['Month'] / 12)
        df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['Month'] / 12)
        df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['Hour'] / 24)
        df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['Hour'] / 24)

    # 构建特征列表
    hist_feats = []
    fcst_feats = []

    # 获取天气特征类别
    weather_category = config.get('weather_category', 'irradiance')

    # PV特征（历史发电量）
    if config.get('use_pv', False):
        # 创建历史Capacity Factor特征（重命名避免与目标变量冲突）
        df_clean['Capacity_Factor_hist'] = df_clean[TARGET_COL]
        hist_feats.append('Capacity_Factor_hist')

    # 历史天气特征
    if config.get('use_hist_weather', False):
        hist_feats += get_weather_features_by_category(weather_category)

    # 时间编码特征（根据开关决定）
    if use_time_encoding:
        hist_feats += TIME_FEATURES

    # 预测特征
    if config.get('use_forecast', False):
        fcst_feats += get_weather_features_by_category(weather_category)

    # 确保所有特征都存在
    available_hist_feats = [f for f in hist_feats if f in df_clean.columns]
    available_fcst_feats = [f for f in fcst_feats if f in df_clean.columns]

    # 删除缺失值
    na_check_feats = available_hist_feats + available_fcst_feats + [TARGET_COL]
    df_clean = df_clean.dropna(subset=na_check_feats).reset_index(drop=True)

    # 标准化特征
    scaler_hist = MinMaxScaler()
    if available_hist_feats:
        # 检查特征是否有足够的变异性
        for feat in available_hist_feats:
            if df_clean[feat].std() == 0:
                print(f"⚠️ 特征 {feat} 标准差为0，添加微小噪声避免除零错误")
                df_clean[feat] += np.random.normal(0, 1e-8, len(df_clean))
        df_clean[available_hist_feats] = scaler_hist.fit_transform(df_clean[available_hist_feats])

    scaler_fcst = MinMaxScaler()
    if available_fcst_feats:
        # 检查特征是否有足够的变异性
        for feat in available_fcst_feats:
            if df_clean[feat].std() == 0:
                print(f"⚠️ 特征 {feat} 标准差为0，添加微小噪声避免除零错误")
                df_clean[feat] += np.random.normal(0, 1e-8, len(df_clean))
        df_clean[available_fcst_feats] = scaler_fcst.fit_transform(df_clean[available_fcst_feats])

    # Capacity Factor不需要标准化（范围0-100）
    scaler_target = None

    df_clean = df_clean.sort_values('Datetime').reset_index(drop=True)

    return df_clean, available_hist_feats, available_fcst_feats, scaler_hist, scaler_fcst, scaler_target

def create_sliding_windows(df, past_hours, future_hours, hist_feats, fcst_feats, no_hist_power=False):
    """
    创建滑动窗口样本，允许时间不连续
    每个样本包含：前n天历史数据 + 预测当天的预测数据
    
    Args:
        no_hist_power: 如果为True，不使用历史发电量数据，只使用预测天气
    """
    X_hist, y, hours, dates = [], [], [], []
    X_fcst = [] if fcst_feats else None  # 只有在需要预测特征时才初始化
    n = len(df)
    
    # 按天分组，每天24小时
    df['Date'] = df['Datetime'].dt.date
    daily_groups = df.groupby('Date')
    daily_dates = list(daily_groups.groups.keys())
    
    # 确保有足够的历史天数
    if no_hist_power:
        min_days = 0  # 仅预测天气模式不需要历史数据
    else:
        min_days = past_hours // 24 + 1  # 至少需要这么多天
    
    if len(daily_dates) < min_days + 1:  # +1 for prediction day
        raise ValueError(f"数据不足：需要至少{min_days + 1}天的数据")
    
    # 为每个预测日创建样本
    for pred_date_idx in range(min_days, len(daily_dates)):
        pred_date = daily_dates[pred_date_idx]
        pred_day_data = daily_groups.get_group(pred_date)
        
        if no_hist_power:
            # 无历史发电量模式：只使用预测天气数据
            fut_win = pred_day_data.head(future_hours)
            
            if len(fut_win) < future_hours:
                continue
            
            # 构建样本（只有预测特征）
            if fcst_feats:
                X_fcst.append(fut_win[fcst_feats].values)
            
            y.append(fut_win[TARGET_COL].values)
            hours.append(fut_win['Hour'].values)
            dates.append(fut_win['Datetime'].iloc[-1])
            
            # 对于无历史发电量模式，X_hist为空
            X_hist.append(np.array([]).reshape(0, len(hist_feats)) if hist_feats else np.array([]).reshape(0, 0))
        else:
            # 正常模式：使用历史数据
            # 收集历史数据（前n天）
            hist_data = []
            for hist_date_idx in range(max(0, pred_date_idx - min_days), pred_date_idx):
                hist_date = daily_dates[hist_date_idx]
                hist_day_data = daily_groups.get_group(hist_date)
                hist_data.append(hist_day_data)
            
            if len(hist_data) == 0:
                continue
                
            # 合并历史数据
            hist_combined = pd.concat(hist_data, ignore_index=True)
            
            # 如果历史数据不足past_hours，跳过
            if len(hist_combined) < past_hours:
                continue
                
            # 取最后past_hours小时的历史数据
            hist_win = hist_combined.tail(past_hours)
            
            # 预测数据（预测当天的数据）
            fut_win = pred_day_data.head(future_hours)
            
            if len(fut_win) < future_hours:
                continue
            
            # 构建样本
            X_hist.append(hist_win[hist_feats].values)
            
            if fcst_feats:
                # 预测天气：使用预测当天的天气数据
                X_fcst.append(fut_win[fcst_feats].values)
            
            y.append(fut_win[TARGET_COL].values)
            hours.append(fut_win['Hour'].values)
            dates.append(fut_win['Datetime'].iloc[-1])
    
    if len(X_hist) == 0:
        raise ValueError("无法创建任何有效样本")
    
    X_hist = np.stack(X_hist)
    y = np.stack(y)
    hours = np.stack(hours)
    X_fcst = np.stack(X_fcst) if fcst_feats else None

    return X_hist, X_fcst, y, hours, dates

def split_data(X_hist, X_fcst, y, hours, dates, train_ratio=0.8, val_ratio=0.1, shuffle=True, random_state=42):
    """
    分割数据为训练集、验证集和测试集
    由于样本已经是非连续的时间窗口，可以安全地shuffle和按比例分割
    每个样本都是独立的预测日，不存在数据泄漏问题
    """
    N = X_hist.shape[0]
    
    # 创建随机索引
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(N)
    else:
        indices = np.arange(N)
    
    # 计算分割点
    i_tr = int(N * train_ratio)
    i_val = int(N * (train_ratio + val_ratio))
    
    # 分割索引
    train_idx = indices[:i_tr]
    val_idx = indices[i_tr:i_val]
    test_idx = indices[i_val:]
    
    # 定义切片函数
    def slice_array(arr, indices):
        if isinstance(arr, np.ndarray):
            return arr[indices]
        else:
            # 处理列表类型
            return [arr[i] for i in indices]

    # 分割所有数组
    Xh_tr, Xh_va, Xh_te = slice_array(X_hist, train_idx), slice_array(X_hist, val_idx), slice_array(X_hist, test_idx)
    y_tr, y_va, y_te = slice_array(y, train_idx), slice_array(y, val_idx), slice_array(y, test_idx)
    hrs_tr, hrs_va, hrs_te = slice_array(hours, train_idx), slice_array(hours, val_idx), slice_array(hours, test_idx)
    
    # 处理日期列表
    dates_tr = [dates[i] for i in train_idx]
    dates_va = [dates[i] for i in val_idx]
    dates_te = [dates[i] for i in test_idx]

    # 处理预测特征
    if X_fcst is not None:
        Xf_tr, Xf_va, Xf_te = slice_array(X_fcst, train_idx), slice_array(X_fcst, val_idx), slice_array(X_fcst, test_idx)
    else:
        Xf_tr = Xf_va = Xf_te = None

    return (
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
        Xh_va, Xf_va, y_va, hrs_va, dates_va,
        Xh_te, Xf_te, y_te, hrs_te, dates_te
    )

