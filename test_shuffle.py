#!/usr/bin/env python3
"""
测试数据shuffle功能
"""

import pandas as pd
import numpy as np
from data.data_utils import load_raw_data, preprocess_features, create_sliding_windows, split_data

def test_shuffle():
    """测试shuffle功能"""
    print("🔍 测试数据shuffle功能")
    print("=" * 50)
    
    # 加载数据
    df = load_raw_data('data/Project1033.csv')
    print(f"原始数据形状: {df.shape}")
    
    # 预处理
    config = {
        'use_hist_weather': True,
        'use_forecast': True,
        'past_days': 3,
        'future_hours': 24
    }
    
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, config)
    print(f"预处理后数据形状: {df_clean.shape}")
    print(f"历史特征: {hist_feats}")
    print(f"预测特征: {fcst_feats}")
    
    # 创建滑动窗口
    Xh, Xf, y, hrs, dates = create_sliding_windows(
        df_clean,
        past_hours=config['past_days'] * 24,
        future_hours=config['future_hours'],
        hist_feats=hist_feats,
        fcst_feats=fcst_feats
    )
    
    print(f"滑动窗口数据形状:")
    print(f"  Xh: {Xh.shape}")
    print(f"  Xf: {Xf.shape if Xf is not None else None}")
    print(f"  y: {y.shape}")
    print(f"  dates: {len(dates)}")
    
    # 显示原始日期顺序
    print(f"\n原始日期顺序 (前10个):")
    for i, date in enumerate(dates[:10]):
        print(f"  {i}: {date}")
    
    # 测试不shuffle
    print(f"\n🔍 测试不shuffle:")
    splits_no_shuffle = split_data(Xh, Xf, y, hrs, dates, shuffle=False)
    Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
    Xh_va, Xf_va, y_va, hrs_va, dates_va, \
    Xh_te, Xf_te, y_te, hrs_te, dates_te = splits_no_shuffle
    
    print(f"测试集日期顺序 (前10个):")
    for i, date in enumerate(dates_te[:10]):
        print(f"  {i}: {date}")
    
    # 测试shuffle
    print(f"\n🔍 测试shuffle:")
    splits_shuffle = split_data(Xh, Xf, y, hrs, dates, shuffle=True, random_state=42)
    Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
    Xh_va, Xf_va, y_va, hrs_va, dates_va, \
    Xh_te, Xf_te, y_te, hrs_te, dates_te = splits_shuffle
    
    print(f"测试集日期顺序 (前10个):")
    for i, date in enumerate(dates_te[:10]):
        print(f"  {i}: {date}")
    
    # 检查是否真的shuffle了
    print(f"\n📊 检查shuffle效果:")
    print(f"原始前5个日期: {dates[:5]}")
    print(f"测试集前5个日期: {dates_te[:5]}")
    
    # 检查日期是否连续
    is_continuous = all(dates_te[i] < dates_te[i+1] for i in range(len(dates_te)-1))
    print(f"测试集日期是否连续: {is_continuous}")
    
    if not is_continuous:
        print("✅ 数据已成功shuffle，日期不再连续")
    else:
        print("❌ 数据未shuffle，日期仍然连续")
    
    # 保存测试结果
    test_results = pd.DataFrame({
        'date': dates_te,
        'hour': hrs_te.flatten() if len(hrs_te.shape) > 1 else hrs_te,
        'prediction': y_te.flatten() if len(y_te.shape) > 1 else y_te
    })
    
    test_results.to_csv('test_shuffle_results.csv', index=False)
    print(f"\n✅ 测试结果保存到: test_shuffle_results.csv")
    print(f"   测试集样本数: {len(test_results)}")
    print(f"   日期范围: {test_results['date'].min()} 到 {test_results['date'].max()}")

if __name__ == "__main__":
    test_shuffle()
