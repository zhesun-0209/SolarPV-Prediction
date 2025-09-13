#!/usr/bin/env python3
"""
调试训练脚本 - 逐步检查训练过程
"""

import sys
import os
sys.path.append('.')

import yaml
import traceback

def debug_training():
    """调试训练过程"""
    print("🔍 开始调试训练过程...")
    
    try:
        # 1. 加载配置
        print("\n1️⃣ 加载配置...")
        with open('config/projects/1140/LSTM_low_PV_24h_TE.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置加载成功: {config['model']}")
        
        # 2. 检查训练参数
        print("\n2️⃣ 检查训练参数...")
        train_params = config.get('train_params', {})
        print(f"train_params: {train_params}")
        
        if 'batch_size' not in train_params:
            print("❌ 缺少batch_size参数")
            return False
        else:
            print(f"✅ batch_size: {train_params['batch_size']}")
            
        if 'learning_rate' not in train_params:
            print("❌ 缺少learning_rate参数")
            return False
        else:
            print(f"✅ learning_rate: {train_params['learning_rate']}")
        
        # 3. 导入模块
        print("\n3️⃣ 导入模块...")
        try:
            from data.data_utils import load_raw_data, preprocess_features, create_sliding_windows, split_data
            print("✅ 数据工具模块导入成功")
        except Exception as e:
            print(f"❌ 数据工具模块导入失败: {e}")
            return False
            
        try:
            from train.train_dl import train_dl_model
            print("✅ 深度学习训练模块导入成功")
        except Exception as e:
            print(f"❌ 深度学习训练模块导入失败: {e}")
            return False
        
        # 4. 数据加载
        print("\n4️⃣ 加载数据...")
        try:
            df = load_raw_data('data/Project1140.csv')
            df_proj = df[df['ProjectID'] == 1140.0]
            print(f"✅ 数据加载成功: {df_proj.shape}")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            traceback.print_exc()
            return False
        
        # 5. 数据预处理
        print("\n5️⃣ 数据预处理...")
        try:
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df_proj, config)
            print(f"✅ 数据预处理成功: {df_clean.shape}")
        except Exception as e:
            print(f"❌ 数据预处理失败: {e}")
            traceback.print_exc()
            return False
        
        # 6. 创建滑动窗口
        print("\n6️⃣ 创建滑动窗口...")
        try:
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean,
                past_hours=config['past_hours'],
                future_hours=config['future_hours'],
                hist_feats=hist_feats,
                fcst_feats=fcst_feats,
                no_hist_power=not config.get('use_pv', True)
            )
            print(f"✅ 滑动窗口创建成功: X_hist={X_hist.shape}, y={y.shape}")
        except Exception as e:
            print(f"❌ 滑动窗口创建失败: {e}")
            traceback.print_exc()
            return False
        
        # 7. 数据分割
        print("\n7️⃣ 数据分割...")
        try:
            Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
            Xh_va, Xf_va, y_va, hrs_va, dates_va, \
            Xh_te, Xf_te, y_te, hrs_te, dates_te = split_data(
                X_hist, X_fcst, y, hours, dates,
                train_ratio=config['train_ratio'],
                val_ratio=config['val_ratio']
            )
            print(f"✅ 数据分割成功: 训练集={Xh_tr.shape}, 测试集={Xh_te.shape}")
        except Exception as e:
            print(f"❌ 数据分割失败: {e}")
            traceback.print_exc()
            return False
        
        # 8. 检查模型参数
        print("\n8️⃣ 检查模型参数...")
        try:
            complexity = config.get('model_complexity', 'low')
            if complexity in config['model_params']:
                mp = config['model_params'][complexity].copy()
                mp['use_forecast'] = config.get('use_forecast', False)
                mp['past_hours'] = config['past_hours']
                mp['future_hours'] = config['future_hours']
                print(f"✅ 模型参数获取成功: {mp}")
            else:
                print(f"❌ 模型参数获取失败: 复杂度 {complexity} 不存在")
                return False
        except Exception as e:
            print(f"❌ 模型参数检查失败: {e}")
            traceback.print_exc()
            return False
        
        # 9. 检查训练参数获取
        print("\n9️⃣ 检查训练参数获取...")
        try:
            # 模拟train_dl.py中的参数获取
            bs = int(config['train_params']['batch_size'])
            lr = float(config['train_params']['learning_rate'])
            print(f"✅ 训练参数获取成功: batch_size={bs}, learning_rate={lr}")
        except Exception as e:
            print(f"❌ 训练参数获取失败: {e}")
            traceback.print_exc()
            return False
        
        # 10. 尝试创建模型
        print("\n🔟 尝试创建模型...")
        try:
            from models.rnn_models import LSTM
            import torch
            
            hist_dim = Xh_tr.shape[2]
            fcst_dim = Xf_tr.shape[2] if Xf_tr is not None else 0
            
            model = LSTM(hist_dim, fcst_dim, mp)
            print(f"✅ 模型创建成功: 参数数量={sum(p.numel() for p in model.parameters())}")
            
            # 测试前向传播
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(2, config['past_hours'], hist_dim)
                output = model(test_input)
                print(f"✅ 前向传播测试成功: 输入={test_input.shape}, 输出={output.shape}")
                
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            traceback.print_exc()
            return False
        
        print("\n🎉 所有检查通过! 训练应该可以正常进行。")
        return True
        
    except Exception as e:
        print(f"\n💥 调试过程中出现异常: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_training()
    if success:
        print("\n✅ 调试完成，没有发现问题")
    else:
        print("\n❌ 调试发现问题，请检查上述错误")
