#!/usr/bin/env python3
"""
100个Project消融实验系统测试脚本
验证所有组件是否正常工作
"""

import os
import sys
import pandas as pd
from pathlib import Path
import yaml

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """测试导入"""
    print("🔍 测试导入...")
    
    try:
        from utils.drive_results_saver import DriveResultsSaver
        from utils.checkpoint_manager import CheckpointManager
        print("✅ 导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_config_generation():
    """测试配置生成"""
    print("📝 测试配置生成...")
    
    try:
        # 测试生成单个Project配置
        from scripts.generate_multi_project_configs import generate_project_configs
        
        configs = generate_project_configs("Project001")
        
        if len(configs) == 360:
            print(f"✅ 配置生成成功: {len(configs)} 个配置")
            return True
        else:
            print(f"❌ 配置数量错误: 期望360，实际{len(configs)}")
            return False
    except Exception as e:
        print(f"❌ 配置生成失败: {e}")
        return False

def test_drive_saver():
    """测试Drive保存器"""
    print("💾 测试Drive保存器...")
    
    try:
        from utils.drive_results_saver import DriveResultsSaver
        
        # 使用临时目录测试
        saver = DriveResultsSaver("./test_drive_results")
        
        # 测试保存结果
        test_result = {
            'config_name': 'Transformer_high_PV_plus_NWP_72h_TE',
            'status': 'completed',
            'duration': 120.5,
            'mae': 0.1234,
            'rmse': 0.1567,
            'r2': 0.8765,
            'model': 'Transformer',
            'model_complexity': 'high',
            'input_category': 'PV_plus_NWP',
            'lookback_hours': 72,
            'use_time_encoding': True
        }
        
        success = saver.save_experiment_result('Project001', test_result)
        
        if success:
            print("✅ Drive保存器测试成功")
            
            # 测试加载结果
            completed = saver.get_completed_experiments('Project001')
            if len(completed) == 1:
                print("✅ 结果加载测试成功")
                return True
            else:
                print(f"❌ 结果加载失败: 期望1个，实际{len(completed)}个")
                return False
        else:
            print("❌ Drive保存器测试失败")
            return False
    except Exception as e:
        print(f"❌ Drive保存器测试异常: {e}")
        return False

def test_checkpoint_manager():
    """测试检查点管理器"""
    print("🔄 测试检查点管理器...")
    
    try:
        from utils.checkpoint_manager import CheckpointManager
        
        # 使用临时目录测试
        manager = CheckpointManager("./test_drive_results")
        
        # 测试获取Project配置
        configs = manager.get_project_configs('Project001')
        
        if len(configs) > 0:
            print(f"✅ 检查点管理器测试成功: 加载{len(configs)}个配置")
            return True
        else:
            print("❌ 检查点管理器测试失败: 未找到配置")
            return False
    except Exception as e:
        print(f"❌ 检查点管理器测试异常: {e}")
        return False

def test_data_structure():
    """测试数据结构"""
    print("📁 测试数据结构...")
    
    # 检查必要目录
    required_dirs = ['data', 'config', 'scripts', 'utils', 'docs']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"❌ 缺少目录: {dir_name}")
            return False
    
    # 检查必要文件
    required_files = [
        'scripts/generate_multi_project_configs.py',
        'scripts/run_multi_project_experiments.py',
        'utils/drive_results_saver.py',
        'utils/checkpoint_manager.py',
        'run_colab_experiments.py'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ 缺少文件: {file_path}")
            return False
    
    print("✅ 数据结构测试成功")
    return True

def test_sample_data():
    """测试示例数据"""
    print("📊 测试示例数据...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("⚠️ data目录不存在，跳过数据测试")
        return True
    
    csv_files = list(data_dir.glob("Project*.csv"))
    
    if len(csv_files) == 0:
        print("⚠️ 未找到Project数据文件，跳过数据测试")
        return True
    
    # 测试第一个CSV文件
    try:
        df = pd.read_csv(csv_files[0])
        
        # 检查必要列
        required_columns = ['Capacity Factor']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ CSV文件缺少列: {missing_columns}")
            return False
        
        print(f"✅ 数据测试成功: {csv_files[0].name} ({len(df)} 行)")
        return True
    except Exception as e:
        print(f"❌ 数据测试失败: {e}")
        return False

def cleanup_test_files():
    """清理测试文件"""
    print("🧹 清理测试文件...")
    
    test_dirs = ['./test_drive_results', './temp_drive_cache']
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            import shutil
            shutil.rmtree(test_dir)
            print(f"   清理: {test_dir}")

def main():
    """主测试函数"""
    print("🧪 100个Project消融实验系统测试")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("配置生成测试", test_config_generation),
        ("Drive保存器测试", test_drive_saver),
        ("检查点管理器测试", test_checkpoint_manager),
        ("数据结构测试", test_data_structure),
        ("示例数据测试", test_sample_data)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"💥 {test_name} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪")
        print("\n📋 下一步操作:")
        print("1. 将100个Project的CSV文件放入data/目录")
        print("2. 运行: python run_colab_experiments.py")
        print("3. 或运行: ./run_100_projects.sh")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    
    # 清理测试文件
    cleanup_test_files()

if __name__ == "__main__":
    main()
