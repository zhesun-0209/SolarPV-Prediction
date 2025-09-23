#!/usr/bin/env python3
"""
测试season and hour analysis配置生成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_season_hour_configs import generate_season_hour_configs, save_season_hour_configs

def test_config_generation():
    """测试配置生成"""
    print("🧪 测试season and hour analysis配置生成")
    print("=" * 50)
    
    # 测试单个项目的配置生成
    test_project_id = "171"
    
    try:
        # 生成配置
        configs = generate_season_hour_configs(test_project_id)
        print(f"✅ 成功生成 {len(configs)} 个配置")
        
        # 显示配置信息
        for i, config_info in enumerate(configs, 1):
            print(f"\n配置 {i}: {config_info['name']}")
            print(f"  模型: {config_info['model']}")
            print(f"  回看小时: {config_info['lookback_hours']}")
            print(f"  复杂度: {config_info['complexity_level']}")
            print(f"  天气设置: {config_info['weather_setup']}")
        
        # 保存配置（测试）
        print(f"\n💾 保存配置到文件...")
        count = save_season_hour_configs(test_project_id, configs)
        print(f"✅ 成功保存 {count} 个配置文件")
        
        print("\n🎉 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_generation()
