#!/usr/bin/env python3
"""
修复配置文件中的save_options，启用结果保存
"""

import os
import yaml
import glob

def fix_save_options():
    """修复所有配置文件的save_options"""
    config_dir = "config/projects/1140"
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    config_files = [f for f in yaml_files if not f.endswith("config_index.yaml")]
    
    print(f"📁 找到 {len(config_files)} 个配置文件")
    
    fixed_count = 0
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查是否需要修复
            save_options = config.get('save_options', {})
            if save_options.get('save_predictions', True) == False:
                # 需要修复
                config['save_options'] = {
                    'save_excel_results': True,
                    'save_model': False,  # 不保存模型文件，节省空间
                    'save_predictions': True,  # 保存预测结果
                    'save_summary': False,  # 不保存summary.csv
                    'save_training_log': True  # 保存训练日志
                }
                
                # 保存修复后的配置
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                print(f"✅ 修复: {os.path.basename(config_file)}")
                fixed_count += 1
            else:
                print(f"⏭️ 跳过: {os.path.basename(config_file)} (已正确)")
                
        except Exception as e:
            print(f"❌ 修复失败: {os.path.basename(config_file)} - {e}")
    
    print(f"\n🎉 修复完成! 共修复 {fixed_count} 个配置文件")

if __name__ == "__main__":
    fix_save_options()
