#!/usr/bin/env python3
"""
Colab配置文件生成脚本
为所有项目生成配置文件
"""

import os
import sys
import subprocess
import glob

def main():
    """主函数"""
    print("🔧 Colab配置文件生成脚本")
    print("=" * 50)
    
    # 检查是否有配置文件生成脚本
    if not os.path.exists("scripts/generate_dynamic_project_configs.py"):
        print("❌ 配置文件生成脚本不存在")
        return
    
    # 检查data目录
    data_files = glob.glob("data/Project*.csv")
    if not data_files:
        print("❌ 未找到任何Project数据文件")
        return
    
    print(f"📊 找到 {len(data_files)} 个Project数据文件")
    
    # 运行配置文件生成脚本
    print("🚀 开始生成配置文件...")
    try:
        result = subprocess.run([
            sys.executable, "scripts/generate_dynamic_project_configs.py"
        ], capture_output=True, text=True, timeout=600)  # 10分钟超时
        
        if result.returncode == 0:
            print("✅ 配置文件生成成功")
            print("📋 生成日志:")
            print(result.stdout)
        else:
            print("❌ 配置文件生成失败")
            print("📋 错误信息:")
            print(result.stderr)
            return
            
    except subprocess.TimeoutExpired:
        print("⏰ 配置文件生成超时")
        return
    except Exception as e:
        print(f"💥 配置文件生成异常: {str(e)}")
        return
    
    # 检查生成的配置文件
    print("\n📁 检查生成的配置文件...")
    config_dirs = glob.glob("config/projects/*")
    config_dirs = [d for d in config_dirs if os.path.isdir(d)]
    
    total_configs = 0
    for config_dir in config_dirs:
        yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
        config_files = [f for f in yaml_files if not f.endswith("config_index.yaml")]
        project_id = os.path.basename(config_dir)
        print(f"  📁 Project {project_id}: {len(config_files)} 个配置文件")
        total_configs += len(config_files)
    
    print(f"\n📊 总计: {total_configs} 个配置文件")
    print(f"📊 项目数: {len(config_dirs)}")
    
    if total_configs > 0:
        print("✅ 配置文件生成完成，可以开始批量实验")
        print("💡 运行命令: !python colab_batch_experiments.py")
    else:
        print("❌ 未生成任何配置文件")

if __name__ == "__main__":
    main()
