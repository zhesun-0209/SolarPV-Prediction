#!/usr/bin/env python3
"""
设置Drive路径和目录
"""

import os

def setup_drive_paths():
    """设置Drive路径和目录"""
    
    print("🔧 设置Drive路径和目录")
    print("=" * 60)
    
    # 检查Drive是否挂载
    drive_root = '/content/drive'
    if not os.path.exists(drive_root):
        print("❌ Drive未挂载，请先运行:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        return False
    
    print("✅ Drive已挂载")
    
    # 创建结果目录
    results_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    os.makedirs(results_dir, exist_ok=True)
    print(f"✅ 结果目录已创建: {results_dir}")
    
    # 检查目录权限
    if os.access(results_dir, os.W_OK):
        print("✅ 目录可写")
    else:
        print("❌ 目录不可写")
        return False
    
    # 创建测试文件
    test_file = os.path.join(results_dir, 'test_write.txt')
    try:
        with open(test_file, 'w') as f:
            f.write('Drive路径测试成功')
        print("✅ 写入测试成功")
        
        # 删除测试文件
        os.remove(test_file)
        print("✅ 删除测试文件成功")
        
    except Exception as e:
        print(f"❌ 写入测试失败: {e}")
        return False
    
    print("\n🎉 Drive路径设置完成!")
    print(f"   结果将保存到: {results_dir}")
    
    return True

if __name__ == "__main__":
    success = setup_drive_paths()
    if success:
        print("\n✅ 可以开始运行实验了!")
    else:
        print("\n❌ 请先解决Drive路径问题")
