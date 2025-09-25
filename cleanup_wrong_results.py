#!/usr/bin/env python3
"""
清理错误的结果文件
删除所有 use_pv=false 且 use_time_encoding=true 的配置产生的结果
这些结果在修复前是错误的，因为时间编码特征被忽略了
"""

import os
import yaml
import shutil
from pathlib import Path

def find_wrong_results():
    """
    找到所有需要删除的错误结果
    条件：use_pv=false 且 use_time_encoding=true
    """
    wrong_results = []
    
    # 查找所有配置文件
    config_dirs = [
        "config/projects/1140",
        "config/projects/171", 
        "config/projects/172",
        "config/projects/186"
    ]
    
    for config_dir in config_dirs:
        if not os.path.exists(config_dir):
            continue
            
        for config_file in os.listdir(config_dir):
            if config_file.endswith('.yaml'):
                config_path = os.path.join(config_dir, config_file)
                
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # 检查是否是需要删除的配置
                    use_pv = config.get('use_pv', True)
                    use_time_encoding = config.get('use_time_encoding', True)
                    
                    if use_pv == False and use_time_encoding == True:
                        # 这是错误的配置，需要删除对应的结果
                        model = config.get('model', '')
                        complexity = config.get('model_complexity', '')
                        past_hours = config.get('past_hours', '')
                        
                        # 构建结果目录名
                        if 'NWP_plus' in config_file:
                            result_name = f"{model}_{complexity}_NWP_plus_{past_hours}h_TE"
                        elif 'NWP' in config_file:
                            result_name = f"{model}_{complexity}_NWP_{past_hours}h_TE"
                        else:
                            continue
                        
                        # 查找对应的结果目录
                        project_id = config_dir.split('/')[-1]
                        result_dir = f"temp_results/{project_id}/{result_name}"
                        
                        if os.path.exists(result_dir):
                            wrong_results.append({
                                'config_file': config_path,
                                'result_dir': result_dir,
                                'model': model,
                                'complexity': complexity,
                                'project_id': project_id
                            })
                            
                except Exception as e:
                    print(f"Error reading {config_path}: {e}")
    
    return wrong_results

def delete_wrong_results(wrong_results, dry_run=True):
    """
    删除错误的结果
    """
    if dry_run:
        print("=== 预览模式 - 将要删除的结果 ===")
    else:
        print("=== 开始删除错误结果 ===")
    
    deleted_count = 0
    total_size = 0
    
    for result in wrong_results:
        result_dir = result['result_dir']
        
        if os.path.exists(result_dir):
            if dry_run:
                print(f"将删除: {result_dir}")
                # 计算目录大小
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(result_dir)
                          for filename in filenames)
                total_size += size
                print(f"  - 模型: {result['model']}")
                print(f"  - 复杂度: {result['complexity']}")
                print(f"  - 项目: {result['project_id']}")
                print(f"  - 大小: {size / 1024:.1f} KB")
                print()
            else:
                try:
                    shutil.rmtree(result_dir)
                    print(f"已删除: {result_dir}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败 {result_dir}: {e}")
        else:
            print(f"结果目录不存在: {result_dir}")
    
    if dry_run:
        print(f"总计将删除 {len(wrong_results)} 个结果目录")
        print(f"总大小: {total_size / 1024 / 1024:.1f} MB")
    else:
        print(f"已删除 {deleted_count} 个结果目录")

def main():
    print("查找需要删除的错误结果...")
    wrong_results = find_wrong_results()
    
    if not wrong_results:
        print("没有找到需要删除的错误结果")
        return
    
    print(f"找到 {len(wrong_results)} 个需要删除的错误结果")
    
    # 按项目分组显示
    by_project = {}
    for result in wrong_results:
        project_id = result['project_id']
        if project_id not in by_project:
            by_project[project_id] = []
        by_project[project_id].append(result)
    
    for project_id, results in by_project.items():
        print(f"\n项目 {project_id}: {len(results)} 个错误结果")
        for result in results:
            print(f"  - {result['model']}_{result['complexity']}_NWP*_TE")
    
    # 预览删除
    print("\n" + "="*50)
    delete_wrong_results(wrong_results, dry_run=True)
    
    # 询问是否确认删除
    print("\n" + "="*50)
    confirm = input("确认删除这些错误结果吗？(y/N): ")
    
    if confirm.lower() == 'y':
        delete_wrong_results(wrong_results, dry_run=False)
        print("清理完成！")
    else:
        print("取消删除操作")

if __name__ == "__main__":
    main()
