#!/usr/bin/env python3
"""
Google Drive保存工具
用于将实验结果保存到Google Drive
"""

import os
import shutil
from typing import Optional

def mount_drive():
    """挂载Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive已挂载到 /content/drive")
        return True
    except ImportError:
        print("⚠️ 不在Colab环境中，跳过Drive挂载")
        return False
    except Exception as e:
        print(f"❌ Drive挂载失败: {e}")
        return False

def save_to_drive(local_file_path: str, drive_folder: str = "/content/drive/MyDrive/SolarPV_Results") -> Optional[str]:
    """
    将本地文件保存到Google Drive
    
    Args:
        local_file_path: 本地文件路径
        drive_folder: Drive保存文件夹
        
    Returns:
        Drive文件路径，如果失败返回None
    """
    try:
        # 确保Drive文件夹存在
        os.makedirs(drive_folder, exist_ok=True)
        
        # 获取文件名
        filename = os.path.basename(local_file_path)
        drive_file_path = os.path.join(drive_folder, filename)
        
        # 复制文件到Drive
        shutil.copy2(local_file_path, drive_file_path)
        
        print(f"✅ 文件已保存到Drive: {drive_file_path}")
        return drive_file_path
        
    except Exception as e:
        print(f"❌ 保存到Drive失败: {e}")
        return None

def save_project_results_to_drive(project_id: str, local_results_dir: str, drive_folder: str = "/content/drive/MyDrive/SolarPV_Results") -> bool:
    """
    将项目结果保存到Google Drive
    
    Args:
        project_id: 项目ID
        local_results_dir: 本地结果目录
        drive_folder: Drive保存文件夹
        
    Returns:
        是否保存成功
    """
    try:
        # 创建项目专用文件夹
        project_drive_folder = os.path.join(drive_folder, f"Project_{project_id}")
        os.makedirs(project_drive_folder, exist_ok=True)
        
        # 查找CSV结果文件
        csv_files = []
        for file in os.listdir(local_results_dir):
            if file.endswith('.csv') and project_id in file:
                csv_files.append(file)
        
        if not csv_files:
            print(f"⚠️ 项目 {project_id} 未找到CSV结果文件")
            return False
        
        # 保存每个CSV文件
        success_count = 0
        for csv_file in csv_files:
            local_file_path = os.path.join(local_results_dir, csv_file)
            drive_file_path = os.path.join(project_drive_folder, csv_file)
            
            try:
                shutil.copy2(local_file_path, drive_file_path)
                print(f"✅ {csv_file} 已保存到Drive")
                success_count += 1
            except Exception as e:
                print(f"❌ 保存 {csv_file} 失败: {e}")
        
        print(f"📊 项目 {project_id} 保存完成: {success_count}/{len(csv_files)} 个文件")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 保存项目 {project_id} 结果失败: {e}")
        return False

def list_drive_results(drive_folder: str = "/content/drive/MyDrive/SolarPV_Results") -> list:
    """
    列出Drive中的结果文件
    
    Args:
        drive_folder: Drive结果文件夹
        
    Returns:
        结果文件列表
    """
    try:
        if not os.path.exists(drive_folder):
            print(f"⚠️ Drive文件夹不存在: {drive_folder}")
            return []
        
        files = []
        for item in os.listdir(drive_folder):
            item_path = os.path.join(drive_folder, item)
            if os.path.isdir(item_path):
                # 项目文件夹
                project_files = os.listdir(item_path)
                csv_files = [f for f in project_files if f.endswith('.csv')]
                files.append({
                    'project': item,
                    'type': 'folder',
                    'csv_count': len(csv_files),
                    'files': csv_files
                })
            else:
                # 直接文件
                files.append({
                    'project': item,
                    'type': 'file',
                    'size': os.path.getsize(item_path)
                })
        
        return files
        
    except Exception as e:
        print(f"❌ 列出Drive结果失败: {e}")
        return []
