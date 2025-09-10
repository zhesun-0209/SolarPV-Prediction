#!/usr/bin/env python3
"""
批量生成Excel结果文件
为每个厂生成包含所有实验结果的Excel文件
"""

import os
import glob
import pandas as pd
from eval.excel_utils import collect_plant_results, save_plant_excel_results

def generate_all_excel_results():
    """为所有厂生成Excel结果文件"""
    
    print("📊 批量生成Excel结果文件")
    print("=" * 60)
    
    # 检查Drive和本地结果
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    local_dir = 'result'
    
    result_dirs = []
    if os.path.exists(drive_dir):
        result_dirs.append(drive_dir)
        print(f"✅ Drive目录存在: {drive_dir}")
    if os.path.exists(local_dir):
        result_dirs.append(local_dir)
        print(f"✅ 本地目录存在: {local_dir}")
    
    if not result_dirs:
        print("❌ 未找到任何结果目录")
        return
    
    # 查找所有厂数据文件
    data_dir = 'data'
    plant_files = []
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    for file in csv_files:
        filename = os.path.basename(file)
        if filename.endswith('.csv'):
            plant_id = filename.replace('.csv', '')
            plant_files.append(plant_id)
    
    if not plant_files:
        print("❌ 未找到任何厂数据文件")
        return
    
    print(f"✅ 找到 {len(plant_files)} 个厂数据文件")
    
    # 为每个厂生成Excel文件
    success_count = 0
    failed_count = 0
    
    for plant_id in sorted(plant_files):
        print(f"\n🏭 处理厂: {plant_id}")
        
        try:
            # 收集该厂的所有实验结果
            results = collect_plant_results(plant_id, result_dirs)
            
            if not results:
                print(f"   ❌ 未找到实验结果")
                failed_count += 1
                continue
            
            print(f"   📊 找到 {len(results)} 个实验结果")
            
            # 生成Excel文件
            excel_file = save_plant_excel_results(
                plant_id=plant_id,
                results=results,
                save_dir=result_dirs[0]  # 保存到第一个结果目录
            )
            
            print(f"   ✅ Excel文件已生成: {excel_file}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ 生成Excel文件失败: {e}")
            failed_count += 1
    
    # 统计结果
    print(f"\n🎉 Excel文件生成完成!")
    print("=" * 60)
    print(f"总厂数: {len(plant_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    
    # 生成汇总统计
    generate_summary_statistics(result_dirs[0])

def generate_summary_statistics(result_dir):
    """生成汇总统计信息"""
    
    print(f"\n📈 生成汇总统计信息...")
    
    # 查找所有Excel文件
    excel_files = glob.glob(os.path.join(result_dir, '*_results.xlsx'))
    
    if not excel_files:
        print("❌ 未找到任何Excel文件")
        return
    
    print(f"✅ 找到 {len(excel_files)} 个Excel文件")
    
    # 合并所有Excel文件
    all_data = []
    
    for excel_file in excel_files:
        try:
            df = pd.read_excel(excel_file)
            plant_id = os.path.basename(excel_file).replace('_results.xlsx', '')
            df['plant_id'] = plant_id
            all_data.append(df)
        except Exception as e:
            print(f"❌ 读取Excel文件失败 {excel_file}: {e}")
    
    if not all_data:
        print("❌ 没有成功读取任何Excel文件")
        return
    
    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 保存汇总文件
    summary_file = os.path.join(result_dir, 'all_plants_summary.xlsx')
    combined_df.to_excel(summary_file, index=False, engine='openpyxl')
    
    print(f"✅ 汇总文件已保存: {summary_file}")
    print(f"   总实验数: {len(combined_df)}")
    print(f"   总厂数: {combined_df['plant_id'].nunique()}")
    
    # 按模型统计
    model_stats = combined_df.groupby('model').size()
    print(f"\n🤖 各模型实验数量:")
    for model, count in model_stats.items():
        print(f"   {model}: {count} 个")

if __name__ == "__main__":
    generate_all_excel_results()
