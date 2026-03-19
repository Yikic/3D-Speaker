#!/usr/bin/env python3
"""
图片垂直拼接脚本
功能：将同一数据集的三张图片（grid_search_3d.png, slice_best_alpha.png, slice_best_pval.png）
     垂直拼接成3行1列的大图
使用：python merge_images_vertical.py
"""

import os
import re
from pathlib import Path
from PIL import Image

def merge_dataset_images_vertical(image_dir, output_dir="merged_results_vertical"):
    """
    垂直合并同一数据集的三张图片（3行1列）
    
    Args:
        image_dir: 包含图片的目录路径
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 获取所有png文件
    image_files = list(Path(image_dir).glob("*.png"))
    
    # 按数据集分组
    dataset_groups = {}
    
    # 从文件名中提取数据集名称和图片类型
    # 文件名模式示例：aishell4_grid_search_3d.png
    pattern = r"(.+?)_(grid_search_3d|slice_best_alpha|slice_best_pval)\.png"
    
    for img_file in image_files:
        match = re.match(pattern, img_file.name)
        if match:
            dataset = match.group(1)  # 数据集名称，如aishell4
            img_type = match.group(2)  # 图片类型
            
            if dataset not in dataset_groups:
                dataset_groups[dataset] = {}
            
            dataset_groups[dataset][img_type] = str(img_file)
    
    # 处理每个数据集
    for dataset, img_dict in dataset_groups.items():
        # 检查是否包含所有三种图片
        required_types = ["grid_search_3d", "slice_best_alpha", "slice_best_pval"]
        
        if not all(t in img_dict for t in required_types):
            print(f"警告：数据集 {dataset} 缺少部分图片，跳过处理")
            print(f"  已有的图片: {list(img_dict.keys())}")
            continue
        
        print(f"处理数据集: {dataset}")
        
        # 按指定顺序加载图片
        image_paths = [
            img_dict["grid_search_3d"],
            img_dict["slice_best_alpha"], 
            img_dict["slice_best_pval"]
        ]
        
        # 打开所有图片
        images = [Image.open(img_path) for img_path in image_paths]
        
        # 获取图片尺寸
        widths, heights = zip(*(img.size for img in images))
        
        # 计算新图片尺寸（垂直拼接）
        total_height = sum(heights)  # 总高度
        max_width = max(widths)      # 最大宽度
        
        # 创建新图片（垂直方向）
        merged_image = Image.new('RGB', (max_width, total_height), color='white')
        
        # 垂直拼接图片
        y_offset = 0
        for img in images:
            # 将图片居中放置
            x_offset = (max_width - img.width) // 2
            merged_image.paste(img, (x_offset, y_offset))
            y_offset += img.height
        
        # 保存图片
        output_file = output_path / f"{dataset}_merged_vertical.png"
        merged_image.save(output_file)
        print(f"  已保存: {output_file}")
        
        # 关闭所有图片
        for img in images:
            img.close()

def main():
    # 设置图片目录（根据你提供的路径）
    image_dir = "/home/yukaichen/work/.sub-repos/diarization/diar_grad/speakerlab/results/results_pval_alpha_exp"
    
    # 设置输出目录
    output_dir = "merged_results_vertical"
    
    print("开始垂直拼接图片...")
    print(f"输入目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 检查输入目录是否存在
    if not Path(image_dir).exists():
        print(f"错误：目录 {image_dir} 不存在！")
        return
    
    # 执行图片合并
    merge_dataset_images_vertical(image_dir, output_dir)
    
    print("-" * 50)
    print("图片垂直拼接完成！")

if __name__ == "__main__":
    main()