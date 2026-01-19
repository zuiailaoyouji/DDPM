"""
工具函数：文件匹配、路径处理等
(轻量修复版：仅增加对 BEST.png 的支持)
"""
import os
from glob import glob


def collect_original_images(orig_dir, label):
    """
    从原始图像目录收集图片
    
    Args:
        orig_dir: 原始图像目录
        label: 标签 (0: NORM, 1: TUM)
    
    Returns:
        list: 图片文件路径列表
    """
    all_files = []
    
    if not os.path.exists(orig_dir):
        return all_files
    
    # 查找所有图片文件
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    for pattern in patterns:
        all_files.extend(glob(os.path.join(orig_dir, pattern)))
        all_files.extend(glob(os.path.join(orig_dir, '**', pattern), recursive=True))
    
    # 去重
    all_files = list(set(all_files))
    return all_files


def collect_enhanced_images(enh_dir):
    """
    从增强图像目录收集 BEST 图片
    支持结构: 
      1. {enh_dir}/{subfolder}/BEST.png   (新增支持)
      2. {enh_dir}/{subfolder}/BEST_*.png (原有支持)
    
    Args:
        enh_dir: 增强图像根目录（必须包含子文件夹）
    
    Returns:
        dict: {prefix: best_file_path} 映射字典
    """
    enh_dict = {}
    
    if not os.path.exists(enh_dir):
        return enh_dict
    
    # 获取所有子文件夹
    subfolders = [d for d in os.listdir(enh_dir) 
                  if os.path.isdir(os.path.join(enh_dir, d))]
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(enh_dir, subfolder)
        
        # 修改点：同时查找 BEST.png 和 BEST_*.png
        patterns = ["BEST.png", "BEST_*.png"]
        best_files = []
        for pat in patterns:
            best_files.extend(glob(os.path.join(subfolder_path, pat)))
        
        for best_file in best_files:
            # 使用子文件夹名作为前缀
            prefix = subfolder
            
            # 简单去重：如果已经有这个前缀，保留第一个找到的
            if prefix not in enh_dict:
                enh_dict[prefix] = best_file
    
    return enh_dict


def build_enh_map(orig_files, enh_dir_tum, enh_dir_norm):
    """
    建立原始文件名 -> 增强文件路径的映射字典
    
    Args:
        orig_files: 原始文件路径列表
        enh_dir_tum: TUM 增强图像目录
        enh_dir_norm: NORM 增强图像目录
    
    Returns:
        dict: {original_basename: enhanced_file_path}
    """
    mapping = {}
    
    # 收集所有增强图像
    enh_dict_tum = collect_enhanced_images(enh_dir_tum)
    enh_dict_norm = collect_enhanced_images(enh_dir_norm)
    
    # 合并
    all_enh_dict = {**enh_dict_tum, **enh_dict_norm}
    
    if len(all_enh_dict) == 0:
        print("  ⚠️ 警告: 未找到任何增强图像！请检查目录下是否存在子文件夹及 BEST.png 文件。")
        return mapping
    
    print(f"  找到 {len(all_enh_dict)} 个增强图像子文件夹")
    
    # 匹配原始文件和增强文件
    count = 0
    for orig_path in orig_files:
        orig_basename = os.path.basename(orig_path)
        orig_name_no_ext = os.path.splitext(orig_basename)[0]
        
        matched = False
        
        # 策略1: 直接匹配文件名
        if orig_name_no_ext in all_enh_dict:
            mapping[orig_basename] = all_enh_dict[orig_name_no_ext]
            count += 1
            matched = True
        else:
            # 策略2: 子文件夹名包含原始文件名的一部分
            for prefix, enh_path in all_enh_dict.items():
                if orig_name_no_ext in prefix or prefix in orig_name_no_ext:
                    mapping[orig_basename] = enh_path
                    count += 1
                    matched = True
                    break
            
            # 策略3: 提取关键部分匹配 (如 TCGA-XXX)
            if not matched and 'TCGA-' in orig_name_no_ext:
                orig_tcga_part = orig_name_no_ext.split('TCGA-')[-1]
                for prefix, enh_path in all_enh_dict.items():
                    if 'TCGA-' in prefix:
                        tcga_part = prefix.split('TCGA-')[-1]
                        if tcga_part == orig_tcga_part:
                            mapping[orig_basename] = enh_path
                            count += 1
                            matched = True
                            break
    
    print(f"  成功匹配到增强图: {count}/{len(orig_files)} ({count/len(orig_files)*100:.1f}%)")
    return mapping