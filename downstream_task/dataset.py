"""
数据集类：支持原始图像和增强图像的切换
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CRCDataset(Dataset):
    """
    CRC 分类数据集
    
    支持在训练时使用增强图像，测试时使用原始图像
    """
    def __init__(self, file_paths, labels, transform=None, use_enhanced=False, enh_map=None):
        """
        Args:
            file_paths: 原始文件的完整路径列表
            labels: 对应的标签 (0: NORM, 1: TUM)
            transform: 图像变换
            use_enhanced: 是否使用增强版图片（仅训练集Ours模式开启）
            enh_map: 一个字典，将原始文件名映射到增强文件路径
                   格式: {original_basename: enhanced_file_path}
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.use_enhanced = use_enhanced
        self.enh_map = enh_map or {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        orig_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 如果指定使用增强版，且能找到对应的增强图，就替换路径
        if self.use_enhanced and self.enh_map:
            orig_basename = os.path.basename(orig_path)
            if orig_basename in self.enh_map:
                path = self.enh_map[orig_basename]
            else:
                # 找不到就回退用原图（防止报错）
                path = orig_path
        else:
            path = orig_path
        
        # 读取图片
        img = cv2.imread(path)
        if img is None:
            # 创建黑图防止崩溃
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            print(f"⚠️ 警告: 无法读取图片 {path}，使用黑图替代")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

