"""
数据集加载模块
核心改变：移除所有 -1 到 1 的归一化，仅保留 / 255.0
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
import random


class NCTDataset(Dataset):
    """
    NCT (Normal vs Tumor) 数据集
    支持过采样以平衡正常样本和肿瘤样本
    """
    def __init__(self, tum_dir, norm_dir, oversample=True):
        """
        Args:
            tum_dir: 肿瘤图像目录路径
            norm_dir: 正常图像目录路径
            oversample: 是否对正常样本进行过采样，使其数量接近肿瘤样本
        """
        self.files = []
        
        # 1. 获取所有文件路径，并打上标签
        # 肿瘤样本标签为 1
        if os.path.exists(tum_dir):
            tum_files = [os.path.join(tum_dir, f) for f in os.listdir(tum_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            self.files.extend([(f, 1) for f in tum_files])
        
        # 正常样本标签为 0
        if os.path.exists(norm_dir):
            norm_files = [os.path.join(norm_dir, f) for f in os.listdir(norm_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            norm_labeled = [(f, 0) for f in norm_files]
            
            # 2. 过采样逻辑：让 len(norm) 接近 len(tum)
            if oversample and len(tum_files) > len(norm_files) and len(norm_files) > 0:
                # 计算需要重复的次数
                repeat_times = (len(tum_files) + len(norm_files) - 1) // len(norm_files)
                norm_labeled = norm_labeled * repeat_times
                # 随机打乱
                random.shuffle(norm_labeled)
                # 截取到接近肿瘤样本数量
                norm_labeled = norm_labeled[:len(tum_files)]
            
            self.files.extend(norm_labeled)
        
        # 打乱所有样本
        random.shuffle(self.files)
        
        num_tum = len([item for _, item in self.files if item == 1])
        num_norm = len([item for _, item in self.files if item == 0])
        print(f"数据集加载完成: 肿瘤样本={num_tum}, "
              f"正常样本={num_norm}, "
              f"总计={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        
        # 1. 读取图像 (OpenCV 读取的是 BGR, 0-255)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Resize 到 256x256
        img = cv2.resize(img, (256, 256))
        
        # 3. 关键修改：归一化到 [0, 1]
        # 不做 (img - 0.5) / 0.5，直接除以 255
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        # 4. 调整维度 [H, W, C] -> [C, H, W]
        img_tensor = img_tensor.permute(2, 0, 1)
        
        return img_tensor, torch.tensor(label).long()

