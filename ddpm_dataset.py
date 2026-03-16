"""
ddpm_dataset.py
语义引导型超分辨率 DDPM 使用的 HR/LR 成对数据集。

相较于旧的单图版本，主要变化是：
  __getitem__ 现在返回 {"hr": tensor, "lr": tensor, "label": tensor}，
  其中 lr 通过 degradation.py 从 hr 在线合成。

旧代码中的过采样逻辑保持不变。
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
import random

from degradation import degrade


class NCTDataset(Dataset):
    """
    NCT-CRC-HE 超分辨率数据集。
    返回 HR / LR 成对样本；LR 通过退化流水线在线生成。

    参数:
        tum_dir    : 肿瘤图像目录路径
        norm_dir   : 正常组织图像目录路径
        oversample : 是否通过过采样平衡两类样本（默认 True）
        scale      : 在线合成 LR 时的下采样倍率（默认 2，表示 ×2）
        blur_sigma_range   : 退化时的高斯模糊 sigma 范围
        noise_std_range    : 退化时的加性噪声标准差范围
        stain_jitter       : 退化时 H&E 染色扰动强度
    """

    def __init__(
        self,
        tum_dir,
        norm_dir,
        oversample: bool = True,
        scale: int = 2,
        blur_sigma_range: tuple = (0.5, 1.5),
        noise_std_range:  tuple = (0.0, 0.02),
        stain_jitter:     float = 0.05,
    ):
        self.scale             = scale
        self.blur_sigma_range  = blur_sigma_range
        self.noise_std_range   = noise_std_range
        self.stain_jitter      = stain_jitter
        self.files             = []

        # ── 收集文件路径 ────────────────────────────────────────────
        _ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

        tum_files = []
        if os.path.exists(tum_dir):
            tum_files = [os.path.join(tum_dir, f)
                         for f in os.listdir(tum_dir)
                         if f.lower().endswith(_ext)]

        norm_files = []
        if os.path.exists(norm_dir):
            norm_files = [os.path.join(norm_dir, f)
                          for f in os.listdir(norm_dir)
                          if f.lower().endswith(_ext)]

        if not tum_files and not norm_files:
            raise ValueError("No image files found. Check tum_dir and norm_dir.")

        orig_tum  = len(tum_files)
        orig_norm = len(norm_files)

        # ── 过采样（逻辑与旧代码完全一致）──────────────────────────
        if oversample and tum_files and norm_files:
            if len(norm_files) < len(tum_files):
                reps       = (len(tum_files) + len(norm_files) - 1) // len(norm_files)
                norm_files = (norm_files * reps)[:len(tum_files)]
                random.shuffle(norm_files)
                print(f"  Oversample: NORM {orig_norm} → {len(norm_files)}")
            elif len(tum_files) < len(norm_files):
                reps      = (len(norm_files) + len(tum_files) - 1) // len(tum_files)
                tum_files = (tum_files * reps)[:len(norm_files)]
                random.shuffle(tum_files)
                print(f"  Oversample: TUM {orig_tum} → {len(tum_files)}")

        self.files.extend([(f, 1) for f in tum_files])
        self.files.extend([(f, 0) for f in norm_files])
        random.shuffle(self.files)

        n_tum  = sum(1 for _, l in self.files if l == 1)
        n_norm = sum(1 for _, l in self.files if l == 0)
        print(f"Dataset loaded: TUM={n_tum}, NORM={n_norm}, total={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]

        # ── 加载并缩放到 256（HR 目标尺寸）────────────────────────
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # HR：归一化到 [0, 1]
        hr = torch.from_numpy(img).float() / 255.0
        hr = hr.permute(2, 0, 1)                        # [3, 256, 256]

        # LR：在线退化（×2 模糊→下采样→上采样 + 染色扰动 + 噪声）
        lr = degrade(
            hr,
            scale            = self.scale,
            blur_sigma_range = self.blur_sigma_range,
            noise_std_range  = self.noise_std_range,
            stain_jitter_strength = self.stain_jitter,
        )                                               # [3, 256, 256]

        return {
            "hr":    hr,
            "lr":    lr,
            "label": torch.tensor(label, dtype=torch.long),
        }
