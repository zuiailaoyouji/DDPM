"""
ddpm_config.py
语义引导型 SR DDPM 的训练配置。

所有旧的极化相关参数（alpha_tumor、alpha_normal、
feedback_weight_prob、feedback_weight_entropy、use_feedback_from_epoch）
都已移除，并替换为针对 SR 任务的新参数。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TrainingConfig:

    # ── 数据路径 ─────────────────────────────────────────────────────
    tum_dir:       str          = '/data/xuwen/NCT-CRC-HE-100K/TUM'
    norm_dir:      str          = '/data/xuwen/NCT-CRC-HE-100K/NORM'
    hovernet_path: Optional[str] = None
    val_vis_dir:   Optional[str] = None

    # ── 基础训练配置 ────────────────────────────────────────────────
    epochs:     int   = 100
    batch_size: int   = 8
    lr:         float = 1e-4
    device:     str   = 'cuda'

    # ── 模型保存 ────────────────────────────────────────────────────
    save_dir:      str = './logs/checkpoints_sr'
    save_interval: int = 10

    # ── 扩散配置 ────────────────────────────────────────────────────
    num_train_timesteps: int = 1000
    t_max:               int = 400    # 语义监督启用的最大时间步
    sample_size:         int = 256
    in_channels:         int = 6      # [lr_image ‖ noisy_hr]
    out_channels:        int = 3

    # ── 在线退化设置 ────────────────────────────────────────────────
    scale:             int   = 2                  # LR 下采样倍率（×2）
    blur_sigma_range:  Tuple = (0.5, 1.5)
    noise_std_range:   Tuple = (0.0, 0.02)
    stain_jitter:      float = 0.05

    # ── 各项损失权重 ────────────────────────────────────────────────
    lambda_noise: float = 1.0    # 扩散噪声 MSE
    lambda_rec:   float = 1.0    # 像素级 L1 重建
    lambda_grad:  float = 0.1    # 边缘 / 梯度 L1
    lambda_sem:   float = 0.05   # 语义 SmoothL1（软目标）
    lambda_dir:   float = 0.005   # 方向约束 hinge
    lambda_tv:    float = 0.001  # 漏斗型相对 TV

    # ── 语义软目标参数 ─────────────────────────────────────────────
    # 动态 delta：肿瘤像素偏移 = delta_t * (1 - p_clean)
    #             正常像素偏移 = delta_n * p_clean
    # 在置信度已较高的像素处减小扰动。
    delta_t: float = 0.05   # 肿瘤区域 delta 缩放（0.03 ~ 0.08）
    delta_n: float = 0.05   # 正常区域 delta 缩放（0.03 ~ 0.08）

    # 语义监督掩膜的高置信度阈值
    tau_pos: float = 0.65   # p_clean >= tau_pos → 高置信肿瘤像素
    tau_neg: float = 0.35   # p_clean <= tau_neg → 高置信正常像素

    # ── 双阶段训练策略 ─────────────────────────────────────────────
    # 阶段 1（epoch < semantic_start_epoch）：
    #   仅使用 L_noise + L_rec + L_grad + L_tv（纯 SR 重建）
    # 阶段 2（epoch >= semantic_start_epoch）：
    #   加入 L_sem + L_dir，并在 semantic_warmup_epochs 内线性升权
    semantic_start_epoch:  int = 5
    semantic_warmup_epochs: int = 5   # 将 lambda_sem/lambda_dir 从 0 线性升至完整权重

    # ── 优化器设置 ─────────────────────────────────────────────────
    weight_decay:       float = 0.01
    max_grad_norm:      float = 1.0
    accumulation_steps: int   = 1

    # ── 数据加载 ───────────────────────────────────────────────────
    num_workers: int  = 4
    pin_memory:  bool = True
    oversample:  bool = True


def get_default_config() -> TrainingConfig:
    return TrainingConfig()
