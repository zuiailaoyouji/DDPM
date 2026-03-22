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
    epochs:     int   = 400
    batch_size: int   = 4
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
    blur_sigma_range:  Tuple = (1.0, 1.0)
    noise_std_range:   Tuple = (0.0, 0.0)
    stain_jitter:      float = 0.0

    # ── 各项损失权重 ────────────────────────────────────────────────
    lambda_noise: float = 1.0    # 扩散噪声 MSE
    lambda_rec:   float = 15.0    # 像素级 L1 重建
    lambda_grad:  float = 0.8    # 边缘 / 梯度 L1
    lambda_sem:   float = 0.01   # 语义总权重（主项）
    lambda_dir:   float = 0.001  # 语义辅助权重（兼容旧字段名）
    lambda_tv:    float = 0.0005  # 漏斗型相对 TV

    # ── 语义目标参数 ───────────────────────────────────────────────
    # 多类别核语义监督区域：
    # valid_mask = (nuc_mask > tau_nuc) & (tp_conf > tau_conf)
    tau_nuc:  float = 0.4
    tau_conf: float = 0.6
    # 语义子项权重（loss 内部组合）
    lambda_sem_dist: float = 0.3
    lambda_sem_cls:  float = 0.05
    lambda_sem_conf: float = 0.02

    # ── 三阶段训练策略 ─────────────────────────────────────────────
    # 阶段 1（epoch < semantic_start_epoch）：
    #   仅使用 L_noise + L_rec + L_grad + L_tv（纯 SR 重建）
    # 阶段 2（semantic_start_epoch <= epoch < semantic_end_epoch）：
    #   加入 L_sem + L_dir，并在 semantic_warmup_epochs 内线性升权
    #   同时启用架构层语义注入（semantic=S）
    # 阶段 3（epoch >= semantic_end_epoch）：
    #   纯像素收尾：再次关闭 L_sem / L_dir 与 semantic injection
    #   仅使用 L_noise + L_rec + L_grad + L_tv
    semantic_start_epoch:   int = 30
    semantic_end_epoch:     int = 110
    semantic_warmup_epochs: int = 15

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
