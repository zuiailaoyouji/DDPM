"""
ddpm_config.py
判别器标签纠正模型的训练配置（单阶段版）。

【任务】
对 HR 病理切片做微小修改，使 CellViT 对修改后图像的预测结果更接近 GT。

【关键设计】
- 模型输入：[HR, noisy_HR]（6 通道），无 LR，无退化，无增广
- sem_tensor：CellViT(HR) 软标签 [type_prob(6), nuc_prob(1)]
- 监督区域：GT ∩ CellViT(HR) 交集
- 损失：Focal-CE + 类别逆频率权重 + correction_boost
- 单阶段训练：从第一个 epoch 起全量开启语义监督
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainingConfig:

    # ── 数据集 ──────────────────────────────────────────────────────
    dataset_type: str = 'pannuke'

    pannuke_root:           Optional[str] = '/data/xuwen/PanNuke'
    pannuke_train_fold_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 1'
    pannuke_val_fold_dir:   Optional[str] = '/data/xuwen/PanNuke/Fold 2'
    pannuke_test_fold_dir:  Optional[str] = '/data/xuwen/PanNuke/Fold 3'
    pannuke_folds:          Optional[Tuple[str, ...]] = None

    tum_dir:    str  = '/data/xuwen/NCT-CRC-HE-100K/TUM'
    norm_dir:   str  = '/data/xuwen/NCT-CRC-HE-100K/NORM'
    oversample: bool = False

    # ── CellViT ─────────────────────────────────────────────────────
    cellvit_path: Optional[str] = '/home/xuwen/DDPM/CellViT/CellViT-256-x40.pth'
    cellvit_repo: str           = '/home/xuwen/DDPM/CellViT'

    val_vis_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 2'

    # ── 基础训练 ─────────────────────────────────────────────────────
    epochs:     int   = 200
    batch_size: int   = 4
    lr:         float = 1e-4
    device:     str   = 'cuda'

    # ── 保存 ─────────────────────────────────────────────────────────
    save_dir:      str = './logs/checkpoints_correction'
    save_interval: int = 10

    # ── 扩散 ─────────────────────────────────────────────────────────
    num_train_timesteps: int          = 1000
    t_max:               int          = 150
    sample_size:         int          = 256
    in_channels:         int          = 6     # [HR(3), noisy_HR(3)]
    out_channels:        int          = 3
    target_size:         Optional[int] = 256

    # ── 损失权重 ─────────────────────────────────────────────────────
    lambda_noise: float = 1.0
    lambda_rec:   float = 3.0    # 像素保真：防止过度修改图像外观
    lambda_grad:  float = 0.8    # 边缘保真：防止模糊
    lambda_sem:   float = 2.0    # 语义纠正：向交集目标靠近
    lambda_tv:    float = 0.0005 # 抑制幻觉伪影

    # ── 语义损失参数 ─────────────────────────────────────────────────
    lambda_sem_cls:   float = 1.0
    correction_boost: float = 2.0   # 纠正候选区额外像素权重
    focal_gamma:      float = 2.0   # Focal Loss gamma

    # ── 优化器 ───────────────────────────────────────────────────────
    weight_decay:       float = 0.01
    max_grad_norm:      float = 1.0
    accumulation_steps: int   = 1

    # ── 数据加载 ─────────────────────────────────────────────────────
    num_workers:     int  = 4
    pin_memory:      bool = True
    train_drop_last: bool = True

    # ── 推理 ─────────────────────────────────────────────────────────
    test_output_dir: str = '/data/xuwen/ddpm_inference_results/pannuke_fold3'
    infer_noise_t:   int = 100

    # ── 以下字段保留兼容，不再使用 ──────────────────────────────────
    tau_nuc:                float          = 0.4
    scale:                  int            = 4
    blur_sigma_range:       Tuple[float, float] = (2.0, 3.0)
    noise_std_range:        Tuple[float, float] = (0.03, 0.08)
    stain_jitter:           float          = 0.15
    semantic_start_epoch:   int            = 0
    semantic_end_epoch:     int            = 200
    semantic_warmup_epochs: int            = 0


def get_default_config() -> TrainingConfig:
    return TrainingConfig()