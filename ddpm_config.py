"""
ddpm_config.py
语义引导型 SR DDPM 的训练配置（PanNuke 主线版）。

v4 变更：CellViT 替换 HoVer-Net
────────────────────────────────
- 语义教师从 HoVer-Net 替换为 CellViT-256-x40
- 退化参数升级为强退化（scale=4，强模糊+噪声+染色扰动）
- t_max 从 400 收紧到 150（只在低噪声时间步计算语义损失）
- lambda_rec 从 15 降到 6（为语义梯度让出空间）
- lambda_sem 从 0.01 升到 0.15
- Stage 3 已移除，semantic_end_epoch 不再使用
- 两阶段：Stage 1（骨干预训练）+ Stage 2（CellViT 语义联合训练）
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainingConfig:

    # ── 数据集类型与路径 ────────────────────────────────────────────
    dataset_type: str = 'pannuke'

    # PanNuke 三折划分
    pannuke_root: Optional[str] = '/data/xuwen/PanNuke'
    pannuke_train_fold_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 1'
    pannuke_val_fold_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 2'
    pannuke_test_fold_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 3'
    pannuke_folds: Optional[Tuple[str, ...]] = None

    # 旧 NCT 路径（legacy 兼容）
    tum_dir: str = '/data/xuwen/NCT-CRC-HE-100K/TUM'
    norm_dir: str = '/data/xuwen/NCT-CRC-HE-100K/NORM'
    oversample: bool = False

    # ── CellViT 配置（替换原 HoVer-Net）───────────────────────────
    cellvit_path: Optional[str] = '/home/xuwen/DDPM/CellViT/CellViT-256-x40.pth'
    cellvit_repo: str = '/home/xuwen/DDPM/CellViT'

    # 验证可视化目录
    val_vis_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 2'

    # ── 基础训练配置 ────────────────────────────────────────────────
    epochs: int = 200
    batch_size: int = 4
    lr: float = 1e-4
    device: str = 'cuda'

    # ── 模型保存 ────────────────────────────────────────────────────
    save_dir: str = './logs/checkpoints_cellvit'
    save_interval: int = 10

    # ── 扩散配置 ────────────────────────────────────────────────────
    num_train_timesteps: int = 1000
    t_max: int = 150          # 只在低噪声时间步计算语义损失，x0_hat质量更好
    sample_size: int = 256
    in_channels: int = 6
    out_channels: int = 3

    # ── 输入尺寸 ────────────────────────────────────────────────────
    target_size: Optional[int] = 256

    # ── 在线退化设置（强退化，使CellViT对LR/HR预测产生显著差距）───
    scale: int = 4
    blur_sigma_range: Tuple[float, float] = (2.0, 3.0)
    noise_std_range: Tuple[float, float] = (0.03, 0.08)
    stain_jitter: float = 0.15

    # ── 各项损失权重 ────────────────────────────────────────────────
    lambda_noise: float = 1.0
    lambda_rec: float = 3.0       # 从15降到6，为语义梯度让出空间
    lambda_grad: float = 0.8
    lambda_sem: float = 2.0      # 从0.01升到0.15
    lambda_tv: float = 0.0005

    # ── 语义目标参数 ────────────────────────────────────────────────
    tau_nuc: float = 0.4
    lambda_sem_cls: float = 0.5

    # ── 两阶段训练策略（Stage 3 已移除）────────────────────────────
    semantic_start_epoch: int = 28
    semantic_end_epoch: int = 200   # 保留字段兼容旧代码，实际不再使用
    semantic_warmup_epochs: int = 5

    # ── 优化器设置 ─────────────────────────────────────────────────
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1

    # ── 数据加载 ───────────────────────────────────────────────────
    num_workers: int = 4
    pin_memory: bool = True
    train_drop_last: bool = True

    # ── 推理 / 测试常用默认项 ───────────────────────────────────────
    use_semantic_injection: bool = True
    test_output_dir: str = '/data/xuwen/ddpm_inference_results/pannuke_fold3'
    infer_iters: int = 5
    infer_noise_t: int = 200
    infer_max_fidelity_loss: float = 0.05
    infer_use_noise_in_fidelity: bool = False


def get_default_config() -> TrainingConfig:
    return TrainingConfig()