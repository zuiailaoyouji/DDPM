"""
ddpm_config.py
语义引导型 SR DDPM 的训练配置（PanNuke 主线版）。

当前默认以 PanNuke 三折划分为主：
- 一个 Fold 训练
- 一个 Fold 验证
- 一个 Fold 测试

语义监督模式：GT mask 驱动
- 语义损失仅保留 CE(pred_prob, gt_label_map)，在 GT 核掩膜区域内监督
- KL 损失、confidence supervision、lambda_dir 均已移除
- sem_tensor 格式：[gt_tp_onehot(6), gt_nuc_mask(1)]，共 7 通道
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainingConfig:

    # ── 数据集类型与路径 ────────────────────────────────────────────
    dataset_type: str = 'pannuke'   # 'pannuke' 或 'nct'

    # PanNuke 推荐显式三折划分
    pannuke_root: Optional[str] = '/data/xuwen/PanNuke'
    pannuke_train_fold_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 1'
    pannuke_val_fold_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 2'
    pannuke_test_fold_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 3'

    # 兼容旧写法：可选地直接传 folds 列表
    pannuke_folds: Optional[Tuple[str, ...]] = None

    # 旧 NCT 路径：仅保留为 legacy 兼容
    tum_dir: str = '/data/xuwen/NCT-CRC-HE-100K/TUM'
    norm_dir: str = '/data/xuwen/NCT-CRC-HE-100K/NORM'
    oversample: bool = False

    # HoVer-Net 与验证可视化
    hovernet_path: Optional[str] = '/home/xuwen/DDPM/HoVer-net/hovernet_fast_pannuke_type_tf2pytorch.tar'
    val_vis_dir: Optional[str] = '/data/xuwen/PanNuke/Fold 2'
    hovernet_upsample_factor: float = 1.0   # PanNuke 256×256，pred 侧不额外上采样

    # ── 基础训练配置 ────────────────────────────────────────────────
    epochs: int = 200
    batch_size: int = 4
    lr: float = 1e-4
    device: str = 'cuda'

    # ── 模型保存 ────────────────────────────────────────────────────
    save_dir: str = './logs/checkpoints_sr'
    save_interval: int = 10

    # ── 扩散配置 ────────────────────────────────────────────────────
    num_train_timesteps: int = 1000
    t_max: int = 400
    sample_size: int = 256
    in_channels: int = 6      # [lr_image || noisy_hr]
    out_channels: int = 3

    # ── 输入尺寸 ────────────────────────────────────────────────────
    target_size: Optional[int] = 256

    # ── 在线退化设置 ────────────────────────────────────────────────
    scale: int = 2
    blur_sigma_range: Tuple[float, float] = (1.0, 1.0)
    noise_std_range: Tuple[float, float] = (0.0, 0.0)
    stain_jitter: float = 0.0

    # ── 各项损失权重 ────────────────────────────────────────────────
    lambda_noise: float = 1.0
    lambda_rec: float = 6.0
    lambda_grad: float = 0.8
    lambda_sem: float = 0.15
    lambda_tv: float = 0.0005

    # ── 语义目标参数 ────────────────────────────────────────────────
    tau_nuc: float = 0.4          # GT 核掩膜阈值
    lambda_sem_cls: float = 0.3  # CE 损失权重（唯一保留的语义子项）

    # ── 三阶段训练策略 ─────────────────────────────────────────────
    semantic_start_epoch: int = 50
    semantic_end_epoch: int = 200
    semantic_warmup_epochs: int = 15

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