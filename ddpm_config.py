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

【数据集划分】
- 训练集：Fold 1 + Fold 2 合并，从中分层抽样 100 张作为验证集
- 验证集：按 tissue type 分层抽样，每种类型都有覆盖，从训练集剔除
- 测试集：Fold 3（不参与训练和验证）
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class TrainingConfig:

    # ── 数据集 ──────────────────────────────────────────────────────
    dataset_type: str = 'pannuke'

    pannuke_root:          Optional[str]       = '/data/xuwen/PanNuke'
    # 训练用折目录列表（支持多折合并，如 Fold1 + Fold2）
    pannuke_fold_dirs:     Optional[List[str]] = field(default_factory=lambda: [
        '/data/xuwen/PanNuke/Fold 1',
        '/data/xuwen/PanNuke/Fold 2',
    ])
    pannuke_test_fold_dir: Optional[str]       = '/data/xuwen/PanNuke/Fold 3'

    # 验证集分层抽样参数
    n_val:    int = 100   # 从训练数据中抽取的验证集样本数
    val_seed: int = 42    # 随机种子，保证可复现

    tum_dir:    str  = '/data/xuwen/NCT-CRC-HE-100K/TUM'
    norm_dir:   str  = '/data/xuwen/NCT-CRC-HE-100K/NORM'
    oversample: bool = False

    # ── CellViT ─────────────────────────────────────────────────────
    # 'cellvit_256' = 旧 CellViT-256 教师
    # 'sam_h'       = CellViT-SAM-H 教师(更强,约 25× 参数,但准确率更高)
    cellvit_variant: str           = 'sam_h'
    cellvit_path:    Optional[str] = '/home/xuwen/DDPM/CellViT/CellViT-SAM-H-x40.pth'
    cellvit_repo:    str           = '/home/xuwen/DDPM/CellViT'

    # ── 基础训练 ─────────────────────────────────────────────────────
    epochs:     int   = 20
    batch_size: int   = 4
    lr:         float = 1e-4
    device:     str   = 'cuda'

    # ── 保存 ─────────────────────────────────────────────────────────
    save_dir:      str = './logs/checkpoints_correction_samh'
    save_interval: int = 10

    # ── 扩散 ─────────────────────────────────────────────────────────
    num_train_timesteps: int          = 1000
    t_max:               int          = 200
    sample_size:         int          = 256
    in_channels:         int          = 6      # [HR(3), noisy_HR(3)]
    out_channels:        int          = 3
    target_size:         Optional[int] = 256

    # ── 损失权重 ─────────────────────────────────────────────────────
    lambda_noise: float = 1.0
    lambda_rec:   float = 2.0    # 降低像素保真约束，给模型更大修改空间
    lambda_grad:  float = 0.8
    lambda_sem:   float = 2.0
    lambda_tv:    float = 0.0005

    # ── 语义损失参数 ─────────────────────────────────────────────────
    lambda_sem_cls:   float = 1.0
    correction_boost: float = 3.0
    focal_gamma:      float = 3.0
    # confusion_penalty 在 SemanticSRLoss 内部默认设置，无需 config 字段

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