"""
配置文件
集中管理训练参数和超参数
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据路径
    tum_dir: str = '/data/xuwen/NCT-CRC-HE-100K/TUM'
    norm_dir: str = '/data/xuwen/NCT-CRC-HE-100K/NORM'
    hovernet_path: Optional[str] = None
    
    # 训练参数
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    device: str = 'cuda'
    
    # 模型保存
    save_dir: str = './logs/checkpoints'
    save_interval: int = 10  # 每 N 个 epoch 保存一次
    
    # 损失权重
    feedback_weight_prob: float = 0.05
    feedback_weight_entropy: float = 0.0
    tv_weight: float = 0.001  # TV 全变分损失权重，压制高频噪点（推荐 0.001～0.002）
    use_feedback_from_epoch: int = 5  # 从第几个 epoch 开始使用反馈损失
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    
    # 模型参数
    sample_size: int = 256
    in_channels: int = 6
    out_channels: int = 3
    num_train_timesteps: int = 1000
    
    # 优化器参数
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 数据增强（可选）
    oversample: bool = True  # 是否对正常样本过采样


def get_default_config():
    """获取默认配置"""
    return TrainingConfig()

