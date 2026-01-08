"""
实验日志模块
封装 TensorBoard 的 SummaryWriter，用于记录训练过程中的指标和图像
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class ExperimentLogger:
    """
    实验日志记录器
    封装 TensorBoard 功能，用于记录训练指标和可视化图像
    """
    def __init__(self, use_tensorboard=True, log_dir='./logs', experiment_name='exp_default'):
        """
        初始化日志记录器
        
        Args:
            use_tensorboard: 是否启用 TensorBoard
            log_dir: 日志根目录
            experiment_name: 实验名称（用于区分不同次运行）
        """
        self.use_tb = use_tensorboard
        self.writer = None
        
        if self.use_tb:
            # 最终路径例如: ./logs/exp_default
            save_path = os.path.join(log_dir, experiment_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            self.writer = SummaryWriter(log_dir=save_path)
            print(f"✅ TensorBoard 已启动，日志路径: {save_path}")
            print(f"   查看日志: tensorboard --logdir {save_path}")
        else:
            print("⚠️ TensorBoard 未启用，仅进行控制台打印。")

    def log_metrics(self, metrics_dict, step, prefix="Train"):
        """
        记录标量数据 (Loss, 准确率, 置信度等)
        
        Args:
            metrics_dict: 字典, 如 {'loss': 0.01, 'conf': 0.9}
            step: 当前 global_step 或 epoch
            prefix: 前缀, 用于在 TensorBoard 中分组 (如 Train/Loss, Val/Loss)
        """
        if self.use_tb and self.writer:
            for key, value in metrics_dict.items():
                # 自动拼接 tag, 例如 "Train/loss"
                tag = f"{prefix}/{key}"
                # 确保 value 是标量
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else float(value)
                self.writer.add_scalar(tag, value, step)

    def log_images(self, tag, images, step, max_images=8, normalize=True):
        """
        记录图像到 TensorBoard
        
        Args:
            tag: 图像标签 (如 "Reconstruction")
            images: Tensor (B, C, H, W) 或列表
            step: 当前步数
            max_images: 最多记录多少张图（防止日志文件过大）
            normalize: 是否归一化图像到 [0, 1]
        """
        if self.use_tb and self.writer:
            # 确保在 CPU 且不需要梯度
            if isinstance(images, torch.Tensor):
                images = images.detach().cpu()
            elif isinstance(images, list):
                images = torch.stack([img.detach().cpu() if isinstance(img, torch.Tensor) else img 
                                     for img in images])
            
            # 取前 N 张图
            images = images[:max_images]
            
            # 确保图像在 [0, 1] 范围内
            if images.max() > 1.0 or images.min() < 0.0:
                images = torch.clamp(images, 0.0, 1.0)
            
            # 生成网格
            grid = make_grid(images, nrow=min(4, len(images)), normalize=normalize, scale_each=False)
            
            self.writer.add_image(tag, grid, step)

    def log_histogram(self, tag, values, step):
        """
        记录直方图（用于查看权重分布等）
        
        Args:
            tag: 标签
            values: 数值张量
            step: 当前步数
        """
        if self.use_tb and self.writer:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            self.writer.add_histogram(tag, values, step)

    def flush(self):
        """刷新写入缓冲区"""
        if self.writer:
            self.writer.flush()

    def close(self):
        """关闭日志记录器"""
        if self.writer:
            self.writer.close()
            print("✅ TensorBoard 日志已关闭")

