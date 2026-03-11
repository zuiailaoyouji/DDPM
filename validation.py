"""
验证集模块
用于加载和管理固定验证集（定性可视化），以及定量验证集的创建与加载。
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from ddpm_utils import predict_x0_from_noise_shared
from ddpm_dataset import NCTDataset


def load_fixed_validation_batch(val_dir, sample_size=256, device='cuda'):
    """
    从验证目录加载固定的图片用于可视化 (4 TUM + 4 NORM)
    
    Args:
        val_dir: 验证集目录，应包含 'TUM' 和 'NORM' 子目录
        sample_size: 图像尺寸，默认 256
        device: 设备类型
    
    Returns:
        batch_tensor: [B, 3, H, W] 的 Tensor，B=8 (4 TUM + 4 NORM)
        labels_tensor: [B] 的 Tensor，标签 (1=TUM, 0=NORM)
    """
    images = []
    labels = []
    
    # 1. 读取 Tumor (前4张)
    tum_path = os.path.join(val_dir, 'TUM')
    if os.path.exists(tum_path):
        files = sorted([f for f in os.listdir(tum_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])[:4]
        for f in files:
            img = cv2.imread(os.path.join(tum_path, f))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(1)  # TUM 标签
    
    # 2. 读取 Normal (前4张)
    norm_path = os.path.join(val_dir, 'NORM')
    if os.path.exists(norm_path):
        files = sorted([f for f in os.listdir(norm_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])[:4]
        for f in files:
            img = cv2.imread(os.path.join(norm_path, f))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(0)  # NORM 标签
    
    if not images:
        return None, None
    
    # 3. 预处理 (归一化 + Tensor转换)
    # Resize 并堆叠
    batch_resized = []
    for img in images:
        batch_resized.append(cv2.resize(img, (sample_size, sample_size)))
    batch_np = np.array(batch_resized)  # [B, H, W, C]
    
    # 归一化 / 255.0 -> permute -> tensor
    batch_tensor = torch.from_numpy(batch_np).float() / 255.0
    batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(device)  # [B, 3, H, W]
    
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    return batch_tensor, labels_tensor


class ValidationSet:
    """
    验证集管理器
    封装验证集的加载、加噪和可视化逻辑
    """
    def __init__(self, val_dir, scheduler, device='cuda', sample_size=256, fixed_timestep=100):
        """
        初始化验证集管理器
        
        Args:
            val_dir: 验证集目录，应包含 'TUM' 和 'NORM' 子目录
            scheduler: DDPMScheduler 实例
            device: 设备类型
            sample_size: 图像尺寸
            fixed_timestep: 固定的时间步（用于加噪），默认 100
        """
        self.val_dir = val_dir
        self.scheduler = scheduler
        self.device = device
        self.sample_size = sample_size
        self.fixed_timestep = fixed_timestep
        
        self.images = None
        self.labels = None
        self.noisy_images = None
        self.timesteps = None
        self.noise = None
        
    def load(self):
        """
        加载验证集图片
        
        Returns:
            bool: 是否成功加载
        """
        if self.val_dir is None or not os.path.exists(self.val_dir):
            return False
        
        print(f"加载固定验证集用于可视化: {self.val_dir}")
        self.images, self.labels = load_fixed_validation_batch(
            self.val_dir, 
            sample_size=self.sample_size, 
            device=self.device
        )
        
        if self.images is not None:
            print(f"✓ 验证集加载成功，Shape: {self.images.shape}")
            # 生成固定的噪声和时间步
            self._prepare_noise()
            return True
        else:
            print("⚠️ 警告: 未找到验证集图片")
            return False
    
    def _prepare_noise(self):
        """准备固定的噪声和时间步（用于公平对比）"""
        # 固定随机种子以确保每次使用相同的噪声
        original_seed = torch.initial_seed()
        torch.manual_seed(42)
        
        self.noise = torch.randn_like(self.images).to(self.device)
        # 固定时间步 t=fixed_timestep (轻微增强，便于观察效果)
        self.timesteps = torch.full(
            (self.images.shape[0],), 
            self.fixed_timestep, 
            device=self.device, 
            dtype=torch.long
        )
        self.noisy_images = self.scheduler.add_noise(self.images, self.noise, self.timesteps)
        
        # 恢复随机种子
        torch.manual_seed(original_seed)
    
    def generate_enhanced_images(self, unet, feedback_criterion=None):
        """
        使用模型生成增强后的图像
        
        Args:
            unet: U-Net 模型
            feedback_criterion: FeedbackLoss 实例（可选，用于还原 x0）
        
        Returns:
            enhanced_images: 增强后的图像 [B, 3, H, W]
        """
        if self.images is None:
            return None
        
        with torch.no_grad():
            # 构建模型输入
            model_input = torch.cat([self.images, self.noisy_images], dim=1)
            # 预测噪声
            pred_noise = unet(model_input, self.timesteps).sample
            
            # 还原图像
            if feedback_criterion is not None:
                enhanced = feedback_criterion.predict_x0_from_noise(
                    self.noisy_images, pred_noise, self.timesteps
                )
            else:
                # 使用共享的 x0 还原函数
                enhanced = predict_x0_from_noise_shared(
                    self.noisy_images, pred_noise, self.timesteps, self.scheduler
                )
            
            return enhanced
    
    def is_available(self):
        """检查验证集是否可用"""
        return self.images is not None


def create_val_dataloader(val_vis_dir, batch_size, device='cuda'):
    """
    创建定量验证集 DataLoader（TUM + NORM 子目录）。

    Args:
        val_vis_dir: 验证集根目录，需包含 TUM 与 NORM 子目录
        batch_size: 批次大小
        device: 设备

    Returns:
        val_dataloader: DataLoader 或 None（目录不存在时）
    """
    if val_vis_dir is None or not os.path.exists(val_vis_dir):
        return None
    val_tum_dir = os.path.join(val_vis_dir, 'TUM')
    val_norm_dir = os.path.join(val_vis_dir, 'NORM')
    if not os.path.exists(val_tum_dir) or not os.path.exists(val_norm_dir):
        print("⚠️ 警告: 未找到验证集的 TUM/NORM 子目录，跳过定量验证环节。")
        return None
    val_dataset = NCTDataset(val_tum_dir, val_norm_dir, oversample=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == 'cuda'),
    )
    print(f"验证集加载成功，共 {len(val_dataset)} 张图片。")
    return val_dataloader

