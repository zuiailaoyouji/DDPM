"""
可视化模块 (DDPM/visualization.py)
负责训练过程中的图像拼接、差异热力图计算与保存
"""
import torch
import os
import torchvision


def save_training_progress_images(clean_images, noisy_images, x0_pred, epoch, save_dir, num_vis=4, return_tensor=False):
    """
    保存训练过程对比图：[原图 | 加噪图 | 增强图 | 差异热力图]
    
    Args:
        clean_images: 原始图像 [B, 3, H, W]
        noisy_images: 加噪图像 [B, 3, H, W]
        x0_pred: 模型预测/增强后的图像 [B, 3, H, W]
        epoch: 当前轮数
        save_dir: 保存目录
        num_vis: 保存的样本数量
        return_tensor: 是否返回拼接好的 tensor（用于 TensorBoard）
    
    Returns:
        vis_tensor: 如果 return_tensor=True，返回拼接好的 tensor [4*N, 3, H, W]
    """
    with torch.no_grad():
        # 1. 截取前 N 张
        n = min(num_vis, clean_images.shape[0])
        clean = clean_images[:n].cpu()
        noisy = noisy_images[:n].cpu()
        enhanced = x0_pred[:n].cpu()
        
        # 2. 差异图：|增强 - 原始|，clamp 到 [0, 1]
        diff = torch.abs(enhanced - clean).clamp(0, 1)
        
        # 为了让差异图更酷炫，可以只取单通道亮度和伪彩色 (这里为了简单先保持 RGB 灰度风格)
        # 如果想变成热力图风格，通常需要 matplotlib，但为了 tensor 拼接方便，直接存亮度即可
        
        # 3. 拼接: 垂直拼接 (每行一张图的四个阶段) 或者 水平拼接 (四行)
        # 这里采用: 第一行原图，第二行加噪，第三行增强，第四行差异
        vis_tensor = torch.cat([clean, noisy, enhanced, diff], dim=0)
        
        # 4. 保存
        # nrow=n 表示每行显示 n 张图片 (即每一行对应一种类型)
        filename = os.path.join(save_dir, f'epoch_{epoch}_vis.png')
        torchvision.utils.save_image(vis_tensor, filename, nrow=n, normalize=False)
        
        # 5. 返回拼接好的 tensor（用于 TensorBoard）
        if return_tensor:
            return vis_tensor

