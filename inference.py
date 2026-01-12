"""
DDPM 推理脚本
基于训练时的逻辑进行严格一致的推理，支持迭代增强
"""
import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from diffusers import DDPMScheduler

from unet_wrapper import create_model
from ddpm_utils import load_hovernet, predict_x0_from_noise_shared


def preprocess_image(img_path, target_size=256, device='cuda'):
    """
    预处理图像
    
    Args:
        img_path: 图像路径
        target_size: 目标尺寸，默认 256
        device: 设备类型
    
    Returns:
        img_tensor: [1, 3, H, W] 的 Tensor，范围 [0, 1]
        img_rgb: 原始 RGB 图像 (用于可视化)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整尺寸
    img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 归一化到 [0, 1] 范围
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    
    # 转换为 CHW 格式并添加 batch 维度
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    return img_tensor, img_rgb


def get_hovernet_confidence(hovernet, img_tensor):
    """
    获取 HoVer-Net 的肿瘤置信度
    
    Args:
        hovernet: HoVer-Net 模型
        img_tensor: [1, 3, H, W] 的 Tensor，范围 [0, 1]
    
    Returns:
        tum_conf: 肿瘤置信度 (0-1)
    """
    with torch.no_grad():
        # 关键：HoVer-Net 内部会执行 / 255.0，所以需要传入 0-255 范围的输入
        hover_input = img_tensor * 255.0
        output = hovernet(hover_input)
        probs = torch.softmax(output['tp'], dim=1)
        # 获取肿瘤通道 (Index 1) 的平均概率
        tum_conf = probs[:, 1, :, :].mean().item()
    return tum_conf


def run_inference(
    img_path, 
    unet, 
    hovernet, 
    scheduler, 
    output_dir, 
    device='cuda', 
    num_iters=5, 
    noise_t=100
):
    """
    运行推理（迭代增强）
    
    Args:
        img_path: 输入图像路径
        unet: U-Net 模型
        hovernet: HoVer-Net 模型
        scheduler: DDPMScheduler
        output_dir: 输出目录
        device: 设备类型
        num_iters: 迭代次数
        noise_t: 加噪时间步（t=100 约等于重绘10%）
    """
    filename = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    # 1. 预处理图像
    original_tensor, _ = preprocess_image(img_path, device=device)
    if original_tensor is None:
        print(f"⚠️ 警告: 无法读取图像 {img_path}")
        return
    
    # current_tensor: 当前最好的增强结果，初始为原图
    current_tensor = original_tensor.clone()
    
    print(f"\n处理: {filename}")
    
    # 2. 迭代增强
    for i in range(num_iters):
        # --- A. 模拟训练中的"加噪"步骤 ---
        # 训练时：noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
        # 推理时：对当前最好的猜测(current_tensor)进行加噪，让模型继续优化
        
        t_tensor = torch.tensor([noise_t], device=device).long()
        noise = torch.randn_like(current_tensor)
        
        # 产生加噪图像 x_t
        x_t = scheduler.add_noise(current_tensor, noise, t_tensor)
        
        # --- B. 构建模型输入 (Conditioning) ---
        # 训练时：model_input = torch.cat([clean_images, noisy_images], dim=1)
        # 关键决策：condition 始终使用最原始的 original_tensor，
        # 这保证了无论循环多少次，解剖结构都不会漂移
        model_input = torch.cat([original_tensor, x_t], dim=1)
        
        # --- C. 模型预测噪声 ---
        with torch.no_grad():
            noise_pred = unet(model_input, t_tensor).sample
        
        # --- D. 还原 x0 (使用与训练时完全一致的公式) ---
        # 使用共享的 x0 还原函数，确保与训练逻辑一致
        pred_x0 = predict_x0_from_noise_shared(x_t, noise_pred, t_tensor, scheduler)
        
        # --- E. 更新当前图像 ---
        current_tensor = pred_x0
        
        # 计算 HoVer-Net 置信度
        conf = get_hovernet_confidence(hovernet, current_tensor)
        print(f"  Iter {i+1}/{num_iters}: TUM Conf = {conf:.4f}")
        
        # 保存图片
        save_path = os.path.join(output_dir, f"{name_no_ext}_iter{i+1}.png")
        save_img_np = (current_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        save_img_bgr = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img_bgr)
    
    # 保存原始图像（用于对比）
    original_path = os.path.join(output_dir, f"{name_no_ext}_original.png")
    if not os.path.exists(original_path):
        original_img_np = (original_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, original_img_bgr)


def main():
    parser = argparse.ArgumentParser(description='DDPM 推理脚本')
    parser.add_argument('--input_path', type=str, required=True, 
                       help='输入图像路径或目录（支持目录时会处理目录下所有图片）')
    parser.add_argument('--output_dir', type=str, default='./results/inference',
                       help='输出目录')
    parser.add_argument('--unet_path', type=str, required=True,
                       help='U-Net 模型检查点路径')
    parser.add_argument('--hovernet_path', type=str, required=True,
                       help='HoVer-Net 模型路径')
    parser.add_argument('--iters', type=int, default=5,
                       help='迭代增强次数（默认 5）')
    parser.add_argument('--noise_t', type=int, default=100,
                       help='加噪时间步（默认 100，约等于重绘10%%）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备类型（cuda/cpu），默认自动检测')
    
    args = parser.parse_args()
    
    # 设备检测
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("\n加载模型...")
    
    # 加载 U-Net
    print(f"  加载 U-Net: {args.unet_path}")
    unet = create_model().to(device)
    checkpoint = torch.load(args.unet_path, map_location=device)
    
    # 处理检查点格式（可能是完整的字典或直接是 state_dict）
    if 'model_state_dict' in checkpoint:
        unet.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        # 如果是直接的 state_dict
        unet.load_state_dict(checkpoint)
    unet.eval()
    print("  ✓ U-Net 加载完成")
    
    # 加载 HoVer-Net
    print(f"  加载 HoVer-Net: {args.hovernet_path}")
    hovernet = load_hovernet(args.hovernet_path, device=device)
    print("  ✓ HoVer-Net 加载完成")
    
    # 初始化调度器
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 获取文件列表
    if os.path.isdir(args.input_path):
        # 如果是目录，获取所有图片文件
        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        files = [
            os.path.join(args.input_path, f) 
            for f in os.listdir(args.input_path) 
            if f.lower().endswith(image_extensions)
        ]
        print(f"\n找到 {len(files)} 个图像文件")
    else:
        # 如果是单个文件
        files = [args.input_path]
    
    if not files:
        print("❌ 错误: 未找到任何图像文件")
        return
    
    # 运行推理
    print(f"\n开始推理（迭代次数: {args.iters}, 加噪时间步: {args.noise_t}）...")
    print("=" * 60)
    
    for img_path in tqdm(files, desc="处理图像"):
        try:
            run_inference(
                img_path, 
                unet, 
                hovernet, 
                scheduler, 
                args.output_dir, 
                device, 
                args.iters, 
                args.noise_t
            )
        except Exception as e:
            print(f"\n❌ 处理 {img_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ 推理完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()

