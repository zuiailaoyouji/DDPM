"""
DDPM 推理脚本
基于训练时的逻辑进行严格一致的推理，支持迭代增强
"""
import torch
import cv2
import numpy as np
import os
import argparse
import csv
from tqdm import tqdm
from diffusers import DDPMScheduler

from unet_wrapper import create_model
from ddpm_utils import load_hovernet, predict_x0_from_noise_shared, get_device, print_gpu_info


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
    计算逻辑与 feedback_loss.py 中的逻辑保持一致
    
    Args:
        hovernet: HoVer-Net 模型
        img_tensor: [1, 3, H, W] 的 Tensor，范围 [0, 1]
    
    Returns:
        tum_conf: 肿瘤置信度 (0-1)，只计算有细胞核区域的加权平均
    """
    with torch.no_grad():
        # 关键：HoVer-Net 内部会执行 / 255.0，所以需要传入 0-255 范围的输入
        hover_input = img_tensor * 255.0
        output = hovernet(hover_input)
        
        # 与 feedback_loss.py 保持一致的计算逻辑
        probs = torch.softmax(output['tp'], dim=1)
        
        # 获取细胞核 mask（与 feedback_loss.py 第 117 行一致）
        mask = torch.softmax(output['np'], dim=1)[:, 1, :, :]  # [B, H, W]
        
        # 获取肿瘤通道 (Index 1) 的概率
        p_neo = probs[:, 1, :, :]  # [B, H, W]
        
        # 使用 mask 加权平均（与 feedback_loss.py 第 125 行一致）
        # 只计算有细胞核区域的概率，忽略背景区域
        avg_prob = (p_neo * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        
        tum_conf = avg_prob.item()
    return tum_conf


def run_inference(
    img_path, 
    unet, 
    hovernet, 
    scheduler, 
    output_dir, 
    device='cuda', 
    num_iters=5, 
    noise_t=100,
    patience=0.05
):
    """
    运行推理（完全自导向推理，根据第一轮趋势自动锁定方向）
    
    Args:
        img_path: 输入图像路径
        unet: U-Net 模型
        hovernet: HoVer-Net 模型
        scheduler: DDPMScheduler
        output_dir: 输出目录
        device: 设备类型
        num_iters: 最大迭代次数
        noise_t: 加噪时间步（t=100 约等于重绘10%）
        patience: 容忍度，如果分数变化超过此值则停止（默认 0.05）
    
    Returns:
        stats: 统计数据字典，包含文件名、模式、分数历史等信息
    """
    filename = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    # 1. 为每张图像创建独立的文件夹
    image_output_dir = os.path.join(output_dir, name_no_ext)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # 2. 预处理图像
    original_tensor, _ = preprocess_image(img_path, device=device)
    if original_tensor is None:
        print(f"⚠️ 警告: 无法读取图像 {img_path}")
        return
    
    # current_tensor: 当前最好的增强结果，初始为原图
    current_tensor = original_tensor.clone()
    
    print(f"\n处理: {filename}")
    print(f"  输出目录: {image_output_dir}")
    
    # ================= 计算初始评分 (Iter 0) =================
    init_conf = get_hovernet_confidence(hovernet, original_tensor)
    print(f"  Iter 0: {init_conf:.4f}", end="")
    
    # 保存原图到独立文件夹
    save_path = os.path.join(image_output_dir, f"original_conf{init_conf:.2f}.png")
    save_img_np = (original_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    save_img_bgr = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img_bgr)
    # ==============================================================
    
    # --- 初始状态：完全未知，等待第一轮结果锁定方向 ---
    mode = 'pending'
    print(" -> [方向未知] 正在试探 Iter 1...")
    
    # --- 数据记录器（用于 CSV 导出）---
    stats = {
        'Filename': filename,
        'Original_Score': init_conf,
        'Best_Score': init_conf,
        'Best_Iter': 0,
        'Stop_Reason': 'Max_Iters',  # 默认原因
        'Scores_History': [init_conf],  # 记录每一步的分数
        'Mode': 'Pending'  # 先标记为 Pending
    }
    
    # --- 状态追踪变量（用于早停和最佳结果保存）---
    best_conf = init_conf
    best_tensor = original_tensor.clone()
    best_iter = 0
    
    # 2. 迭代增强
    for i in range(num_iters):
        # --- A. 模拟训练中的"加噪"步骤 ---
        # 训练时：noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
        # 推理时：在【上一轮的结果】(current_tensor) 上加噪，让模型继续优化
        
        t_tensor = torch.tensor([noise_t], device=device).long()
        noise = torch.randn_like(current_tensor)
        
        # 产生加噪图像 x_t（基于上一轮的结果）
        x_t = scheduler.add_noise(current_tensor, noise, t_tensor)
        
        # --- B. 构建模型输入 (Conditioning) ---
        # 关键修改：Condition 使用【上一轮的结果】(current_tensor) 而不是 original_tensor
        # 逻辑：告诉模型"目前的图像长 current_tensor 这样，请在它的基础上修补/增强"
        # 这样每次迭代都是基于上一轮的结果进行累积增强，而不是每次都回到原图
        model_input = torch.cat([current_tensor, x_t], dim=1)
        
        # --- C. 模型预测噪声 ---
        with torch.no_grad():
            noise_pred = unet(model_input, t_tensor).sample
        
        # --- D. 还原 x0 (使用与训练时完全一致的公式) ---
        # 使用共享的 x0 还原函数，确保与训练逻辑一致
        pred_x0 = predict_x0_from_noise_shared(x_t, noise_pred, t_tensor, scheduler)
        
        # --- E. 计算当前轮次的置信度 ---
        current_conf = get_hovernet_confidence(hovernet, pred_x0)
        stats['Scores_History'].append(current_conf)  # 记录历史分数
        
        # 打印当前轮次结果
        print(f"  Iter {i+1}: {current_conf:.4f}", end="")
        
        # --- F. 核心逻辑：在第一轮结束时锁定方向 ---
        if i == 0:
            # 比较 Iter 1 和 Iter 0
            if current_conf >= init_conf:
                mode = 'maximize'
                print(" -> [趋势上升] 锁定方向: Maximize (TUM)")
            else:
                mode = 'minimize'
                print(" -> [趋势下降] 锁定方向: Minimize (NORM)")
            stats['Mode'] = mode  # 记录锁定后的模式
        
        # --- G. 智能判断逻辑（基于锁定的 Mode）---
        is_new_best = False
        should_stop = False
        stop_reason = ""
        
        # 只有在方向锁定后才进行判断（第一轮用于锁定方向）
        if mode != 'pending':
            # 根据模式设置目标
            target_goal = 0.98 if mode == 'maximize' else 0.02
            
            if mode == 'maximize':
                # 目标：越高越好
                if current_conf > best_conf:
                    is_new_best = True
                
                # 早停判断
                if current_conf >= target_goal:
                    should_stop = True
                    stop_reason = 'Target_Reached'
                elif current_conf < (best_conf - patience):
                    # 如果不是第一轮（第一轮定义了方向，肯定符合方向），且分数下降
                    if i > 0:
                        should_stop = True
                        stop_reason = 'Degradation'
                        
            elif mode == 'minimize':
                # 目标：越低越好
                if current_conf < best_conf:
                    is_new_best = True
                
                # 早停判断
                if current_conf <= target_goal:
                    should_stop = True
                    stop_reason = 'Target_Reached'
                elif current_conf > (best_conf + patience):
                    if i > 0:
                        should_stop = True
                        stop_reason = 'Hallucination'
        
        # 更新最佳记录（第一轮时，如果方向已锁定，也可以更新）
        if is_new_best:
            best_conf = current_conf
            best_tensor = pred_x0.clone()
            best_iter = i + 1
            stats['Best_Score'] = current_conf
            stats['Best_Iter'] = i + 1
            print(" (New Best!)")
        else:
            print("")  # 换行
        
        # 保存中间结果（用于分析过程）
        save_path = os.path.join(image_output_dir, f"iter{i+1}_conf{current_conf:.2f}.png")
        save_img_np = (pred_x0.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        save_img_bgr = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img_bgr)
        
        # 更新 current_tensor 用于下一轮
        current_tensor = pred_x0
        
        # --- H. 早停判断 ---
        if should_stop:
            print(f"  [早停] {stop_reason}")
            stats['Stop_Reason'] = stop_reason
            break
    
    # --- I. 循环结束，保存最佳结果 ---
    print(f"  >> 完成。最佳结果在 Iter {best_iter}，分数为 {best_conf:.4f}")
    save_path = os.path.join(image_output_dir, f"BEST_iter{best_iter}_conf{best_conf:.2f}.png")
    save_img_np = (best_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    save_img_bgr = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img_bgr)
    
    # 记录绝对变化量
    stats['Improvement_Abs'] = abs(stats['Best_Score'] - stats['Original_Score'])
    
    return stats


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
                       help='设备类型（cuda/cpu），如果指定 gpu_id 则此参数将被忽略')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='指定使用的 GPU ID（例如：0, 1, 2），默认为 None 自动选择')
    parser.add_argument('--patience', type=float, default=0.05,
                       help='容忍度，如果分数变化超过此值则停止（默认 0.05）')
    
    args = parser.parse_args()
    
    # 设备选择逻辑（优先级：gpu_id > device > 自动检测）
    if args.gpu_id is not None:
        # 如果指定了 gpu_id，使用指定的 GPU
        device = get_device(gpu_id=args.gpu_id)
    elif args.device is not None:
        # 如果指定了 device 字符串（兼容旧代码）
        device = args.device
    else:
        # 自动检测
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            device = 'cuda:0'  # 默认使用第一个 GPU
    
    # 打印 GPU 信息
    if device.startswith('cuda'):
        print_gpu_info()
    
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
    
    # 准备 CSV 文件路径
    csv_path = os.path.join(args.output_dir, 'inference_results.csv')
    print(f"\n数据将保存到: {csv_path}")
    
    # --- CSV 初始化 ---
    # 定义表头：基础信息 + 每一轮的分数占位符
    fieldnames = ['Filename', 'Mode', 'Original_Score', 'Best_Score', 'Improvement_Abs', 'Best_Iter', 'Stop_Reason']
    # 动态添加 Iter_0 到 Iter_Max 列
    score_cols = [f'Iter_{i}' for i in range(args.iters + 1)]
    fieldnames.extend(score_cols)
    
    # 运行推理
    print(f"\n开始推理（迭代次数: {args.iters}, 加噪时间步: {args.noise_t}）...")
    print("=" * 60)
    
    # 打开 CSV 文件准备写入
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for img_path in tqdm(files, desc="处理图像"):
            try:
                # 运行推理，获取统计数据
                stats = run_inference(
                    img_path, 
                    unet, 
                    hovernet, 
                    scheduler, 
                    args.output_dir, 
                    device, 
                    args.iters, 
                    args.noise_t,
                    args.patience
                )
                
                if stats is not None:
                    # 整理数据格式写入 CSV
                    row = {
                        'Filename': stats['Filename'],
                        'Mode': stats['Mode'],
                        'Original_Score': f"{stats['Original_Score']:.6f}",
                        'Best_Score': f"{stats['Best_Score']:.6f}",
                        'Improvement_Abs': f"{stats['Improvement_Abs']:.6f}",
                        'Best_Iter': stats['Best_Iter'],
                        'Stop_Reason': stats['Stop_Reason']
                    }
                    # 填充具体的 Iter 分数
                    for i in range(args.iters + 1):
                        col_name = f'Iter_{i}'
                        if i < len(stats['Scores_History']):
                            row[col_name] = f"{stats['Scores_History'][i]:.6f}"
                        else:
                            row[col_name] = ""  # 没跑到的轮次留空
                    
                    writer.writerow(row)
                    # 强制刷新缓冲区，确保哪怕程序崩了数据也写进去了
                    csv_file.flush()
                    
            except Exception as e:
                print(f"\n❌ 处理 {img_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ 推理完成！结果保存在: {args.output_dir}")
    print(f"✓ 数据表格已保存: {csv_path}")


if __name__ == '__main__':
    main()

