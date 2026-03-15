"""
DDPM 像素级自导向推理脚本 (Pixel-wise Self-Guided Inference)
核心工作流（四阶段）：
1. 状态初始化：原图 → HoVer-Net 得 init_p_neo、cell_mask，pixel_active_mask 全活跃。
2. 试探与方向锁定 (Iter 0)：盲目去噪 → 读意图 Δ → 像素级锁定 Maximize(0.98) / Minimize(0.02)。
3. 迭代演化：DDPM 去噪 + 宏观三态/跳变监控 + 像素级微观结算（失真/达标/退化/集体停机）+ 空间状态拼图。
4. 终态输出：合成“历史最佳”图像，背景自然平滑，日志写入 CSV。
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
import csv
from tqdm import tqdm
from diffusers import DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from collections import deque

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
    获取 HoVer-Net 的肿瘤置信度（单样本）
    计算逻辑与 feedback_loss.py 中的逻辑保持一致
    
    Args:
        hovernet: HoVer-Net 模型
        img_tensor: [1, 3, H, W] 的 Tensor，范围 [0, 1]
    
    Returns:
        tum_conf: 肿瘤置信度 (0-1)，只计算有细胞核区域的加权平均
    """
    with torch.no_grad():
        # 确保输入张量在正确的设备上（与 HoVer-Net 模型一致）
        hovernet_device = next(hovernet.parameters()).device
        if img_tensor.device != hovernet_device:
            img_tensor = img_tensor.to(hovernet_device)
        
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


def get_hovernet_confidence_batch(hovernet, img_tensor):
    """[Batch] 获取肿瘤置信度（图像级标量）"""
    with torch.no_grad():
        hovernet_device = next(hovernet.parameters()).device
        if img_tensor.device != hovernet_device:
            img_tensor = img_tensor.to(hovernet_device)
        hover_input = img_tensor * 255.0
        output = hovernet(hover_input)
        probs = torch.softmax(output['tp'], dim=1)
        mask = torch.softmax(output['np'], dim=1)[:, 1, :, :]
        p_neo = probs[:, 1, :, :]
        numerator = (p_neo * mask).sum(dim=(1, 2))
        denominator = mask.sum(dim=(1, 2)) + 1e-6
        avg_probs = numerator / denominator
    return avg_probs


def get_hovernet_pixel_maps(hovernet, img_tensor):
    """
    获取 HoVer-Net 的像素级肿瘤概率图与细胞核掩膜（用于像素级自导向）。
    HoVer-Net 输出空间尺寸可能与输入不一致，会插值到与 img_tensor 相同的 (H, W)，保证与后续 pixel_target 等张量维度一致。
    
    Args:
        hovernet: HoVer-Net 模型
        img_tensor: [B, 3, H, W]，范围 [0, 1]
    
    Returns:
        p_neo: [B, H, W] 肿瘤概率图
        cell_mask: [B, H, W] 细胞核区域掩膜（软掩膜或二值）
    """
    with torch.no_grad():
        hovernet_device = next(hovernet.parameters()).device
        if img_tensor.device != hovernet_device:
            img_tensor = img_tensor.to(hovernet_device)
        target_h, target_w = img_tensor.shape[2], img_tensor.shape[3]
        hover_input = img_tensor * 255.0
        output = hovernet(hover_input)
        probs = torch.softmax(output['tp'], dim=1)
        cell_mask = torch.softmax(output['np'], dim=1)[:, 1, :, :]  # [B, H', W']
        p_neo = probs[:, 1, :, :]  # [B, H', W']
        # 若 HoVer-Net 输出尺寸与输入图像不一致，插值到 (H, W)
        if p_neo.shape[-2] != target_h or p_neo.shape[-1] != target_w:
            p_neo = F.interpolate(p_neo.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
            cell_mask = F.interpolate(cell_mask.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
    return p_neo, cell_mask


# ==========================================
# 宏观分布与跳变监控器 (三态 + 跳变率 + 收敛判定)
# ==========================================

class MacroDistributionMonitor:
    """
    宏观分布监控：三态占比、跨越 0.5 的跳变率、两极稳定即“分布锁死”。
    """
    def __init__(self, patience_rounds=3, polar_tolerance=0.005):
        """
        Args:
            patience_rounds: 连续 N 轮两极占比波动 < polar_tolerance 即判定收敛
            polar_tolerance: 两极占比允许波动（如 0.5%）
        """
        self.patience_rounds = patience_rounds
        self.polar_tolerance = polar_tolerance
        self.history_normal_ratio = deque(maxlen=patience_rounds + 1)
        self.history_tumor_ratio = deque(maxlen=patience_rounds + 1)

    def update(self, p_neo, cell_mask):
        """
        基于细胞区域像素概率统计三态占比。
        p_neo: [B, H, W], cell_mask: [B, H, W]
        """
        valid = cell_mask > 0.5
        if valid.sum() == 0:
            return 0.0, 0.0, 0.0
        p_flat = p_neo[valid].float()
        normal_ratio = (p_flat < 0.3).float().mean().item()
        mid_ratio = ((p_flat >= 0.3) & (p_flat <= 0.7)).float().mean().item()
        tumor_ratio = (p_flat > 0.7).float().mean().item()
        self.history_normal_ratio.append(normal_ratio)
        self.history_tumor_ratio.append(tumor_ratio)
        return normal_ratio, mid_ratio, tumor_ratio

    def add_polar_ratios(self, normal_ratio, tumor_ratio):
        """外部已算好两极占比时可直接填入（用于与 update 一致的历史）"""
        self.history_normal_ratio.append(normal_ratio)
        self.history_tumor_ratio.append(tumor_ratio)

    def is_distribution_locked(self):
        """最近 N 轮两极占比波动均小于阈值则判定分布锁死"""
        if len(self.history_normal_ratio) < self.patience_rounds + 1:
            return False
        last_n = self.history_normal_ratio[-1]
        last_t = self.history_tumor_ratio[-1]
        for i in range(len(self.history_normal_ratio) - 1):
            if abs(self.history_normal_ratio[i] - last_n) > self.polar_tolerance:
                return False
            if abs(self.history_tumor_ratio[i] - last_t) > self.polar_tolerance:
                return False
        return True


def compute_jump_rate(init_p_neo, curr_p_neo, cell_mask):
    """
    对比 init_p_neo，计算跨越 0.5 中轴线的像素比例（Jump Rate）。
    init_p_neo, curr_p_neo, cell_mask: [B, H, W]
    """
    with torch.no_grad():
        init_side = (init_p_neo >= 0.5).float()
        curr_side = (curr_p_neo >= 0.5).float()
        crossed = (init_side != curr_side).float() * cell_mask
        total_cell = cell_mask.sum().clamp(min=1e-6)
        jump_rate = crossed.sum() / total_cell
    return jump_rate.item()


# ==========================================
# 旧版：图像级分布收敛监控器（保留兼容）
# ==========================================

class DistributionConvergenceMonitor:
    """
    监控 Batch 内样本分类比例的变化
    目标：当 TUM/NORM 比例不再跳变时，认为群体增强完成，允许停止。
    """
    def __init__(self, patience=2, tolerance=0.0):
        """
        Args:
            patience: 连续 N 轮比例保持稳定才算收敛
            tolerance: 允许的浮动误差（通常设为 0，要求严格稳定）
        """
        self.history_ratios = deque(maxlen=patience + 1)
        self.patience = patience
        self.tolerance = tolerance
        
    def update(self, confs):
        """
        输入当前 Batch 的所有置信度，计算 TUM 占比
        """
        # 判定阈值 0.5：大于 0.5 视为模型认为是 TUM
        tum_count = (confs > 0.5).sum().item()
        total = len(confs)
        ratio = tum_count / total
        
        self.history_ratios.append(ratio)
        return ratio, tum_count, total - tum_count

    def is_converged(self):
        """判断分布是否已稳定"""
        if len(self.history_ratios) < self.patience + 1:
            return False
        
        # 检查最近 patience+1 次记录是否基本一致
        # 例如 patience=2，需要 [R_t-2, R_t-1, R_t] 都相等
        latest = self.history_ratios[-1]
        for r in list(self.history_ratios)[:-1]:
            if abs(r - latest) > self.tolerance:
                return False
        return True


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
    运行推理（完全自导向推理，根据初始置信度自动锁定方向）
    
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
    
    # --- 核心：完全自导向模式锁定 ---
    # 信任模型：>0.5 就是 TUM(Maximize), <=0.5 就是 NORM(Minimize)
    mode = 'maximize' if init_conf > 0.5 else 'minimize'
    mode_label = 'Maximize (TUM)' if mode == 'maximize' else 'Minimize (NORM)'
    
    print(f"  Iter 0: {init_conf:.4f} -> [方向锁定] {mode_label}")
    
    # 保存原图到独立文件夹
    save_path = os.path.join(image_output_dir, f"original_conf{init_conf:.2f}.png")
    save_img_np = (original_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    save_img_bgr = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img_bgr)
    # ==============================================================
    
    # --- 数据记录器（用于 CSV 导出）---
    stats = {
        'Filename': filename,
        'Original_Score': init_conf,
        'Best_Score': init_conf,
        'Best_Iter': 0,
        'Stop_Reason': 'Max_Iters',  # 默认原因
        'Scores_History': [init_conf],  # 记录每一步的分数
        'Mode': mode.capitalize()  # 直接记录锁定的模式
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
        
        # --- F. 智能判断逻辑（基于锁定的 Mode）---
        # 根据模式设置目标
        target_goal = 0.98 if mode == 'maximize' else 0.02
        
        is_new_best = False
        should_stop = False
        stop_reason = ""
        
        if mode == 'maximize':
            # 目标：越高越好
            if current_conf > best_conf:
                is_new_best = True
            
            # 早停判断
            if current_conf >= target_goal:
                should_stop = True
                stop_reason = 'Target_Reached'
            elif current_conf < (best_conf - patience):
                # 如果分数下降超过容忍度，停止
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
                # 如果分数上升超过容忍度，停止（退化）
                should_stop = True
                stop_reason = 'Degradation'
        
        # 更新最佳记录
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


# ==========================================
# 批量推理相关函数
# ==========================================

class InferenceDataset(Dataset):
    """高效图像加载器"""
    def __init__(self, file_paths, target_size=256):
        self.file_paths = file_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = cv2.imread(path)
            if img is None:
                return torch.zeros((3, self.target_size, self.target_size)), "ERROR"
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1) 
            filename = os.path.basename(path)
            return img_tensor, filename
        except Exception as e:
            return torch.zeros((3, self.target_size, self.target_size)), "ERROR"


def save_batch_images(tensors, filenames, output_dir, suffix):
    """批量保存"""
    for i, fname in enumerate(filenames):
        if fname == "ERROR": continue
        name_no_ext = os.path.splitext(fname)[0]
        save_dir = os.path.join(output_dir, name_no_ext)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{suffix}.png")
        
        img_tensor = tensors[i].detach().cpu()
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)


def run_batch_inference(dataloader, unet, hovernet, scheduler, args):
    """
    像素级自导向推理（四阶段）：
    阶段一：状态初始化 → init_p_neo、cell_mask、pixel_active_mask 全活跃
    阶段二：Iter 0 试探 → 盲目去噪，像素级方向锁定 (Maximize 0.98 / Minimize 0.02)
    阶段三：Iter 1~Max 迭代 → DDPM 去噪 + 宏观三态/跳变监控 + 像素级微观结算 + 空间拼图
    阶段四：终态合成 → 输出历史最佳拼图，写 CSV（跳变率、迭代数、三态比例、提升量等）
    """
    csv_path = os.path.join(args.output_dir, 'batch_results.csv')
    # 像素级早停下无整图统一 Best_Iter；Total_Iters = 触发集体停机的终态轮次（分布锁死或达最大迭代）
    fieldnames = [
        'Filename', 'Original_Score', 'Best_Score', 'Improvement_Abs', 'Stop_Reason',
        'Jump_Rate', 'Final_Normal_Ratio', 'Final_Mid_Ratio', 'Final_Tumor_Ratio', 'Total_Iters'
    ]
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    device = args.device
    max_distortion = getattr(args, 'max_distortion', 0.1)
    polar_tolerance = getattr(args, 'polar_tolerance', 0.005)
    macro_patience = getattr(args, 'macro_patience', 3)

    for batch_idx, (img_tensors, filenames) in enumerate(tqdm(dataloader, desc="Batch Processing")):
        valid_indices = [i for i, f in enumerate(filenames) if f != "ERROR"]
        if not valid_indices:
            continue
        img_tensors = img_tensors[valid_indices].to(device)
        filenames = [filenames[i] for i in valid_indices]
        B, C, H, W = img_tensors.shape

        # ========== 阶段一：状态初始化 ==========
        init_p_neo, cell_mask = get_hovernet_pixel_maps(hovernet, img_tensors)
        pixel_active_mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
        original_tensor = img_tensors.clone()
        current_tensor = img_tensors.clone()
        best_snapshot = img_tensors.clone()
        pixel_target = torch.full((B, H, W), 0.5, device=device, dtype=img_tensors.dtype)
        best_p_neo = init_p_neo.clone()
        cell_binary = (cell_mask > 0.5).float()

        save_batch_images(img_tensors, filenames, args.output_dir, "original")
        init_confs = get_hovernet_confidence_batch(hovernet, img_tensors)
        best_confs = init_confs.clone()
        stop_reasons = ["Max_Iters"] * B
        macro_monitor = MacroDistributionMonitor(patience_rounds=macro_patience, polar_tolerance=polar_tolerance)

        print(f"\n--- Batch {batch_idx+1} (Size: {B}) ---")

        # ========== 阶段二：Iter 0 试探与方向锁定 ==========
        t_tensor = torch.tensor([args.noise_t], device=device).long().expand(B)
        noise = torch.randn_like(current_tensor)
        x_t = scheduler.add_noise(current_tensor, noise, t_tensor)
        model_input = torch.cat([current_tensor, x_t], dim=1)
        with torch.no_grad():
            noise_pred = unet(model_input, t_tensor).sample
        pred_x0 = predict_x0_from_noise_shared(x_t, noise_pred, t_tensor, scheduler)
        probe_p_neo, _ = get_hovernet_pixel_maps(hovernet, pred_x0)
        delta = probe_p_neo - init_p_neo
        pixel_target = torch.where(cell_binary > 0.5, torch.where(delta >= 0, torch.full_like(delta, 0.98), torch.full_like(delta, 0.02)), pixel_target)
        current_tensor = pred_x0.clone()
        best_p_neo = torch.where(cell_binary > 0.5, probe_p_neo, best_p_neo)
        better_max = (cell_binary > 0.5) & (pixel_target > 0.5) & (probe_p_neo > best_p_neo)
        better_min = (cell_binary > 0.5) & (pixel_target < 0.5) & (probe_p_neo < best_p_neo)
        best_p_neo = torch.where(better_max | better_min, probe_p_neo, best_p_neo)
        best_snapshot = torch.where((better_max | better_min).unsqueeze(1).expand_as(best_snapshot), pred_x0, best_snapshot)

        # ========== 阶段三：Iter 1 ~ Max_Iters ==========
        total_iters_done = 1
        for i in range(1, args.iters):
            if (pixel_active_mask & (cell_mask > 0.5)).sum() == 0:
                break
            total_iters_done = i + 1

            # 3.1 DDPM 去噪
            t_tensor = torch.tensor([args.noise_t], device=device).long().expand(B)
            noise = torch.randn_like(current_tensor)
            x_t = scheduler.add_noise(current_tensor, noise, t_tensor)
            model_input = torch.cat([current_tensor, x_t], dim=1)
            with torch.no_grad():
                noise_pred = unet(model_input, t_tensor).sample
            pred_x0 = predict_x0_from_noise_shared(x_t, noise_pred, t_tensor, scheduler)
            curr_p_neo, _ = get_hovernet_pixel_maps(hovernet, pred_x0)

            # 3.2 宏观分布与跳变监控
            normal_ratio, mid_ratio, tumor_ratio = macro_monitor.update(curr_p_neo, cell_mask)
            jump_rate = compute_jump_rate(init_p_neo, curr_p_neo, cell_mask)
            distribution_locked = macro_monitor.is_distribution_locked()

            print(f"  Iter {i+1}: Normal<0.3={normal_ratio:.2%} | Mid={mid_ratio:.2%} | Tumor>0.7={tumor_ratio:.2%} | Jump={jump_rate:.2%}", end="")
            if distribution_locked:
                print(" [分布锁死]")
            else:
                print("")

            # 3.3 像素级微观结算（仅细胞区域）
            mse_per_pixel = ((pred_x0 - original_tensor) ** 2).mean(dim=1)
            cond_a_distortion = (mse_per_pixel > max_distortion) & (cell_mask > 0.5)
            reached_max = (curr_p_neo >= pixel_target - 1e-4) & (pixel_target > 0.5) & (cell_mask > 0.5)
            reached_min = (curr_p_neo <= pixel_target + 1e-4) & (pixel_target < 0.5) & (cell_mask > 0.5)
            cond_b_target = reached_max | reached_min
            regress_max = (curr_p_neo < best_p_neo - args.patience) & (pixel_target > 0.5) & (cell_mask > 0.5)
            regress_min = (curr_p_neo > best_p_neo + args.patience) & (pixel_target < 0.5) & (cell_mask > 0.5)
            cond_c_degradation = regress_max | regress_min
            cond_d_macro = distribution_locked & (cell_mask > 0.5)
            freeze_this = pixel_active_mask & (cell_mask > 0.5) & (cond_a_distortion | cond_b_target | cond_c_degradation | cond_d_macro)

            # 3.4 空间状态更新：记录最佳、冻结剔除、拼图
            better_max = (cell_mask > 0.5) & (pixel_target > 0.5) & (curr_p_neo > best_p_neo)
            better_min = (cell_mask > 0.5) & (pixel_target < 0.5) & (curr_p_neo < best_p_neo)
            improved = better_max | better_min
            best_p_neo = torch.where(improved, curr_p_neo, best_p_neo)
            best_snapshot = torch.where(improved.unsqueeze(1).expand_as(best_snapshot), pred_x0, best_snapshot)
            pixel_active_mask = pixel_active_mask & ~freeze_this

            current_tensor = pred_x0.clone()
            current_tensor = torch.where(~pixel_active_mask.unsqueeze(1).expand_as(current_tensor), best_snapshot, current_tensor)

            if distribution_locked:
                break

        # ========== 阶段四：终态输出 ==========
        # 使用 current_tensor 而非 best_snapshot：后者仅更新细胞区，背景始终是原图；
        # current_tensor 已是“缝合体”：冻结细胞=best_snapshot，活跃区（含背景）= 最后一轮 DDPM 结果，背景自然平滑
        final_tensor = current_tensor
        final_p_neo, _ = get_hovernet_pixel_maps(hovernet, final_tensor)
        valid = cell_mask > 0.5
        if valid.sum() > 0:
            p_flat = final_p_neo[valid].float()
            final_norm = (p_flat < 0.3).float().mean().item()
            final_mid = ((p_flat >= 0.3) & (p_flat <= 0.7)).float().mean().item()
            final_tum = (p_flat > 0.7).float().mean().item()
        else:
            final_norm = final_mid = final_tum = 0.0
        final_jump = compute_jump_rate(init_p_neo, final_p_neo, cell_mask)
        save_batch_images(final_tensor, filenames, args.output_dir, "BEST")
        final_confs = get_hovernet_confidence_batch(hovernet, final_tensor)
        for idx in range(B):
            best_confs[idx] = final_confs[idx]

        rows = []
        for idx in range(B):
            row = {
                'Filename': filenames[idx],
                'Original_Score': f"{init_confs[idx].item():.6f}",
                'Best_Score': f"{best_confs[idx].item():.6f}",
                'Improvement_Abs': f"{abs(best_confs[idx].item() - init_confs[idx].item()):.6f}",
                'Stop_Reason': stop_reasons[idx],
                'Jump_Rate': f"{final_jump:.4f}",
                'Final_Normal_Ratio': f"{final_norm:.4f}",
                'Final_Mid_Ratio': f"{final_mid:.4f}",
                'Final_Tumor_Ratio': f"{final_tum:.4f}",
                'Total_Iters': total_iters_done,
            }
            rows.append(row)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)
        print(f"  >> Batch {batch_idx+1} 完成 (Jump={final_jump:.2%}, N/M/T={final_norm:.2%}/{final_mid:.2%}/{final_tum:.2%})")


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
                       help='微观早停：概率相对历史最佳恶化超过此值则冻结该像素（默认 0.05）')
    parser.add_argument('--max_distortion', type=float, default=0.1,
                       help='微观早停：像素与原图 MSE 超过此阈值则冻结，防结构崩坏（默认 0.1）')
    parser.add_argument('--polar_tolerance', type=float, default=0.005,
                       help='宏观收敛：两极占比波动小于此值连续 N 轮即判定分布锁死（默认 0.005）')
    parser.add_argument('--macro_patience', type=int, default=3,
                       help='宏观收敛：连续 N 轮两极稳定即触发集体停机（默认 3）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批量大小，若指定则启用像素级批量推理，否则单样本处理')
    
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

    args.device = device  # 将计算出的 device 更新回 args 对象
    
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

    # 统一使用像素级四阶段推理（结果与 CSV 写入 run_batch_inference 内的 batch_results.csv）（batch_size=1 即为逐张像素级推理）
    batch_size = args.batch_size if args.batch_size is not None else 1
    print(f"\n🚀 像素级自导向推理 (Batch Size: {batch_size})")
    print(f"迭代: {args.iters}, 加噪步: {args.noise_t}, 失真阈值: {getattr(args, 'max_distortion', 0.1)}, 宏观稳定轮数: {getattr(args, 'macro_patience', 3)}")
    print("=" * 60)

    dataset = InferenceDataset(files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    run_batch_inference(dataloader, unet, hovernet, scheduler, args)

    print("\n" + "=" * 60)
    print(f"✅ 推理完成！结果与 CSV 见: {args.output_dir}")


if __name__ == '__main__':
    main()

