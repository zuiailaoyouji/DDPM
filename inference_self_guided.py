"""
DDPM 自导向分布监控推理脚本 (Batch Inference with Distribution Convergence)
核心特性：
1. 自导向推理：完全信任模型 Iter 0 的判断，自动锁定方向。
2. 双层早停：
   - 微观：单样本达标即停 (Target Reached)。
   - 宏观：Batch 分布稳定即停 (Distribution Convergence)。
3. 实时监控：打印每一轮的 TUM 占比，可视化增强进程。
"""
import torch
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
    """[Batch] 获取肿瘤置信度"""
    with torch.no_grad():
        # 确保输入张量在正确的设备上（与 HoVer-Net 模型一致）
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


# ==========================================
# 核心组件：分布收敛监控器 (Distribution Monitor)
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
                # 如果分数上升超过容忍度，停止（可能是幻觉）
                should_stop = True
                stop_reason = 'Hallucination'
        
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
    """执行 Batch 推理（带分布监控和动态校正）"""
    
    # 初始化 CSV
    csv_path = os.path.join(args.output_dir, 'batch_results.csv')
    fieldnames = ['Filename', 'Mode', 'Original_Score', 'Best_Score', 'Improvement_Abs', 'Best_Iter', 'Stop_Reason']
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    
    device = args.device
    
    # 定义经验阈值（用于动态校正）
    AMBIGUOUS_LOW = 0.35   # 模糊区域下界
    AMBIGUOUS_HIGH = 0.65  # 模糊区域上界（对称的）
    
    for batch_idx, (img_tensors, filenames) in enumerate(tqdm(dataloader, desc="Batch Processing")):
        
        # 0. 数据清洗
        valid_indices = [i for i, f in enumerate(filenames) if f != "ERROR"]
        if not valid_indices: continue
        img_tensors = img_tensors[valid_indices].to(device)
        filenames = [filenames[i] for i in valid_indices]
        batch_size = len(filenames)
        
        # ==========================================
        # 状态初始化
        # ==========================================
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        current_tensors = img_tensors.clone()
        best_tensors = img_tensors.clone()
        
        # 1. 计算初始分数和模式锁定（依然遵循 0.5 的贝叶斯原则）
        init_confs = get_hovernet_confidence_batch(hovernet, img_tensors)
        best_confs = init_confs.clone()
        best_iters = torch.zeros(batch_size, dtype=torch.long, device=device)
        stop_reasons = ["Max_Iters"] * batch_size
        
        # 初始模式锁定：信任模型 >0.5 就是 TUM(Maximize), <=0.5 就是 NORM(Minimize)
        modes = ['maximize' if s > 0.5 else 'minimize' for s in init_confs.tolist()]
        target_goals = torch.tensor([0.98 if m == 'maximize' else 0.02 for m in modes], device=device)
        
        # 2. 标记"有资格反悔"的样本（0.35 <= Score <= 0.65）
        # 只有落在这个区间的样本，才会被监控 Iter 1 的趋势，允许动态校正
        is_ambiguous = [(s >= AMBIGUOUS_LOW and s <= AMBIGUOUS_HIGH) for s in init_confs.tolist()]
        
        # 初始化分布监控器
        dist_monitor = DistributionConvergenceMonitor(patience=2) # 连续 2 轮比例不变即停止
        
        # 保存原图
        save_batch_images(img_tensors, filenames, args.output_dir, "original")
        
        print(f"\n--- Batch {batch_idx+1} Start (Size: {batch_size}) ---")
        
        # ==========================================
        # 迭代循环
        # ==========================================
        for i in range(args.iters):
            # 如果所有样本都微观停止了，退出
            if not active_mask.any():
                break
                
            # A. 加噪 & 预测 & 还原 (标准流程)
            t_tensor = torch.tensor([args.noise_t], device=device).long().expand(batch_size)
            noise = torch.randn_like(current_tensors)
            x_t = scheduler.add_noise(current_tensors, noise, t_tensor)
            model_input = torch.cat([current_tensors, x_t], dim=1)
            
            with torch.no_grad():
                noise_pred = unet(model_input, t_tensor).sample
            
            pred_x0 = predict_x0_from_noise_shared(x_t, noise_pred, t_tensor, scheduler)
            
            # E. 评分
            current_confs = get_hovernet_confidence_batch(hovernet, pred_x0)
            
            # ===【核心修改：基于 0.35 阈值的动态校正】===
            # 在 Iter 1 结束时进行动态判定（只在第一轮校正）
            if i == 0:
                for idx in range(batch_size):
                    if not active_mask[idx]: 
                        continue
                    
                    # 只有处于 0.35-0.65 之间的样本才允许校正
                    # < 0.35 的样本即使涨分了，也视为幻觉，坚决不改方向
                    if is_ambiguous[idx]:
                        delta = current_confs[idx].item() - init_confs[idx].item()
                        current_mode = modes[idx]
                        
                        # 场景：初始 0.4 (Minimize)，但增强后变成了 0.45 (+0.05)
                        # 因为它 > 0.35，我们允许它"反水"变成 Maximize
                        if current_mode == 'minimize' and delta > 0.02:
                            print(f"  🔄 [校正] {filenames[idx]}: Minimize -> Maximize (Init={init_confs[idx].item():.2f}, Δ={delta:.4f})")
                            modes[idx] = 'maximize'
                            target_goals[idx] = 0.98
                            # 如果之前没有设置过停止原因，标记为校正切换
                            if stop_reasons[idx] == "Max_Iters":
                                stop_reasons[idx] = "Correction_Switch"
                        
                        # 场景：初始 0.55 (Maximize)，但增强后变成了 0.50 (-0.05)
                        elif current_mode == 'maximize' and delta < -0.02:
                            print(f"  🔄 [校正] {filenames[idx]}: Maximize -> Minimize (Init={init_confs[idx].item():.2f}, Δ={delta:.4f})")
                            modes[idx] = 'minimize'
                            target_goals[idx] = 0.02
                            # 如果之前没有设置过停止原因，标记为校正切换
                            if stop_reasons[idx] == "Max_Iters":
                                stop_reasons[idx] = "Correction_Switch"
            
            # ==========================================
            # 分布监控 (Distribution Check)
            # ==========================================
            # 记录当前轮次的分布状态
            ratio, tum_num, norm_num = dist_monitor.update(current_confs)
            
            print(f"  Iter {i+1}: TUM={tum_num} | NORM={norm_num} | Ratio={ratio:.2f}", end="")
            
            # 检查宏观收敛 (Batch Convergence)
            # 只有在跑了几轮之后(例如 >= 2)才检查，给一点震荡空间
            batch_converged = False
            if i >= 2 and dist_monitor.is_converged():
                print(" -> [稳定] 分布不再跳变，Batch 级早停触发！")
                batch_converged = True
            else:
                print(" -> [波动]" if i < 2 else " -> [未稳]")
            
            # ==========================================
            # 微观控制 (Individual Control)
            # ==========================================
            for idx in range(batch_size):
                # 如果这个样本已经停了，就不管它
                if not active_mask[idx]: 
                    continue
                
                # 如果 Batch 收敛了，强制停止所有 Active 样本
                if batch_converged:
                    active_mask[idx] = False
                    stop_reasons[idx] = "Batch_Converged"
                    continue
                
                # 正常的单样本逻辑
                curr_score = current_confs[idx].item()
                best_score = best_confs[idx].item()
                mode = modes[idx]
                patience = args.patience
                target = target_goals[idx].item()
                
                is_new_best = False
                should_stop = False
                reason = ""
                
                if mode == 'maximize': # TUM
                    if curr_score > best_score: is_new_best = True
                    if curr_score >= target:
                        should_stop = True; reason = "Target_Reached"
                    elif curr_score < (best_score - patience):
                        should_stop = True; reason = "Degradation"
                else: # NORM
                    if curr_score < best_score: is_new_best = True
                    if curr_score <= target:
                        should_stop = True; reason = "Target_Reached"
                    elif curr_score > (best_score + patience):
                        should_stop = True; reason = "Hallucination"
                
                if is_new_best:
                    best_confs[idx] = curr_score
                    best_tensors[idx] = pred_x0[idx].clone()
                    best_iters[idx] = i + 1
                
                if should_stop:
                    active_mask[idx] = False
                    stop_reasons[idx] = reason
                
                current_tensors[idx] = pred_x0[idx]
                
            # 如果 Batch 收敛触发了，跳出循环
            if batch_converged:
                break

        # ==========================================
        # Batch 结束：保存
        # ==========================================
        save_batch_images(best_tensors, filenames, args.output_dir, "BEST")
        
        rows = []
        for idx in range(batch_size):
            row = {
                'Filename': filenames[idx],
                'Mode': modes[idx],
                'Original_Score': f"{init_confs[idx].item():.6f}",
                'Best_Score': f"{best_confs[idx].item():.6f}",
                'Improvement_Abs': f"{abs(best_confs[idx].item() - init_confs[idx].item()):.6f}",
                'Best_Iter': best_iters[idx].item(),
                'Stop_Reason': stop_reasons[idx]
            }
            rows.append(row)
            
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)
        
        print(f"  >> Batch {batch_idx+1} 完成")


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
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批量大小，如果指定则启用批量处理模式（带分布监控），否则单样本处理')
    
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
    
    # 准备 CSV 文件路径
    csv_path = os.path.join(args.output_dir, 'inference_results.csv')
    print(f"\n数据将保存到: {csv_path}")
    
    # --- CSV 初始化 ---
    # 定义表头：基础信息 + 每一轮的分数占位符
    fieldnames = ['Filename', 'Mode', 'Original_Score', 'Best_Score', 'Improvement_Abs', 'Best_Iter', 'Stop_Reason']
    # 动态添加 Iter_0 到 Iter_Max 列
    score_cols = [f'Iter_{i}' for i in range(args.iters + 1)]
    fieldnames.extend(score_cols)
    
    # 判断使用批量模式还是单样本模式
    use_batch_mode = args.batch_size is not None and args.batch_size > 1
    
    if use_batch_mode:
        # ==========================================
        # 批量处理模式（带分布监控）
        # ==========================================
        print(f"\n🚀 启动批量推理模式 (Batch Size: {args.batch_size})")
        print(f"迭代次数: {args.iters}, 加噪时间步: {args.noise_t}")
        print("=" * 60)
        
        dataset = InferenceDataset(files)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        run_batch_inference(dataloader, unet, hovernet, scheduler, args)
        
        print("\n" + "=" * 60)
        print(f"✅ 批量推理完成！结果保存在: {args.output_dir}")
    else:
        # ==========================================
        # 单样本处理模式（逐个处理）
        # ==========================================
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
        print(f"✅ 推理完成！结果保存在: {args.output_dir}")
        print(f"✅ 数据表格已保存: {csv_path}")


if __name__ == '__main__':
    main()

