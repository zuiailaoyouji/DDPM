"""
训练主循环模块
核心改变：明确 clean_images 是如何变成 noisy_images 的，并修正输入拼接逻辑
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os
import argparse
import csv
import datetime
from tqdm import tqdm

from ddpm_dataset import NCTDataset
from unet_wrapper import create_model
from feedback_loss import FeedbackLoss
from ddpm_utils import load_hovernet, get_device, print_gpu_info, predict_x0_from_noise_shared
# 新增: 导入可视化模块
from visualization import save_training_progress_images
# 新增: 导入日志模块
from logger import ExperimentLogger
# 新增: 导入验证集模块
from validation import ValidationSet, create_val_dataloader


def train(
    tum_dir,
    norm_dir,
    hovernet,
    epochs=100,
    batch_size=8,
    lr=1e-4,
    device='cuda',
    save_dir='./checkpoints',
    feedback_weight_prob=0.05,
    feedback_weight_entropy=0.01,
    use_feedback_from_epoch=5,
    resume_path=None,
    logger=None,
    val_vis_dir=None,  # 验证集目录（用于固定可视化）
    accumulation_steps=1,  # 梯度累积步数，等效 batch = batch_size * accumulation_steps
    alpha_tumor=0.2559,   # 全局癌细胞静态 Focal Loss 权重（可用 compute_cell_priors.py 估计）
    alpha_normal=0.7441,   # 全局正常细胞静态 Focal Loss 权重
    tv_weight=0.0015,     # TV 全变分损失权重，压制高频噪点
):
    """
    训练 DDPM 模型

    Args:
        tum_dir: 肿瘤图像目录
        norm_dir: 正常图像目录
        hovernet: HoVer-Net 模型（用于反馈损失）
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 设备 ('cuda' 或 'cpu')
        save_dir: 模型保存目录
        feedback_weight_prob: 概率损失权重
        feedback_weight_entropy: 熵损失权重
        use_feedback_from_epoch: 从第几个 epoch 开始使用反馈损失
        resume_path: 恢复训练的检查点路径（可选）
        logger: 日志记录器（可选）
        val_vis_dir: 验证集目录（用于固定可视化），应包含 TUM 和 NORM 子目录（可选）
        accumulation_steps: 梯度累积步数，等效有效 batch = batch_size * accumulation_steps，不增加显存
        alpha_tumor: 癌细胞的全局静态 Focal Loss 权重（可用 compute_cell_priors.py 估计）
        alpha_normal: 正常细胞的全局静态 Focal Loss 权重
        tv_weight: TV 全变分损失权重，压制高频噪点
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 新增: 准备可视化目录
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 新增: 初始化 CSV 日志
    log_path = os.path.join(save_dir, 'training_log.csv')
    csv_mode = 'a' if os.path.exists(log_path) else 'w'
    with open(log_path, csv_mode, newline='') as f:
        writer = csv.writer(f)
        if csv_mode == 'w':
            # 写入表头：增加了 TUM_Conf, NORM_Conf, L1_Diff
            writer.writerow(['Epoch', 'Total_Loss', 'MSE', 'Prob_Loss', 'Entropy', 'TV_Loss', 'Tumor_Conf', 'Normal_Conf', 'Conf_Gap', 'L1_Diff'])
    
    # 1. 初始化模型和调度器
    print("初始化模型...")
    unet = create_model().to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 初始化反馈损失（仅在 hovernet 可用时）
    if hovernet is not None:
        feedback_criterion = FeedbackLoss(
            hovernet,
            scheduler,
            alpha_tumor=alpha_tumor,
            alpha_normal=alpha_normal,
        ).to(device)
        print("运行模式：带反馈损失的 DDPM 训练")
    else:
        feedback_criterion = None
        print("运行模式：基础 DDPM 训练（无反馈损失）")
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    
    # 如果需要断点续训加载代码
    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        print(f"加载检查点: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"从 Epoch {start_epoch} 继续训练")
    
    # 2. 初始化验证集管理器
    val_set = None
    if val_vis_dir is not None:
        val_set = ValidationSet(
            val_dir=val_vis_dir,
            scheduler=scheduler,
            device=device,
            sample_size=256,
            fixed_timestep=100
        )
        if not val_set.load():
            val_set = None
            print(f"⚠️ 警告: 验证集目录不存在或无法加载: {val_vis_dir}，将使用训练集图片进行可视化")

    # ================= 3. 加载定量验证集 DataLoader =================
    val_dataloader = create_val_dataloader(val_vis_dir, batch_size, device)
    best_val_conf_gap = -float('inf')

    # 4. 加载训练数据集
    print("加载数据集...")
    dataset = NCTDataset(tum_dir, norm_dir, oversample=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
        drop_last=True  # 确保批次大小一致
    )
    
    # 5. 训练循环
    print(f"开始训练，共 {epochs} 个 epoch...")
    global_step = 0
    for epoch in range(start_epoch, epochs):
        unet.train()
        
        # 初始化累加器
        acc = {
            'loss': 0.0, 'mse': 0.0, 'prob': 0.0, 'entropy': 0.0, 'tv': 0.0, 'l1': 0.0
        }
        tumor_conf_sum = 0.0
        tumor_conf_count = 0
        normal_conf_sum = 0.0
        normal_conf_count = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # 1. 准备数据
            clean_images, labels = batch  # clean_images 是 [0, 1] 范围
            clean_images = clean_images.to(device)
            labels = labels.to(device)
            bs = clean_images.shape[0]

            # 2. 采样随机噪声 (Gaussian Noise)
            noise = torch.randn_like(clean_images).to(device)
            
            # 3. 采样随机时间步 (Timesteps)
            timesteps = torch.randint(0, 1000, (bs,), device=device).long()
            
            # 4. 前向加噪过程 (Forward Diffusion)
            # ---> 这里就是加噪图像的来源 <---
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            
            # 5. 构建模型输入 (Concatenation)
            # 维度变成 6 通道: [Batch, 6, 256, 256]
            # clean_images 作为条件(Condition)，noisy_images 作为输入(Input)
            model_input = torch.cat([clean_images, noisy_images], dim=1)
            
            # 6. U-Net 预测噪声
            # 注意：虽然输入了 clean_images，但模型的目标是预测 noisy_images 里的 noise
            noise_pred = unet(model_input, timesteps).sample
            
            # 7. 计算 MSE Loss (基础重建能力)
            loss_mse = F.mse_loss(noise_pred, noise)
            
            # 关键修改: 总是计算增强图 x0_pred 以便监控 L1
            # 即使在阶段一 (无反馈)，我们也想看看模型生成的图和原图差多少
            with torch.no_grad():
                if feedback_criterion is not None:
                    x0_pred = feedback_criterion.predict_x0_from_noise(noisy_images, noise_pred, timesteps)
                else:
                    # 如果没有 feedback_criterion，手动计算 x0_pred（用于可视化）
                    device_img = noisy_images.device
                    dtype_img = noisy_images.dtype
                    # 先把 alphas_cumprod 移动到 device，然后再用 timesteps 去取值
                    # 这样可以避免设备不匹配的问题（timesteps 在 GPU，alphas_cumprod 在 CPU）
                    alpha_prod_t = scheduler.alphas_cumprod.to(device_img)[timesteps].to(dtype_img).view(-1, 1, 1, 1)
                    beta_prod_t = 1 - alpha_prod_t
                    x0_pred = (noisy_images - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)
                    x0_pred = torch.clamp(x0_pred, 0.0, 1.0)
                l1_diff = F.l1_loss(x0_pred, clean_images)
            
            # 8. 计算反馈 Loss（使用 Focal Loss 和原图引导）
            if epoch >= use_feedback_from_epoch and feedback_criterion is not None:
                loss_prob, loss_entropy, tumor_conf, normal_conf, loss_tv = feedback_criterion(
                    noisy_images, noise_pred, timesteps, clean_images
                )
            else:
                loss_prob = torch.tensor(0.0, device=device)
                loss_entropy = torch.tensor(0.0, device=device)
                tumor_conf = torch.tensor(-1.0, device=device)
                normal_conf = torch.tensor(-1.0, device=device)
                loss_tv = torch.tensor(0.0, device=device)

            # 9. 总 Loss（含 TV Loss，压制高频噪点）
            if epoch >= use_feedback_from_epoch and feedback_criterion is not None:
                loss_total = (
                    loss_mse
                    + feedback_weight_prob * loss_prob
                    + feedback_weight_entropy * loss_entropy
                    + tv_weight * loss_tv
                )
            else:
                loss_total = loss_mse

            # 10. 梯度累积：先缩放再反传，每 accumulation_steps 步或最后一个 batch 才更新参数
            loss_for_log = loss_total.item()
            loss_total = loss_total / accumulation_steps
            loss_total.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 累加记录（用未缩放的 loss，便于与「无累积」时数值一致）
            acc['loss'] += loss_for_log
            acc['mse'] += loss_mse.item()
            acc['prob'] += loss_prob.item() if isinstance(loss_prob, torch.Tensor) else loss_prob
            acc['entropy'] += loss_entropy.item() if isinstance(loss_entropy, torch.Tensor) else loss_entropy
            acc['tv'] += loss_tv.item() if isinstance(loss_tv, torch.Tensor) else loss_tv
            acc['l1'] += l1_diff.item()

            t_conf_val = tumor_conf.item() if isinstance(tumor_conf, torch.Tensor) else tumor_conf
            n_conf_val = normal_conf.item() if isinstance(normal_conf, torch.Tensor) else normal_conf
            if t_conf_val >= 0:
                tumor_conf_sum += t_conf_val
                tumor_conf_count += 1
            if n_conf_val >= 0:
                normal_conf_sum += n_conf_val
                normal_conf_count += 1

            # 实时记录指标到 TensorBoard (每隔几步记录一次，避免IO过高)
            if logger is not None and global_step % 10 == 0:
                metrics = {
                    "Total_Loss": loss_for_log,
                    "MSE_Loss": loss_mse.item(),
                    "Prob_Loss": loss_prob.item() if isinstance(loss_prob, torch.Tensor) else loss_prob,
                    "Entropy_Loss": loss_entropy.item() if isinstance(loss_entropy, torch.Tensor) else loss_entropy,
                    "TV_Loss": loss_tv.item() if isinstance(loss_tv, torch.Tensor) else loss_tv,
                    "L1_Diff": l1_diff.item(),
                }
                if t_conf_val >= 0:
                    metrics["Tumor_Conf"] = t_conf_val
                if n_conf_val >= 0:
                    metrics["Normal_Conf"] = n_conf_val
                if t_conf_val >= 0 and n_conf_val >= 0:
                    metrics["Conf_Gap"] = t_conf_val - n_conf_val
                logger.log_metrics(metrics, step=global_step, prefix="Train")
                logger.flush()
            
            # 可视化逻辑 (仅在每个 Epoch 的第一个 Batch)
            if batch_idx == 0:
                # 优先使用验证集进行可视化（更公平的对比）
                if val_set is not None and val_set.is_available():
                    # 使用验证集生成增强图像
                    val_enhanced = val_set.generate_enhanced_images(unet, feedback_criterion)
                    
                    if val_enhanced is not None:
                        # 保存验证集可视化图片（用于论文插图），并获取拼接好的 tensor
                        vis_grid = save_training_progress_images(
                            val_set.images, val_set.noisy_images, val_enhanced, 
                            epoch=epoch+1, save_dir=vis_dir, num_vis=8, return_tensor=True
                        )
                        
                        # 记录拼接好的对比图到 TensorBoard（替代原来的三张分开的图）
                        if logger is not None:
                            logger.log_images("Validation/Comparison", vis_grid, step=global_step, max_images=1)
                else:
                    # 如果没有验证集，使用训练集的当前 batch
                    vis_grid = save_training_progress_images(
                        clean_images, noisy_images, x0_pred, 
                        epoch=epoch+1, save_dir=vis_dir, return_tensor=True
                    )
                    
                    # 记录拼接好的对比图到 TensorBoard
                    if logger is not None:
                        logger.log_images("Train/Comparison", vis_grid, step=global_step, max_images=1)
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f"{loss_for_log:.4f}"})
            
            global_step += 1
        
        # Epoch 结束: 计算平均值
        avg_loss = acc['loss'] / len(dataloader)
        avg_mse = acc['mse'] / len(dataloader)
        avg_prob = acc['prob'] / len(dataloader)
        avg_entropy = acc['entropy'] / len(dataloader)
        avg_tv = acc['tv'] / len(dataloader)
        avg_l1 = acc['l1'] / len(dataloader)
        avg_tumor_conf = tumor_conf_sum / tumor_conf_count if tumor_conf_count > 0 else 0.0
        avg_normal_conf = normal_conf_sum / normal_conf_count if normal_conf_count > 0 else 0.0
        avg_conf_gap = avg_tumor_conf - avg_normal_conf

        print(f"\nEpoch {epoch+1}/{epochs} 完成:")
        print(f"  平均总损失: {avg_loss:.4f}")
        print(f"  平均MSE损失: {avg_mse:.4f}")
        print(f"  平均概率损失: {avg_prob:.4f}")
        print(f"  平均熵损失: {avg_entropy:.4f}")
        print(f"  癌细胞平均置信度 (应趋向1): {avg_tumor_conf:.4f}")
        print(f"  正常细胞平均置信度 (应趋向0): {avg_normal_conf:.4f}")
        print(f"  类别区分度 Gap (应趋向1): {avg_conf_gap:.4f}")
        print(f"  平均TV损失: {avg_tv:.4f}")
        print(f"  L1修改幅度: {avg_l1:.4f}")

        # 记录 Epoch 级别的指标到 TensorBoard
        if logger is not None:
            epoch_metrics = {
                "Total_Loss": avg_loss,
                "MSE_Loss": avg_mse,
                "Prob_Loss": avg_prob,
                "Entropy_Loss": avg_entropy,
                "TV_Loss": avg_tv,
                "Tumor_Conf": avg_tumor_conf,
                "Normal_Conf": avg_normal_conf,
                "Conf_Gap": avg_conf_gap,
                "L1_Diff": avg_l1,
            }
            logger.log_metrics(epoch_metrics, step=epoch+1, prefix="Epoch")
            logger.flush()

        # 写入 CSV 日志
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, avg_loss, avg_mse, avg_prob, avg_entropy, avg_tv,
                avg_tumor_conf, avg_normal_conf, avg_conf_gap, avg_l1
            ])

        # =========================================================
        # 定量验证：在验证集上评估 MSE / Focal / 置信度 Gap
        # =========================================================
        if val_dataloader is not None:
            unet.eval()
            print(f"\n--- 开始 Epoch {epoch+1} 验证集评估 ---")

            val_acc = {'mse': 0.0, 'prob': 0.0, 'entropy': 0.0, 'tv': 0.0}
            val_tumor_conf_sum = 0.0
            val_tumor_conf_count = 0
            val_normal_conf_sum = 0.0
            val_normal_conf_count = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_clean_images, val_labels = val_batch
                    val_clean_images = val_clean_images.to(device)
                    bs = val_clean_images.shape[0]

                    val_noise = torch.randn_like(val_clean_images).to(device)
                    val_timesteps = torch.randint(0, 1000, (bs,), device=device).long()
                    val_noisy_images = scheduler.add_noise(val_clean_images, val_noise, val_timesteps)

                    val_model_input = torch.cat([val_clean_images, val_noisy_images], dim=1)
                    val_noise_pred = unet(val_model_input, val_timesteps).sample

                    val_loss_mse = F.mse_loss(val_noise_pred, val_noise)
                    val_acc['mse'] += val_loss_mse.item()

                    if epoch >= use_feedback_from_epoch and feedback_criterion is not None:
                        val_loss_prob, val_loss_entropy, val_t_conf, val_n_conf, val_loss_tv = feedback_criterion(
                            val_noisy_images, val_noise_pred, val_timesteps, val_clean_images
                        )
                        val_acc['prob'] += val_loss_prob.item() if isinstance(val_loss_prob, torch.Tensor) else val_loss_prob
                        val_acc['entropy'] += val_loss_entropy.item() if isinstance(val_loss_entropy, torch.Tensor) else val_loss_entropy
                        val_acc['tv'] += val_loss_tv.item() if isinstance(val_loss_tv, torch.Tensor) else val_loss_tv

                        v_t_conf_val = val_t_conf.item() if isinstance(val_t_conf, torch.Tensor) else val_t_conf
                        v_n_conf_val = val_n_conf.item() if isinstance(val_n_conf, torch.Tensor) else val_n_conf
                        if v_t_conf_val >= 0:
                            val_tumor_conf_sum += v_t_conf_val
                            val_tumor_conf_count += 1
                        if v_n_conf_val >= 0:
                            val_normal_conf_sum += v_n_conf_val
                            val_normal_conf_count += 1

            val_steps = len(val_dataloader)
            val_avg_mse = val_acc['mse'] / val_steps
            print(f"  [Val] MSE Loss: {val_avg_mse:.4f}")

            if epoch >= use_feedback_from_epoch and feedback_criterion is not None:
                val_avg_prob = val_acc['prob'] / val_steps
                val_avg_entropy = val_acc['entropy'] / val_steps
                val_avg_tv = val_acc['tv'] / val_steps
                val_avg_tumor_conf = val_tumor_conf_sum / val_tumor_conf_count if val_tumor_conf_count > 0 else 0.0
                val_avg_normal_conf = val_normal_conf_sum / val_normal_conf_count if val_normal_conf_count > 0 else 0.0
                val_avg_conf_gap = val_avg_tumor_conf - val_avg_normal_conf

                print(f"  [Val] Focal Loss: {val_avg_prob:.4f} | Entropy Loss: {val_avg_entropy:.4f} | TV Loss: {val_avg_tv:.4f}")
                print(f"  [Val] Tumor Conf: {val_avg_tumor_conf:.4f}")
                print(f"  [Val] Normal Conf: {val_avg_normal_conf:.4f}")
                print(f"  [Val] Conf Gap: {val_avg_conf_gap:.4f}")

                if logger is not None:
                    val_metrics = {
                        "MSE_Loss": val_avg_mse,
                        "Prob_Loss": val_avg_prob,
                        "Entropy_Loss": val_avg_entropy,
                        "TV_Loss": val_avg_tv,
                        "Tumor_Conf": val_avg_tumor_conf,
                        "Normal_Conf": val_avg_normal_conf,
                        "Conf_Gap": val_avg_conf_gap,
                    }
                    logger.log_metrics(val_metrics, step=epoch + 1, prefix="Val")

                if val_avg_conf_gap > best_val_conf_gap:
                    best_val_conf_gap = val_avg_conf_gap
                    best_checkpoint_path = os.path.join(save_dir, 'best_unet_model.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': unet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_conf_gap': best_val_conf_gap,
                    }, best_checkpoint_path)
                    print(f"  🔥 发现更优模型！验证集 Gap 达到 {best_val_conf_gap:.4f}，已保存至 {best_checkpoint_path}")

            unet.train()

        # 保存模型
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'unet_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"模型已保存到: {checkpoint_path}")
    
    print("训练完成！")
    
    # 关闭日志记录器
    if logger is not None:
        logger.close()


def main():
    parser = argparse.ArgumentParser(description='训练 DDPM 模型')
    parser.add_argument('--tum_dir', type=str, required=True, help='肿瘤图像目录')
    parser.add_argument('--norm_dir', type=str, required=True, help='正常图像目录')
    parser.add_argument('--hovernet_path', type=str, default=None, help='HoVer-Net 模型路径（可选）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)，如果指定 gpu_id 则此参数将被忽略')
    parser.add_argument('--gpu_id', type=int, default=None, help='指定使用的 GPU ID（例如：0, 1, 2），默认为 None 自动选择')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--feedback_weight_prob', type=float, default=0.05, help='概率损失权重')
    parser.add_argument('--feedback_weight_entropy', type=float, default=0.01, help='熵损失权重')
    parser.add_argument('--use_feedback_from_epoch', type=int, default=5, help='从第几个epoch开始使用反馈损失')
    parser.add_argument('--resume_path', type=str, default=None, help='恢复训练的检查点路径（可选）')
    parser.add_argument('--no_tb', action='store_true', help='如果加上这个参数，则关闭 TensorBoard')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard 日志根目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称（用于区分不同次运行），默认为时间戳')
    parser.add_argument('--val_vis_dir', type=str, default=None, help='验证集目录（用于固定可视化），应包含 tumor 和 normal 子目录')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数，等效 batch = batch_size * accumulation_steps（默认 1 即不累积）')
    parser.add_argument('--alpha_tumor', type=float, default=0.2559, help='全局癌细胞静态 Focal Loss 权重（可用 compute_cell_priors.py 估计）')
    parser.add_argument('--alpha_normal', type=float, default=0.7441, help='全局正常细胞静态 Focal Loss 权重')
    parser.add_argument('--tv_weight', type=float, default=0.0015, help='TV 全变分损失权重，压制高频噪点（推荐 0.001～0.002）')

    args = parser.parse_args()
    
    # 打印 GPU 信息
    print_gpu_info()
    
    # 确定实验名称（如果没有指定，使用时间戳）
    import datetime
    if args.exp_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"DDPM_{timestamp}"
    
    # 初始化日志记录器
    logger = ExperimentLogger(
        use_tensorboard=not args.no_tb,
        log_dir=args.log_dir,
        experiment_name=args.exp_name
    )
    
    # 确定使用的设备
    if args.gpu_id is not None:
        # 如果指定了 gpu_id，使用指定的 GPU
        device = get_device(gpu_id=args.gpu_id)
    elif args.device is not None:
        # 如果指定了 device 字符串（兼容旧代码）
        device = args.device
        if device.startswith('cuda'):
            print(f"使用设备: {device}")
    else:
        # 默认自动检测
        device = get_device()
        print(f"自动检测设备: {device}")
    
    # 加载 HoVer-Net 模型（使用确定的设备）
    hovernet = None
    if args.hovernet_path:
        print(f"加载 HoVer-Net 模型: {args.hovernet_path}")
        hovernet = load_hovernet(args.hovernet_path, device=device)
        
        if hovernet is None:
            print("警告: HoVer-Net 模型未加载，反馈损失将无法使用")
    else:
        print("警告: 未提供 HoVer-Net 模型路径，反馈损失将无法使用")
    
    train(
        tum_dir=args.tum_dir,
        norm_dir=args.norm_dir,
        hovernet=hovernet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        feedback_weight_prob=args.feedback_weight_prob,
        feedback_weight_entropy=args.feedback_weight_entropy,
        use_feedback_from_epoch=args.use_feedback_from_epoch,
        resume_path=args.resume_path,
        logger=logger,
        val_vis_dir=args.val_vis_dir,
        accumulation_steps=args.accumulation_steps,
        alpha_tumor=args.alpha_tumor,
        alpha_normal=args.alpha_normal,
        tv_weight=args.tv_weight,
    )


if __name__ == '__main__':
    main()

