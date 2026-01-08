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
from ddpm_utils import load_hovernet, get_device, print_gpu_info
# 新增: 导入可视化模块
from visualization import save_training_progress_images
# 新增: 导入日志模块
from logger import ExperimentLogger


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
            writer.writerow(['Epoch', 'Total_Loss', 'MSE', 'Prob_Loss', 'Entropy', 'TUM_Conf', 'NORM_Conf', 'L1_Diff'])
    
    # 1. 初始化模型和调度器
    print("初始化模型...")
    unet = create_model().to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 初始化反馈损失（仅在 hovernet 可用时）
    if hovernet is not None:
        feedback_criterion = FeedbackLoss(hovernet, scheduler).to(device)
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
    
    # 2. 加载数据集
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
    
    # 3. 训练循环
    print(f"开始训练，共 {epochs} 个 epoch...")
    global_step = 0
    for epoch in range(start_epoch, epochs):
        unet.train()
        
        # 初始化累加器
        acc = {
            'loss': 0.0, 'mse': 0.0, 'prob': 0.0, 'entropy': 0.0,
            'tum_conf': 0.0, 'norm_conf': 0.0, 'l1': 0.0,
            'tum_count': 0, 'norm_count': 0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
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
                    alpha_prod_t = scheduler.alphas_cumprod[timesteps].to(device_img).to(dtype_img).view(-1, 1, 1, 1)
                    beta_prod_t = 1 - alpha_prod_t
                    x0_pred = (noisy_images - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)
                    x0_pred = torch.clamp(x0_pred, 0.0, 1.0)
                l1_diff = F.l1_loss(x0_pred, clean_images)
            
            # 8. 计算反馈 Loss (特征增强能力)
            # 建议：前 N 个 Epoch 可以先不加这个 Loss，或者权重设为 0
            if epoch >= use_feedback_from_epoch and feedback_criterion is not None:
                # 接收 4 个返回值
                loss_prob, loss_entropy, conf_tum, conf_norm = feedback_criterion(
                    noisy_images, noise_pred, timesteps, labels
                )
            else:
                # 占位符
                loss_prob = torch.tensor(0.0, device=device)
                loss_entropy = torch.tensor(0.0, device=device)
                conf_tum = torch.tensor(0.0, device=device)
                conf_norm = torch.tensor(0.0, device=device)
            
            # 9. 总 Loss
            # lambda 系数需要微调，建议从 0.01 开始
            if epoch >= use_feedback_from_epoch and feedback_criterion is not None:
                loss_total = loss_mse + feedback_weight_prob * loss_prob + feedback_weight_entropy * loss_entropy
            else:
                loss_total = loss_mse
            
            # 10. 反向传播
            optimizer.zero_grad()
            loss_total.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累加记录
            acc['loss'] += loss_total.item()
            acc['mse'] += loss_mse.item()
            acc['prob'] += loss_prob.item() if isinstance(loss_prob, torch.Tensor) else loss_prob
            acc['entropy'] += loss_entropy.item() if isinstance(loss_entropy, torch.Tensor) else loss_entropy
            acc['l1'] += l1_diff.item()
            
            if conf_tum > 0:
                acc['tum_conf'] += conf_tum.item()
                acc['tum_count'] += 1
            if conf_norm > 0:
                acc['norm_conf'] += conf_norm.item()
                acc['norm_count'] += 1
            
            # 实时记录指标到 TensorBoard (每隔几步记录一次，避免IO过高)
            if logger is not None and global_step % 10 == 0:
                metrics = {
                    "Total_Loss": loss_total.item(),
                    "MSE_Loss": loss_mse.item(),
                    "Prob_Loss": loss_prob.item() if isinstance(loss_prob, torch.Tensor) else loss_prob,
                    "Entropy_Loss": loss_entropy.item() if isinstance(loss_entropy, torch.Tensor) else loss_entropy,
                    "L1_Diff": l1_diff.item(),
                }
                # 添加置信度（如果有）
                if conf_tum > 0:
                    metrics["Tumor_Conf"] = conf_tum.item()
                if conf_norm > 0:
                    metrics["Normal_Conf"] = conf_norm.item()
                
                logger.log_metrics(metrics, step=global_step, prefix="Train")
                logger.flush()
            
            # 调用可视化模块 (仅在每个 Epoch 的第一个 Batch)
            if batch_idx == 0:
                save_training_progress_images(
                    clean_images, noisy_images, x0_pred, 
                    epoch=epoch+1, save_dir=vis_dir
                )
                
                # 记录图像到 TensorBoard
                if logger is not None:
                    logger.log_images("Images/Original", clean_images, step=global_step, max_images=4)
                    logger.log_images("Images/Noisy", noisy_images, step=global_step, max_images=4)
                    logger.log_images("Images/Enhanced", x0_pred, step=global_step, max_images=4)
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f"{loss_total.item():.4f}"})
            
            global_step += 1
        
        # Epoch 结束: 计算平均值并写入 CSV
        avg_loss = acc['loss'] / len(dataloader)
        avg_mse = acc['mse'] / len(dataloader)
        avg_prob = acc['prob'] / len(dataloader)
        avg_entropy = acc['entropy'] / len(dataloader)
        avg_l1 = acc['l1'] / len(dataloader)
        
        # 防止除以 0
        avg_tum_conf = acc['tum_conf'] / max(acc['tum_count'], 1)
        avg_norm_conf = acc['norm_conf'] / max(acc['norm_count'], 1)
        
        print(f"\nEpoch {epoch+1}/{epochs} 完成:")
        print(f"  平均总损失: {avg_loss:.4f}")
        print(f"  平均MSE损失: {avg_mse:.4f}")
        print(f"  平均概率损失: {avg_prob:.4f}")
        print(f"  平均熵损失: {avg_entropy:.4f}")
        print(f"  TUM置信度: {avg_tum_conf:.4f} (目标->1.0)")
        print(f"  NORM置信度: {avg_norm_conf:.4f} (目标->0.0)")
        print(f"  L1修改幅度: {avg_l1:.4f}")
        
        # 记录 Epoch 级别的指标到 TensorBoard
        if logger is not None:
            epoch_metrics = {
                "Total_Loss": avg_loss,
                "MSE_Loss": avg_mse,
                "Prob_Loss": avg_prob,
                "Entropy_Loss": avg_entropy,
                "Tumor_Conf": avg_tum_conf,
                "Normal_Conf": avg_norm_conf,
                "L1_Diff": avg_l1,
            }
            logger.log_metrics(epoch_metrics, step=epoch+1, prefix="Epoch")
            logger.flush()
        
        # 写入 CSV 日志
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, avg_loss, avg_mse, avg_prob, avg_entropy, 
                avg_tum_conf, avg_norm_conf, avg_l1
            ])
        
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
    )


if __name__ == '__main__':
    main()

