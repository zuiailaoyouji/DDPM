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
import csv  # 新增
from tqdm import tqdm

from dataset import NCTDataset
from unet_wrapper import create_model
from feedback_loss import FeedbackLoss
from utils import load_hovernet, get_device
# 新增: 导入可视化模块
from visualization import save_training_progress_images


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
            
            # 调用可视化模块 (仅在每个 Epoch 的第一个 Batch)
            if batch_idx == 0:
                save_training_progress_images(
                    clean_images, noisy_images, x0_pred, 
                    epoch=epoch+1, save_dir=vis_dir
                )
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f"{loss_total.item():.4f}"})
        
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


def main():
    parser = argparse.ArgumentParser(description='训练 DDPM 模型')
    parser.add_argument('--tum_dir', type=str, required=True, help='肿瘤图像目录')
    parser.add_argument('--norm_dir', type=str, required=True, help='正常图像目录')
    parser.add_argument('--hovernet_path', type=str, default=None, help='HoVer-Net 模型路径（可选）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--feedback_weight_prob', type=float, default=0.05, help='概率损失权重')
    parser.add_argument('--feedback_weight_entropy', type=float, default=0.01, help='熵损失权重')
    parser.add_argument('--use_feedback_from_epoch', type=int, default=5, help='从第几个epoch开始使用反馈损失')
    parser.add_argument('--resume_path', type=str, default=None, help='恢复训练的检查点路径（可选）')
    
    args = parser.parse_args()
    
    # 加载 HoVer-Net 模型
    hovernet = None
    if args.hovernet_path:
        print(f"加载 HoVer-Net 模型: {args.hovernet_path}")
        hovernet = load_hovernet(args.hovernet_path, device=args.device)
        
        if hovernet is None:
            print("警告: HoVer-Net 模型未加载，反馈损失将无法使用")
            print("请在 utils.py 中实现 load_hovernet 函数")
    else:
        print("警告: 未提供 HoVer-Net 模型路径，反馈损失将无法使用")
    
    train(
        tum_dir=args.tum_dir,
        norm_dir=args.norm_dir,
        hovernet=hovernet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        feedback_weight_prob=args.feedback_weight_prob,
        feedback_weight_entropy=args.feedback_weight_entropy,
        use_feedback_from_epoch=args.use_feedback_from_epoch,
        resume_path=args.resume_path,
    )


if __name__ == '__main__':
    main()

