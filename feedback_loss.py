"""
反馈损失模块
核心改变：实现从 ε 还原 x0 的数学公式，并确保梯度链不断
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedbackLoss(nn.Module):
    """
    反馈损失：利用 HoVer-Net 的分类结果来引导去噪过程
    """
    def __init__(self, hovernet, noise_scheduler):
        """
        Args:
            hovernet: HoVer-Net 模型，需要冻结参数
            noise_scheduler: DDPM 噪声调度器，需要用它的 alpha 参数来还原 x0
        """
        super().__init__()
        self.hovernet = hovernet
        self.scheduler = noise_scheduler
        
        # 冻结 HoVer-Net 参数
        for param in self.hovernet.parameters():
            param.requires_grad = False
        self.hovernet.eval()

    def predict_x0_from_noise(self, x_t, noise_pred, t):
        """
        数学核心：利用 DDPM 公式逆向推导 x0
        x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
        所以: x0 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)
        
        Args:
            x_t: 加噪图像 [B, C, H, W]
            noise_pred: U-Net 预测的噪声 [B, C, H, W]
            t: 时间步 [B]
        
        Returns:
            pred_x0: 预测的原始图像 [B, C, H, W]，范围 [0, 1]
        """
        # 获取当前 timestep 对应的 alpha_bar (cumulative alphas)
        # 注意：需要根据 t 的 shape 提取对应的 alpha 值，并 reshape 成 [B, 1, 1, 1] 以便广播
        # t 可能是 [B] 形状，需要提取每个样本对应的 alpha_bar
        device = x_t.device
        dtype = x_t.dtype
        
        # 先把 alphas_cumprod 移动到 device，然后再用 t 去取值
        # 这样可以避免设备不匹配的问题（t 在 GPU，alphas_cumprod 在 CPU）
        alpha_prod_t = self.scheduler.alphas_cumprod.to(device)[t].to(dtype)
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 预测的 x0 (这就是我们要喂给 HoVer-Net 的图)
        # 这里的 x_t 是加噪图，noise_pred 是 U-Net 的输出
        pred_x0 = (x_t - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)
        
        # 裁剪到 [0, 1] 之间，防止数值溢出导致 NaN，但要保留梯度
        pred_x0 = torch.clamp(pred_x0, 0.0, 1.0)
        return pred_x0

    def forward(self, x_t, noise_pred, t, target_label):
        """
        计算反馈损失（密集监督：细胞级伪标签 + 图像级 NORM 一票否决）

        Args:
            x_t: 加噪图像 [B, C, H, W]
            noise_pred: U-Net 预测的噪声 [B, C, H, W]
            t: 时间步 [B]
            target_label: 目标标签 [B]，1 表示肿瘤，0 表示正常

        Returns:
            loss_prob: 概率损失（逐像素 BCE，按细胞核 mask 加权）
            loss_entropy: 熵损失（分类器置信度）
            conf_tum: TUM 样本的平均置信度
            conf_norm: NORM 样本的平均置信度
        """
        # 1. DDPM 逆推还原 x0_hat 并转换到 HoVer-Net 所在的设备及取值范围 (0-255)
        x0_hat = self.predict_x0_from_noise(x_t, noise_pred, t)
        hovernet_device = next(self.hovernet.parameters()).device
        x0_input = x0_hat.to(hovernet_device) if x0_hat.device != hovernet_device else x0_hat
        x0_input = x0_input * 255.0

        # 2. HoVer-Net 推理，获取概率分布与细胞核掩膜
        output = self.hovernet(x0_input)
        probs = torch.softmax(output['tp'], dim=1)
        mask = torch.softmax(output['np'], dim=1)[:, 1, :, :]  # [B, 164, 164] 细胞核掩膜
        p_neo = probs[:, 1, :, :]  # [B, 164, 164] 获取 Neoplastic 概率

        # ==================== 核心修改区块 ====================
        # 3. 生成细胞级伪标签 (以 0.5 为硬性分界线，充当特征放大器)
        # 注意：必须使用 .detach() 截断伪标签的梯度
        pseudo_target = (p_neo.detach() >= 0.5).float()

        # 4. 结合图像级真实标签 (防范 NORM 图像的假阳性)
        is_norm = (target_label == 0).view(-1, 1, 1).to(p_neo.device)
        # NORM 图像一票否决：全图伪标签强制归 0
        pseudo_target = torch.where(is_norm, torch.zeros_like(pseudo_target), pseudo_target)

        # 5. 逐像素/逐细胞计算 BCE 损失 (密集监督 Dense Supervision)
        pixel_bce_loss = F.binary_cross_entropy(p_neo, pseudo_target, reduction='none')

        # 6. 利用细胞核 Mask 进行加权求均值 (只关注有细胞核的像素)
        loss_prob_per_sample = (pixel_bce_loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        loss_prob = loss_prob_per_sample.mean()
        # ====================================================

        # ---------------------------------------------------------
        # (可选记录) 计算平均置信度用于 TensorBoard 监控 (不参与梯度)
        # ---------------------------------------------------------
        with torch.no_grad():
            avg_prob_per_sample = (p_neo * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
            is_tum_mask = (target_label == 1)
            is_norm_mask = (target_label == 0)
            conf_tum = avg_prob_per_sample[is_tum_mask].mean() if is_tum_mask.any() else torch.tensor(0.0, device=x_t.device)
            conf_norm = avg_prob_per_sample[is_norm_mask].mean() if is_norm_mask.any() else torch.tensor(0.0, device=x_t.device)

        # 7. 熵 Loss: 保持全局熵最小化作为辅助
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        loss_entropy = entropy.mean()

        return loss_prob, loss_entropy, conf_tum, conf_norm

