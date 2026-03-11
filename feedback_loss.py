"""
反馈损失模块
核心改变：实现从 ε 还原 x0 的数学公式，并确保梯度链不断
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedbackLoss(nn.Module):
    """
    反馈损失：利用 HoVer-Net 的分类结果来引导去噪过程；
    使用原图生成伪标签，并采用动态 Focal Loss。
    """
    def __init__(self, hovernet, noise_scheduler, gamma=2.0):
        """
        Args:
            hovernet: HoVer-Net 模型，需要冻结参数
            noise_scheduler: DDPM 噪声调度器，需要用它的 alpha 参数来还原 x0
            gamma: Focal Loss 的聚焦参数，默认 2.0
        """
        super().__init__()
        self.hovernet = hovernet
        self.scheduler = noise_scheduler
        self.gamma = gamma

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

    def forward(self, x_t, noise_pred, t, clean_images):
        """
        利用原图作为伪标签基准，计算动态 Focal Loss。

        Args:
            x_t: 加噪图像 [B, C, H, W]
            noise_pred: U-Net 预测的噪声 [B, C, H, W]
            t: 时间步 [B]
            clean_images: 干净原图 x0 [B, C, H, W]，范围 [0, 1]

        Returns:
            loss_prob: Focal 概率损失
            loss_entropy: 熵损失
            avg_tumor_conf: 原图癌细胞区域的 p_neo 平均值（无该类时为 -1.0）
            avg_normal_conf: 原图正常细胞区域的 p_neo 平均值（无该类时为 -1.0）
        """
        # 1. 获取模型当前预测的 x0_hat
        x0_hat = self.predict_x0_from_noise(x_t, noise_pred, t)
        hovernet_device = next(self.hovernet.parameters()).device

        x0_input = x0_hat.to(hovernet_device) if x0_hat.device != hovernet_device else x0_hat
        x0_input = x0_input * 255.0

        clean_input = clean_images.to(hovernet_device) if clean_images.device != hovernet_device else clean_images
        clean_input = clean_input * 255.0

        # 2. 生成原图的伪标签（无需梯度）并计算动态 alpha
        with torch.no_grad():
            clean_output = self.hovernet(clean_input)
            clean_probs = torch.softmax(clean_output['tp'], dim=1)
            mask = torch.softmax(clean_output['np'], dim=1)[:, 1, :, :]  # 细胞核掩膜
            clean_p_neo = clean_probs[:, 1, :, :]
            pseudo_target = (clean_p_neo >= 0.5).float()

            total_cells = mask.sum()
            if total_cells < 1:
                return (
                    torch.tensor(0.0, device=x_t.device),
                    torch.tensor(0.0, device=x_t.device),
                    torch.tensor(-1.0, device=x_t.device),
                    torch.tensor(-1.0, device=x_t.device),
                )

            tumor_cells = (pseudo_target * mask).sum()
            normal_cells = total_cells - tumor_cells
            # 反比权重：数量越少，权重越大
            alpha_tumor = (normal_cells / total_cells).item()
            alpha_normal = (tumor_cells / total_cells).item()

        # 3. 对预测的 x0_hat 进行 HoVer-Net 推理（需要梯度）
        output = self.hovernet(x0_input)
        probs = torch.softmax(output['tp'], dim=1)
        p_neo = probs[:, 1, :, :]

        # =========================================================
        # Focal Loss: FL = -alpha * (1 - pt)^gamma * log(pt)
        # =========================================================
        pt = torch.where(pseudo_target == 1.0, p_neo, 1.0 - p_neo)
        alpha_matrix = torch.where(
            pseudo_target == 1.0,
            torch.full_like(p_neo, alpha_tumor),
            torch.full_like(p_neo, alpha_normal),
        )
        pt = torch.clamp(pt, min=1e-6, max=1.0 - 1e-6)
        focal_weight = alpha_matrix * torch.pow(1.0 - pt, self.gamma)

        pixel_bce_loss = F.binary_cross_entropy(p_neo, pseudo_target, reduction='none')
        pixel_focal_loss = focal_weight * pixel_bce_loss
        loss_prob_per_sample = (pixel_focal_loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        loss_prob = loss_prob_per_sample.mean()

        # 4. 熵损失
        entropy_matrix = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        entropy_per_sample = (entropy_matrix * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        loss_entropy = entropy_per_sample.mean()

        # 5. 计算用于监控的分类别平均置信度 (不参与反向传播)
        with torch.no_grad():
            tumor_mask = pseudo_target * mask
            normal_mask = (1.0 - pseudo_target) * mask
            has_tumor = tumor_mask.sum() > 0
            has_normal = normal_mask.sum() > 0

            if has_tumor:
                avg_tumor_conf = (p_neo * tumor_mask).sum() / tumor_mask.sum()
            else:
                avg_tumor_conf = torch.tensor(-1.0, device=p_neo.device)
            if has_normal:
                avg_normal_conf = (p_neo * normal_mask).sum() / normal_mask.sum()
            else:
                avg_normal_conf = torch.tensor(-1.0, device=p_neo.device)

        return loss_prob, loss_entropy, avg_tumor_conf, avg_normal_conf

