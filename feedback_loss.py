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

    def forward(self, x_t, noise_pred, t):
        """
        计算纯自监督反馈损失（无 target_label，不区分 TUM/NORM）

        Args:
            x_t: 加噪图像 [B, C, H, W]
            noise_pred: U-Net 预测的噪声 [B, C, H, W]
            t: 时间步 [B]

        Returns:
            loss_prob: 自监督概率损失（锐化损失）
            loss_entropy: 自监督熵损失
            avg_conf: 当前 Batch 细胞的全局平均癌变置信度（用于日志记录）
        """
        # 1. 还原 x0_hat 并转换到 HoVer-Net 设备及 0-255 范围
        x0_hat = self.predict_x0_from_noise(x_t, noise_pred, t)
        hovernet_device = next(self.hovernet.parameters()).device
        x0_input = x0_hat.to(hovernet_device) if x0_hat.device != hovernet_device else x0_hat
        x0_input = x0_input * 255.0

        # 2. HoVer-Net 推理
        output = self.hovernet(x0_input)
        probs = torch.softmax(output['tp'], dim=1)
        mask = torch.softmax(output['np'], dim=1)[:, 1, :, :]  # 细胞核掩膜

        # 获取 Neoplastic (癌变) 概率
        p_neo = probs[:, 1, :, :]  # [B, 164, 164]

        # =========================================================
        # 纯自监督核心：直接生成伪标签，无视图像真实归属
        # =========================================================
        # 以 0.5 为分界线，直接定极点。必须 detach 截断梯度！
        pseudo_target = (p_neo.detach() >= 0.5).float()

        # 3. 计算自监督概率损失 (矩阵级掩膜计算)
        pixel_bce_loss = F.binary_cross_entropy(p_neo, pseudo_target, reduction='none')
        # 只在有细胞核的地方计算平均损失
        loss_prob_per_sample = (pixel_bce_loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        loss_prob = loss_prob_per_sample.mean()

        # 4. 计算自监督熵损失 (矩阵级掩膜计算)
        entropy_matrix = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        # 只要求网络对细胞核区域的判断确信，不干涉背景的熵
        entropy_per_sample = (entropy_matrix * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        loss_entropy = entropy_per_sample.mean()

        # 5. 计算用于监控的全局平均置信度 (不参与反向传播)
        with torch.no_grad():
            avg_conf_per_sample = (p_neo * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
            avg_conf = avg_conf_per_sample.mean()

        return loss_prob, loss_entropy, avg_conf

