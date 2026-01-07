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
        
        # 获取 alpha_bar，shape: [B, 1, 1, 1]
        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device).to(dtype)
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
        计算反馈损失
        
        Args:
            x_t: 加噪图像 [B, C, H, W]
            noise_pred: U-Net 预测的噪声 [B, C, H, W]
            t: 时间步 [B]
            target_label: 目标标签 [B]，1 表示肿瘤，0 表示正常
        
        Returns:
            loss_prob: 概率损失（BCE）
            loss_entropy: 熵损失（分类器置信度）
        """
        # 1. 还原 x0
        x0_hat = self.predict_x0_from_noise(x_t, noise_pred, t)
        
        # 2. 裁剪中心 164x164 (匹配 HoVer-Net 输出)
        # 256 - 164 = 92,  92 / 2 = 46
        x0_crop = x0_hat[:, :, 46:210, 46:210]
        
        # 3. HoVer-Net 推理
        # 注意：虽然 HoVer-Net 参数被冻结，但我们需要保留计算图以便梯度能从 loss 传播到 x0_hat
        # 因此不使用 torch.no_grad()，而是依赖 requires_grad=False 的参数来阻止参数更新
        output = self.hovernet(x0_crop)
        
        probs = torch.softmax(output['tp'], dim=1)
        mask = torch.sigmoid(output['np'])  # 细胞核掩膜
        
        # 4. 计算 Loss (概率极大化 + 熵最小化)
        # 获取 Neoplastic (Index 1) 通道
        p_neo = probs[:, 1, :, :]
        
        # 掩膜引导的平均概率 (只看细胞核区域)
        # 需要保持梯度，所以不能完全 detach
        # 对每个样本计算平均概率
        avg_prob_per_sample = (p_neo * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        
        # 引导 Loss: 如果是 TUM(1) -> avg_prob 趋向 1; 如果是 NORM(0) -> avg_prob 趋向 0
        # 使用逐样本的 BCE，而不是批次平均
        target_float = target_label.float()
        loss_prob = F.binary_cross_entropy(avg_prob_per_sample, target_float)
        
        # 熵 Loss: 全局熵最小化 (让分类器不犹豫)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        loss_entropy = entropy.mean()
        
        return loss_prob, loss_entropy

