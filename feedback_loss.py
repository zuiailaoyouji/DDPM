"""
反馈损失模块
核心改变：引入全局静态 Alpha 权重解决单类切片梯度坍塌，引入定向熵损失压制异常反弹
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedbackLoss(nn.Module):
    """
    反馈损失：利用 HoVer-Net 的分类结果来引导去噪过程；
    使用原图生成伪标签，并采用静态 Focal Loss 和定向熵损失。
    """
    def __init__(self, hovernet, noise_scheduler, gamma=2.0, alpha_tumor=0.2559, alpha_normal=0.7441):
        """
        Args:
            hovernet: HoVer-Net 模型，需要冻结参数
            noise_scheduler: DDPM 噪声调度器
            gamma: Focal Loss 的聚焦参数，默认 2.0
            alpha_tumor: 癌细胞的全局静态权重 (推荐值: 0.2559)
            alpha_normal: 正常细胞的全局静态权重 (推荐值: 0.7441)
        """
        super().__init__()
        self.hovernet = hovernet
        self.scheduler = noise_scheduler
        self.gamma = gamma
        self.alpha_tumor = alpha_tumor
        self.alpha_normal = alpha_normal

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
            loss_tv: 全变分损失（压制高频噪点）
        """
        # =========================================================
        # 0. 核心修正：时间步截断 (Timestep Cutoff)
        # 只在噪声较小 (t < t_max) 时，pred_x0 才具有细胞语义意义，
        # HoVer-Net 才能给出正确指导，防止高 t 阶段的梯度爆炸。
        # =========================================================
        t_max = 400
        valid_idx = (t < t_max).nonzero(as_tuple=True)[0]

        # 如果当前 batch 里所有的图加噪都太深，直接返回 0，跳过反馈计算
        if len(valid_idx) == 0:
            zero = torch.tensor(0.0, device=x_t.device)
            return zero, zero, torch.tensor(-1.0, device=x_t.device), torch.tensor(-1.0, device=x_t.device), zero

        # 仅截取有效的样本继续前向传播 (大幅节省 HoVer-Net 推理算力)
        x_t = x_t[valid_idx]
        noise_pred = noise_pred[valid_idx]
        t = t[valid_idx]
        clean_images = clean_images[valid_idx]

        # 1. 获取模型当前预测的 x0_hat
        x0_hat = self.predict_x0_from_noise(x_t, noise_pred, t)
        hovernet_device = next(self.hovernet.parameters()).device

        x0_input = x0_hat.to(hovernet_device) if x0_hat.device != hovernet_device else x0_hat
        x0_input = x0_input * 255.0

        clean_input = clean_images.to(hovernet_device) if clean_images.device != hovernet_device else clean_images
        clean_input = clean_input * 255.0

        # 2. 生成原图的伪标签（无需梯度）
        with torch.no_grad():
            clean_output = self.hovernet(clean_input)
            clean_probs = torch.softmax(clean_output['tp'], dim=1)
            mask = torch.softmax(clean_output['np'], dim=1)[:, 1, :, :]  # 细胞核掩膜
            clean_p_neo = clean_probs[:, 1, :, :]
            pseudo_target = (clean_p_neo >= 0.5).float()

            total_cells = mask.sum()
            # 如果整张图没有任何细胞，直接返回 0
            if total_cells < 1:
                zero = torch.tensor(0.0, device=x_t.device)
                return zero, zero, torch.tensor(-1.0, device=x_t.device), torch.tensor(-1.0, device=x_t.device), zero

            # 注意：已删除原有的动态 alpha_tumor 和 alpha_normal 计算逻辑

        # 3. 对预测的 x0_hat 进行 HoVer-Net 推理（需要梯度）
        output = self.hovernet(x0_input)
        probs = torch.softmax(output['tp'], dim=1)
        p_neo = probs[:, 1, :, :]

        # =========================================================
        # Focal Loss: FL = -alpha * (1 - pt)^gamma * log(pt)
        # =========================================================
        pt = torch.where(pseudo_target == 1.0, p_neo, 1.0 - p_neo)

        # 核心修改：直接使用初始化时传入的静态参数
        alpha_matrix = torch.where(
            pseudo_target == 1.0,
            torch.full_like(p_neo, self.alpha_tumor),
            torch.full_like(p_neo, self.alpha_normal),
        )

        pt = torch.clamp(pt, min=1e-6, max=1.0 - 1e-6)
        focal_weight = alpha_matrix * torch.pow(1.0 - pt, self.gamma)

        pixel_bce_loss = F.binary_cross_entropy(p_neo, pseudo_target, reduction='none')
        pixel_focal_loss = focal_weight * pixel_bce_loss
        loss_prob_per_sample = (pixel_focal_loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)
        loss_prob = loss_prob_per_sample.mean()

        # =========================================================
        # 4. 熵损失 (修改为定向熵损失 Masked Entropy Loss)
        # =========================================================
        entropy_matrix = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # 找出当前预测方向正确的细胞核像素
        # (预测为阳性且标签为阳性) 或 (预测为阴性且标签为阴性)
        with torch.no_grad():
            correct_direction = ((p_neo >= 0.5) == (pseudo_target == 1.0)).float()

        # 将原有的细胞核 mask 乘以正确方向的 mask
        effective_mask = mask * correct_direction

        # 仅对方向正确的区域计算熵损失
        entropy_per_sample = (entropy_matrix * effective_mask).sum(dim=(1, 2)) / (effective_mask.sum(dim=(1, 2)) + 1e-6)
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

        # =========================================================
        # 6. 相对 TV Loss (Relative Total Variation)
        # 允许保留自然生物学纹理，仅惩罚超出原图复杂度的高频对抗噪点
        # =========================================================
        def calc_tv(img):
            # 计算每张图片的 TV 值
            diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            # 注意：这里在空间维度上求均值，保留 batch 维度
            return diff_h.mean(dim=(1, 2, 3)) + diff_w.mean(dim=(1, 2, 3))

        # 计算生成图像的 TV
        tv_hat = calc_tv(x0_input)

        # 计算原始干净图像的自然 TV (不参与反向传播)
        with torch.no_grad():
            tv_clean = calc_tv(clean_input)

        # 核心逻辑：引入相对松弛边界 (Margin)
        # 允许生成的图像 TV 值比原图高出 5% (1.05)，给特征强化留出轻微空间
        # 只有超出这个自然边界的“作弊噪点”才会被狠狠惩罚
        loss_tv = F.relu(tv_hat - tv_clean * 1.05).mean()

        return loss_prob, loss_entropy, avg_tumor_conf, avg_normal_conf, loss_tv

