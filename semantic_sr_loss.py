"""
semantic_sr_loss.py
连续语义先验约束的 SR 损失函数。

完全替代旧的 feedback_loss.py / FeedbackLoss。
与旧的极化损失的主要设计差异
--------------------------------
旧方案（polarization）：
  - pseudo_target = 对 p_clean 在 0.5 处做硬阈值 →  {0, 1}
  - Focal-BCE 将 p_neo_hat 推向 {0, 1}
  - 指标：Conf_Gap（肿瘤置信度 − 正常置信度，越大越好）

新方案（SR 保真度 + 软语义引导）：
  - p_target = p_clean  （连续值，无需阈值）
  - L_sem = SmoothL1(p_neo_hat, p_target)，只在高置信细胞区域内计算
  - L_dir = hinge 损失，用于约束概率变化方向
  - L_rec = L1(x0_hat, hr)
  - L_grad = L1(∇x0_hat, ∇hr)
  - L_tv  = 漏斗型相对 TV（与旧代码一致）
  - 指标：PSNR / SSIM / Masked_Semantic_MAE

新的语义引导损失函数，不再使用 delta 和方向约束。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSRLoss(nn.Module):
    """
    语义引导型 SR DDPM 的完整损失。

    各项损失（都会单独返回以便记录）：
      L_noise : MSE(noise_pred, noise)              — 扩散骨干损失
      L_rec   : L1(x0_hat, hr)                      — 像素保真度
      L_grad  : L1(∇x0_hat, ∇hr)                   — 边缘锐度
      L_sem   : SmoothL1(p_neo_hat, p_target)·mask  — 语义软目标
      L_dir   : 针对方向错误像素的 hinge 损失       — 方向约束
      L_tv    : 漏斗型相对 TV                        — 抗伪影
    """

    def __init__(
        self,
        hovernet,
        noise_scheduler,
        # 语义软目标参数
        delta_t: float = 0.05,       # 肿瘤区域的偏移尺度
        delta_n: float = 0.05,       # 正常区域的偏移尺度
        tau_pos: float = 0.65,       # p_clean >= tau_pos → 高置信肿瘤
        tau_neg: float = 0.35,       # p_clean <= tau_neg → 高置信正常
        # 各项损失权重（可在 forward 中按 batch 覆盖）
        lambda_noise: float = 1.0,
        lambda_rec:   float = 1.0,
        lambda_grad:  float = 0.1,
        lambda_sem:   float = 0.05,
        lambda_dir:   float = 0.02,
        lambda_tv:    float = 0.001,
        # 扩散 / HoVer-Net 的门控相关参数
        t_max: int = 400,
        tv_margin_factor: float = 1.05,
        tv_leaky_alpha:   float = 0.10,
    ):
        super().__init__()
        self.hovernet  = hovernet
        self.scheduler = noise_scheduler

        self.delta_t  = delta_t
        self.delta_n  = delta_n
        self.tau_pos  = tau_pos
        self.tau_neg  = tau_neg

        self.lambda_noise = lambda_noise
        self.lambda_rec   = lambda_rec
        self.lambda_grad  = lambda_grad
        self.lambda_sem   = lambda_sem
        self.lambda_dir   = lambda_dir
        self.lambda_tv    = lambda_tv

        self.t_max            = t_max
        self.tv_margin_factor = tv_margin_factor
        self.tv_leaky_alpha   = tv_leaky_alpha

        # 冻结 HoVer-Net 参数
        for p in self.hovernet.parameters():
            p.requires_grad = False
        self.hovernet.eval()

    # ─────────────────────────────────────────────────────────────────────
    # 公有接口
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        noise_pred:   torch.Tensor,   # [B, 3, H, W]  U-Net output
        noise:        torch.Tensor,   # [B, 3, H, W]  ground-truth noise
        noisy_hr:     torch.Tensor,   # [B, 3, H, W]  x_t
        hr:           torch.Tensor,   # [B, 3, H, W]  clean HR in [0,1]
        t:            torch.Tensor,   # [B]            timesteps
        # 可选：每次调用覆盖权重（为 None 时使用 __init__ 中的默认值）
        lambda_sem:   float = None,
        lambda_dir:   float = None,
        semantic_on:  bool  = True,   # set False during warm-up stage 1
    ):
        """
        返回
        -------
        total_loss : 标量张量，包含梯度
        breakdown  : 字典，包含各个指标的标量张量（已 detach，便于记录）
        """
        lam_sem = lambda_sem if lambda_sem is not None else self.lambda_sem
        lam_dir = lambda_dir if lambda_dir is not None else self.lambda_dir

        # ── 1. 扩散噪声损失（始终开启）─────────────────────────────
        l_noise = F.mse_loss(noise_pred, noise)

        # ── 2. 重建 x0_hat ─────────────────────────────────────────
        x0_hat = self._predict_x0(noisy_hr, noise_pred, t)  # [B,3,H,W]

        # ── 3. 重建损失（始终开启）────────────────────────────────
        l_rec  = F.l1_loss(x0_hat, hr)
        l_grad = self._gradient_loss(x0_hat, hr)

        # ── 4. 语义相关损失（由 t_max 与 semantic_on 共同控制）────
        valid_idx = (t < self.t_max).nonzero(as_tuple=True)[0]

        l_sem = hr.new_zeros(())
        l_dir = hr.new_zeros(())
        l_tv  = hr.new_zeros(())

        # 监控用标量（不回传梯度）
        mon = dict(
            sem_mae    = -1.0,
            dir_acc    = -1.0,
            p_hat_mean = -1.0,
            p_tgt_mean = -1.0,
        )

        if len(valid_idx) > 0:
            x0_v  = x0_hat[valid_idx]
            hr_v  = hr[valid_idx]
            t_v   = t[valid_idx]
            npred_v = noise_pred[valid_idx]
            noisy_v = noisy_hr[valid_idx]

            # TV 损失（始终计算，用于抑制伪影）
            l_tv = self._tv_loss(x0_v, hr_v)

            if semantic_on:
                l_sem, l_dir, mon = self._semantic_losses(x0_v, hr_v)

        # ── 5. 加权求和得到总损失 ─────────────────────────────────
        total = (self.lambda_noise * l_noise
                 + self.lambda_rec   * l_rec
                 + self.lambda_grad  * l_grad
                 + lam_sem           * l_sem
                 + lam_dir           * l_dir
                 + self.lambda_tv    * l_tv)

        breakdown = dict(
            l_noise    = l_noise.detach(),
            l_rec      = l_rec.detach(),
            l_grad     = l_grad.detach(),
            l_sem      = l_sem.detach(),
            l_dir      = l_dir.detach(),
            l_tv       = l_tv.detach(),
            **{k: torch.tensor(v) for k, v in mon.items()},
        )

        return total, breakdown

    # ─────────────────────────────────────────────────────────────────────
    # 内部辅助函数
    # ─────────────────────────────────────────────────────────────────────

    def _predict_x0(self, x_t, noise_pred, t):
        dev   = x_t.device
        dtype = x_t.dtype
        alpha = self.scheduler.alphas_cumprod.to(dev)[t].to(dtype).view(-1,1,1,1)
        beta  = 1.0 - alpha
        x0    = (x_t - beta**0.5 * noise_pred) / (alpha**0.5 + 1e-8)
        return x0.clamp(0.0, 1.0)

    def _run_hovernet(self, img_01):
        """img_01: [B,3,H,W]，取值范围 [0,1] → (p_neo [B,H,W], cell_mask [B,H,W])"""
        dev = next(self.hovernet.parameters()).device
        out = self.hovernet(img_01.to(dev) * 255.0)
        probs     = torch.softmax(out['tp'], dim=1)
        p_neo     = probs[:, 1, :, :]
        cell_mask = torch.softmax(out['np'], dim=1)[:, 1, :, :]
        return p_neo, cell_mask

    def _build_soft_target(self, p_clean):
        """
        构建语义目标图。
        对于语义引导 SR，希望 p_pred ≈ p_clean，不再人为加减 delta。
        """
        p_target = p_clean.detach()  # 直接用原始 HoVer-Net 概率图作为目标
        return p_target

    def _semantic_losses(self, x0_hat, hr):
        """
        计算 L_sem 与 L_dir。
        L_sem: 使 p_pred ≈ p_clean；L_dir: 可选方向约束 hinge。
        """
        with torch.no_grad():
            p_clean, cell_mask = self._run_hovernet(hr)

            conf_mask = ((p_clean >= self.tau_pos) | (p_clean <= self.tau_neg)).float()
            eff_mask  = cell_mask * conf_mask          # [B,H,W]

            if eff_mask.sum() < 1:
                zero = hr.new_zeros(())
                return zero, zero, dict(
                    sem_mae=-1.0, dir_acc=-1.0,
                    p_hat_mean=-1.0, p_tgt_mean=-1.0)

            p_target = self._build_soft_target(p_clean)  # 纯 p_clean 目标，无 delta

        # 预测的 p_neo（梯度通过 x0_hat 传回 U-Net）
        p_hat, _ = self._run_hovernet(x0_hat)

        # 若 HoVer-Net 输出的空间尺寸更小，则对齐
        if p_hat.shape != eff_mask.shape:
            def _resize(t):
                return F.interpolate(t.unsqueeze(1),
                                     size=p_hat.shape[1:],
                                     mode='bilinear',
                                     align_corners=False).squeeze(1)
            eff_mask = _resize(eff_mask)
            p_target = _resize(p_target)
            p_clean  = _resize(p_clean)

        # SmoothL1 语义回归损失：p_pred ≈ p_clean
        raw_sem   = F.smooth_l1_loss(p_hat, p_target, reduction='none')
        denom_sem = eff_mask.sum().clamp(min=1.0)
        l_sem     = (raw_sem * eff_mask).sum() / denom_sem

        # 可选方向约束 hinge
        is_tumor = (p_clean > 0.5).float()
        dir_violation = (is_tumor       * F.relu(p_clean.detach() - p_hat)
                         + (1-is_tumor) * F.relu(p_hat - p_clean.detach()))
        l_dir         = (dir_violation * eff_mask).sum() / denom_sem

        # 监控指标
        with torch.no_grad():
            sem_mae    = ((p_hat - p_target).abs() * eff_mask).sum() / denom_sem
            is_t_f     = (p_clean > 0.5).float() * eff_mask
            is_n_f     = (p_clean <= 0.5).float() * eff_mask
            dir_corr_t = (p_hat >= p_clean).float() * is_t_f
            dir_corr_n = (p_hat <= p_clean).float() * is_n_f
            dir_acc    = (dir_corr_t + dir_corr_n).sum() / eff_mask.sum().clamp(1)
            p_hat_mean = (p_hat * eff_mask).sum() / denom_sem
            p_tgt_mean = (p_target * eff_mask).sum() / denom_sem

        mon = dict(
            sem_mae    = sem_mae.item(),
            dir_acc    = dir_acc.item(),
            p_hat_mean = p_hat_mean.item(),
            p_tgt_mean = p_tgt_mean.item(),
        )

        return l_sem, l_dir, mon

    def _gradient_loss(self, pred, target):
        """在有限差分（边缘）上的 L1 损失。"""
        def _grad(x):
            dh = x[:, :, 1:, :] - x[:, :, :-1, :]
            dw = x[:, :, :, 1:] - x[:, :, :, :-1]
            return dh, dw
        ph, pw = _grad(pred)
        th, tw = _grad(target)
        return (ph - th).abs().mean() + (pw - tw).abs().mean()

    def _tv_loss(self, x0_hat, hr):
        """漏斗型相对 TV 损失（与旧 feedback_loss.py 中的公式一致）。"""
        def _tv(img):
            dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
            dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
            return dh.mean(dim=(1,2,3)) + dw.mean(dim=(1,2,3))

        # 在 0-255 空间中计算，以保持与旧代码的尺度兼容
        dev = next(self.hovernet.parameters()).device
        tv_hat  = _tv(x0_hat.to(dev) * 255.0)
        with torch.no_grad():
            tv_ref  = _tv(hr.to(dev)     * 255.0)

        margin = tv_ref * self.tv_margin_factor
        alpha  = self.tv_leaky_alpha
        loss_batch = torch.where(
            tv_hat <= margin,
            alpha * tv_hat,
            alpha * margin + (tv_hat - margin),
        )
        return loss_batch.mean()
