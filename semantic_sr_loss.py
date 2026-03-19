"""
semantic_sr_loss.py
多类别核语义分布引导的 SR 损失函数。

设计要点
--------
1) SR 主损失保持不变：
   - L_noise / L_rec / L_grad / L_tv

2) 语义损失升级为“多类别分布一致性”：
   - clean 与 pred 都通过 HoVer-Net 得到 tp 分支概率分布（[B,C,H,W]）
   - 仅在高置信核区域监督：valid_mask = (nuc_mask > tau_nuc) & (tp_conf > tau_conf)
   - 语义主项：KL(P_clean || P_pred)
   - 语义辅助：CE(pred_logits, argmax(P_clean))
   - 可选置信度一致性：|conf_pred - conf_clean|

3) 为兼容现有训练日志字段，仍返回 l_sem / l_dir：
   - l_sem = lambda_sem_dist * KL + lambda_sem_cls * CE
   - l_dir = lambda_sem_conf * L_conf  （占位沿用旧字段名，不再是“方向 hinge”）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSRLoss(nn.Module):
    """
    语义引导型 SR DDPM 的完整损失。

    各项损失（都会单独返回以便记录）：
      L_noise : MSE(noise_pred, noise)                  — 扩散骨干损失
      L_rec   : L1(x0_hat, hr)                          — 像素保真度
      L_grad  : L1(∇x0_hat, ∇hr)                       — 边缘锐度
      L_sem   : KL + CE（多类别语义分布一致性）         — 语义主项
      L_dir   : 置信度一致性项（兼容旧日志字段名）       — 语义辅助
      L_tv    : 漏斗型相对 TV                            — 抗伪影
    """

    def __init__(
        self,
        hovernet,
        noise_scheduler,
        # 多类别语义监督掩膜参数
        tau_nuc: float = 0.5,        # nuclei 置信阈值（来自 np[:,1]）
        tau_conf: float = 0.7,       # tp top1 置信阈值（来自 max(tp_prob)）
        # 语义子项权重（语义总权重仍由 lambda_sem / lambda_dir 外层控制）
        lambda_sem_dist: float = 1.0,   # KL
        lambda_sem_cls:  float = 0.3,   # CE
        lambda_sem_conf: float = 0.1,   # |conf_pred-conf_clean|
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

        self.tau_nuc  = tau_nuc
        self.tau_conf = tau_conf
        self.lambda_sem_dist = lambda_sem_dist
        self.lambda_sem_cls  = lambda_sem_cls
        self.lambda_sem_conf = lambda_sem_conf

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
        l_sem_dist = hr.new_zeros(())
        l_sem_cls  = hr.new_zeros(())
        l_sem_conf = hr.new_zeros(())

        # 监控用标量（不回传梯度）
        mon = dict(sem_mae=-1.0, dir_acc=-1.0, p_hat_mean=-1.0, p_tgt_mean=-1.0)

        if len(valid_idx) > 0:
            x0_v  = x0_hat[valid_idx]
            hr_v  = hr[valid_idx]
            t_v   = t[valid_idx]
            npred_v = noise_pred[valid_idx]
            noisy_v = noisy_hr[valid_idx]

            # TV 损失（始终计算，用于抑制伪影）
            l_tv = self._tv_loss(x0_v, hr_v)

            if semantic_on:
                l_sem, l_dir, sem_terms, mon = self._semantic_losses(x0_v, hr_v)
                l_sem_dist = sem_terms['l_sem_dist']
                l_sem_cls  = sem_terms['l_sem_cls']
                l_sem_conf = sem_terms['l_sem_conf']

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
            l_sem_dist = l_sem_dist.detach(),
            l_sem_cls  = l_sem_cls.detach(),
            l_sem_conf = l_sem_conf.detach(),
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
        """
        img_01: [B,3,H,W] in [0,1]
        返回：
          tp_logits: [B,C,H,W]
          tp_prob  : [B,C,H,W]
          tp_label : [B,H,W]
          tp_conf  : [B,H,W]
          nuc_mask : [B,H,W]
        """
        dev = next(self.hovernet.parameters()).device
        out = self.hovernet(img_01.to(dev) * 255.0)
        tp_logits = out['tp']
        tp_prob   = torch.softmax(tp_logits, dim=1)
        tp_conf, tp_label = torch.max(tp_prob, dim=1)
        nuc_mask = torch.softmax(out['np'], dim=1)[:, 1, :, :]
        return dict(
            tp_logits=tp_logits,
            tp_prob=tp_prob,
            tp_label=tp_label,
            tp_conf=tp_conf,
            nuc_mask=nuc_mask,
        )

    def _semantic_losses(self, x0_hat, hr):
        """
        计算 L_sem 与 L_dir。
        L_sem: 使 p_pred ≈ p_clean；L_dir: 可选方向约束 hinge。
        """
        with torch.no_grad():
            clean = self._run_hovernet(hr)
            clean_prob = clean['tp_prob']     # [B,C,H,W]
            clean_lbl  = clean['tp_label']    # [B,H,W]
            clean_conf = clean['tp_conf']     # [B,H,W]
            nuc_mask   = clean['nuc_mask']    # [B,H,W]

            valid_mask = ((nuc_mask > self.tau_nuc) & (clean_conf > self.tau_conf)).float()  # [B,H,W]
            if valid_mask.sum() < 1:
                z = hr.new_zeros(())
                terms = dict(l_sem_dist=z, l_sem_cls=z, l_sem_conf=z)
                mon = dict(sem_mae=-1.0, dir_acc=-1.0, p_hat_mean=-1.0, p_tgt_mean=-1.0)
                return z, z, terms, mon

        pred = self._run_hovernet(x0_hat)
        pred_prob = pred['tp_prob']     # [B,C,H,W]
        pred_lbl  = pred['tp_label']    # [B,H,W]
        pred_conf = pred['tp_conf']     # [B,H,W]

        # KL(P_clean || P_pred)
        eps = 1e-8
        kl_map = F.kl_div(
            torch.log(pred_prob.clamp(min=eps)),
            clean_prob.detach(),
            reduction='none'
        ).sum(dim=1)  # [B,H,W]

        # CE(logits_pred, label_clean)
        ce_map = F.nll_loss(
            torch.log(pred_prob.clamp(min=eps)),
            clean_lbl.detach().long(),
            reduction='none'
        )  # [B,H,W]

        # 置信度一致性
        conf_map = (pred_conf - clean_conf.detach()).abs()  # [B,H,W]

        denom = valid_mask.sum().clamp(min=1.0)
        l_kl   = (kl_map * valid_mask).sum() / denom
        l_ce   = (ce_map * valid_mask).sum() / denom
        l_conf = (conf_map * valid_mask).sum() / denom

        l_sem = self.lambda_sem_dist * l_kl + self.lambda_sem_cls * l_ce
        l_dir = self.lambda_sem_conf * l_conf  # 兼容旧日志字段名

        with torch.no_grad():
            # 监控：多类别概率 MAE（按 mask）
            mae_map = (pred_prob - clean_prob).abs().mean(dim=1)  # [B,H,W]
            sem_mae = (mae_map * valid_mask).sum() / denom
            cls_acc = ((pred_lbl == clean_lbl).float() * valid_mask).sum() / denom
            p_hat_mean = (pred_conf * valid_mask).sum() / denom
            p_tgt_mean = (clean_conf * valid_mask).sum() / denom

        mon = dict(
            sem_mae=sem_mae.item(),
            dir_acc=cls_acc.item(),
            p_hat_mean=p_hat_mean.item(),
            p_tgt_mean=p_tgt_mean.item(),
        )
        terms = dict(l_sem_dist=l_kl, l_sem_cls=l_ce, l_sem_conf=l_conf)
        return l_sem, l_dir, terms, mon

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
