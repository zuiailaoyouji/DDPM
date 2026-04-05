"""
semantic_sr_loss.py
多类别核语义分布引导的 SR 损失函数。

v3 升级：CellViT 双侧推理语义监督
────────────────────────────────────
clean 侧：CellViT(HR) 实时推理得到伪标签（no_grad）
pred  侧：CellViT(x0_hat) 实时推理，CE 监督向 HR 伪标签靠近

教师与目标统一在 CellViT 预测空间，梯度方向有意义。
valid_mask 仍由 gt_nuc_mask > tau_nuc 确定监督区域。

类别映射（与 PanNuke GT 一致）
────────────────────────────────
  class 0: 背景
  class 1: Neoplastic
  class 2: Inflammatory
  class 3: Connective
  class 4: Dead
  class 5: Epithelial

损失各项
────────
  L_noise : MSE(noise_pred, noise)              — 扩散骨干损失
  L_rec   : L1(x0_hat, hr)                     — 像素保真度
  L_grad  : L1(∇x0_hat, ∇hr)                  — 边缘锐度
  L_sem   : CE(CellViT(x0_hat), CellViT(HR))   — 语义类别监督
  L_tv    : 漏斗型相对 TV                        — 抗伪影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# GT mask → sem_tensor 工具（架构注入用，保持不变）
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_sem_tensor(
    gt_label_map: torch.Tensor,   # [B, H, W]  int64
    gt_nuc_mask:  torch.Tensor,   # [B, H, W]  float32
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    将 GT 字段拼接为 sem_tensor [B, 7, H, W]，用于 SPMUNet 架构注入。
    通道：[gt_tp_onehot(6), gt_nuc_mask(1)]
    """
    if device is None:
        device = gt_label_map.device
    onehot = F.one_hot(gt_label_map.to(device).long(), num_classes=6) \
              .permute(0, 3, 1, 2).float()           # [B, 6, H, W]
    nuc    = gt_nuc_mask.unsqueeze(1).to(device)     # [B, 1, H, W]
    return torch.cat([onehot, nuc], dim=1)           # [B, 7, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# CellViT 推理封装
# ─────────────────────────────────────────────────────────────────────────────

def run_cellvit(
    cellvit:  nn.Module,
    img_01:   torch.Tensor,        # [B, 3, H, W] in [0, 1]
) -> Dict[str, torch.Tensor]:
    """
    对输入图像运行 CellViT，返回：
      nuclei_type_prob  : [B, 6, H, W]  softmax 概率
      nuclei_type_label : [B, H, W]     argmax 类别索引
      nuclei_binary_prob: [B, 2, H, W]  核/非核 softmax 概率
    """
    # CellViT 输入范围 [0, 1]，内部不需要 ×255
    out = cellvit(img_01)

    type_prob  = F.softmax(out['nuclei_type_map'], dim=1)   # [B, 6, H, W]
    type_label = type_prob.argmax(dim=1)                    # [B, H, W]
    bin_prob   = F.softmax(out['nuclei_binary_map'], dim=1) # [B, 2, H, W]

    return dict(
        nuclei_type_prob  = type_prob,
        nuclei_type_label = type_label,
        nuclei_binary_prob = bin_prob,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主损失模块
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSRLoss(nn.Module):
    """
    语义引导型 SR DDPM 的完整损失（CellViT 双侧推理版）。

    关键设计
    ────────
    1. clean 侧：CellViT(HR) no_grad 推理，得到伪标签作为 CE 目标。
       教师和目标在同一预测空间，梯度方向有意义。
    2. pred  侧：CellViT(x0_hat) 实时推理，CE 损失驱动其向 CellViT(HR) 靠近。
    3. valid_mask：gt_nuc_mask > tau_nuc，框定有细胞核的监督区域。
    4. 监控指标：pred 预测与 HR 伪标签的一致率（dir_acc）和概率 MAE（sem_mae）。
    """

    def __init__(
        self,
        cellvit,
        noise_scheduler,
        # 核掩膜阈值
        tau_nuc:  float = 0.5,
        # 语义子项权重
        lambda_sem_cls: float = 0.3,
        # 各项总体权重
        lambda_noise: float = 1.0,
        lambda_rec:   float = 1.0,
        lambda_grad:  float = 0.1,
        lambda_sem:   float = 0.05,
        lambda_tv:    float = 0.001,
        # TV 与时间步配置
        t_max: int = 150,
        tv_margin_factor: float = 1.05,
        tv_leaky_alpha:   float = 0.10,
    ):
        super().__init__()
        self.cellvit   = cellvit
        self.scheduler = noise_scheduler

        self.tau_nuc        = tau_nuc
        self.lambda_sem_cls = lambda_sem_cls

        self.lambda_noise = lambda_noise
        self.lambda_rec   = lambda_rec
        self.lambda_grad  = lambda_grad
        self.lambda_sem   = lambda_sem
        self.lambda_tv    = lambda_tv

        self.t_max             = t_max
        self.tv_margin_factor  = tv_margin_factor
        self.tv_leaky_alpha    = tv_leaky_alpha

        # 冻结 CellViT，仅用于推理
        if self.cellvit is not None:
            for p in self.cellvit.parameters():
                p.requires_grad = False
            self.cellvit.eval()

    # ─────────────────────────────────────────────────────────────────────
    # 公有接口
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        noise_pred:    torch.Tensor,        # [B, 3, H, W]
        noise:         torch.Tensor,        # [B, 3, H, W]
        noisy_hr:      torch.Tensor,        # [B, 3, H, W]  x_t
        hr:            torch.Tensor,        # [B, 3, H, W]  clean HR [0,1]
        t:             torch.Tensor,        # [B]
        gt_nuc_mask:   torch.Tensor,        # [B, H, W]  GT 核掩膜
        gt_label_map:  torch.Tensor,        # [B, H, W]  GT 类别（保留用于 valid_mask）
        lambda_sem:    Optional[float] = None,
        semantic_on:   bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        lam_sem = lambda_sem if lambda_sem is not None else self.lambda_sem

        # ── 1. 扩散噪声损失 ────────────────────────────────────────
        l_noise = F.mse_loss(noise_pred, noise)

        # ── 2. 重建 x0_hat ─────────────────────────────────────────
        x0_hat = self._predict_x0(noisy_hr, noise_pred, t)

        # ── 3. 像素重建损失 ────────────────────────────────────────
        l_rec  = F.l1_loss(x0_hat, hr)
        l_grad = self._gradient_loss(x0_hat, hr)

        # ── 4. 语义损失（t_max 与 semantic_on 联合控制）────────────
        valid_idx = (t < self.t_max).nonzero(as_tuple=True)[0]

        l_sem     = hr.new_zeros(())
        l_tv      = hr.new_zeros(())
        l_sem_cls = hr.new_zeros(())
        mon = dict(sem_mae=-1.0, dir_acc=-1.0)

        if len(valid_idx) > 0:
            x0_v  = x0_hat[valid_idx]
            hr_v  = hr[valid_idx]
            nuc_v = gt_nuc_mask[valid_idx].to(hr.device)

            # TV 损失（始终开启）
            l_tv = self._tv_loss(x0_v, hr_v)

            if semantic_on and self.cellvit is not None:
                l_sem, l_sem_cls, mon = self._semantic_losses_cellvit(
                    x0_v, hr_v, nuc_v,
                )

        # ── 5. 加权求和 ────────────────────────────────────────────
        total = (
            self.lambda_noise * l_noise
            + self.lambda_rec  * l_rec
            + self.lambda_grad * l_grad
            + lam_sem          * l_sem
            + self.lambda_tv   * l_tv
        )

        breakdown = dict(
            l_noise   = l_noise.detach(),
            l_rec     = l_rec.detach(),
            l_grad    = l_grad.detach(),
            l_sem     = l_sem.detach(),
            l_tv      = l_tv.detach(),
            l_sem_cls = l_sem_cls.detach(),
            **{k: torch.tensor(v) for k, v in mon.items()},
        )
        return total, breakdown

    # ─────────────────────────────────────────────────────────────────────
    # CellViT 双侧语义损失
    # ─────────────────────────────────────────────────────────────────────

    def _semantic_losses_cellvit(
        self,
        x0_hat:      torch.Tensor,   # [N, 3, H, W]  pred 侧
        hr:          torch.Tensor,   # [N, 3, H, W]  clean HR
        gt_nuc_mask: torch.Tensor,   # [N, H, W]     GT 核掩膜
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        clean 侧：CellViT(HR) no_grad → 伪标签
        pred  侧：CellViT(x0_hat) → CE 损失

        valid_mask = gt_nuc_mask > tau_nuc
        损失：CE(CellViT(x0_hat)_prob, CellViT(HR)_label)
        """
        N, _, H, W = x0_hat.shape

        # GT 核掩膜尺寸对齐
        if gt_nuc_mask.shape[-2:] != (H, W):
            gt_nuc_mask = F.interpolate(
                gt_nuc_mask.unsqueeze(1), size=(H, W), mode="nearest"
            ).squeeze(1)

        valid_mask = (gt_nuc_mask > self.tau_nuc).float()   # [N, H, W]
        denom      = valid_mask.sum().clamp(min=1.0)

        if valid_mask.sum() < 1:
            z = x0_hat.new_zeros(())
            return z, z, dict(sem_mae=-1.0, dir_acc=-1.0)

        # ── clean 侧：CellViT(HR) → 伪标签（no_grad）────────────────
        with torch.no_grad():
            hr_out    = run_cellvit(self.cellvit, hr)
            hr_label  = hr_out['nuclei_type_label']   # [N, H, W]  伪标签
            hr_prob   = hr_out['nuclei_type_prob']    # [N, 6, H, W]

        # ── pred 侧：CellViT(x0_hat)（需要梯度）──────────────────────
        pred_out  = run_cellvit(self.cellvit, x0_hat)
        pred_prob = pred_out['nuclei_type_prob']      # [N, 6, H, W]
        pred_lbl  = pred_out['nuclei_type_label']     # [N, H, W]

        # ── CE(pred_prob, hr_label) 在 valid_mask 内平均 ─────────────
        eps       = 1e-8
        ce_map    = F.nll_loss(
            torch.log(pred_prob.clamp(min=eps)),
            hr_label.long(),
            reduction="none",
        )                                              # [N, H, W]

        l_sem_cls = (ce_map * valid_mask).sum() / denom
        l_sem     = self.lambda_sem_cls * l_sem_cls

        # ── 监控指标（不回传梯度）────────────────────────────────────
        with torch.no_grad():
            # pred 和 HR 伪标签的一致率
            cls_acc = ((pred_lbl == hr_label).float()
                       * valid_mask).sum() / denom
            # pred_prob 和 hr_prob 的 MAE
            mae_map = (pred_prob - hr_prob).abs().mean(dim=1)
            sem_mae = (mae_map * valid_mask).sum() / denom

        mon = dict(sem_mae=sem_mae.item(), dir_acc=cls_acc.item())
        return l_sem, l_sem_cls, mon

    # ─────────────────────────────────────────────────────────────────────
    # 共用辅助函数
    # ─────────────────────────────────────────────────────────────────────

    def _predict_x0(
        self,
        x_t:        torch.Tensor,
        noise_pred: torch.Tensor,
        t:          torch.Tensor,
    ) -> torch.Tensor:
        dev   = x_t.device
        dtype = x_t.dtype
        alpha = self.scheduler.alphas_cumprod.to(dev)[t].to(dtype).view(-1, 1, 1, 1)
        beta  = 1.0 - alpha
        x0    = (x_t - beta ** 0.5 * noise_pred) / (alpha ** 0.5 + 1e-8)
        return x0.clamp(0.0, 1.0)

    def _gradient_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        def _grad(x):
            dh = x[:, :, 1:, :] - x[:, :, :-1, :]
            dw = x[:, :, :, 1:] - x[:, :, :, :-1]
            return dh, dw
        ph, pw = _grad(pred)
        th, tw = _grad(target)
        return (ph - th).abs().mean() + (pw - tw).abs().mean()

    def _tv_loss(
        self,
        x0_hat: torch.Tensor,
        hr:     torch.Tensor,
    ) -> torch.Tensor:
        def _tv(img: torch.Tensor) -> torch.Tensor:
            dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
            dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
            return dh.mean(dim=(1, 2, 3)) + dw.mean(dim=(1, 2, 3))

        dev    = x0_hat.device
        tv_hat = _tv(x0_hat * 255.0)
        with torch.no_grad():
            tv_ref = _tv(hr.to(dev) * 255.0)

        margin     = tv_ref * self.tv_margin_factor
        alpha      = self.tv_leaky_alpha
        loss_batch = torch.where(
            tv_hat <= margin,
            alpha * tv_hat,
            alpha * margin + (tv_hat - margin),
        )
        return loss_batch.mean()