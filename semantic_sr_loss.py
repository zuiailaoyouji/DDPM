"""
semantic_sr_loss.py
面向"判别器标签纠正"的核心损失函数。

【任务定义】
─────────────────────────────────────────────────────────────────────────────
对 HR 病理切片做微小修改，使 CellViT 对修改后图像的预测结果更接近 GT。
核心命题：诊断准确性与视觉感知质量是可分离的——模型可以修改人眼不敏感
但 CellViT 敏感的特征，在视觉质量略有下降的情况下提升诊断准确性。

【核心设计】
─────────────────────────────────────────────────────────────────────────────
1. 监督目标 = GT ∩ CellViT(HR) 的交集区域
   intersect_mask = (gt_label > 0) & (cellvit_hr_label == gt_label)

2. correction_boost：交集中 pred 仍判错的区域给更高权重

3. confusion_penalty：对特定混淆方向额外惩罚
   默认惩罚退步分析中最主要的两个方向：
   - Connective(3)→Epithelial(5)：3倍惩罚
   - Neoplastic(1)→Connective(3)：2倍惩罚
   这些方向的错误不需要视觉上明显的修改，只需要改变
   CellViT 敏感的微小特征（纹理频率、局部颜色分布等）

4. Focal CE + 类别逆频率权重（Inflammatory 权重加倍）

5. lambda_rec 降低到 1.5，给模型更大的像素修改空间，
   支持"诊断准确性与视觉感知质量可分离"的命题

【损失组成】
─────────────────────────────────────────────────────────────────────────────
L_noise : MSE(noise_pred, noise)
L_rec   : L1(x0_hat, hr)
L_grad  : L1(∇x0_hat, ∇hr)
L_sem   : Focal-CE with confusion_penalty + correction_boost
L_tv    : 相对 TV 惩罚
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# sem_tensor 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_sem_tensor_from_cellvit(
    cellvit_type_prob: torch.Tensor,
    cellvit_nuc_prob:  torch.Tensor,
) -> torch.Tensor:
    nuc = cellvit_nuc_prob.unsqueeze(1)
    return torch.cat([cellvit_type_prob, nuc], dim=1)


def build_gt_sem_tensor(
    gt_label_map: torch.Tensor,
    gt_nuc_mask:  torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = gt_label_map.device
    onehot = F.one_hot(gt_label_map.to(device).long(), num_classes=6) \
              .permute(0, 3, 1, 2).float()
    nuc = gt_nuc_mask.unsqueeze(1).to(device)
    return torch.cat([onehot, nuc], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# CellViT 推理封装
# ─────────────────────────────────────────────────────────────────────────────

def run_cellvit(
    cellvit: nn.Module,
    img_01:  torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out        = cellvit(img_01)
    type_prob  = F.softmax(out['nuclei_type_map'], dim=1)
    type_label = type_prob.argmax(dim=1)
    bin_prob   = F.softmax(out['nuclei_binary_map'], dim=1)
    nuc_prob   = bin_prob[:, 1]
    return dict(
        nuclei_type_prob   = type_prob,
        nuclei_type_label  = type_label,
        nuclei_binary_prob = bin_prob,
        nuclei_nuc_prob    = nuc_prob,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Focal CE
# ─────────────────────────────────────────────────────────────────────────────

def focal_ce_loss(
    log_prob:       torch.Tensor,
    target:         torch.Tensor,
    gamma:          float = 2.0,
    class_weights:  Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_pt  = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    pt      = log_pt.exp()
    focal_w = (1.0 - pt) ** gamma

    if class_weights is not None:
        focal_w = focal_w * class_weights.to(target.device)[target]

    if sample_weights is not None:
        focal_w = focal_w * sample_weights.to(focal_w.device)

    return -(focal_w * log_pt).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 主损失模块
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSRLoss(nn.Module):

    # Inflammatory 频率调低到 0.01，逆频率权重加倍
    _DEFAULT_CLASS_FREQ = torch.tensor(
        [0.85, 0.07, 0.01, 0.04, 0.005, 0.015],
        dtype=torch.float32,
    )

    def __init__(
        self,
        cellvit,
        noise_scheduler,
        lambda_sem_cls:    float = 1.0,
        correction_boost:  float = 3.0,
        # 混淆方向惩罚表：{(gt_cls, pred_cls): penalty_weight}
        # 对特定错误分类方向额外加权
        # None 时使用默认值（惩罚主要退步方向）
        confusion_penalty: Optional[dict] = None,
        focal_gamma:       float = 3.0,
        use_class_weights: bool  = True,
        lambda_noise: float = 1.0,
        lambda_rec:   float = 1.5,   # 降低像素保真约束，给模型更大修改空间
        lambda_grad:  float = 0.8,
        lambda_sem:   float = 2.0,
        lambda_tv:    float = 0.0005,
        t_max:            int   = 200,
        tv_margin_factor: float = 1.05,
        tv_leaky_alpha:   float = 0.10,
        class_freq: Optional[torch.Tensor] = None,
        tau_nuc:    float = 0.4,
        lambda_sem_cls_compat: float = None,
    ):
        super().__init__()
        self.cellvit   = cellvit
        self.scheduler = noise_scheduler

        self.lambda_sem_cls   = lambda_sem_cls
        self.correction_boost = correction_boost
        self.focal_gamma      = focal_gamma
        self.use_class_weights = use_class_weights

        # 混淆方向惩罚表：默认惩罚退步分析中最主要的两个方向
        # Connective→Epithelial 和 Neoplastic→Connective
        if confusion_penalty is None:
            self.confusion_penalty = {
                (3, 5): 3.0,   # Connective→Epithelial
                (1, 3): 2.0,   # Neoplastic→Connective
            }
        else:
            self.confusion_penalty = confusion_penalty

        self.lambda_noise = lambda_noise
        self.lambda_rec   = lambda_rec
        self.lambda_grad  = lambda_grad
        self.lambda_sem   = lambda_sem
        self.lambda_tv    = lambda_tv

        self.t_max            = t_max
        self.tv_margin_factor = tv_margin_factor
        self.tv_leaky_alpha   = tv_leaky_alpha

        freq = class_freq if class_freq is not None else self._DEFAULT_CLASS_FREQ
        inv_freq = 1.0 / (freq + 1e-6)
        self.register_buffer('class_weights', (inv_freq / inv_freq.mean()).float())

        if self.cellvit is not None:
            for p in self.cellvit.parameters():
                p.requires_grad = False
            self.cellvit.eval()

    # ─────────────────────────────────────────────────────────────────────
    # 公有接口
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        noise_pred:    torch.Tensor,
        noise:         torch.Tensor,
        noisy_hr:      torch.Tensor,
        hr:            torch.Tensor,
        t:             torch.Tensor,
        gt_nuc_mask:   torch.Tensor,
        gt_label_map:  torch.Tensor,
        lambda_sem:    Optional[float] = None,
        semantic_on:   bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        lam_sem = lambda_sem if lambda_sem is not None else self.lambda_sem

        l_noise = F.mse_loss(noise_pred, noise)
        x0_hat  = self._predict_x0(noisy_hr, noise_pred, t)
        l_rec   = F.l1_loss(x0_hat, hr)
        l_grad  = self._gradient_loss(x0_hat, hr)

        valid_idx = (t < self.t_max).nonzero(as_tuple=True)[0]
        l_sem     = hr.new_zeros(())
        l_tv      = hr.new_zeros(())
        l_sem_cls = hr.new_zeros(())
        mon = dict(
            sem_mae=-1.0, dir_acc=-1.0,
            intersect_ratio=-1.0, correction_ratio=-1.0,
        )

        if len(valid_idx) > 0:
            x0_v  = x0_hat[valid_idx]
            hr_v  = hr[valid_idx]
            lbl_v = gt_label_map[valid_idx].to(hr.device)

            l_tv = self._tv_loss(x0_v, hr_v)

            if semantic_on and self.cellvit is not None:
                l_sem, l_sem_cls, mon = self._semantic_losses_intersection(
                    x0_v, hr_v, lbl_v,
                )

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
    # 核心：基于交集 mask 的语义损失 + 混淆方向惩罚
    # ─────────────────────────────────────────────────────────────────────

    def _semantic_losses_intersection(
        self,
        x0_hat:   torch.Tensor,
        hr:       torch.Tensor,
        gt_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        在交集 mask 内计算语义损失，加入三层权重：
          1. correction_boost：pred 判错的区域权重更高
          2. confusion_penalty：特定混淆方向额外惩罚
             Connective→Epithelial(3x) 和 Neoplastic→Connective(2x)
          3. class_weights + focal_gamma：类别不平衡处理
        """
        N, _, H, W = x0_hat.shape

        if gt_label.shape[-2:] != (H, W):
            gt_label = F.interpolate(
                gt_label.unsqueeze(1).float(),
                size=(H, W), mode='nearest',
            ).squeeze(1).long()

        # CellViT(HR) no_grad
        with torch.no_grad():
            hr_out   = run_cellvit(self.cellvit, hr)
            hr_label = hr_out['nuclei_type_label']
            hr_prob  = hr_out['nuclei_type_prob']

        # 交集 mask
        gt_has_cell    = (gt_label > 0)
        hr_correct     = (hr_label == gt_label)
        intersect_mask = gt_has_cell & hr_correct

        n_intersect = intersect_mask.sum().item()
        if n_intersect < 1:
            z = x0_hat.new_zeros(())
            return z, z, dict(
                sem_mae=-1.0, dir_acc=-1.0,
                intersect_ratio=0.0, correction_ratio=0.0,
            )

        # CellViT(x0_hat) 需要梯度
        pred_out   = run_cellvit(self.cellvit, x0_hat)
        pred_prob  = pred_out['nuclei_type_prob']
        pred_label = pred_out['nuclei_type_label']

        # ── 像素权重：三层叠加 ────────────────────────────────────────
        pixel_weight = torch.ones(N, H, W, dtype=torch.float32, device=hr.device)

        # 第一层：correction_boost（pred 判错的区域）
        correction_mask = intersect_mask & (pred_label != gt_label)
        pixel_weight[correction_mask] *= self.correction_boost

        # 第二层：confusion_penalty（特定混淆方向额外惩罚）
        for (gt_cls, pred_cls), penalty in self.confusion_penalty.items():
            # 在交集内，GT 是 gt_cls 且 pred 判成 pred_cls 的像素
            conf_mask = (intersect_mask
                         & (gt_label == gt_cls)
                         & (pred_label == pred_cls))
            if conf_mask.sum() > 0:
                pixel_weight[conf_mask] *= penalty

        # ── Focal-CE 在 intersect_mask 内 ────────────────────────────
        mask_flat   = intersect_mask.reshape(-1)
        prob_flat   = pred_prob.permute(0, 2, 3, 1).reshape(-1, 6)
        target_flat = gt_label.reshape(-1)
        pw_flat     = pixel_weight.reshape(-1)

        prob_sel   = prob_flat[mask_flat]
        target_sel = target_flat[mask_flat]
        pw_sel     = pw_flat[mask_flat]

        log_prob  = torch.log(prob_sel.clamp(min=1e-8))
        cw        = self.class_weights if self.use_class_weights else None
        l_sem_cls = focal_ce_loss(
            log_prob       = log_prob,
            target         = target_sel,
            gamma          = self.focal_gamma,
            class_weights  = cw,
            sample_weights = pw_sel,
        )
        l_sem = self.lambda_sem_cls * l_sem_cls

        # 监控指标
        with torch.no_grad():
            dir_acc = (
                (pred_label[intersect_mask] == gt_label[intersect_mask])
                .float().mean().item()
            )
            sem_mae = (
                (pred_prob - hr_prob).abs().mean(dim=1)[intersect_mask]
                .mean().item()
            )
            n_gt_cell        = gt_has_cell.sum().item()
            intersect_ratio  = n_intersect / max(n_gt_cell, 1)
            correction_ratio = correction_mask.sum().item() / max(n_intersect, 1)

        return l_sem, l_sem_cls, dict(
            sem_mae          = sem_mae,
            dir_acc          = dir_acc,
            intersect_ratio  = intersect_ratio,
            correction_ratio = correction_ratio,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 辅助函数
    # ─────────────────────────────────────────────────────────────────────

    def _predict_x0(self, x_t, noise_pred, t):
        dev   = x_t.device
        dtype = x_t.dtype
        alpha = self.scheduler.alphas_cumprod.to(dev)[t].to(dtype).view(-1, 1, 1, 1)
        beta  = 1.0 - alpha
        return ((x_t - beta ** 0.5 * noise_pred) / (alpha ** 0.5 + 1e-8)).clamp(0.0, 1.0)

    def _gradient_loss(self, pred, target):
        def _grad(x):
            return (x[:, :, 1:, :] - x[:, :, :-1, :],
                    x[:, :, :, 1:] - x[:, :, :, :-1])
        ph, pw = _grad(pred)
        th, tw = _grad(target)
        return (ph - th).abs().mean() + (pw - tw).abs().mean()

    def _tv_loss(self, x0_hat, hr):
        def _tv(img):
            dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
            dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
            return dh.mean(dim=(1, 2, 3)) + dw.mean(dim=(1, 2, 3))

        tv_hat = _tv(x0_hat * 255.0)
        with torch.no_grad():
            tv_ref = _tv(hr.to(x0_hat.device) * 255.0)

        margin     = tv_ref * self.tv_margin_factor
        loss_batch = torch.where(
            tv_hat <= margin,
            self.tv_leaky_alpha * tv_hat,
            self.tv_leaky_alpha * margin + (tv_hat - margin),
        )
        return loss_batch.mean()