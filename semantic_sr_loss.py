"""
semantic_sr_loss.py
多类别核语义分布引导的 SR 损失函数。

v2 升级：GT mask 驱动的语义监督
────────────────────────────────
clean 参考来源切换为 Dataset 的 masks.npy GT 标注（由 ddpm_dataset.pannuke_mask_to_semantic 转换），
不再对 HR 图像实时跑 HoVer-Net。

语义先验张量（sem_tensor）格式（7 通道，与 unet_wrapper 兼容）
────────────────────────────────────────────────────────────────
  [0:6]  gt_tp_onehot  : one-hot 硬标签，HoVer-Net tp 类别空间
                          class 0=背景, 1=Neoplastic, 2=Inflammatory,
                          3=Connective, 4=Dead, 5=Epithelial
  [6]    gt_nuc_mask   : GT 细胞核二值掩膜（任意类型有实例=1.0）
  （已移除 gt_conf 通道：恒为 1 会导致 SemanticEncoder 对所有像素过度增强）

类别映射（masks.npy → HoVer-Net tp 空间）
──────────────────────────────────────────
  PanNuke masks.npy 通道  →  HoVer-Net tp class（本代码语义空间）
      ch 0  Neoplastic    →  class 1
      ch 1  Inflammatory  →  class 2
      ch 2  Connective    →  class 3
      ch 3  Dead          →  class 4
      ch 4  Epithelial    →  class 5
      （背景）             →  class 0
  该映射由 ddpm_dataset.pannuke_mask_to_semantic() 在数据加载阶段完成。

损失各项
────────
  L_noise : MSE(noise_pred, noise)              — 扩散骨干损失
  L_rec   : L1(x0_hat, hr)                     — 像素保真度
  L_grad  : L1(∇x0_hat, ∇hr)                  — 边缘锐度
  L_sem   : CE(pred_prob, gt_label_map)          — 语义类别监督
  L_tv    : 漏斗型相对 TV                        — 抗伪影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from hovernet_input_preprocess import run_hovernet_semantics_aligned


# ─────────────────────────────────────────────────────────────────────────────
# GT mask → sem_tensor 工具（可在 train.py / inference 中直接调用）
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_sem_tensor(
    gt_label_map: torch.Tensor,   # [B, H, W]  int64，来自 Dataset
    gt_nuc_mask:  torch.Tensor,   # [B, H, W]  float32，来自 Dataset
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    将 Dataset 提供的 GT 字段拼接为 sem_tensor [B, 7, H, W]。

    通道含义：
      [0:6]  gt_tp_onehot  — one-hot 硬标签（6 类，HoVer-Net tp 空间），
                             由 gt_label_map 在此处用 F.one_hot 按需生成
      [6]    gt_nuc_mask   — GT 核掩膜（0/1 float）

    该格式与 unet_wrapper.SPMUNet 的 SemanticEncoder(in_channels=7) 兼容。
    """
    if device is None:
        device = gt_label_map.device

    onehot = F.one_hot(gt_label_map.to(device).long(), num_classes=6) \
              .permute(0, 3, 1, 2).float()           # [B, 6, H, W]
    nuc    = gt_nuc_mask.unsqueeze(1).to(device)     # [B, 1, H, W]
    return torch.cat([onehot, nuc], dim=1)           # [B, 7, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# 主损失模块
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSRLoss(nn.Module):
    """
    语义引导型 SR DDPM 的完整损失（GT mask 驱动版）。

    关键变化（相比旧版）
    ────────────────────
    1. clean 侧完全使用 GT：gt_tp_onehot / gt_nuc_mask / gt_label_map。
       → 不再对 HR 图实时跑 HoVer-Net，省去 clean 侧的 HoVer-Net 推理开销。
    2. pred 侧仍用 HoVer-Net 对 x0_hat 做实时预测，评估重建质量。
    3. valid_mask 仅由 gt_nuc_mask > tau_nuc 决定（GT 硬标签无需置信度过滤）。
    4. confidence supervision 已移除：不对 pred_conf 施加任何显式监督目标。
    """

    def __init__(
        self,
        hovernet,
        noise_scheduler,
        # 核掩膜阈值
        tau_nuc:  float = 0.5,
        # 语义子项权重
        lambda_sem_cls: float = 0.3,    # CE
        # 各项总体权重
        lambda_noise: float = 1.0,
        lambda_rec:   float = 1.0,
        lambda_grad:  float = 0.1,
        lambda_sem:   float = 0.05,
        lambda_tv:    float = 0.001,
        # HoVer-Net 推理配置
        hovernet_upsample_factor: float = 1.0,
        # TV 与时间步配置
        t_max: int = 400,
        tv_margin_factor: float = 1.05,
        tv_leaky_alpha:   float = 0.10,
    ):
        super().__init__()
        self.hovernet  = hovernet
        self.scheduler = noise_scheduler

        self.tau_nuc         = tau_nuc
        self.lambda_sem_cls  = lambda_sem_cls

        self.lambda_noise = lambda_noise
        self.lambda_rec   = lambda_rec
        self.lambda_grad  = lambda_grad
        self.lambda_sem   = lambda_sem
        self.lambda_tv    = lambda_tv

        self.t_max                    = t_max
        self.tv_margin_factor         = tv_margin_factor
        self.tv_leaky_alpha           = tv_leaky_alpha
        self.hovernet_upsample_factor = float(hovernet_upsample_factor)

        # 冻结 HoVer-Net（仅用于 pred 侧推理，不参与训练）
        if self.hovernet is not None:
            for p in self.hovernet.parameters():
                p.requires_grad = False
            self.hovernet.eval()

    # ─────────────────────────────────────────────────────────────────────
    # 公有接口
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        noise_pred:    torch.Tensor,                  # [B, 3, H, W]
        noise:         torch.Tensor,                  # [B, 3, H, W]
        noisy_hr:      torch.Tensor,                  # [B, 3, H, W]  x_t
        hr:            torch.Tensor,                  # [B, 3, H, W]  clean HR [0,1]
        t:             torch.Tensor,                  # [B]            timesteps
        gt_nuc_mask:   torch.Tensor,                  # [B, H, W]     GT 核掩膜（必须）
        gt_label_map:  torch.Tensor,                  # [B, H, W]     GT 类别索引 long（必须）
        lambda_sem:    Optional[float] = None,
        semantic_on:   bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        返回
        ────
        total_loss : 标量张量（含梯度）
        breakdown  : 各项 detach 标量字典（用于日志/TensorBoard）
        """
        lam_sem = lambda_sem if lambda_sem is not None else self.lambda_sem

        # ── 1. 扩散噪声损失 ────────────────────────────────────────
        l_noise = F.mse_loss(noise_pred, noise)

        # ── 2. 重建 x0_hat ─────────────────────────────────────────
        x0_hat = self._predict_x0(noisy_hr, noise_pred, t)

        # ── 3. 像素重建损失 ────────────────────────────────────────
        l_rec  = F.l1_loss(x0_hat, hr)
        l_grad = self._gradient_loss(x0_hat, hr)

        # ── 4. 语义损失（由 t_max 与 semantic_on 联合控制）────────
        valid_idx = (t < self.t_max).nonzero(as_tuple=True)[0]

        l_sem = hr.new_zeros(())
        l_tv  = hr.new_zeros(())
        l_sem_cls = hr.new_zeros(())
        mon = dict(sem_mae=-1.0, dir_acc=-1.0)

        if len(valid_idx) > 0:
            x0_v  = x0_hat[valid_idx]
            hr_v  = hr[valid_idx]
            nuc_v = gt_nuc_mask[valid_idx].to(hr.device)    # [N, H, W]
            lbl_v = gt_label_map[valid_idx].to(hr.device)   # [N, H, W]

            # TV 损失（始终开启，抑制伪影）
            l_tv = self._tv_loss(x0_v, hr_v)

            if semantic_on:
                l_sem, l_sem_cls, mon = self._semantic_losses_gt(
                    x0_v, nuc_v, lbl_v,
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
    # GT mask 语义损失
    # ─────────────────────────────────────────────────────────────────────

    def _semantic_losses_gt(
        self,
        x0_hat:       torch.Tensor,   # [N, 3, H, W]
        gt_nuc_mask:  torch.Tensor,   # [N, H, W]     GT 核掩膜（float）
        gt_label_map: torch.Tensor,   # [N, H, W]     GT 类别索引（long，0-5）
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        clean 侧：GT masks.npy（不跑 HoVer-Net）
        pred  侧：HoVer-Net 对 x0_hat 的实时预测

        valid_mask = gt_nuc_mask > tau_nuc

        损失：
          l_ce = CE(pred_prob, gt_label_map)   逐像素，在 valid_mask 内平均

        返回：(l_sem, l_sem_cls, mon)
        """
        N, _, H, W = x0_hat.shape

        # ── GT 尺寸对齐 ───────────────────────────────────────────
        if gt_nuc_mask.shape[-2:] != (H, W):
            gt_nuc_mask = F.interpolate(
                gt_nuc_mask.unsqueeze(1), size=(H, W), mode="nearest"
            ).squeeze(1)
            gt_label_map = F.interpolate(
                gt_label_map.float().unsqueeze(1), size=(H, W), mode="nearest"
            ).squeeze(1).long()

        valid_mask = (gt_nuc_mask > self.tau_nuc).float()   # [N, H, W]
        denom = valid_mask.sum().clamp(min=1.0)

        # 空掩膜快速返回
        if valid_mask.sum() < 1:
            z = x0_hat.new_zeros(())
            mon = dict(sem_mae=-1.0, dir_acc=-1.0)
            return z, z, mon

        # ── HoVer-Net 对 x0_hat 预测（pred 侧）──────────────────
        pred      = self._run_hovernet(x0_hat)
        pred_prob = pred["tp_prob"]    # [N, 6, H, W]
        pred_lbl  = pred["tp_label"]   # [N, H, W]

        # 对齐 pred 尺寸
        if pred_prob.shape[-2:] != (H, W):
            pred_prob = F.interpolate(
                pred_prob, size=(H, W), mode="bilinear", align_corners=False
            )
            pred_lbl = F.interpolate(
                pred_lbl.float().unsqueeze(1), size=(H, W), mode="nearest"
            ).squeeze(1).long()

        # ── CE(pred_prob, gt_label_map) ──────────────────────────
        eps = 1e-8
        ce_map = F.nll_loss(
            torch.log(pred_prob.clamp(min=eps)),
            gt_label_map.long(),
            reduction="none",
        )                                          # [N, H, W]

        l_sem_cls = (ce_map * valid_mask).sum() / denom
        l_sem     = self.lambda_sem_cls * l_sem_cls

        # ── 监控指标（不回传梯度）────────────────────────────────
        with torch.no_grad():
            # 概率 MAE：pred_prob vs GT one-hot（按需生成，不存储）
            gt_onehot = F.one_hot(gt_label_map.long(), num_classes=6) \
                         .permute(0, 3, 1, 2).float()        # [N, 6, H, W]
            mae_map = (pred_prob - gt_onehot).abs().mean(dim=1)   # [N, H, W]
            sem_mae = (mae_map * valid_mask).sum() / denom
            cls_acc = ((pred_lbl == gt_label_map).float() * valid_mask).sum() / denom

        mon = dict(sem_mae=sem_mae.item(), dir_acc=cls_acc.item())
        return l_sem, l_sem_cls, mon

    # ─────────────────────────────────────────────────────────────────────
    # HoVer-Net 推理封装（仅用于 pred 侧，即 x0_hat）
    # ─────────────────────────────────────────────────────────────────────

    def _run_hovernet(self, img_01: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        img_01 : [B, 3, H, W] in [0, 1]
        返回   : tp_logits / tp_prob / tp_label / tp_conf / nuc_mask
        """
        assert self.hovernet is not None, \
            "语义损失 pred 侧需要 HoVer-Net 模型，但 hovernet=None。"
        return run_hovernet_semantics_aligned(
            self.hovernet,
            img_01,
            upsample_factor=self.hovernet_upsample_factor,
        )

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
        """有限差分边缘 L1 损失。"""
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
        """漏斗型相对 TV 损失，防止重建图引入过多高频伪影。"""
        def _tv(img: torch.Tensor) -> torch.Tensor:
            dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
            dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
            return dh.mean(dim=(1, 2, 3)) + dw.mean(dim=(1, 2, 3))

        # 在 0-255 空间计算以保持尺度兼容
        dev = x0_hat.device
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