"""
semantic_sr_loss.py
面向"判别器标签纠正"的核心损失函数。

【任务定义】
─────────────────────────────────────────────────────────────────────────────
对 HR 病理切片做微小修改，使 CellViT 对修改后图像的预测结果更接近 GT。
推理时没有 GT，只有 HR 和 CellViT。训练时用 PanNuke（有 GT）来监督。

【核心设计】
─────────────────────────────────────────────────────────────────────────────
1. 监督目标 = GT ∩ CellViT(HR) 的交集区域
   intersect_mask = (gt_label > 0) & (cellvit_hr_label == gt_label)
   含义：GT 有细胞标注，且 CellViT(HR) 也判对了的像素。
   这是有意义的监督信号：模型学习让 CellViT(pred) 在这些区域也判对。
   交集之外（CellViT 连 HR 都判错的区域）不施加语义监督，
   因为模型无法从 CellViT 的错误预测中学到正确的细胞外观。

2. 额外在"纠正候选区域"施加更强监督
   correction_mask = intersect_mask & (pred_label != gt_label)
   即 CellViT(HR) 判对但 CellViT(pred) 还判错的区域，
   给予 correction_boost 倍像素权重。

3. Focal CE + 类别逆频率权重
   解决 Neoplastic 主导、少数类（Inflammatory/Dead）梯度不足的问题。

4. sem_tensor 用 CellViT(HR) 软标签构建，训练推理一致
   [cellvit_hr_type_prob(6), cellvit_hr_nuc_prob(1)] → 7 通道

【损失组成】
─────────────────────────────────────────────────────────────────────────────
L_noise : MSE(noise_pred, noise)               — 扩散骨干稳定性
L_rec   : L1(x0_hat, hr)                      — 像素保真（防过度修改）
L_grad  : L1(∇x0_hat, ∇hr)                   — 边缘保真（防模糊）
L_sem   : Focal-CE(CellViT(x0_hat), gt_label) — 向交集目标靠近
          仅在 intersect_mask 内有效，correction_mask 额外加权
L_tv    : 相对 TV 惩罚                         — 抑制幻觉伪影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# sem_tensor 构建（CellViT 软标签，训练推理一致）
# ─────────────────────────────────────────────────────────────────────────────

def build_sem_tensor_from_cellvit(
    cellvit_type_prob: torch.Tensor,  # [B, 6, H, W]  CellViT(HR) softmax 概率
    cellvit_nuc_prob:  torch.Tensor,  # [B, H, W]     CellViT(HR) 核概率
) -> torch.Tensor:
    """
    用 CellViT(HR) 软标签构建 sem_tensor [B, 7, H, W]。
    通道：[cellvit_hr_type_prob(6), cellvit_hr_nuc_prob(1)]
    全部为连续概率值，不做 argmax 硬化。
    训练和推理均调用此函数，保证一致性。
    与 unet_wrapper.SemanticEncoder(in_channels=7) 接口兼容。
    """
    nuc = cellvit_nuc_prob.unsqueeze(1)                  # [B, 1, H, W]
    return torch.cat([cellvit_type_prob, nuc], dim=1)    # [B, 7, H, W]


def build_gt_sem_tensor(
    gt_label_map: torch.Tensor,
    gt_nuc_mask:  torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    旧接口保留，仅在无 CellViT 时降级回退使用。
    正常训练流程应使用 build_sem_tensor_from_cellvit。
    """
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
    img_01:  torch.Tensor,  # [B, 3, H, W] in [0, 1]
) -> Dict[str, torch.Tensor]:
    """
    对输入图像运行 CellViT，返回：
      nuclei_type_prob  : [B, 6, H, W]  softmax 概率
      nuclei_type_label : [B, H, W]     argmax 类别索引
      nuclei_binary_prob: [B, 2, H, W]  核/非核 softmax 概率
      nuclei_nuc_prob   : [B, H, W]     核概率（binary[:,1]）
    """
    out        = cellvit(img_01)
    type_prob  = F.softmax(out['nuclei_type_map'], dim=1)    # [B, 6, H, W]
    type_label = type_prob.argmax(dim=1)                     # [B, H, W]
    bin_prob   = F.softmax(out['nuclei_binary_map'], dim=1)  # [B, 2, H, W]
    nuc_prob   = bin_prob[:, 1]                              # [B, H, W]
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
    log_prob:       torch.Tensor,                    # [N, C]  log softmax
    target:         torch.Tensor,                    # [N]     int64
    gamma:          float = 2.0,
    class_weights:  Optional[torch.Tensor] = None,   # [C]  逐类别权重
    sample_weights: Optional[torch.Tensor] = None,   # [N]  逐像素权重
) -> torch.Tensor:
    """
    Focal Cross-Entropy：(1 - p_t)^gamma * CE
    class_weights  处理类别不平衡，
    sample_weights 用于 correction_boost。
    """
    log_pt  = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]
    pt      = log_pt.exp()                                         # [N]
    focal_w = (1.0 - pt) ** gamma                                  # [N]

    if class_weights is not None:
        focal_w = focal_w * class_weights.to(target.device)[target]

    if sample_weights is not None:
        focal_w = focal_w * sample_weights.to(focal_w.device)

    return -(focal_w * log_pt).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 主损失模块
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSRLoss(nn.Module):
    """
    面向"判别器标签纠正"的完整损失模块。

    模型输入约定（与 train.py 对应）
    ──────────────────────────────────
    model_input : [HR, noisy_HR]（6 通道，不再有 LR）
    sem_tensor  : build_sem_tensor_from_cellvit(cellvit_hr_prob, cellvit_hr_nuc)
    监督区域    : intersect_mask = (gt_label>0) & (cellvit_hr_label==gt_label)
    """

    # PanNuke 各类别近似像素频率
    # 0:bg   1:Neo  2:Inf   3:Con   4:Dead  5:Epi
    _DEFAULT_CLASS_FREQ = torch.tensor(
        [0.85, 0.07, 0.02, 0.04, 0.005, 0.015],
        dtype=torch.float32,
    )

    def __init__(
        self,
        cellvit,
        noise_scheduler,
        # 语义子项
        lambda_sem_cls:    float = 1.0,
        correction_boost:  float = 2.0,
        # Focal Loss
        focal_gamma:       float = 2.0,
        use_class_weights: bool  = True,
        # 各项总体权重
        lambda_noise: float = 1.0,
        lambda_rec:   float = 3.0,
        lambda_grad:  float = 0.8,
        lambda_sem:   float = 2.0,
        lambda_tv:    float = 0.0005,
        # 时间步配置
        t_max:            int   = 150,
        tv_margin_factor: float = 1.05,
        tv_leaky_alpha:   float = 0.10,
        # 自定义类别频率
        class_freq: Optional[torch.Tensor] = None,
        # 旧接口兼容（不再使用）
        tau_nuc:        float = 0.4,
        lambda_sem_cls_compat: float = None,
    ):
        super().__init__()
        self.cellvit   = cellvit
        self.scheduler = noise_scheduler

        self.lambda_sem_cls   = lambda_sem_cls
        self.correction_boost = correction_boost
        self.focal_gamma      = focal_gamma
        self.use_class_weights = use_class_weights

        self.lambda_noise = lambda_noise
        self.lambda_rec   = lambda_rec
        self.lambda_grad  = lambda_grad
        self.lambda_sem   = lambda_sem
        self.lambda_tv    = lambda_tv

        self.t_max            = t_max
        self.tv_margin_factor = tv_margin_factor
        self.tv_leaky_alpha   = tv_leaky_alpha

        # 类别逆频率权重，归一化到均值 = 1
        freq = class_freq if class_freq is not None else self._DEFAULT_CLASS_FREQ
        inv_freq = 1.0 / (freq + 1e-6)
        self.register_buffer('class_weights', (inv_freq / inv_freq.mean()).float())

        # 冻结 CellViT
        if self.cellvit is not None:
            for p in self.cellvit.parameters():
                p.requires_grad = False
            self.cellvit.eval()

    # ─────────────────────────────────────────────────────────────────────
    # 公有接口
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        noise_pred:    torch.Tensor,  # [B, 3, H, W]
        noise:         torch.Tensor,  # [B, 3, H, W]
        noisy_hr:      torch.Tensor,  # [B, 3, H, W]  x_t
        hr:            torch.Tensor,  # [B, 3, H, W]  clean HR [0,1]
        t:             torch.Tensor,  # [B]
        gt_nuc_mask:   torch.Tensor,  # [B, H, W]  保留接口兼容，内部不再用于定义监督区域
        gt_label_map:  torch.Tensor,  # [B, H, W]  0=bg, 1-5=细胞类型
        lambda_sem:    Optional[float] = None,
        semantic_on:   bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        lam_sem = lambda_sem if lambda_sem is not None else self.lambda_sem

        # ── 1. 扩散噪声损失 ────────────────────────────────────────
        l_noise = F.mse_loss(noise_pred, noise)

        # ── 2. 重建 x0_hat ─────────────────────────────────────────
        x0_hat = self._predict_x0(noisy_hr, noise_pred, t)

        # ── 3. 像素保真损失 ────────────────────────────────────────
        l_rec  = F.l1_loss(x0_hat, hr)
        l_grad = self._gradient_loss(x0_hat, hr)

        # ── 4. 语义损失（t < t_max 且 semantic_on）─────────────────
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

        # ── 5. 加权汇总 ────────────────────────────────────────────
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
    # 核心：基于交集 mask 的语义损失
    # ─────────────────────────────────────────────────────────────────────

    def _semantic_losses_intersection(
        self,
        x0_hat:   torch.Tensor,  # [N, 3, H, W]
        hr:       torch.Tensor,  # [N, 3, H, W]
        gt_label: torch.Tensor,  # [N, H, W]  0=bg
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        1. CellViT(HR) no_grad → hr_label
        2. intersect_mask = (gt_label>0) & (hr_label==gt_label)
        3. correction_mask = intersect_mask & (pred_label!=gt_label)
        4. Focal-CE(CellViT(x0_hat), gt_label) 在 intersect_mask 内，
           correction_mask 区域给 correction_boost 倍像素权重
        """
        N, _, H, W = x0_hat.shape

        if gt_label.shape[-2:] != (H, W):
            gt_label = F.interpolate(
                gt_label.unsqueeze(1).float(),
                size=(H, W), mode='nearest',
            ).squeeze(1).long()

        # ── CellViT(HR) no_grad ───────────────────────────────────
        with torch.no_grad():
            hr_out   = run_cellvit(self.cellvit, hr)
            hr_label = hr_out['nuclei_type_label']   # [N, H, W]
            hr_prob  = hr_out['nuclei_type_prob']    # [N, 6, H, W]

        # ── 交集 mask ─────────────────────────────────────────────
        gt_has_cell    = (gt_label > 0)
        hr_correct     = (hr_label == gt_label)
        intersect_mask = gt_has_cell & hr_correct    # [N, H, W]

        n_intersect = intersect_mask.sum().item()
        if n_intersect < 1:
            z = x0_hat.new_zeros(())
            return z, z, dict(
                sem_mae=-1.0, dir_acc=-1.0,
                intersect_ratio=0.0, correction_ratio=0.0,
            )

        # ── CellViT(x0_hat) 需要梯度 ─────────────────────────────
        pred_out   = run_cellvit(self.cellvit, x0_hat)
        pred_prob  = pred_out['nuclei_type_prob']    # [N, 6, H, W]
        pred_label = pred_out['nuclei_type_label']   # [N, H, W]

        # ── correction_mask + 像素权重 ────────────────────────────
        correction_mask = intersect_mask & (pred_label != gt_label)
        pixel_weight    = torch.ones(N, H, W, dtype=torch.float32, device=hr.device)
        pixel_weight[correction_mask] = self.correction_boost

        # ── Focal-CE 在 intersect_mask 内 ────────────────────────
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

        # ── 监控指标（无梯度）────────────────────────────────────
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
        return ((x_t - beta ** 0.5 * noise_pred) / (alpha ** 0.5 + 1e-8)).clamp(0.0, 1.0)

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def _grad(x):
            return (x[:, :, 1:, :] - x[:, :, :-1, :],
                    x[:, :, :, 1:] - x[:, :, :, :-1])
        ph, pw = _grad(pred)
        th, tw = _grad(target)
        return (ph - th).abs().mean() + (pw - tw).abs().mean()

    def _tv_loss(self, x0_hat: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
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