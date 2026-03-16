"""
metrics.py
用于评估语义引导 SR 重建质量的度量函数。

所有函数均在取值范围 [0, 1] 的 float32 CPU/GPU 张量上工作，
支持 batch 维度，标量输出为对一个 batch 的平均值。

主要指标
--------
compute_psnr                — 标准峰值信噪比
compute_ssim                — 结构相似度（单尺度）
compute_masked_semantic_mae — 仅在细胞掩膜内计算的 p_pred 与 p_target 的 MAE
compute_artifact_penalty    — TV 比值：TV(pred) / TV(hr)，> 1 表示引入了额外纹理
compute_composite_score     — 用于模型选择的加权综合评分
"""

import torch
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# 图像保真度指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 1.0) -> float:
    """
    以 dB 为单位的 PSNR。
    pred, target : [B, C, H, W] 或 [C, H, W]，float32，取值范围 [0, 1]
    """
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0.0:
        return float('inf')
    return 10.0 * math.log10((data_range ** 2) / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11,
                 data_range: float = 1.0,
                 reduction: str = 'mean') -> float:
    """
    单尺度 SSIM。
    pred, target : [B, C, H, W]，float32，取值范围 [0, 1]
    返回一个标量（在 batch 与通道维度上取平均）。
    """
    if pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # 构建高斯窗口
    k1d  = _gaussian_kernel_1d(window_size, sigma=1.5)
    win  = k1d.unsqueeze(0) * k1d.unsqueeze(1)          # [w, w]
    B_sz, nC, H, W = pred.shape
    win  = win.unsqueeze(0).unsqueeze(0).to(pred.device) # [1,1,w,w]
    win  = win.expand(nC, 1, window_size, window_size)   # [C,1,w,w]
    pad  = window_size // 2

    def _conv(x):
        return F.conv2d(x, win, padding=pad, groups=nC)

    mu1    = _conv(pred)
    mu2    = _conv(target)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = _conv(pred   ** 2) - mu1_sq
    sigma2_sq = _conv(target ** 2) - mu2_sq
    sigma12   = _conv(pred * target) - mu12

    ssim_map = ((2 * mu12   + C1) * (2 * sigma12   + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# 语义相关指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_masked_semantic_mae(p_pred: torch.Tensor,
                                p_target: torch.Tensor,
                                cell_mask: torch.Tensor) -> float:
    """
    仅在细胞掩膜内计算的 p_pred 与 p_target 的平均绝对误差（MAE）。

    参数:
        p_pred   : [B, H, W]，重建图像对应的癌变概率预测图
        p_target : [B, H, W]，软目标图（p_clean ± delta）
        cell_mask: [B, H, W]，HoVer-Net 输出的细胞核二值 / 软掩膜

    返回:
        标量 MAE（float）
    """
    weight = cell_mask.float().clamp(0.0, 1.0)
    denom  = weight.sum().clamp(min=1.0)
    mae    = ((p_pred - p_target).abs() * weight).sum() / denom
    return mae.item()


def compute_directional_accuracy(p_pred: torch.Tensor,
                                 p_clean: torch.Tensor,
                                 cell_mask: torch.Tensor,
                                 threshold: float = 0.5) -> float:
    """
    统计在细胞掩膜内，预测概率相对于 p_clean 是否朝着“正确方向”变化的比例。
      - 肿瘤像素  (p_clean > threshold)：p_pred 应该 >= p_clean
      - 正常像素  (p_clean ≤ threshold)：p_pred 应该 <= p_clean

    返回值位于 [0, 1]；1.0 表示所有像素都朝着正确方向变化。
    """
    is_tumor  = (p_clean > threshold).float()
    is_normal = 1.0 - is_tumor

    correct_t = (p_pred >= p_clean).float() * is_tumor  * cell_mask
    correct_n = (p_pred <= p_clean).float() * is_normal * cell_mask
    correct   = correct_t + correct_n

    denom = cell_mask.sum().clamp(min=1.0)
    return (correct.sum() / denom).item()


# ─────────────────────────────────────────────────────────────────────────────
# 伪影惩罚
# ─────────────────────────────────────────────────────────────────────────────

def compute_artifact_penalty(pred: torch.Tensor,
                             reference: torch.Tensor,
                             eps: float = 1e-6) -> float:
    """
    TV 比值：total_variation(pred) / total_variation(reference)。
    值 > 1 表明模型引入了比参考 HR 图更多的高频纹理，可视为“幻觉细节”的近似度量。

    pred, reference : [B, C, H, W] 或 [C, H, W]
    """
    tv_pred = _total_variation(pred)
    tv_ref  = _total_variation(reference)
    return (tv_pred / (tv_ref + eps)).item()


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """计算 [B,C,H,W] 或 [C,H,W] 张量的平均绝对 TV。"""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


# ─────────────────────────────────────────────────────────────────────────────
# 模型选择用综合评分
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_score(psnr: float,
                             ssim: float,
                             semantic_mae: float,
                             artifact_penalty: float,
                             w_psnr: float = 0.3,
                             w_ssim: float = 0.4,
                             w_sem:  float = 0.2,
                             w_art:  float = 0.1,
                             psnr_ref: float = 40.0) -> float:
    """
    生成一个位于 [0, 1] 的综合评分标量，可用于“值越大越好”的模型选择。

    psnr_ref : 映射到归一化得分 1.0 的 PSNR 参考值（软上限）
    """
    psnr_norm  = min(psnr / psnr_ref, 1.0)        # 0→1
    ssim_norm  = max(min(ssim, 1.0), 0.0)          # already 0→1
    sem_score  = max(1.0 - semantic_mae * 10, 0.0) # MAE 0.1 → score 0
    art_score  = max(1.0 - (artifact_penalty - 1.0), 0.0)  # ratio 2.0 → score 0

    return (w_psnr * psnr_norm
            + w_ssim * ssim_norm
            + w_sem  * sem_score
            + w_art  * art_score)


# ─────────────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    half = size // 2
    xs   = torch.arange(-half, half + 1, dtype=torch.float32)
    k    = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()
