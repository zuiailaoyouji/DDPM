"""
degradation.py
用于 SR 训练阶段的在线 LR 合成流水线。

给定一个干净的 HR 图像张量 (C, H, W)，取值范围 [0, 1]，按如下步骤生成与之配对、
分辨率相同（256×256）的 LR 图像：
  1. 高斯模糊
  2. 双三次插值下采样 ×2  →  128×128
  3. 再次双三次插值上采样 →  256×256   （引入块状 / 模糊退化）
  4. 染色扰动（H&E 颜色增强，病理特异性）
  5. 轻微加性高斯噪声

所有操作都在 CPU 张量上完成，并在每次调用时随机采样，使模型能看到
一个退化分布，而不是固定退化。
"""

import torch
import torch.nn.functional as F
import random
import math


# ─────────────────────────────────────────────────────────────────────────────
# 公共接口
# ─────────────────────────────────────────────────────────────────────────────

def sample_degradation_params(
    blur_sigma_range: tuple,
    noise_std_range: tuple,
    stain_jitter_strength: float,
    device='cpu',
):
    """
    采样一次退化参数，全部返回 tensor，避免 DataLoader collate 时出现 None。
    """
    sigma = random.uniform(*blur_sigma_range)
    noise_std = random.uniform(*noise_std_range)

    if stain_jitter_strength > 0.0:
        stain_scales = 1.0 + (torch.rand(3, 1, 1, device=device) * 2 - 1) * stain_jitter_strength
    else:
        stain_scales = torch.ones(3, 1, 1, device=device)

    return {
        "sigma": torch.tensor(sigma, dtype=torch.float32, device=device),
        "noise_std": torch.tensor(noise_std, dtype=torch.float32, device=device),
        "stain_scales": stain_scales,
    }


def apply_degradation(
    hr: torch.Tensor,
    scale: int = 2,
    sigma: float = 1.0,
    stain_scales: torch.Tensor = None,
    noise_std: float = 0.0,
    noise_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """
    根据给定参数执行一次前向退化。
    """
    x = hr.clone()

    # 1. 高斯模糊
    x = _gaussian_blur(x, sigma)

    # 2. 下采样与上采样
    C, H, W = x.shape
    lr_h, lr_w = H // scale, W // scale
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(lr_h, lr_w), mode='bicubic', align_corners=False)
    x = F.interpolate(x, size=(H, W), mode='bicubic', align_corners=False)
    x = x.squeeze(0).clamp(0.0, 1.0)

    # 3. 染色扰动（直接应用给定 scale）
    if stain_scales is not None:
        eps = 1e-6
        od = -torch.log(x.clamp(min=eps))
        od = od * stain_scales.to(x.device)
        x = torch.exp(-od).clamp(0.0, 1.0)

    # 4. 加噪
    if noise_std > 0.0:
        if noise_tensor is None:
            noise_tensor = torch.randn_like(x)
        x = x + noise_tensor.to(x.device) * noise_std

    return x.clamp(0.0, 1.0)


def degrade(hr: torch.Tensor,
            scale: int = 2,
            blur_sigma_range: tuple = (0.5, 1.5),
            noise_std_range: tuple = (0.0, 0.02),
            stain_jitter_strength: float = 0.05,
            return_params: bool = False):
    """
    从干净的 HR 张量合成 LR 图像。

    当 return_params=True 时，返回 (lr, params) 以便推理阶段复用同一退化参数。
    """
    assert hr.dim() == 3, "Expected [C, H, W] tensor"
    params = sample_degradation_params(
        blur_sigma_range=blur_sigma_range,
        noise_std_range=noise_std_range,
        stain_jitter_strength=stain_jitter_strength,
        device=hr.device,
    )

    noise_tensor = torch.randn_like(hr) if params["noise_std"].item() > 0 else None
    lr = apply_degradation(
        hr,
        scale=scale,
        sigma=params["sigma"].item(),
        stain_scales=params["stain_scales"],
        noise_std=params["noise_std"].item(),
        noise_tensor=noise_tensor,
    )

    if return_params:
        params["noise_tensor"] = noise_tensor if noise_tensor is not None else torch.zeros_like(hr)
        return lr, params
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(sigma: float, kernel_size: int = None) -> torch.Tensor:
    """一维高斯核；返回长度为 k 的张量（先非归一化，再归一化）。"""
    if kernel_size is None:
        # 经验公式：6σ + 1，且保证为奇数
        kernel_size = max(3, 2 * int(math.ceil(3 * sigma)) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    half = kernel_size // 2
    xs = torch.arange(-half, half + 1, dtype=torch.float32)
    k  = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """在 [C, H, W] 张量上施加可分离高斯模糊。"""
    if sigma < 1e-3:
        return x

    k1d = _gaussian_kernel(sigma)                   # [k]
    C   = x.shape[0]
    pad = k1d.shape[0] // 2

    x4 = x.unsqueeze(0)                             # [1, C, H, W]

    # 水平方向 depthwise conv
    kh = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1) # [C,1,1,k]
    x4 = F.conv2d(x4, kh, padding=(0, pad), groups=C)

    # 竖直方向 depthwise conv
    kv = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1) # [C,1,k,1]
    x4 = F.conv2d(x4, kv, padding=(pad, 0), groups=C)

    return x4.squeeze(0).clamp(0.0, 1.0)


def _stain_jitter(x: torch.Tensor, strength: float) -> torch.Tensor:
    """
    旧接口保留：当前主流程已改为在 apply_degradation() 中直接使用 stain_scales。
    如需一次性随机扰动可继续调用本函数。

    在每个通道上施加轻微的乘性染色扰动，以模拟 H&E 染色差异。
    每个通道会乘以 (1 + U(-strength, +strength))。
    在 OD（光密度）空间中操作以保持物理合理性。
    """
    # 转换到 OD 空间：OD = -log(I + eps)
    eps = 1e-6
    od  = -torch.log(x.clamp(min=eps))

    # 在 OD 空间中为每个通道生成随机缩放因子
    scales = 1.0 + (torch.rand(x.shape[0], 1, 1) * 2 - 1) * strength
    od     = od * scales

    # 转回强度空间：I = exp(-OD)
    out = torch.exp(-od)
    return out.clamp(0.0, 1.0)
