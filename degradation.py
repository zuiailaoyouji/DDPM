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

def degrade(hr: torch.Tensor,
            scale: int = 2,
            blur_sigma_range: tuple = (0.5, 1.5),
            noise_std_range: tuple = (0.0, 0.02),
            stain_jitter_strength: float = 0.05,
            ) -> torch.Tensor:
    """
    从干净的 HR 张量合成 LR 图像。

    参数:
        hr                  : [C, H, W]，float32，取值范围 [0, 1]
        scale               : 下采样倍率（2 → 128→256）
        blur_sigma_range    : 高斯模糊 sigma 的最小/最大值
        noise_std_range     : 加性噪声标准差的最小/最大值
        stain_jitter_strength: 每个通道的最大乘性染色扰动幅度

    返回:
        lr : [C, H, W]，float32，取值范围 [0, 1]，与 hr 具有相同空间尺寸
    """
    assert hr.dim() == 3, "Expected [C, H, W] tensor"
    x = hr.clone()

    # 1. 高斯模糊（模拟衍射 / 对焦极限）
    sigma = random.uniform(*blur_sigma_range)
    x = _gaussian_blur(x, sigma)

    # 2. 下采样 → 上采样（模拟传感器分辨率上限）
    C, H, W = x.shape
    lr_h, lr_w = H // scale, W // scale
    x = x.unsqueeze(0)                                           # [1,C,H,W]
    x = F.interpolate(x, size=(lr_h, lr_w), mode='bicubic', align_corners=False)
    x = F.interpolate(x, size=(H, W),       mode='bicubic', align_corners=False)
    x = x.squeeze(0).clamp(0.0, 1.0)                            # [C,H,W]

    # 3. H&E 染色扰动（病理特异性的颜色偏移）
    if stain_jitter_strength > 0.0:
        x = _stain_jitter(x, stain_jitter_strength)

    # 4. 加性高斯噪声（传感器 / 数字化噪声）
    noise_std = random.uniform(*noise_std_range)
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std

    return x.clamp(0.0, 1.0)


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
    k1d = _gaussian_kernel(sigma)                       # [k]
    k   = k1d.unsqueeze(0).unsqueeze(0)                 # [1, 1, k]
    C   = x.shape[0]
    pad = k1d.shape[0] // 2

    # 水平方向卷积
    xh  = x.unsqueeze(0)                                # [1, C, H, W]
    xh  = xh.view(1 * C, 1, x.shape[1], x.shape[2])
    kh  = k.expand(C, 1, 1, k1d.shape[0])              # [C, 1, 1, k]
    xh  = F.conv2d(xh, kh, padding=(0, pad), groups=C)

    # 竖直方向卷积
    kv  = k.permute(0, 1, 3, 2).expand(C, 1, k1d.shape[0], 1)
    xv  = F.conv2d(xh, kv, padding=(pad, 0), groups=C)

    return xv.squeeze(0).clamp(0.0, 1.0)


def _stain_jitter(x: torch.Tensor, strength: float) -> torch.Tensor:
    """
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
