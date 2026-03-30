"""
unet_wrapper.py
语义先验调制 U-Net（SPM-UNet），用于病理图像超分辨率（SR）扩散模型。

架构概述
--------
基础扩散骨干仍是 diffusers 的 `UNet2DModel`（输入 6 通道：`[LR ‖ noisy_HR]`）。
在其上新增两类模块，用"辅助"的方式把语义先验注入到解码器特征中：

- **SemanticEncoder（语义编码器）**
  输入：`S = [gt_tp_onehot(6), gt_nuc_mask(1)]`（7 通道 GT 语义先验）
        注意：不含 gt_conf 通道，避免恒为 1 导致背景区域也被过度调制。
  输出：多尺度语义特征（1/4 与 1/2 分辨率）

- **SemanticModBlock（语义调制块）**
  对解码器特征做 gated 的 FiLM / SPADE 风格调制：

    F' = F + g ⊙ (γ(S) ⊙ GN(F) + β(S))

  其中 g=σ(conv(S)) 是逐像素空间门控，网络可以自动在背景区域抑制语义信号。

注入位置
--------
仅在解码器的高分辨率阶段注入语义：
- Layer B：128×128（1/2 分辨率，up_block[2]，通道 256）

sem_tensor 格式（7 通道）
--------------------------
  [0:6]  gt_tp_onehot — one-hot 硬标签（class 0=背景, 1-5=5种细胞类型）
  [6]    gt_nuc_mask  — GT 核掩膜（0/1 float，有效监督区域）

API 兼容性
----------
`create_model()` 返回 `SPMUNet`，前向签名：

  noise_pred = model(model_input, timestep, semantic=S).sample

model_input : [B, 6, H, W]（[LR ‖ noisy_HR]）
semantic    : [B, 7, H, W] 或 None
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput


# ─────────────────────────────────────────────────────────────────────────────
# 语义编码器（SemanticEncoder）
# ─────────────────────────────────────────────────────────────────────────────

class SemanticEncoder(nn.Module):
    """
    轻量级 CNN：把 GT 语义先验映射为多尺度特征，供语义调制块使用。

    输入：
      s : [B, 7, H, W]
          [gt_tp_onehot(6), gt_nuc_mask(1)]，取值范围 [0,1]

    输出：
      s_quarter : [B, 64, H/4, W/4]  （64×64，备用低分辨率注入点）
      s_half    : [B, 16, H/2, W/2]  （128×128，当前启用的注入点）
    """

    def __init__(self, in_channels: int = 7):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # → H/2
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # → H/4
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )  # 保持在 H/4

    def forward(self, s: torch.Tensor):
        s_half = self.enc1(s)       # [B, 16, H/2, W/2]
        s_q2   = self.enc2(s_half)  # [B, 32, H/4, W/4]
        s_quar = self.enc3(s_q2)    # [B, 64, H/4, W/4]
        return s_quar, s_half


# ─────────────────────────────────────────────────────────────────────────────
# 语义调制块（SemanticModBlock）
# ─────────────────────────────────────────────────────────────────────────────

class SemanticModBlock(nn.Module):
    """
    带空间门控的 FiLM / SPADE 风格调制块（残差注入）。

    给定解码器特征 feat ∈ [B, C_f, H, W] 与语义特征 sem ∈ [B, C_s, H_s, W_s]：
      1) 若需要，将 sem 上采样到 (H, W)
      2) 由 sem 预测逐像素 γ、β
      3) 由 sem 预测逐像素 gate g ∈ (0,1)
      4) 输出：feat + g ⊙ (γ ⊙ GN(feat) + β)

    初始化时输出层权重为零，保证初始为恒等映射，不破坏已有骨干权重。
    """

    def __init__(self, feat_channels: int, sem_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=min(num_groups, feat_channels),
            num_channels=feat_channels,
            affine=False,
        )

        hidden = max(sem_channels, 32)
        self.gamma_conv = nn.Sequential(
            nn.Conv2d(sem_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, feat_channels, 1),
        )
        self.beta_conv = nn.Sequential(
            nn.Conv2d(sem_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, feat_channels, 1),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(sem_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, feat_channels, 1),
            nn.Sigmoid(),
        )

        # 输出层初始化为零 → 初始恒等路径，训练更稳定
        for m in [self.gamma_conv[-1], self.beta_conv[-1], self.gate_conv[-2]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, sem: torch.Tensor) -> torch.Tensor:
        H, W = feat.shape[2], feat.shape[3]
        if sem.shape[2] != H or sem.shape[3] != W:
            sem = F.interpolate(sem, size=(H, W), mode='bilinear', align_corners=False)

        gamma  = self.gamma_conv(sem)
        beta   = self.beta_conv(sem)
        gate   = self.gate_conv(sem)
        normed = self.norm(feat)
        mod    = gamma * normed + beta
        return feat + gate * mod


# ─────────────────────────────────────────────────────────────────────────────
# 注入点常量
# ─────────────────────────────────────────────────────────────────────────────

# UNet2DModel up_blocks 通道（block_out_channels = (128,128,256,256,512,512)，倒序）：
#   up_block[0] → 512  (32×32)
#   up_block[1] → 512  (64×64)
#   up_block[2] → 256  (128×128)  ← 当前唯一启用的注入点 B
#   up_block[3] → 256  (256×256)
#   up_block[4] → 128  (256×256)
#   up_block[5] → 128  (256×256)

_INJECT_B = 2    # up_block 索引
_CHAN_B   = 256  # 该层通道数
_SEM_B    = 16   # SemanticEncoder s_half 通道数（对应 H/2=128×128）


# ─────────────────────────────────────────────────────────────────────────────
# SPMUNet
# ─────────────────────────────────────────────────────────────────────────────

class SPMUNet(nn.Module):
    """
    语义先验调制 U-Net（SPM-UNet）。

    输入：
      sample   : [B, 6, H, W]   [LR ‖ noisy_HR]
      timestep : [B] 或 int
      semantic : [B, 7, H, W]   [gt_tp_onehot(6), gt_nuc_mask(1)]，可选

    输出：
      UNet2DOutput，.sample 为 [B, 3, H, W] 噪声预测
    """

    def __init__(
        self,
        sample_size:  int = 256,
        in_channels:  int = 6,
        out_channels: int = 3,
        sem_channels: int = 7,    # 7 通道：gt_tp_onehot(6) + gt_nuc_mask(1)
        use_semantic: bool = True,
    ):
        super().__init__()
        self.use_semantic = use_semantic

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
        )

        self.sem_encoder: SemanticEncoder | None = None
        self.mod_B: SemanticModBlock | None = None

        self._sem_feats: dict[str, torch.Tensor] = {}
        self._hooks: list = []

        if self.use_semantic:
            self.sem_encoder = SemanticEncoder(in_channels=sem_channels)
            self.mod_B = SemanticModBlock(feat_channels=_CHAN_B, sem_channels=_SEM_B)
            self._register_hooks()

    # ─────────────────────────────────────────────────────────────────
    # Hook 管理
    # ─────────────────────────────────────────────────────────────────

    def _register_hooks(self):
        def make_hook(mod_block: SemanticModBlock, sem_key: str):
            def hook(module, input, output):
                sem = self._sem_feats.get(sem_key)
                if sem is None:
                    return output
                if isinstance(output, tuple):
                    feat = output[0]
                    modded = mod_block(feat, sem)
                    return (modded,) + output[1:]
                return mod_block(output, sem)
            return hook

        assert self.mod_B is not None
        h_B = self.unet.up_blocks[_INJECT_B].register_forward_hook(
            make_hook(self.mod_B, "sem_B")
        )
        self._hooks = [h_B]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def disable_semantic_modulation(self):
        self.remove_hooks()
        self.use_semantic = False
        self._sem_feats.clear()

    def enable_semantic_modulation(self):
        self.use_semantic = True
        if self.sem_encoder is None or self.mod_B is None:
            return
        if not self._hooks:
            self._register_hooks()

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        sample:   torch.Tensor,
        timestep: torch.Tensor | int,
        semantic: torch.Tensor | None = None,  # [B, 7, H, W]
        **kwargs,
    ) -> UNet2DOutput:
        if self.use_semantic and (semantic is not None) and (self.sem_encoder is not None):
            _, s_half = self.sem_encoder(semantic)
            self._sem_feats["sem_B"] = s_half   # [B, 16, H/2, W/2] → 128×128 注入点
        else:
            self._sem_feats.clear()

        out = self.unet(sample, timestep, **kwargs)
        self._sem_feats.clear()
        return out

    # ─────────────────────────────────────────────────────────────────
    # 便捷接口
    # ─────────────────────────────────────────────────────────────────

    @property
    def config(self):
        return self.unet.config

    def semantic_parameters(self):
        if (not self.use_semantic) or self.sem_encoder is None or self.mod_B is None:
            return iter([])
        return iter(list(self.sem_encoder.parameters()) + list(self.mod_B.parameters()))

    def backbone_parameters(self):
        return self.unet.parameters()


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def create_model(
    sample_size:  int = 256,
    in_channels:  int = 6,
    out_channels: int = 3,
    use_semantic: bool = True,
) -> SPMUNet:
    """
    创建 SPMUNet。
    semantic=True 时 SemanticEncoder 接受 7 通道输入
    [gt_tp_onehot(6), gt_nuc_mask(1)]。
    """
    return SPMUNet(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        sem_channels=7,
        use_semantic=use_semantic,
    )


def create_base_model(
    sample_size:  int = 256,
    in_channels:  int = 6,
    out_channels: int = 3,
) -> SPMUNet:
    """消融基线：关闭语义注入。"""
    return create_model(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        use_semantic=False,
    )


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if isinstance(model, SPMUNet) and model.use_semantic:
        sem = sum(p.numel() for p in model.semantic_parameters() if p.requires_grad)
        bb  = sum(p.numel() for p in model.backbone_parameters() if p.requires_grad)
        return dict(total=total, backbone=bb, semantic=sem)
    return dict(total=total)