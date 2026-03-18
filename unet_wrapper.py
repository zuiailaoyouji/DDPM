"""
unet_wrapper.py
语义先验调制 U-Net（SPM-UNet），用于病理图像超分辨率（SR）扩散模型。

架构概述
--------
基础扩散骨干仍是 diffusers 的 `UNet2DModel`（输入 6 通道：`[LR ‖ noisy_HR]`）。
在其上新增两类模块，用“辅助”的方式把语义先验注入到解码器特征中：

- **SemanticEncoder（语义编码器）**
  输入：`S = [p_clean, nuc_mask, conf_mask]`（3 通道语义先验）
  输出：多尺度语义特征（1/4 与 1/2 分辨率）

- **SemanticModBlock（语义调制块）**
  对解码器特征做 gated 的 FiLM / SPADE 风格调制：

    \(F' = F + g \odot (\gamma(S) \odot GN(F) + \beta(S))\)

  其中 `g=σ(conv(S))` 是逐像素空间门控，网络可以自动在不确定区域抑制语义信号。

注入位置（Injection points）
----------------------------
遵循你的推荐，仅在解码器的两个分辨率阶段注入语义：
- Layer A：64×64（约 1/4 分辨率）
- Layer B：128×128（约 1/2 分辨率）

API 兼容性
----------
`create_model()` 返回 `SPMUNet`，前向签名兼容：

  `noise_pred = model(model_input, timestep, semantic=S).sample`

其中 `model_input` 仍为 `[B,6,H,W]`；`semantic` 可选，为 `[B,3,H,W]`。
当 `semantic=None` 或 `use_semantic=False` 时，模型行为等价于普通 `UNet2DModel`。
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
    轻量级 CNN：把 3 通道语义先验映射为多尺度特征，供语义调制块使用。

    输入：
      s : [B, 3, H, W]，通道含义为 (p_clean, nuc_mask, conf_mask)，取值范围 [0,1]

    输出：
      s_quarter : [B, 64, H/4, W/4]  （用于 64×64 注入点）
      s_half    : [B, 16, H/2, W/2]  （用于 128×128 注入点）

    该分支刻意做得很小，避免参数量与计算量显著增加。
    """

    def __init__(self, in_channels: int = 3):
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
        s_half = self.enc1(s)      # [B, 16, H/2, W/2]
        s_q2   = self.enc2(s_half) # [B, 32, H/4, W/4]
        s_quar = self.enc3(s_q2)   # [B, 64, H/4, W/4]
        return s_quar, s_half


# ─────────────────────────────────────────────────────────────────────────────
# 语义调制块（SemanticModBlock）
# ─────────────────────────────────────────────────────────────────────────────

class SemanticModBlock(nn.Module):
    """
    带空间门控的 FiLM / SPADE 风格调制块（残差注入）。

    给定解码器特征 `feat ∈ [B, C_f, H, W]` 与语义特征 `sem ∈ [B, C_s, H_s, W_s]`：
      1) 若需要，将 sem 上采样到 (H, W)
      2) 由 sem 预测逐像素 γ、β
      3) 由 sem 预测逐像素 gate g ∈ (0,1)
      4) 输出：feat + g ⊙ (γ ⊙ GN(feat) + β)

    残差形式保证在初始化时近似恒等映射，不会破坏已有骨干。
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

        # 让输出层接近 0：初始为恒等路径，更稳定
        for m in [self.gamma_conv[-1], self.beta_conv[-1], self.gate_conv[-2]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, sem: torch.Tensor) -> torch.Tensor:
        H, W = feat.shape[2], feat.shape[3]
        if sem.shape[2] != H or sem.shape[3] != W:
            sem = F.interpolate(sem, size=(H, W), mode='bilinear', align_corners=False)

        gamma = self.gamma_conv(sem)
        beta  = self.beta_conv(sem)
        gate  = self.gate_conv(sem)

        normed = self.norm(feat)
        mod    = gamma * normed + beta
        return feat + gate * mod


# ─────────────────────────────────────────────────────────────────────────────
# SPMUNet：在 UNet2DModel 上做语义注入的包装器
# ─────────────────────────────────────────────────────────────────────────────

# UNet2DModel 的 up_blocks 输出通道顺序（与你当前配置一致）：
# block_out_channels = (128, 128, 256, 256, 512, 512)
# up_blocks 的通道是“倒序”生成：
#   up_block[0] → 512  (32×32,  bottleneck)
#   up_block[1] → 512  (64×64,  ← 低分辨率注入点 A，当前已停用，仅保留注释)
#   up_block[2] → 256  (128×128,← 高分辨率注入点 B，当前唯一启用的注入点)
#   up_block[3] → 256  (256×256)
#   up_block[4] → 128  (256×256)
#   up_block[5] → 128  (256×256)
# _INJECT_A = 1                # 低分辨率注入点 A（64×64）—— 已注释，仅保留作参考
_INJECT_B = 2                  # 高分辨率注入点 B（128×128）
# _CHAN_A   = 512              # 注入点 A 的通道数
_CHAN_B   = 256                # 注入点 B 的通道数
# _SEM_A    = 64               # quarter-res 语义特征通道数（对应注入点 A）
_SEM_B    = 16                 # half-res    语义特征通道数（对应注入点 B）


class SPMUNet(nn.Module):
    """
    语义先验调制 U-Net（SPM-UNet）。

    - 输入：sample=[B,6,H,W]（[LR ‖ noisy_HR]）
    - 可选语义先验：semantic=[B,3,H,W]（[p_clean, nuc_mask, conf_mask]）
    - 输出：UNet2DOutput，其中 `.sample` 为 [B,3,H,W] 的噪声预测
    """

    def __init__(
        self,
        sample_size: int = 256,
        in_channels: int = 6,
        out_channels: int = 3,
        sem_channels: int = 3,
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
        # self.mod_A: SemanticModBlock | None = None   # 低分辨率注入块 A（64×64），当前停用
        self.mod_B: SemanticModBlock | None = None     # 高分辨率注入块 B（128×128）

        self._sem_feats: dict[str, torch.Tensor] = {}
        self._hooks: list = []

        if self.use_semantic:
            self.sem_encoder = SemanticEncoder(in_channels=sem_channels)
            # self.mod_A = SemanticModBlock(feat_channels=_CHAN_A, sem_channels=_SEM_A)
            self.mod_B = SemanticModBlock(feat_channels=_CHAN_B, sem_channels=_SEM_B)
            self._register_hooks()

    # ─────────────────────────────────────────────────────────────────
    # Hook 管理：在指定 up_block 输出后注入语义调制
    # ─────────────────────────────────────────────────────────────────

    def _register_hooks(self):
        def make_hook(mod_block: SemanticModBlock, sem_key: str):
            def hook(module, input, output):
                sem = self._sem_feats.get(sem_key)
                if sem is None:
                    return output

                # UNet2DModel 的 up_blocks 通常返回 tuple；第 0 项为特征张量
                if isinstance(output, tuple):
                    feat = output[0]
                    rest = output[1:]
                    modded = mod_block(feat, sem)
                    return (modded,) + rest
                return mod_block(output, sem)
            return hook

        # 仅保留高分辨率注入点 B 的 hook；
        # 低分辨率注入点 A 的 hook 暂时停用，仅保留作注释参考。
        assert self.mod_B is not None
        # h_A = self.unet.up_blocks[_INJECT_A].register_forward_hook(make_hook(self.mod_A, "sem_A"))
        h_B = self.unet.up_blocks[_INJECT_B].register_forward_hook(make_hook(self.mod_B, "sem_B"))
        self._hooks = [h_B]

    def remove_hooks(self):
        """如需在保存前移除 hook，可调用该函数。"""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def disable_semantic_modulation(self):
        """
        关闭“架构层语义调制”。

        行为：
        - 移除 forward hooks（不再拦截 up_blocks 输出）
        - 标记 use_semantic=False
        - 清空缓存的语义特征

        注意：语义分支模块仍保留在模型里（参数不丢失），只是前向不会使用它们。
        """
        self.remove_hooks()
        self.use_semantic = False
        self._sem_feats.clear()

    def enable_semantic_modulation(self):
        """
        开启“架构层语义调制”。

        行为：
        - 标记 use_semantic=True
        - 若语义分支已构建但 hooks 不存在，则注册 hooks

        注意：只有在 forward 时传入 semantic!=None 才会真正注入语义特征。
        """
        self.use_semantic = True
        if self.sem_encoder is None or self.mod_B is None:
            # 理论上不会发生：构造时 use_semantic=True 才会创建语义分支
            return
        if not self._hooks:
            self._register_hooks()

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        sample: torch.Tensor,                # [B, 6, H, W]
        timestep: torch.Tensor | int,
        semantic: torch.Tensor | None = None,  # [B, 3, H, W]
        **kwargs,
    ) -> UNet2DOutput:
        if self.use_semantic and (semantic is not None) and (self.sem_encoder is not None):
            s_quarter, s_half = self.sem_encoder(semantic)
            # self._sem_feats["sem_A"] = s_quarter    # 注入点 A（64×64）当前停用
            self._sem_feats["sem_B"] = s_half         # 仅使用高分辨率注入点 B（128×128）
        else:
            self._sem_feats.clear()

        out = self.unet(sample, timestep, **kwargs)

        # 清理，避免跨 batch 意外复用
        self._sem_feats.clear()
        return out

    # ─────────────────────────────────────────────────────────────────
    # 一些便捷接口（用于分组训练/统计）
    # ─────────────────────────────────────────────────────────────────

    @property
    def config(self):
        return self.unet.config

    def semantic_parameters(self):
        """仅返回新增语义分支参数（可用于 Stage-1 冻结骨干时训练）。"""
        if (not self.use_semantic) or self.sem_encoder is None or self.mod_B is None:
            return iter([])
        # 仅返回当前启用的高分辨率注入分支参数；低分辨率注入块 A 已停用。
        return iter(list(self.sem_encoder.parameters()) + list(self.mod_B.parameters()))

    def backbone_parameters(self):
        """返回原始 UNet2DModel 的参数。"""
        return self.unet.parameters()


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def create_model(
    sample_size: int = 256,
    in_channels: int = 6,
    out_channels: int = 3,
    use_semantic: bool = True,
) -> SPMUNet:
    """创建 SPMUNet。将 use_semantic=False 可得到不注入语义的消融基线。"""
    return SPMUNet(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        use_semantic=use_semantic,
    )


def create_base_model(
    sample_size: int = 256,
    in_channels: int = 6,
    out_channels: int = 3,
) -> SPMUNet:
    """消融基线：关闭语义注入（行为等价于普通 UNet2DModel）。"""
    return create_model(sample_size=sample_size, in_channels=in_channels, out_channels=out_channels, use_semantic=False)


def count_parameters(model: nn.Module) -> dict:
    """
    统计可训练参数量。
    若是 SPMUNet，会额外给出骨干与语义分支的拆分统计。
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if isinstance(model, SPMUNet) and model.use_semantic:
        sem = sum(p.numel() for p in model.semantic_parameters() if p.requires_grad)
        bb  = sum(p.numel() for p in model.backbone_parameters() if p.requires_grad)
        return dict(total=total, backbone=bb, semantic=sem)
    return dict(total=total)

