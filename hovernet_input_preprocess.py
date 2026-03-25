"""
hovernet_input_preprocess.py

把“输入分辨率（例如 20x 对应的 HR）与 HoVer-Net 预训练分辨率（例如 40x）不一致”带来的误差，
通过在语义提取前对输入做上采样，并在语义输出后再对齐回原分辨率来处理。

核心思想：
1) 将 img_01: [B,3,H,W] 上采样到 (H*upsample_factor, W*upsample_factor)
2) 前向 HoVer-Net 得到 tp_prob / tp_conf / nuc_mask
3) 将这些语义输出下采样回原始 (H,W)，保证下游损失/注入张量形状不变
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def upsample_img_01(
    img_01: torch.Tensor,
    upsample_factor: float = 2.0,
    mode: str = "bicubic",
) -> torch.Tensor:
    """
    img_01: [B,3,H,W] in [0,1]
    返回上采样图像 [B,3,H*factor,W*factor]
    """
    if upsample_factor == 1.0:
        return img_01
    return F.interpolate(
        img_01,
        scale_factor=upsample_factor,
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )


def resize_semantic_maps_to_hw(
    tp_prob: torch.Tensor,
    tp_conf: torch.Tensor,
    nuc_mask: torch.Tensor,
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    tp_prob: [B,C,H,W]
    tp_conf: [B,H,W]
    nuc_mask: [B,H,W]

    返回：
      tp_prob_down: [B,C,target_h,target_w]
      tp_conf_down: [B,target_h,target_w]
      nuc_mask_down:[B,target_h,target_w]
    """
    th, tw = target_hw
    tp_prob_down = F.interpolate(tp_prob, size=(th, tw), mode="bilinear", align_corners=False)
    tp_conf_down = F.interpolate(tp_conf.unsqueeze(1), size=(th, tw), mode="bilinear", align_corners=False).squeeze(1)
    nuc_mask_down = F.interpolate(nuc_mask.unsqueeze(1), size=(th, tw), mode="bilinear", align_corners=False).squeeze(1)
    return tp_prob_down, tp_conf_down, nuc_mask_down


@torch.no_grad()
def run_hovernet_semantics_aligned(
    hovernet,
    img_01: torch.Tensor,
    upsample_factor: float = 2.0,
    upsample_mode: str = "bicubic",
) -> Dict[str, torch.Tensor]:
    """
    对齐语义输出到输入原分辨率，返回与 semantic_sr_loss._run_hovernet 相同 key 的结构。

    返回字段：
      tp_logits: [B,C,H,W]（下采样对齐后的 logits）
      tp_prob  : [B,C,H,W]
      tp_label : [B,H,W]
      tp_conf  : [B,H,W]
      nuc_mask : [B,H,W]
    """
    dev = next(hovernet.parameters()).device
    img_01 = img_01.to(dev)

    target_h, target_w = int(img_01.shape[-2]), int(img_01.shape[-1])

    if upsample_factor != 1.0:
        img_in = upsample_img_01(img_01, upsample_factor=upsample_factor, mode=upsample_mode)
    else:
        img_in = img_01

    out = hovernet(img_in * 255.0)
    tp_logits = out["tp"]  # [B,C,Hup,Wup]
    tp_prob_hup = F.softmax(tp_logits, dim=1)
    tp_conf_hup, tp_label_hup = torch.max(tp_prob_hup, dim=1)  # [B,Hup,Wup]
    nuc_mask_hup = F.softmax(out["np"], dim=1)[:, 1, :, :]  # [B,Hup,Wup]

    # 下采样对齐回原分辨率
    if upsample_factor != 1.0:
        tp_prob, tp_conf, nuc_mask = resize_semantic_maps_to_hw(
            tp_prob_hup, tp_conf_hup, nuc_mask_hup, target_hw=(target_h, target_w)
        )
        tp_label = torch.argmax(tp_prob, dim=1)
        # logits 没有严格意义（因为我们用了 softmax 再 resize），但为了保持结构一致，做个对齐兜底
        tp_logits = F.interpolate(tp_logits, size=(target_h, target_w), mode="bilinear", align_corners=False)
    else:
        tp_prob = tp_prob_hup
        tp_conf = tp_conf_hup
        nuc_mask = nuc_mask_hup
        tp_label = tp_label_hup

    return dict(
        tp_logits=tp_logits,
        tp_prob=tp_prob,
        tp_label=tp_label,
        tp_conf=tp_conf,
        nuc_mask=nuc_mask,
    )

