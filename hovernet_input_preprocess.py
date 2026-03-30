"""
hovernet_input_preprocess.py

语义提取前可选上采样；语义提取后采用 HoVer-Net 原生风格的 patch 回拼对齐，
避免“中心贴回”导致外围有效区域不足。
"""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Tuple

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


def _get_last_steps(length: int, step_size: int) -> int:
    nr_step = int(ceil((float(length) - float(step_size)) / float(step_size)))
    return int((nr_step + 1) * step_size)


def _prepare_padded_tensor(img_chw: torch.Tensor, patch_input: int, patch_output: int):
    """
    参考 HoVer-Net tile 推理的 patching 规则。
    img_chw: [3,H,W]
    """
    _, h, w = img_chw.shape
    last_h = _get_last_steps(h, patch_output)
    last_w = _get_last_steps(w, patch_output)

    diff = patch_input - patch_output
    pad_t = pad_l = diff // 2
    pad_b = last_h + patch_input - h
    pad_r = last_w + patch_input - w

    img_pad = _safe_reflect_pad_2d(img_chw.unsqueeze(0), pad_l, pad_r, pad_t, pad_b).squeeze(0)
    coord_y = list(range(0, last_h, patch_output))
    coord_x = list(range(0, last_w, patch_output))
    return img_pad, coord_y, coord_x, h, w


def _safe_reflect_pad_2d(x: torch.Tensor, pad_l: int, pad_r: int, pad_t: int, pad_b: int) -> torch.Tensor:
    """
    torch 的 reflect pad 要求每次 pad < 对应维度。
    这里把大 padding 拆成多次小步 padding，保证与 HoVer-Net 原生 tile 的大范围反射填充兼容。
    x: [N,C,H,W]
    """
    out = x
    rem_l, rem_r, rem_t, rem_b = int(pad_l), int(pad_r), int(pad_t), int(pad_b)

    while rem_l > 0 or rem_r > 0 or rem_t > 0 or rem_b > 0:
        h = int(out.shape[-2])
        w = int(out.shape[-1])
        # reflect 约束：pad 必须 < 当前维度
        step_l = min(rem_l, max(0, w - 1))
        step_r = min(rem_r, max(0, w - 1))
        step_t = min(rem_t, max(0, h - 1))
        step_b = min(rem_b, max(0, h - 1))

        if step_l == 0 and step_r == 0 and step_t == 0 and step_b == 0:
            # 理论上不会到这里；兜底避免死循环
            out = F.pad(out, (rem_l, rem_r, rem_t, rem_b), mode="replicate")
            break

        out = F.pad(out, (step_l, step_r, step_t, step_b), mode="reflect")
        rem_l -= step_l
        rem_r -= step_r
        rem_t -= step_t
        rem_b -= step_b
    return out


@torch.no_grad()
def _infer_patch_output_size(hovernet, img_in: torch.Tensor, patch_input: int) -> int:
    """
    用单个 probe patch 推断 HoVer-Net 的有效输出 patch 尺寸（fast 模式通常为 164）。
    """
    sample = img_in[0:1]  # [1,3,H,W]
    _, _, h, w = sample.shape
    if h < patch_input or w < patch_input:
        pad_h = max(0, patch_input - h)
        pad_w = max(0, patch_input - w)
        sample = F.pad(sample, (0, pad_w, 0, pad_h), mode="reflect")
    patch = sample[:, :, :patch_input, :patch_input]
    out = hovernet(patch * 255.0)
    return int(out["tp"].shape[-1])


@torch.no_grad()
def _run_hovernet_stitch_semantics(
    hovernet,
    img_in: torch.Tensor,
    patch_input: int = 256,
    infer_batch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    原生风格 patch 推理 + 回拼，返回：
      tp_logits [B,C,H,W], tp_prob [B,C,H,W], tp_conf [B,H,W], nuc_mask [B,H,W]
    """
    dev = img_in.device
    patch_output = _infer_patch_output_size(hovernet, img_in, patch_input)

    tp_logits_all: List[torch.Tensor] = []
    tp_prob_all: List[torch.Tensor] = []
    tp_conf_all: List[torch.Tensor] = []
    nuc_mask_all: List[torch.Tensor] = []

    for b in range(img_in.shape[0]):
        img_chw = img_in[b]
        img_pad, ys, xs, src_h, src_w = _prepare_padded_tensor(img_chw, patch_input, patch_output)
        nr_row, nr_col = len(ys), len(xs)

        patch_list: List[torch.Tensor] = []
        for y in ys:
            for x in xs:
                patch_list.append(img_pad[:, y:y + patch_input, x:x + patch_input])
        patches = torch.stack(patch_list, dim=0).to(dev)

        out_logits_list: List[torch.Tensor] = []
        out_prob_list: List[torch.Tensor] = []
        out_conf_list: List[torch.Tensor] = []
        out_nuc_list: List[torch.Tensor] = []
        for s in range(0, patches.shape[0], infer_batch_size):
            p = patches[s:s + infer_batch_size]
            out = hovernet(p * 255.0)
            tp_logits = out["tp"]  # [N,C,h,w]
            tp_prob = F.softmax(tp_logits, dim=1)
            tp_conf = torch.max(tp_prob, dim=1)[0]  # [N,h,w]
            nuc = F.softmax(out["np"], dim=1)[:, 1, :, :]  # [N,h,w]

            out_logits_list.append(tp_logits)
            out_prob_list.append(tp_prob)
            out_conf_list.append(tp_conf)
            out_nuc_list.append(nuc)

        tp_logits_p = torch.cat(out_logits_list, dim=0)
        tp_prob_p = torch.cat(out_prob_list, dim=0)
        tp_conf_p = torch.cat(out_conf_list, dim=0)
        nuc_p = torch.cat(out_nuc_list, dim=0)

        c = int(tp_logits_p.shape[1])
        ho = int(tp_logits_p.shape[-2])
        wo = int(tp_logits_p.shape[-1])

        logits_canvas = torch.zeros((c, nr_row * ho, nr_col * wo), device=dev, dtype=tp_logits_p.dtype)
        prob_canvas = torch.zeros((c, nr_row * ho, nr_col * wo), device=dev, dtype=tp_prob_p.dtype)
        conf_canvas = torch.zeros((nr_row * ho, nr_col * wo), device=dev, dtype=tp_conf_p.dtype)
        nuc_canvas = torch.zeros((nr_row * ho, nr_col * wo), device=dev, dtype=nuc_p.dtype)

        idx = 0
        for r in range(nr_row):
            y0 = r * ho
            y1 = y0 + ho
            for cc in range(nr_col):
                x0 = cc * wo
                x1 = x0 + wo
                logits_canvas[:, y0:y1, x0:x1] = tp_logits_p[idx]
                prob_canvas[:, y0:y1, x0:x1] = tp_prob_p[idx]
                conf_canvas[y0:y1, x0:x1] = tp_conf_p[idx]
                nuc_canvas[y0:y1, x0:x1] = nuc_p[idx]
                idx += 1

        tp_logits_all.append(logits_canvas[:, :src_h, :src_w])
        tp_prob_all.append(prob_canvas[:, :src_h, :src_w])
        tp_conf_all.append(conf_canvas[:src_h, :src_w])
        nuc_mask_all.append(nuc_canvas[:src_h, :src_w])

    tp_logits_b = torch.stack(tp_logits_all, dim=0)
    tp_prob_b = torch.stack(tp_prob_all, dim=0)
    tp_conf_b = torch.stack(tp_conf_all, dim=0)
    nuc_mask_b = torch.stack(nuc_mask_all, dim=0)
    return tp_logits_b, tp_prob_b, tp_conf_b, nuc_mask_b


@torch.no_grad()
def run_hovernet_semantics_aligned(
    hovernet,
    img_01: torch.Tensor,
    upsample_factor: float = 2.0,
    upsample_mode: str = "bicubic",
) -> Dict[str, torch.Tensor]:
    """
    对齐语义输出到输入原分辨率（原生 patch 回拼），返回与 semantic_sr_loss._run_hovernet 相同 key 的结构。

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

    tp_logits, tp_prob, tp_conf, nuc_mask = _run_hovernet_stitch_semantics(
        hovernet,
        img_in,
        patch_input=256,
        infer_batch_size=16,
    )

    # 若上采样过，则从 img_in 尺寸回到原始 img_01 尺寸
    if (tp_prob.shape[-2] != target_h) or (tp_prob.shape[-1] != target_w):
        tp_logits = F.interpolate(tp_logits, size=(target_h, target_w), mode="bilinear", align_corners=False)
        tp_prob = F.interpolate(tp_prob, size=(target_h, target_w), mode="bilinear", align_corners=False)
        tp_conf = F.interpolate(tp_conf.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(1)
        nuc_mask = F.interpolate(nuc_mask.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(1)

    tp_label = torch.argmax(tp_prob, dim=1)

    return dict(
        tp_logits=tp_logits,
        tp_prob=tp_prob,
        tp_label=tp_label,
        tp_conf=tp_conf,
        nuc_mask=nuc_mask,
    )

