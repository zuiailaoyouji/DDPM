"""
validation.py
语义引导型 SR DDPM 的验证集管理。

v2 变更
────────
1. generate_reconstructions 不再对 HR 跑 HoVer-Net 取 clean 语义参考。
   cls_clean / conf_clean / nuc_mask_clean 改为由 GT masks.npy 提供：
     - cls_clean      : gt_label_map（整数类别图）
     - nuc_mask_clean : gt_nuc_mask（GT 核掩膜）
     - conf_clean     : 已移除，GT 硬标签置信度恒为 1，无可视化意义
2. ValidationSet 新增 gt_label_map / gt_nuc_mask 字段，在 load() 阶段一并缓存。
3. save_validation_debug_images 移除 conf_clean 行（保留 conf_pred 供观测）。
4. 兼容 NCT 数据集：NCT 无 GT mask，GT 相关字段退化为全零占位，不影响可视化。

v3 变更
────────
5. generate_reconstructions 新增对 LR 图像的 HoVer-Net 预测：
     cls_pred_lr / conf_pred_lr / nuc_mask_pred_lr
6. save_validation_debug_images 新增三行，插入在原 SR pred 行之前，
   使 LR pred 与 SR pred 成对出现，便于直观对比超分前后语义质量：
     - "TP overlay LR pred"  : LR 的 HoVer-Net 类别 overlay
     - "tp_conf LR pred"     : LR 的 HoVer-Net 置信度图
     - "nuc_mask LR pred"    : LR 的 HoVer-Net 核掩膜
"""

import os
import random
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ddpm_dataset import build_dataset, _is_pannuke_fold_dir
from ddpm_utils import predict_x0_from_noise_shared
from degradation import degrade
from hovernet_input_preprocess import run_hovernet_semantics_aligned


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：自动推断验证源类型
# ─────────────────────────────────────────────────────────────────────────────

def infer_dataset_type_from_path(val_source: Optional[str]) -> str:
    if not val_source or not os.path.exists(val_source):
        return 'nct'
    if _is_pannuke_fold_dir(val_source):
        return 'pannuke'
    if os.path.isdir(val_source):
        entries = set(os.listdir(val_source))
        if 'TUM' in entries and 'NORM' in entries:
            return 'nct'
        for name in entries:
            cand = os.path.join(val_source, name)
            if _is_pannuke_fold_dir(cand):
                return 'pannuke'
    return 'nct'


def _pannuke_mask_hwc_to_label_map(mask: np.ndarray) -> np.ndarray:
    """
    PanNuke masks.npy 单样本 → 整图类型 id [H, W] int32。
    0=背景, 1..5 对应 Neoplastic/Inflammatory/Connective/Dead/Epithelial。
    """
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError(f"PanNuke mask 期望 3 维，得到 shape={m.shape}")
    if m.shape[0] in (5, 6) and m.shape[-1] not in (5, 6):
        m = np.transpose(m, (1, 2, 0))
    h, w, c = m.shape
    if c not in (5, 6):
        raise ValueError(f"PanNuke mask 通道数应为 5 或 6，得到 shape={m.shape}")
    m = m.astype(np.int32, copy=False)
    lbl = np.zeros((h, w), dtype=np.int32)
    for ch in range(5):
        lbl = np.where(m[..., ch] > 0, np.int32(ch + 1), lbl)
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# 加载固定验证样本
# ─────────────────────────────────────────────────────────────────────────────

def _load_fixed_validation_batch_nct(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
):
    images, labels = [], []
    for subdir, lbl in [('TUM', 1), ('NORM', 0)]:
        path = os.path.join(val_dir, subdir)
        if not os.path.exists(path):
            continue
        files = sorted(
            [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        )
        for f in files[:n_per_class]:
            img = cv2.imread(os.path.join(path, f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (sample_size, sample_size), interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
            labels.append(lbl)

    if not images:
        return None, None, None, None, None, None, None

    hr_np = np.stack(images, axis=0).astype(np.float32) / 255.0
    hr = torch.from_numpy(hr_np).permute(0, 3, 1, 2).to(device)

    torch_rng_state = torch.random.get_rng_state()
    py_rng_state = random.getstate()
    torch.manual_seed(42)
    random.seed(42)
    lr = torch.stack([
        degrade(hr[i].cpu(), scale=scale,
                blur_sigma_range=blur_sigma_range,
                noise_std_range=noise_std_range,
                stain_jitter_strength=stain_jitter)
        for i in range(len(hr))
    ]).to(device)
    torch.random.set_rng_state(torch_rng_state)
    random.setstate(py_rng_state)

    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    titles = (
        [f'TUM-{i+1}' for i in range(sum(1 for x in labels if x == 1))] +
        [f'NORM-{i+1}' for i in range(sum(1 for x in labels if x == 0))]
    )
    # NCT 无 GT mask，返回全零占位
    B, _, H, W = hr.shape
    gt_label_map = torch.zeros(B, H, W, dtype=torch.long, device=device)
    gt_nuc_mask  = torch.zeros(B, H, W, dtype=torch.float32, device=device)
    return hr, lr, labels_t, titles, gt_label_map, gt_nuc_mask


def _load_fixed_validation_batch_pannuke(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
):
    ds = build_dataset(
        dataset_type='pannuke',
        pannuke_folds=[val_dir] if _is_pannuke_fold_dir(val_dir) else None,
        pannuke_root=None if _is_pannuke_fold_dir(val_dir) else val_dir,
        scale=scale,
        blur_sigma_range=blur_sigma_range,
        noise_std_range=noise_std_range,
        stain_jitter=stain_jitter,
        target_size=sample_size,
    )
    if len(ds) == 0:
        return None, None, None, None, None, None

    # round-robin 抽样
    type_to_indices = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        type_name = sample.get('type_name', str(int(sample['label'])))
        type_to_indices.setdefault(type_name, []).append(idx)

    selected_indices = []
    type_names_sorted = sorted(type_to_indices.keys())
    per_type_ptr = {k: 0 for k in type_names_sorted}
    per_type_cap = {k: min(n_per_class, len(type_to_indices[k])) for k in type_names_sorted}
    while True:
        progressed = False
        for type_name in type_names_sorted:
            p = per_type_ptr[type_name]
            if p < per_type_cap[type_name]:
                selected_indices.append(type_to_indices[type_name][p])
                per_type_ptr[type_name] = p + 1
                progressed = True
        if not progressed:
            break

    max_total = max(8, n_per_class * min(4, len(type_to_indices)))
    selected_indices = selected_indices[:max_total]
    if not selected_indices:
        selected_indices = list(range(min(max_total, len(ds))))

    hrs, lrs, labels, titles = [], [], [], []
    gt_label_maps, gt_nuc_masks = [], []

    for idx in selected_indices:
        sample = ds[idx]
        hrs.append(sample['hr'])
        lrs.append(sample['lr'])
        labels.append(sample['label'])
        titles.append(sample.get('type_name', str(int(sample['label']))))
        gt_label_maps.append(sample['gt_label_map'])   # [H, W] long
        gt_nuc_masks.append(sample['gt_nuc_mask'])     # [H, W] float

    hr_t  = torch.stack(hrs).to(device)
    lr_t  = torch.stack(lrs).to(device)
    lbl_t = torch.stack(labels).to(device)
    gt_label_map_t = torch.stack(gt_label_maps).to(device)   # [B, H, W]
    gt_nuc_mask_t  = torch.stack(gt_nuc_masks).to(device)    # [B, H, W]

    return hr_t, lr_t, lbl_t, titles, gt_label_map_t, gt_nuc_mask_t


def load_fixed_validation_batch(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
    dataset_type: str = 'auto',
):
    """
    返回
    ────
    hr, lr, labels, titles, gt_label_map, gt_nuc_mask
    """
    if dataset_type == 'auto':
        dataset_type = infer_dataset_type_from_path(val_dir)

    if dataset_type == 'pannuke':
        return _load_fixed_validation_batch_pannuke(
            val_dir, sample_size, device, scale, n_per_class,
            blur_sigma_range, noise_std_range, stain_jitter,
        )
    return _load_fixed_validation_batch_nct(
        val_dir, sample_size, device, scale, n_per_class,
        blur_sigma_range, noise_std_range, stain_jitter,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ValidationSet
# ─────────────────────────────────────────────────────────────────────────────

class ValidationSet:
    """
    管理一小批固定的验证样本，用于主观可视化。

    新增缓存字段：
      gt_label_map : [B, H, W] long   — GT 像素级类别索引
      gt_nuc_mask  : [B, H, W] float  — GT 核掩膜
    """

    def __init__(self, val_dir, scheduler, device='cuda',
                 sample_size=256, fixed_timestep=100, scale=2,
                 blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02),
                 stain_jitter=0.05, dataset_type: str = 'auto'):
        self.val_dir        = val_dir
        self.scheduler      = scheduler
        self.device         = device
        self.sample_size    = sample_size
        self.fixed_timestep = fixed_timestep
        self.scale          = scale
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range  = noise_std_range
        self.stain_jitter     = stain_jitter
        self.dataset_type     = dataset_type

        self.hr = self.lr = self.noisy_hr = None
        self.labels = self.noise = self.timesteps = None
        self.col_titles   = None
        self.gt_label_map = None   # [B, H, W] long
        self.gt_nuc_mask  = None   # [B, H, W] float

    def load(self) -> bool:
        if not self.val_dir or not os.path.exists(self.val_dir):
            return False
        if self.dataset_type == 'auto':
            self.dataset_type = infer_dataset_type_from_path(self.val_dir)
        print(f"Loading fixed validation batch: {self.val_dir} | dataset_type={self.dataset_type}")

        (self.hr, self.lr, self.labels, self.col_titles,
         self.gt_label_map, self.gt_nuc_mask) = load_fixed_validation_batch(
            self.val_dir, self.sample_size, self.device, self.scale,
            blur_sigma_range=self.blur_sigma_range,
            noise_std_range=self.noise_std_range,
            stain_jitter=self.stain_jitter,
            dataset_type=self.dataset_type,
        )
        if self.hr is None:
            return False
        print(f"  ✓ {self.hr.shape[0]} validation images loaded")
        self._prepare_noise()
        return True

    def _prepare_noise(self):
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        self.noise     = torch.randn_like(self.hr)
        self.timesteps = torch.full(
            (self.hr.shape[0],), self.fixed_timestep,
            device=self.device, dtype=torch.long,
        )
        self.noisy_hr = self.scheduler.add_noise(self.hr, self.noise, self.timesteps)
        torch.random.set_rng_state(torch_rng_state)

    def is_available(self):
        return self.hr is not None

    def generate_reconstructions(
        self,
        unet,
        loss_module=None,
        use_semantic_injection: bool = False,
    ):
        if self.hr is None:
            return None

        with torch.no_grad():
            model_input = torch.cat([self.lr, self.noisy_hr], dim=1)

            # ── Stage 2：用 GT 构造 sem_tensor 做架构注入 ──────────
            sem_tensor = None
            if use_semantic_injection and (self.gt_label_map is not None):
                from semantic_sr_loss import build_gt_sem_tensor
                sem_tensor = build_gt_sem_tensor(
                    self.gt_label_map, self.gt_nuc_mask, device=self.device,
                )

            noise_pred = unet(model_input, self.timesteps, semantic=sem_tensor).sample
            recon      = predict_x0_from_noise_shared(
                self.noisy_hr, noise_pred, self.timesteps, self.scheduler,
            )
            diff_vis = (recon - self.hr).abs().clamp(0, 1)

            B, _, H, W = self.hr.shape
            nr_types = 6

            # ── clean 侧：直接用 GT ────────────────────────────────
            if self.gt_label_map is not None:
                cls_clean      = self.gt_label_map.float().unsqueeze(1)   # [B,1,H,W]
                nuc_mask_clean = self.gt_nuc_mask.unsqueeze(1)             # [B,1,H,W]
            else:
                cls_clean      = torch.zeros(B, 1, H, W, device=self.device)
                nuc_mask_clean = torch.zeros(B, 1, H, W, device=self.device)

            # ── SR pred 侧：HoVer-Net 对重建图预测 ────────────────
            cls_pred      = torch.zeros(B, 1, H, W, device=self.device)
            conf_pred     = torch.zeros(B, 1, H, W, device=self.device)
            nuc_mask_pred = torch.zeros(B, 1, H, W, device=self.device)

            # ── LR pred 侧：HoVer-Net 对 LR 图预测 ───────────────
            cls_pred_lr      = torch.zeros(B, 1, H, W, device=self.device)
            conf_pred_lr     = torch.zeros(B, 1, H, W, device=self.device)
            nuc_mask_pred_lr = torch.zeros(B, 1, H, W, device=self.device)

            if loss_module is not None and hasattr(loss_module, 'hovernet'):
                upsample_factor = float(
                    getattr(loss_module, 'hovernet_upsample_factor', 1.0)
                )

                # SR 重建图预测
                try:
                    p_sr = run_hovernet_semantics_aligned(
                        loss_module.hovernet,
                        recon,
                        upsample_factor=upsample_factor,
                    )
                    cls_pred      = p_sr['tp_label'].float().unsqueeze(1)
                    conf_pred     = p_sr['tp_conf'].unsqueeze(1)
                    nuc_mask_pred = p_sr['nuc_mask'].unsqueeze(1)
                    nr_types      = p_sr['tp_prob'].shape[1]
                except Exception:
                    pass

                # LR 图预测
                try:
                    p_lr = run_hovernet_semantics_aligned(
                        loss_module.hovernet,
                        self.lr,
                        upsample_factor=upsample_factor,
                    )
                    cls_pred_lr      = p_lr['tp_label'].float().unsqueeze(1)
                    conf_pred_lr     = p_lr['tp_conf'].unsqueeze(1)
                    nuc_mask_pred_lr = p_lr['nuc_mask'].unsqueeze(1)
                except Exception:
                    pass

        return dict(
            reconstructed    = recon,
            diff_vis         = diff_vis,
            cls_clean        = cls_clean,
            cls_pred         = cls_pred,
            conf_pred        = conf_pred,
            nuc_mask_clean   = nuc_mask_clean,
            nuc_mask_pred    = nuc_mask_pred,
            cls_pred_lr      = cls_pred_lr,
            conf_pred_lr     = conf_pred_lr,
            nuc_mask_pred_lr = nuc_mask_pred_lr,
            nr_types         = nr_types,
            col_titles       = self.col_titles,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def save_validation_debug_images(
    hr, lr, reconstructed, diff_vis,
    cls_clean, cls_pred,
    conf_pred,                          # SR pred 置信度
    nuc_mask_clean, nuc_mask_pred,
    epoch, save_dir, num_vis=8, return_tensor=False,
    col_titles=None,
    suptitle: str = None,
    nr_types: int = 6,
    # ── v3 新增：LR pred 字段（均有默认值，向后兼容旧调用方）────
    cls_pred_lr=None,                   # [B,1,H,W] float  LR 类别图
    conf_pred_lr=None,                  # [B,1,H,W] float  LR 置信度
    nuc_mask_pred_lr=None,              # [B,1,H,W] float  LR 核掩膜
):
    os.makedirs(save_dir, exist_ok=True)
    num_vis = min(num_vis, hr.shape[0])

    def _rgb(t):
        return t[:num_vis].detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

    def _gray(t):
        return t[:num_vis, 0].detach().cpu().clamp(0, 1).numpy()

    def _label(t):
        return t[:num_vis, 0].detach().cpu().numpy().astype(np.int32)

    def _resize_label_np(arr_np, out_hw):
        t = torch.from_numpy(arr_np).unsqueeze(1).float()
        t = F.interpolate(t, size=out_hw, mode='nearest')
        return t[:, 0].numpy().astype(np.int32)

    def _resize_gray_np(arr_np, out_hw):
        t = torch.from_numpy(arr_np).unsqueeze(1).float()
        t = F.interpolate(t, size=out_hw, mode='bilinear', align_corners=False)
        return t[:, 0].numpy()

    hr_rgb    = _rgb(hr)
    lr_rgb    = _rgb(lr)
    recon_rgb = _rgb(reconstructed)
    diff_rgb  = _rgb(diff_vis)

    cls_clean_int = _label(cls_clean)
    cls_pred_int  = _label(cls_pred)
    conf_pred_np  = _gray(conf_pred)
    nuc_clean_np  = _gray(nuc_mask_clean)
    nuc_pred_np   = _gray(nuc_mask_pred)

    # ── LR pred 数据（若未传入则用零占位，保持向后兼容）──────────
    _, _, H, W = hr.shape
    _zeros_label = np.zeros((num_vis, H, W), dtype=np.int32)
    _zeros_gray  = np.zeros((num_vis, H, W), dtype=np.float32)

    cls_pred_lr_int = _label(cls_pred_lr)      if cls_pred_lr      is not None else _zeros_label.copy()
    conf_pred_lr_np = _gray(conf_pred_lr)       if conf_pred_lr     is not None else _zeros_gray.copy()
    nuc_pred_lr_np  = _gray(nuc_mask_pred_lr)   if nuc_mask_pred_lr is not None else _zeros_gray.copy()

    target_hw = hr_rgb.shape[1:3]

    # 对齐所有标签图到 target_hw
    for arr in [cls_clean_int, cls_pred_int, cls_pred_lr_int]:
        if arr.shape[1:3] != target_hw:
            arr[:] = _resize_label_np(arr, target_hw)

    # 对齐所有灰度图到 target_hw
    for arr in [conf_pred_np, nuc_clean_np, nuc_pred_np, conf_pred_lr_np, nuc_pred_lr_np]:
        if arr.shape[1:3] != target_hw:
            arr[:] = _resize_gray_np(arr, target_hw)

    # 颜色表
    type_info_path = Path(__file__).resolve().parent / 'HoVer-net' / 'type_info.json'
    if type_info_path.exists():
        raw_type_info = json.loads(type_info_path.read_text(encoding='utf-8'))
    else:
        raw_type_info = {str(i): ['', [0, 0, 0]] for i in range(6)}

    tp_colors_hex = []
    tp_color_map_rgb01 = np.zeros((6, 3), dtype=np.float32)
    for tid in range(6):
        t   = raw_type_info.get(str(tid), None)
        rgb = [0, 0, 0] if t is None else t[1]
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        tp_color_map_rgb01[tid] = np.array([r, g, b], dtype=np.float32) / 255.0
        tp_colors_hex.append(f'#{r:02x}{g:02x}{b:02x}')

    def _tp_overlay(base_rgb, tp_label_int, tp_conf_gray, nuc_mask_gray, alpha=0.6):
        tp_label_int = np.clip(tp_label_int, 0, tp_color_map_rgb01.shape[0] - 1)
        color_map    = tp_color_map_rgb01[tp_label_int]
        valid_mask   = (tp_label_int > 0) & (nuc_mask_gray > 0)
        a = (alpha * tp_conf_gray * valid_mask.astype(np.float32))[..., None].clip(0, 1)
        return base_rgb * (1 - a) + color_map * a

    # GT clean 侧用 nuc_clean 作为 conf（GT 核区域置信度视为 1）
    overlay_clean = _tp_overlay(hr_rgb,    cls_clean_int,   nuc_clean_np,   nuc_clean_np)
    overlay_lr    = _tp_overlay(lr_rgb,    cls_pred_lr_int, conf_pred_lr_np, nuc_pred_lr_np)
    overlay_pred  = _tp_overlay(recon_rgb, cls_pred_int,    conf_pred_np,   nuc_pred_np)

    # ── 行定义 ─────────────────────────────────────────────────────
    # LR pred 行紧跟在对应 SR pred 行之前，方便上下对比
    rows_data = [
        (hr_rgb,          'HR'),
        (lr_rgb,          'LR'),
        (recon_rgb,       'Recon'),
        (diff_rgb,        'Residual'),
        (overlay_clean,   'TP overlay GT'),
        (overlay_lr,      'TP overlay LR pred'),    # ← 新增
        (overlay_pred,    'TP overlay SR pred'),
        (conf_pred_lr_np, 'tp_conf LR pred'),       # ← 新增
        (conf_pred_np,    'tp_conf SR pred'),
        (nuc_clean_np,    'nuc_mask GT'),
        (nuc_pred_lr_np,  'nuc_mask LR pred'),      # ← 新增
        (nuc_pred_np,     'nuc_mask SR pred'),
    ]

    n_rows, n_cols = len(rows_data), num_vis
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.8 * n_cols, 2.6 * n_rows),
                              squeeze=False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.995)
    if col_titles is not None:
        for c in range(min(n_cols, len(col_titles))):
            axes[0, c].set_title(str(col_titles[c]), fontsize=10, pad=8)

    for r, (data, title) in enumerate(rows_data):
        axes[r, 0].set_ylabel(title, fontsize=10)
        for c in range(n_cols):
            ax = axes[r, c]
            if data.ndim == 4:
                ax.imshow(data[c])
            else:
                ax.imshow(data[c], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

    legend_handles = [
        Patch(facecolor=tp_colors_hex[0], label='0: background'),
        Patch(facecolor=tp_colors_hex[1], label='1: Neoplastic'),
        Patch(facecolor=tp_colors_hex[2], label='2: Inflammatory'),
        Patch(facecolor=tp_colors_hex[3], label='3: Connective'),
        Patch(facecolor=tp_colors_hex[4], label='4: Dead'),
        Patch(facecolor=tp_colors_hex[5], label='5: Epithelial'),
    ]
    fig.legend(handles=legend_handles, title='TP classes',
               loc='upper right', fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.97 if suptitle else 1))
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_val.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if not return_tensor:
        return None

    grid = make_grid(
        torch.cat([hr[:num_vis], lr[:num_vis],
                   reconstructed[:num_vis], diff_vis[:num_vis]]).detach().cpu(),
        nrow=num_vis, padding=2,
    )
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# 定量验证用 DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def create_val_dataloader(
    val_vis_dir, batch_size, device='cuda',
    scale=2, oversample=False,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
    num_workers: int = 4,
    pin_memory: bool = True,
    dataset_type: str = 'auto',
    target_size: Optional[int] = 256,
):
    if not val_vis_dir or not os.path.exists(val_vis_dir):
        return None

    if dataset_type == 'auto':
        dataset_type = infer_dataset_type_from_path(val_vis_dir)

    if dataset_type == 'pannuke':
        val_ds = build_dataset(
            dataset_type='pannuke',
            pannuke_folds=[val_vis_dir] if _is_pannuke_fold_dir(val_vis_dir) else None,
            pannuke_root=None if _is_pannuke_fold_dir(val_vis_dir) else val_vis_dir,
            scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
            target_size=target_size,
        )
        dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and ('cuda' in str(device)),
            drop_last=False,
        )
        print(f'定量验证集（PanNuke）：{len(val_ds)} 张 patch')
        return dl

    tum_dir  = os.path.join(val_vis_dir, 'TUM')
    norm_dir = os.path.join(val_vis_dir, 'NORM')
    if not os.path.exists(tum_dir) or not os.path.exists(norm_dir):
        print('⚠️  验证目录缺少 TUM/NORM 子目录 —— 跳过定量验证。')
        return None

    val_ds = build_dataset(
        dataset_type='nct',
        tum_dir=tum_dir,
        norm_dir=norm_dir,
        oversample=oversample,
        scale=scale,
        blur_sigma_range=blur_sigma_range,
        noise_std_range=noise_std_range,
        stain_jitter=stain_jitter,
        target_size=target_size,
    )
    dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and ('cuda' in str(device)),
        drop_last=False,
    )
    print(f'定量验证集（NCT）：{len(val_ds)} 张图像')
    return dl