"""
validation_patched.py

改动目标：
1. 修复/缓解 overlay 看起来偏移的问题：
   - 不再用大面积实心 overlay 作为主要视觉表达；
   - 改为“轻量透明填充 + 轮廓描边”，并且 contour 直接从当前分辨率的二值 mask 提取。
2. 提升固定验证图的多样性：
   - PanNuke 下按组织类型 round-robin 抽样，优先覆盖更多 type；
   - NCT 下支持按文件名前缀/组织名分组的均匀抽样，避免前 8 张只落在一两类组织。
3. 与旧训练脚本保持尽量兼容：
   - ValidationSet / save_validation_debug_images / create_val_dataloader 的接口保留；
   - 若项目中已有 build_dataset / _is_pannuke_fold_dir，则自动启用 PanNuke 支持；
   - 若没有，则回退到 legacy NCT 逻辑。
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

try:
    from ddpm_dataset import build_dataset, _is_pannuke_fold_dir, NCTDataset
except Exception:
    build_dataset = None
    _is_pannuke_fold_dir = None
    from ddpm_dataset import NCTDataset  # type: ignore

from ddpm_utils import predict_x0_from_noise_shared
from degradation import degrade

_VALID_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def infer_dataset_type_from_path(val_source: Optional[str]) -> str:
    if not val_source or not os.path.exists(val_source):
        return 'nct'
    if _is_pannuke_fold_dir is not None and _is_pannuke_fold_dir(val_source):
        return 'pannuke'
    if os.path.isdir(val_source):
        entries = set(os.listdir(val_source))
        if 'TUM' in entries and 'NORM' in entries:
            return 'nct'
        if _is_pannuke_fold_dir is not None:
            for name in entries:
                cand = os.path.join(val_source, name)
                if os.path.isdir(cand) and _is_pannuke_fold_dir(cand):
                    return 'pannuke'
    return 'nct'


# ------------------------------
# 固定验证样本抽样
# ------------------------------

def _guess_nct_group_name(filename: str) -> str:
    """
    尝试从 NCT 文件名中提取更粗粒度的组织/来源组名，供可视化分层抽样使用。
    规则尽量保守：
    - 优先按第一个分隔符前缀分组，如 Adrenal_gland_xxx.png -> Adrenal_gland
    - 兼容空格 / 连字符 / 多下划线
    - 提取失败时退回完整 stem
    """
    stem = Path(filename).stem
    m = re.match(r'^([A-Za-z]+(?:[_-][A-Za-z]+)*)', stem)
    if m:
        return m.group(1)
    return stem


def _round_robin_pick(group_to_paths: Dict[str, List[str]], total_limit: int, per_group_cap: int, seed: int = 42) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    groups = sorted(group_to_paths.keys())
    pools: Dict[str, List[str]] = {}
    for g in groups:
        pools[g] = list(group_to_paths[g])
        rng.shuffle(pools[g])

    picked: List[Tuple[str, str]] = []
    cursors = {g: 0 for g in groups}
    while len(picked) < total_limit:
        progressed = False
        for g in groups:
            cur = cursors[g]
            pool = pools[g]
            if cur >= min(len(pool), per_group_cap):
                continue
            picked.append((g, pool[cur]))
            cursors[g] = cur + 1
            progressed = True
            if len(picked) >= total_limit:
                break
        if not progressed:
            break
    return picked


def _load_fixed_validation_batch_nct(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
):
    images, labels, titles = [], [], []
    # 总共仍默认 8 张，但改成“组内上限 + round-robin”抽样。
    target_total = max(8, n_per_class * 2)
    per_group_cap = max(1, n_per_class)

    for subdir, lbl in [('TUM', 1), ('NORM', 0)]:
        path = os.path.join(val_dir, subdir)
        if not os.path.exists(path):
            continue
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(_VALID_EXT)])
        group_to_paths: Dict[str, List[str]] = {}
        for f in files:
            group = _guess_nct_group_name(f)
            group_to_paths.setdefault(group, []).append(os.path.join(path, f))

        picked = _round_robin_pick(
            group_to_paths,
            total_limit=n_per_class,
            per_group_cap=per_group_cap,
            seed=42 if lbl == 1 else 43,
        )
        for group, fp in picked:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (sample_size, sample_size), interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
            labels.append(lbl)
            titles.append(group)

    if not images:
        return None, None, None, None

    hr_np = np.stack(images, axis=0).astype(np.float32) / 255.0
    hr = torch.from_numpy(hr_np).permute(0, 3, 1, 2).to(device)

    torch_rng_state = torch.random.get_rng_state()
    py_rng_state = random.getstate()
    torch.manual_seed(42)
    random.seed(42)
    lr = torch.stack([
        degrade(
            hr[i].cpu(), scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter_strength=stain_jitter,
        )
        for i in range(len(hr))
    ]).to(device)
    torch.random.set_rng_state(torch_rng_state)
    random.setstate(py_rng_state)

    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    return hr, lr, labels_t, titles


def _load_fixed_validation_batch_pannuke(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
):
    if build_dataset is None or _is_pannuke_fold_dir is None:
        raise RuntimeError('当前 ddpm_dataset.py 不包含 PanNuke 构建逻辑。')

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
        return None, None, None, None

    type_to_indices: Dict[str, List[int]] = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        type_name = sample.get('type_name', str(int(sample['label'])))
        type_to_indices.setdefault(type_name, []).append(idx)

    # round-robin：优先让前 num_vis 张覆盖更多组织类型。
    rng = random.Random(42)
    for k in type_to_indices:
        rng.shuffle(type_to_indices[k])
    sorted_types = sorted(type_to_indices.keys())
    max_total = max(8, n_per_class * min(4, len(sorted_types)))
    cursors = {k: 0 for k in sorted_types}
    selected_indices: List[int] = []
    while len(selected_indices) < max_total:
        progressed = False
        for type_name in sorted_types:
            cur = cursors[type_name]
            pool = type_to_indices[type_name]
            if cur >= min(len(pool), n_per_class):
                continue
            selected_indices.append(pool[cur])
            cursors[type_name] = cur + 1
            progressed = True
            if len(selected_indices) >= max_total:
                break
        if not progressed:
            break

    if not selected_indices:
        selected_indices = list(range(min(max_total, len(ds))))

    hrs, lrs, labels, titles = [], [], [], []
    for idx in selected_indices:
        sample = ds[idx]
        hrs.append(sample['hr'])
        lrs.append(sample['lr'])
        labels.append(sample['label'])
        titles.append(sample.get('type_name', str(int(sample['label']))))

    return (
        torch.stack(hrs, dim=0).to(device),
        torch.stack(lrs, dim=0).to(device),
        torch.stack(labels, dim=0).to(device),
        titles,
    )


def load_fixed_validation_batch(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
    dataset_type: str = 'auto',
):
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


class ValidationSet:
    def __init__(self, val_dir, scheduler, device='cuda',
                 sample_size=256, fixed_timestep=100, scale=2,
                 blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02),
                 stain_jitter=0.05, dataset_type: str = 'auto'):
        self.val_dir = val_dir
        self.scheduler = scheduler
        self.device = device
        self.sample_size = sample_size
        self.fixed_timestep = fixed_timestep
        self.scale = scale
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range = noise_std_range
        self.stain_jitter = stain_jitter
        self.dataset_type = dataset_type

        self.hr = self.lr = self.noisy_hr = None
        self.labels = self.noise = self.timesteps = None
        self.col_titles = None

    def load(self) -> bool:
        if not self.val_dir or not os.path.exists(self.val_dir):
            return False
        if self.dataset_type == 'auto':
            self.dataset_type = infer_dataset_type_from_path(self.val_dir)
        print(f"Loading fixed validation batch: {self.val_dir} | dataset_type={self.dataset_type}")
        out = load_fixed_validation_batch(
            self.val_dir, self.sample_size, self.device, self.scale,
            blur_sigma_range=self.blur_sigma_range,
            noise_std_range=self.noise_std_range,
            stain_jitter=self.stain_jitter,
            dataset_type=self.dataset_type,
        )
        self.hr, self.lr, self.labels, self.col_titles = out
        if self.hr is None:
            return False
        print(f"  ✓ {self.hr.shape[0]} validation images loaded")
        self._prepare_noise()
        return True

    def _prepare_noise(self):
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        self.noise = torch.randn_like(self.hr)
        self.timesteps = torch.full((self.hr.shape[0],), self.fixed_timestep, device=self.device, dtype=torch.long)
        self.noisy_hr = self.scheduler.add_noise(self.hr, self.noise, self.timesteps)
        torch.random.set_rng_state(torch_rng_state)

    def is_available(self):
        return self.hr is not None

    def generate_reconstructions(self, unet, loss_module=None, use_semantic_injection: bool = False):
        if self.hr is None:
            return None
        with torch.no_grad():
            model_input = torch.cat([self.lr, self.noisy_hr], dim=1)
            sem_tensor = None
            if use_semantic_injection and (loss_module is not None) and hasattr(loss_module, '_run_hovernet'):
                c = loss_module._run_hovernet(self.hr)
                sem_tensor = torch.cat([c['tp_prob'], c['nuc_mask'].unsqueeze(1), c['tp_conf'].unsqueeze(1)], dim=1)

            noise_pred = unet(model_input, self.timesteps, semantic=sem_tensor).sample
            recon = predict_x0_from_noise_shared(self.noisy_hr, noise_pred, self.timesteps, self.scheduler)
            diff_vis = (recon - self.hr).abs().clamp(0, 1)

            B, _, H, W = self.hr.shape
            nr_types = 6
            cls_clean = cls_pred = conf_clean = conf_pred = torch.zeros(B, 1, H, W, device=self.device)
            nuc_mask_clean = torch.zeros(B, 1, H, W, device=self.device)
            nuc_mask_pred = torch.zeros(B, 1, H, W, device=self.device)
            if loss_module is not None and hasattr(loss_module, '_run_hovernet'):
                c = loss_module._run_hovernet(self.hr)
                p = loss_module._run_hovernet(recon)
                nr_types = c['tp_prob'].shape[1]
                cls_clean = c['tp_label'].float().unsqueeze(1)
                cls_pred = p['tp_label'].float().unsqueeze(1)
                conf_clean = c['tp_conf'].unsqueeze(1)
                conf_pred = p['tp_conf'].unsqueeze(1)
                nuc_mask_clean = c['nuc_mask'].unsqueeze(1)
                nuc_mask_pred = p['nuc_mask'].unsqueeze(1)

        return dict(
            reconstructed=recon,
            diff_vis=diff_vis,
            cls_clean=cls_clean,
            cls_pred=cls_pred,
            conf_clean=conf_clean,
            conf_pred=conf_pred,
            nuc_mask_clean=nuc_mask_clean,
            nuc_mask_pred=nuc_mask_pred,
            nr_types=nr_types,
            col_titles=self.col_titles,
        )


# ------------------------------
# 可视化：轮廓优先，减少“看起来偏移”
# ------------------------------

def save_validation_debug_images(
    hr, lr, reconstructed, diff_vis,
    cls_clean, cls_pred,
    conf_clean, conf_pred,
    nuc_mask_clean, nuc_mask_pred,
    epoch, save_dir, num_vis=8, return_tensor=False,
    col_titles=None,
    suptitle: str = None,
    nr_types: int = 6,
):
    os.makedirs(save_dir, exist_ok=True)
    num_vis = min(num_vis, hr.shape[0])

    def _rgb(t):
        return t[:num_vis].detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

    def _gray(t):
        return t[:num_vis, 0].detach().cpu().clamp(0, 1).numpy()

    def _label(t):
        return t[:num_vis, 0].detach().cpu().numpy().astype(np.int32)

    hr_rgb = _rgb(hr)
    lr_rgb = _rgb(lr)
    recon_rgb = _rgb(reconstructed)
    diff_rgb = _rgb(diff_vis)

    cls_clean_int = _label(cls_clean)
    cls_pred_int = _label(cls_pred)
    conf_clean_np = _gray(conf_clean)
    conf_pred_np = _gray(conf_pred)
    nuc_mask_clean_np = _gray(nuc_mask_clean)
    nuc_mask_pred_np = _gray(nuc_mask_pred)

    default_type_info_path = Path(__file__).resolve().parent / 'HoVer-net' / 'type_info.json'
    if default_type_info_path.exists():
        raw_type_info = json.loads(default_type_info_path.read_text(encoding='utf-8'))
    else:
        raw_type_info = {
            '0': ['Background', [0, 0, 0]],
            '1': ['Neoplastic', [255, 0, 0]],
            '2': ['Inflammatory', [0, 255, 0]],
            '3': ['Connective', [0, 0, 255]],
            '4': ['Dead', [255, 255, 0]],
            '5': ['Non-Neoplastic Epithelial', [255, 165, 0]],
        }

    tp_colors_hex = []
    tp_color_map_rgb01 = np.zeros((max(nr_types, 6), 3), dtype=np.float32)
    for tid in range(tp_color_map_rgb01.shape[0]):
        t = raw_type_info.get(str(tid), None)
        rgb = [0, 0, 0] if t is None else t[1]
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        tp_color_map_rgb01[tid] = np.array([r, g, b], dtype=np.float32) / 255.0
        tp_colors_hex.append(f'#{r:02x}{g:02x}{b:02x}')

    def _tp_overlay(base_rgb, tp_label_int, tp_conf_gray, nuc_mask_gray,
                    nuc_thresh=0.5, conf_thresh=0.55,
                    overlay_alpha=0.20, contour_thickness=1):
        """
        先轻量填充，再画轮廓。
        视觉上更接近 HoVer-Net 输出检查图，也更容易看出是否真的“位置偏移”。
        """
        tp_label_int = np.clip(tp_label_int, 0, tp_color_map_rgb01.shape[0] - 1)
        valid = (tp_label_int > 0) & (nuc_mask_gray >= nuc_thresh) & (tp_conf_gray >= conf_thresh)

        out = base_rgb.copy()
        if np.any(valid):
            alpha = (overlay_alpha * np.clip(tp_conf_gray, 0.0, 1.0) * valid.astype(np.float32))[..., None]
            color_map = tp_color_map_rgb01[tp_label_int]
            out = np.clip(out * (1.0 - alpha) + color_map * alpha, 0.0, 1.0)

        out_bgr = (out[..., ::-1] * 255.0).astype(np.uint8)
        for tid in range(1, tp_color_map_rgb01.shape[0]):
            mask = ((tp_label_int == tid) & valid).astype(np.uint8)
            if mask.sum() == 0:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            color_bgr = tuple(int(round(v * 255.0)) for v in tp_color_map_rgb01[tid][::-1])
            cv2.drawContours(out_bgr, contours, -1, color_bgr, thickness=contour_thickness, lineType=cv2.LINE_AA)
        return out_bgr[..., ::-1].astype(np.float32) / 255.0

    overlay_clean = _tp_overlay(hr_rgb, cls_clean_int, conf_clean_np, nuc_mask_clean_np)
    overlay_pred = _tp_overlay(recon_rgb, cls_pred_int, conf_pred_np, nuc_mask_pred_np)

    rows_data = [
        (hr_rgb, 'HR'),
        (lr_rgb, 'LR'),
        (recon_rgb, 'Recon'),
        (diff_rgb, 'Residual'),
        (overlay_clean, 'TP overlay clean'),
        (overlay_pred, 'TP overlay pred'),
        (conf_clean_np, 'tp_conf clean'),
        (conf_pred_np, 'tp_conf pred'),
        (nuc_mask_clean_np, 'nuc_mask clean'),
        (nuc_mask_pred_np, 'nuc_mask pred'),
    ]

    n_rows, n_cols = len(rows_data), num_vis
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.4 * n_rows), squeeze=False)

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

    legend_labels = {
        0: '0:background',
        1: '1:neoplastic',
        2: '2:inflammatory',
        3: '3:connective',
        4: '4:dead/necrosis',
        5: '5:non-neoplastic epi',
    }
    legend_handles = [
        Patch(facecolor=tp_colors_hex[i], label=legend_labels.get(i, f'{i}:class{i}'))
        for i in range(min(6, len(tp_colors_hex)))
    ]
    fig.legend(handles=legend_handles, title='TP classes', loc='upper right', fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.97 if suptitle else 1))
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_val.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if not return_tensor:
        return None

    grid = make_grid(
        torch.cat([hr[:num_vis], lr[:num_vis], reconstructed[:num_vis], diff_vis[:num_vis]]).detach().cpu(),
        nrow=num_vis, padding=2)
    return grid


# ------------------------------
# 定量验证 DataLoader
# ------------------------------

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

    if dataset_type == 'pannuke' and build_dataset is not None and _is_pannuke_fold_dir is not None:
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
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and ('cuda' in str(device)),
            drop_last=False,
        )
        print(f'定量验证集（PanNuke）：{len(val_ds)} 张 patch')
        return dl

    tum_dir = os.path.join(val_vis_dir, 'TUM')
    norm_dir = os.path.join(val_vis_dir, 'NORM')
    if not os.path.exists(tum_dir) or not os.path.exists(norm_dir):
        print('⚠️  验证目录缺少 TUM/NORM 子目录 —— 跳过定量验证。')
        return None

    try:
        val_ds = NCTDataset(
            tum_dir, norm_dir, oversample=oversample, scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
            target_size=target_size,  # 新版 NCTDataset 支持
        )
    except TypeError:
        val_ds = NCTDataset(
            tum_dir, norm_dir, oversample=oversample, scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
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
