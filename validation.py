"""
validation.py
语义引导型 SR DDPM 的验证集管理。

当前版本特点
----------------
1. 同时兼容 legacy NCT 目录结构与 PanNuke fold(.npy) 结构。
2. 可视化固定样本不再强依赖 TUM/NORM；若传入的是 PanNuke fold，会直接从 fold 中抽样。
3. 定量验证 DataLoader 支持通过 dataset_type 显式指定，也支持基于路径自动推断。
"""

import os
import random
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ddpm_dataset import build_dataset, _is_pannuke_fold_dir
from ddpm_utils import predict_x0_from_noise_shared
from degradation import degrade


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：自动推断验证源类型
# ─────────────────────────────────────────────────────────────────────────────

def infer_dataset_type_from_path(val_source: Optional[str]) -> str:
    """
    根据路径粗略推断验证数据源类型：
      - 若能识别为 PanNuke fold/root，则返回 'pannuke'
      - 若存在 TUM/NORM 子目录，则返回 'nct'
      - 否则默认 'nct'
    """
    if not val_source or not os.path.exists(val_source):
        return 'nct'

    if _is_pannuke_fold_dir(val_source):
        return 'pannuke'

    if os.path.isdir(val_source):
        entries = set(os.listdir(val_source))
        if 'TUM' in entries and 'NORM' in entries:
            return 'nct'
        # 也兼容传入 PanNuke 根目录（其下有 Fold 1/Fold 2/...）
        for name in entries:
            cand = os.path.join(val_source, name)
            if _is_pannuke_fold_dir(cand):
                return 'pannuke'

    return 'nct'


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：加载一小批固定验证样本用于可视化
# ─────────────────────────────────────────────────────────────────────────────

def _load_fixed_validation_batch_nct(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
):
    """从 TUM/NORM 中各取 n_per_class 张图像作为固定可视化样本。"""
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
    titles = [f'TUM-{i+1}' for i in range(sum(1 for x in labels if x == 1))] + \
             [f'NORM-{i+1}' for i in range(sum(1 for x in labels if x == 0))]
    return hr, lr, labels_t, titles


def _load_fixed_validation_batch_pannuke(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
):
    """
    从 PanNuke fold/root 中抽取固定可视化样本。
    优先按组织类型均匀抽样；若某些类型不足，则按顺序补足。
    """
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

    # 建立 type_name -> indices
    type_to_indices = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        type_name = sample.get('type_name', str(int(sample['label'])))
        type_to_indices.setdefault(type_name, []).append(idx)

    selected_indices = []
    for type_name in sorted(type_to_indices.keys()):
        selected_indices.extend(type_to_indices[type_name][:n_per_class])

    # 控制总数，避免过多；至少保留若干类型样本
    max_total = max(8, n_per_class * min(4, len(type_to_indices)))
    selected_indices = selected_indices[:max_total]

    if not selected_indices:
        selected_indices = list(range(min(max_total, len(ds))))

    hrs, lrs, labels, titles = [], [], [], []
    for idx in selected_indices:
        sample = ds[idx]
        hr = sample['hr']
        lr = sample['lr']
        label = sample['label']
        type_name = sample.get('type_name', str(int(label)))

        hrs.append(hr)
        lrs.append(lr)
        labels.append(label)
        titles.append(type_name)

    hr_t = torch.stack(hrs, dim=0).to(device)
    lr_t = torch.stack(lrs, dim=0).to(device)
    labels_t = torch.stack(labels, dim=0).to(device)
    return hr_t, lr_t, labels_t, titles


def load_fixed_validation_batch(
    val_dir, sample_size=256, device='cuda',
    scale=2, n_per_class=4,
    blur_sigma_range=(0.5, 1.5), noise_std_range=(0.0, 0.02), stain_jitter=0.05,
    dataset_type: str = 'auto',
):
    """
    加载一小批固定验证样本用于可视化。

    返回
    -------
    hr_tensor  : [B, 3, H, W]
    lr_tensor  : [B, 3, H, W]
    labels     : [B]
    titles     : list[str]，用于列标题
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
# ValidationSet：用于可视化的一小批固定验证样本
# ─────────────────────────────────────────────────────────────────────────────

class ValidationSet:
    """
    管理一小批固定的验证样本，用于主观可视化。
    在 load() 之后固定 LR、噪声和时间步，以保证不同 epoch 之间的公平对比。
    """

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
        self.dataset_type = infer_dataset_type_from_path(self.val_dir) if self.dataset_type == 'auto' else self.dataset_type
        print(f"Loading fixed validation batch: {self.val_dir} | dataset_type={self.dataset_type}")
        self.hr, self.lr, self.labels, self.col_titles = load_fixed_validation_batch(
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
        self.noise = torch.randn_like(self.hr)
        self.timesteps = torch.full(
            (self.hr.shape[0],), self.fixed_timestep,
            device=self.device, dtype=torch.long)
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
                tp_prob = c['tp_prob']
                sem_tensor = torch.cat(
                    [tp_prob, c['nuc_mask'].unsqueeze(1), c['tp_conf'].unsqueeze(1)],
                    dim=1,
                )

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


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

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
    raw_type_info = json.loads(default_type_info_path.read_text(encoding='utf-8'))

    tp_colors_hex = []
    tp_color_map_rgb01 = np.zeros((6, 3), dtype=np.float32)
    for tid in range(6):
        t = raw_type_info.get(str(tid), None)
        rgb = [0, 0, 0] if t is None else t[1]
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        tp_color_map_rgb01[tid] = np.array([r, g, b], dtype=np.float32) / 255.0
        tp_colors_hex.append(f'#{r:02x}{g:02x}{b:02x}')

    def _tp_overlay(base_rgb, tp_label_int, tp_conf_gray, nuc_mask_gray, overlay_alpha=0.6):
        tp_label_int = np.clip(tp_label_int, 0, tp_color_map_rgb01.shape[0] - 1)
        color_map = tp_color_map_rgb01[tp_label_int]
        valid_mask = (tp_label_int > 0) & (nuc_mask_gray > 0)
        alpha = overlay_alpha * tp_conf_gray * valid_mask.astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        return base_rgb * (1.0 - alpha) + color_map * alpha

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
        (nuc_mask_pred_np, 'nuc_mask pred'),
    ]

    n_rows, n_cols = len(rows_data), num_vis
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.6 * n_rows), squeeze=False)

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
        Patch(facecolor=tp_colors_hex[0], label='0:background(nolabe)'),
        Patch(facecolor=tp_colors_hex[1], label='1:neopla'),
        Patch(facecolor=tp_colors_hex[2], label='2:inflam'),
        Patch(facecolor=tp_colors_hex[3], label='3:connec'),
        Patch(facecolor=tp_colors_hex[4], label='4:necros'),
        Patch(facecolor=tp_colors_hex[5], label='5:no-neo'),
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
    """
    构建定量验证 DataLoader。

    - NCT: 需要 val_vis_dir/TUM 与 val_vis_dir/NORM
    - PanNuke: val_vis_dir 可以是单个 Fold 目录，也可以是 PanNuke 根目录
    """
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

    tum_dir = os.path.join(val_vis_dir, 'TUM')
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
