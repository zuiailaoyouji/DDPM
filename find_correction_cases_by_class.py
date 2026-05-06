"""
find_correction_cases_by_class.py
按细胞类别 + 按 tissue type 分别筛选典型纠正案例，各出一张对比图。

三组对比：
  HR 基线  : HR 直接送 CellViT，不经过任何纠正
  消融模型 : 无语义监督（只有重建损失）
  本文方法 : 交集 Focal-CE + CellViT 软标签 sem_tensor

【推理配置 · 可视化版】
  INFER_T = 200，单步确定性推理（固定随机种子）

【筛选逻辑（两轮，共用同一次推理，扫描一遍数据集）】

  第一轮：按目标细胞类别筛选（原有逻辑，保持不变）
    对每个目标细胞类别 c（1/2/3/5），找改进最大的 patch：
      - GT 核区域内该类别像素数 > min_pixels
      - HR 基线在该类别上的召回率 < max_hr_recall
      - 本文方法比消融模型在该类别召回率上高 > min_improve
    输出：correction_cases_by_class_ablation.png

  第二轮:按 tissue type 筛选(新增)
    对每个 tissue type,找整体 Overall_Acc 提升最大的 patch
    (Full Overall_Acc − HR Overall_Acc 最大,且 > TISSUE_MIN_IMPROVE)。
    输出:correction_cases_by_tissue_ablation.png

可视化每个案例 8 列：
  HR | 消融模型 | 本文方法 |（空列）|
  GT overlay | HR pred | 消融 pred | 本文 pred
"""

import colorsys
import math
import os
import random
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from diffusers import DDPMScheduler

from ddpm_dataset import PanNukeDataset
from unet_wrapper import create_model
from ddpm_utils import load_cellvit, predict_x0_from_noise_shared
from semantic_sr_loss import run_cellvit, build_sem_tensor_from_cellvit

device      = 'cuda'
CLASS_NAMES = ['Background', 'Neoplastic', 'Inflammatory',
               'Connective', 'Dead', 'Epithelial']
CLASS_COLORS = [
    [0,   0,   0  ],
    [255, 0,   0  ],
    [0,   0,   255],
    [255, 255, 0  ],
    [0,   255, 0  ],
    [255, 0,   255],
]
COLOR_MAP = np.array(CLASS_COLORS, dtype=np.float32) / 255.0

# ── 第一轮参数：按细胞类别 ───────────────────────────────────────────
TARGET_CLASSES = {
    1: dict(name='Neoplastic',   min_pixels=300, max_hr_recall=0.75,
            min_improve=0.02, min_cls_ratio=0.35),
    2: dict(name='Inflammatory', min_pixels=50,  max_hr_recall=0.50,
            min_improve=0.02, min_cls_ratio=0.10),
    3: dict(name='Connective',   min_pixels=300, max_hr_recall=0.80,
            min_improve=0.02, min_cls_ratio=0.35),
    5: dict(name='Epithelial',   min_pixels=300, max_hr_recall=0.80,
            min_improve=0.02, min_cls_ratio=0.35),
}

# ── 第二轮参数：按 tissue type ────────────────────────────────────────
TISSUE_MIN_NUC_PIXELS = 100   # GT 核区域最少像素数
TISSUE_MIN_IMPROVE    = 0.005 # Full Overall_Acc − HR Overall_Acc 最低提升阈值

# ── 加载模型 ─────────────────────────────────────────────────────────
print("加载 CellViT...")
cellvit = load_cellvit(
    model_path        = '/home/xuwen/DDPM/CellViT/CellViT-SAM-H-x40.pth',
    cellvit_repo_path = '/home/xuwen/DDPM/CellViT',
    device            = device,
    variant           = 'sam_h',
)


def load_unet(ckpt_path, use_semantic):
    unet = create_model(use_semantic=use_semantic).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    return unet


print("加载 UNet 模型...")
unet_ablation = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_ablation.pth",
    use_semantic=False,
)
unet_full = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_correction.pth",
    use_semantic=True,
)

scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset   = PanNukeDataset(
    fold_dirs   = ['/data/xuwen/PanNuke/Fold 3'],
    target_size = 256,
)
print(f"测试集大小: {len(dataset)} 张\n")

INFER_T = 200
torch.manual_seed(42)
random.seed(42)

# ── 构建 round-robin 扫描顺序（保证所有 tissue 都被早期访问到）────────
_type_to_idx = defaultdict(list)
for _i in range(len(dataset)):
    _s = dataset[_i]
    if _s['gt_nuc_mask'].bool().sum() < TISSUE_MIN_NUC_PIXELS:
        continue
    _type_to_idx[_s['type_name']].append(_i)

all_tissue_types = sorted(_type_to_idx.keys())
print(f"共 {len(all_tissue_types)} 种 tissue type，有效样本分布：")
for _t in all_tissue_types:
    print(f"  {_t:<25} {len(_type_to_idx[_t]):>5} 张")

_ptrs = {t: 0 for t in all_tissue_types}
scan_indices = []
while True:
    added = False
    for _t in all_tissue_types:
        p = _ptrs[_t]
        if p < len(_type_to_idx[_t]):
            scan_indices.append(_type_to_idx[_t][p])
            _ptrs[_t] += 1
            added = True
    if not added:
        break

print(f"\nRound-robin 扫描顺序：共 {len(scan_indices)} 张\n")


# ── 辅助函数 ─────────────────────────────────────────────────────────

def class_recall(pred_np, gt_np, cls_id, mask):
    region = mask & (gt_np == cls_id)
    if region.sum() == 0:
        return 0.0
    return float((pred_np[region] == cls_id).mean())


def overall_acc(pred_np, gt_np, mask):
    if mask.sum() == 0:
        return 0.0
    return float((pred_np[mask] == gt_np[mask]).mean())


def make_overlay(img_rgb, label_map, nuc_prob, alpha=0.65):
    label_map = np.clip(label_map, 0, len(COLOR_MAP) - 1)
    color = COLOR_MAP[label_map]
    valid = (label_map > 0) & (nuc_prob > 0.3)
    a     = (alpha * nuc_prob * valid)[..., None].clip(0, 1)
    return (img_rgb * (1 - a) + color * a).clip(0, 1)


def to_rgb(t):
    return t.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def infer_once(unet, hr, noisy_hr, t, sem=None):
    with torch.no_grad():
        x0  = predict_x0_from_noise_shared(
            noisy_hr,
            unet(torch.cat([hr, noisy_hr], dim=1), t, semantic=sem).sample,
            t, scheduler,
        )
        cv  = run_cellvit(cellvit, x0)
    lbl      = cv['nuclei_type_label'].squeeze(0).cpu().numpy().astype(np.int32)
    nuc_prob = cv['nuclei_nuc_prob'].squeeze().cpu().numpy()
    return x0, lbl, nuc_prob


def pack_case(i, type_name, hr, x0_abl, x0_full,
              gt_np, gt_nuc_np, hr_np, hr_nuc,
              abl_np, abl_nuc, full_np, full_nuc, **extra):
    return dict(
        idx       = i,
        type_name = type_name,
        hr_rgb    = to_rgb(hr),
        abl_rgb   = to_rgb(x0_abl),
        full_rgb  = to_rgb(x0_full),
        gt_lbl    = gt_np,
        gt_nuc    = gt_nuc_np.astype(np.float32),
        hr_lbl    = hr_np,
        hr_nuc    = hr_nuc,
        abl_lbl   = abl_np,
        abl_nuc   = abl_nuc,
        full_lbl  = full_np,
        full_nuc  = full_nuc,
        **extra,
    )


def tissue_color(tname):
    idx = all_tissue_types.index(tname)
    h   = idx / max(len(all_tissue_types), 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.85)
    return [r, g, b]


# ── 初始化两轮的最佳案例容器 ─────────────────────────────────────────
best_cls_case    = {cls_id: None  for cls_id in TARGET_CLASSES}
best_cls_improve = {cls_id: -1.0  for cls_id in TARGET_CLASSES}

best_tissue_case    = {t: None  for t in all_tissue_types}
best_tissue_improve = {t: -1.0  for t in all_tissue_types}

# ── 统一扫描：两轮共用同一次推理 ─────────────────────────────────────
print("扫描测试集，双轮同步筛选中...\n")

for i in scan_indices:

    sample    = dataset[i]
    hr_cpu    = sample['hr']
    gt_lbl    = sample['gt_label_map']
    gt_nuc    = sample['gt_nuc_mask'].bool()
    type_name = sample['type_name']

    if gt_nuc.sum() < TISSUE_MIN_NUC_PIXELS:
        continue

    gt_np     = gt_lbl.numpy()
    gt_nuc_np = gt_nuc.numpy()
    cell_mask = gt_nuc_np & (gt_np > 0)

    hr = hr_cpu.unsqueeze(0).to(device)
    t  = torch.tensor([INFER_T], device=device)

    # 消融和完整模型共用同一次加噪
    noise    = torch.randn_like(hr)
    noisy_hr = scheduler.add_noise(hr, noise, t)

    with torch.no_grad():
        hr_cv  = run_cellvit(cellvit, hr)
        hr_lbl_t = hr_cv['nuclei_type_label'].squeeze(0).cpu()
        hr_nuc = hr_cv['nuclei_nuc_prob'].squeeze().cpu().numpy()
        sem    = build_sem_tensor_from_cellvit(
            hr_cv['nuclei_type_prob'],
            hr_cv['nuclei_nuc_prob'],
        )

    hr_np = hr_lbl_t.numpy()

    x0_abl,  abl_np,  abl_nuc  = infer_once(unet_ablation, hr, noisy_hr, t, sem=None)
    x0_full, full_np, full_nuc = infer_once(unet_full,     hr, noisy_hr, t, sem=sem)

    # ── 第一轮：按细胞类别 ───────────────────────────────────────────
    for cls_id, cfg in TARGET_CLASSES.items():
        cls_pixels = int(((gt_np == cls_id) & gt_nuc_np).sum())
        if cls_pixels < cfg['min_pixels']:
            continue
        cls_ratio = cls_pixels / max(gt_nuc_np.sum(), 1)
        if cls_ratio < cfg['min_cls_ratio']:
            continue

        hr_recall   = class_recall(hr_np,   gt_np, cls_id, gt_nuc_np)
        abl_recall  = class_recall(abl_np,  gt_np, cls_id, gt_nuc_np)
        full_recall = class_recall(full_np, gt_np, cls_id, gt_nuc_np)
        improve     = full_recall - abl_recall

        if hr_recall > cfg['max_hr_recall']:
            continue
        if improve < cfg['min_improve']:
            continue

        if improve > best_cls_improve[cls_id]:
            best_cls_improve[cls_id] = improve
            best_cls_case[cls_id] = pack_case(
                i, type_name, hr, x0_abl, x0_full,
                gt_np, gt_nuc_np, hr_np, hr_nuc,
                abl_np, abl_nuc, full_np, full_nuc,
                target_cls  = cls_id,
                hr_recall   = hr_recall,
                abl_recall  = abl_recall,
                full_recall = full_recall,
                improvement = improve,
                cls_pixels  = cls_pixels,
                cls_ratio   = float(cls_ratio),
            )
            print(f"  [细胞类别] 更新 {cfg['name']:>14} | 样本{i:>4} | "
                  f"tissue={type_name:<20} "
                  f"HR={hr_recall:.3f}  Abl={abl_recall:.3f}  "
                  f"Full={full_recall:.3f}  Δ={improve:+.3f}")

    # ── 第二轮：按 tissue type（整体 Overall_Acc） ─────────────────────
    oa_hr   = overall_acc(hr_np,   gt_np, cell_mask)
    oa_abl  = overall_acc(abl_np,  gt_np, cell_mask)
    oa_full = overall_acc(full_np, gt_np, cell_mask)
    tissue_improve = oa_full - oa_hr   # ← 主表叙事：DGFR 相对 HR 基线的整体改进

    if (tissue_improve >= TISSUE_MIN_IMPROVE and
            tissue_improve > best_tissue_improve[type_name]):
        best_tissue_improve[type_name] = tissue_improve
        best_tissue_case[type_name] = pack_case(
            i, type_name, hr, x0_abl, x0_full,
            gt_np, gt_nuc_np, hr_np, hr_nuc,
            abl_np, abl_nuc, full_np, full_nuc,
            oa_hr       = oa_hr,
            oa_abl      = oa_abl,
            oa_full     = oa_full,
            improvement = tissue_improve,
        )
        print(f"  [Tissue]   更新 {type_name:<22} | 样本{i:>4} | "
              f"HR={oa_hr:.3f}  Abl={oa_abl:.3f}  "
              f"Full={oa_full:.3f}  Δ(Full−HR)={tissue_improve:+.3f}")


# ── 通用绘图函数 ──────────────────────────────────────────────────────

def draw_cases(cases, fig_title, save_path,
               row_label_fn, info_text_fn, border_color_fn):
    n_rows = len(cases)
    if n_rows == 0:
        print(f"  ⚠️  无有效案例，跳过：{save_path}")
        return

    col_titles = [
        'HR', 'Ablation', 'Full model (ours)', '',
        'GT overlay', 'HR pred', 'Ablation pred', 'Full pred (ours)',
    ]
    n_cols = len(col_titles)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    fig.suptitle(fig_title, fontsize=12, y=1.01)

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=10, pad=6)

    for ri, case in enumerate(cases):
        bc = border_color_fn(case)

        axes[ri, 0].text(
            0.02, 0.98, row_label_fn(case),
            transform=axes[ri, 0].transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left',
            color=bc,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'),
        )
        axes[ri, 4].text(
            0.02, 0.98, info_text_fn(case),
            transform=axes[ri, 4].transAxes,
            fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.80, ec='none'),
        )

        for ci, rgb in enumerate([case['hr_rgb'], case['abl_rgb'], case['full_rgb']]):
            axes[ri, ci].imshow(rgb)
            axes[ri, ci].axis('off')
        axes[ri, 3].axis('off')

        axes[ri, 4].imshow(make_overlay(case['hr_rgb'],  case['gt_lbl'],  case['gt_nuc']))
        axes[ri, 4].axis('off')
        axes[ri, 5].imshow(make_overlay(case['hr_rgb'],  case['hr_lbl'],  case['hr_nuc']))
        axes[ri, 5].axis('off')
        axes[ri, 6].imshow(make_overlay(case['abl_rgb'], case['abl_lbl'], case['abl_nuc']))
        axes[ri, 6].axis('off')
        axes[ri, 7].imshow(make_overlay(case['full_rgb'],case['full_lbl'],case['full_nuc']))
        axes[ri, 7].axis('off')

        for col in [4, 5, 6, 7]:
            for spine in axes[ri, col].spines.values():
                spine.set_edgecolor(bc)
                spine.set_linewidth(3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


# ── 输出第一张图：按细胞类别 ─────────────────────────────────────────
found_cls_cases = []
for cls_id in [1, 2, 3, 5]:
    if best_cls_case[cls_id] is not None:
        found_cls_cases.append(best_cls_case[cls_id])
    else:
        print(f"⚠️  未找到 {TARGET_CLASSES[cls_id]['name']} 案例，"
              f"请尝试降低 min_improve 或 min_pixels 阈值。")

draw_cases(
    cases           = found_cls_cases,
    fig_title       = (f'Per-class correction cases — t={INFER_T}, single-step inference\n'
                       'Full model improves over ablation '
                       '(evaluated on GT nucleus region vs GT label)'),
    save_path       = './logs/downstream_Cellvitsamh/correction_cases_by_class_ablation.png',
    row_label_fn    = lambda c: (f"[{CLASS_NAMES[c['target_cls']]}]\n"
                                 f"tissue={c['type_name']}"),
    info_text_fn    = lambda c: (f"HR={c['hr_recall']:.3f}\n"
                                 f"Abl={c['abl_recall']:.3f}\n"
                                 f"Full={c['full_recall']:.3f}  "
                                 f"Δ={c['improvement']:+.3f}"),
    border_color_fn = lambda c: COLOR_MAP[c['target_cls']].tolist(),
)

# 图例
if found_cls_cases:
    legend_patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[c]) / 255., label=CLASS_NAMES[c])
        for c in [1, 2, 3, 5]
    ]
    fig_l, ax_l = plt.subplots(figsize=(7, 0.6))
    ax_l.axis('off')
    fig_l.legend(handles=legend_patches, title='Cell type (focus)',
                 loc='center', ncol=4, fontsize=9)
    plt.tight_layout()
    legend_path = './logs/downstream_Cellvitsamh/correction_cases_by_class_legend.png'
    os.makedirs(os.path.dirname(legend_path) or '.', exist_ok=True)
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()

# ── 输出第二张图：按 tissue type ──────────────────────────────────────
found_tissue_cases = []
missing_tissues    = []
for tname in all_tissue_types:
    if best_tissue_case[tname] is not None:
        found_tissue_cases.append(best_tissue_case[tname])
    else:
        missing_tissues.append(tname)

if missing_tissues:
    print(f"\n⚠️  以下 tissue 未找到满足条件的案例"
          f"（可降低 TISSUE_MIN_IMPROVE={TISSUE_MIN_IMPROVE}）：")
    for t in missing_tissues:
        print(f"    {t}")

draw_cases(
    cases           = found_tissue_cases,
    fig_title       = (f'Per-tissue correction cases — t={INFER_T}, single-step inference\n'
                       'Best overall Overall_Acc improvement: Full vs HR baseline '
                       '(GT nucleus region)'),
    save_path       = './logs/downstream_Cellvitsamh/correction_cases_by_tissue_ablation.png',
    row_label_fn    = lambda c: (f"[{c['type_name']}]\n"
                                 f"idx={c['idx']}"),
    info_text_fn    = lambda c: (f"HR={c['oa_hr']:.3f}\n"
                                 f"Abl={c['oa_abl']:.3f}\n"
                                 f"Full={c['oa_full']:.3f}  "
                                 f"Δ(Full−HR)={c['improvement']:+.3f}"),
    border_color_fn = lambda c: tissue_color(c['type_name']),
)

# ── 汇总打印 ─────────────────────────────────────────────────────────
W = 90
print(f"\n{'='*W}")
print("第一轮：按细胞类别典型案例汇总")
print(f"{'类别':>14}  {'Tissue':<22}  {'样本':>5}  "
      f"{'HR召回':>8}  {'消融召回':>9}  {'本文召回':>9}  {'Δ':>8}")
print("-" * W)
for case in found_cls_cases:
    print(f"{CLASS_NAMES[case['target_cls']]:>14}  "
          f"{case['type_name']:<22}  "
          f"{case['idx']:>5}  "
          f"{case['hr_recall']:>8.4f}  "
          f"{case['abl_recall']:>9.4f}  "
          f"{case['full_recall']:>9.4f}  "
          f"{case['improvement']:>+8.4f}")

print(f"\n{'='*W}")
print("第二轮:按 Tissue Type 典型案例汇总(整体 Overall_Acc)")
print(f"{'Tissue':<22}  {'样本':>5}  "
      f"{'HR Acc':>8}  {'消融 Acc':>9}  {'本文 Acc':>9}  {'Δ(Full−HR)':>10}")
print("-" * W)
for case in found_tissue_cases:
    print(f"{case['type_name']:<22}  "
          f"{case['idx']:>5}  "
          f"{case['oa_hr']:>8.4f}  "
          f"{case['oa_abl']:>9.4f}  "
          f"{case['oa_full']:>9.4f}  "
          f"{case['improvement']:>+10.4f}")
for t in missing_tissues:
    print(f"{t:<22}  {'—':>5}  {'未找到满足条件的案例'}")
print(f"{'='*W}")