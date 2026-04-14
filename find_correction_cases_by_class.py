"""
find_correction_cases_by_class.py
按细胞类别定向筛选典型纠正案例，保证每个目标类别至少出现一个代表性案例。

三组对比：
  HR 基线  : HR 直接送 CellViT，不经过任何纠正
  消融模型 : 无语义监督（只有重建损失）
  本文方法 : 交集 Focal-CE + CellViT 软标签 sem_tensor

【推理配置 · 可视化版】
  INFER_T = 200，单步确定性推理（固定随机种子）
  t=200 加噪幅度适中，图像改动清晰可见，overlay 对比效果好。
  不做集成，保留单次推理图像的真实感，避免均值模糊。

筛选策略：
  对每个目标类别 c，找满足以下条件的 patch：
    1. GT 核区域内该类别像素数 > min_pixels
    2. HR 基线在该类别上的召回率 < max_hr_recall（说明存在误分类）
    3. 本文方法在该类别召回率上比消融模型高 > min_improve

目标类别：
    1: Neoplastic（红）  2: Inflammatory（蓝）
    3: Connective（黄）  5: Epithelial（品红）

可视化每个案例 8 列：
  HR | 消融模型 | 本文方法 |（空列）|
  GT overlay | HR pred | 消融 pred | 本文 pred
"""

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── CellViT ─────────────────────────────────────────────────────────
print("加载 CellViT...")
cellvit = load_cellvit(
    model_path        = '/home/xuwen/DDPM/CellViT/CellViT-256-x40.pth',
    cellvit_repo_path = '/home/xuwen/DDPM/CellViT',
    device            = device,
)


# ── UNet ────────────────────────────────────────────────────────────
def load_unet(ckpt_path: str, use_semantic: bool) -> torch.nn.Module:
    unet = create_model(use_semantic=use_semantic).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    return unet


print("加载 UNet 模型...")
unet_ablation = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction/best_unet_ablation.pth",
    use_semantic=False,
)
unet_full = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction/best_unet_correction.pth",
    use_semantic=True,
)

scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset   = PanNukeDataset(
    fold_dirs   = ['/data/xuwen/PanNuke/Fold 3'],
    target_size = 256,
)
print(f"测试集大小: {len(dataset)} 张\n")

INFER_T = 200   # 可视化：适中加噪幅度，图像改动清晰可见
torch.manual_seed(42)


# ── 辅助函数 ─────────────────────────────────────────────────────────

def class_recall(pred_np, gt_np, cls_id, mask):
    region = mask & (gt_np == cls_id)
    if region.sum() == 0:
        return 0.0
    return (pred_np[region] == cls_id).mean()


def make_overlay(img_rgb, label_map, nuc_prob, alpha=0.65):
    label_map = np.clip(label_map, 0, len(COLOR_MAP) - 1)
    color = COLOR_MAP[label_map]
    valid = (label_map > 0) & (nuc_prob > 0.3)
    a     = (alpha * nuc_prob * valid)[..., None].clip(0, 1)
    return (img_rgb * (1 - a) + color * a).clip(0, 1)


def to_rgb(t):
    return t.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


# ── 单步推理（可视化） ───────────────────────────────────────────────
def infer_once(unet, hr, noisy_hr, t, sem=None):
    """固定加噪输入，单步确定性推理，返回 x0 和 CellViT 结果。"""
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


# ── 按类别定向筛选 ───────────────────────────────────────────────────
best_case    = {cls_id: None  for cls_id in TARGET_CLASSES}
best_improve = {cls_id: -1.0  for cls_id in TARGET_CLASSES}

print("扫描测试集，按类别定向筛选...")

for i in range(len(dataset)):
    if all(v is not None for v in best_case.values()):
        break

    sample  = dataset[i]
    hr_cpu  = sample['hr']
    gt_lbl  = sample['gt_label_map']
    gt_nuc  = sample['gt_nuc_mask'].bool()

    if gt_nuc.sum() < 100:
        continue

    gt_np     = gt_lbl.numpy()
    gt_nuc_np = gt_nuc.numpy()

    hr = hr_cpu.unsqueeze(0).to(device)
    t  = torch.tensor([INFER_T], device=device)

    # 消融和完整模型共用同一次加噪，保证对比公平
    noise    = torch.randn_like(hr)
    noisy_hr = scheduler.add_noise(hr, noise, t)

    with torch.no_grad():
        hr_cv  = run_cellvit(cellvit, hr)
        hr_lbl = hr_cv['nuclei_type_label'].squeeze(0).cpu()
        hr_nuc = hr_cv['nuclei_nuc_prob'].squeeze().cpu().numpy()
        sem    = build_sem_tensor_from_cellvit(
            hr_cv['nuclei_type_prob'],
            hr_cv['nuclei_nuc_prob'],
        )

    hr_np = hr_lbl.numpy()

    x0_abl,  abl_np,  abl_nuc  = infer_once(unet_ablation, hr, noisy_hr, t, sem=None)
    x0_full, full_np, full_nuc = infer_once(unet_full,     hr, noisy_hr, t, sem=sem)

    for cls_id, cfg in TARGET_CLASSES.items():
        cls_pixels = ((gt_np == cls_id) & gt_nuc_np).sum()
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

        if improve > best_improve[cls_id]:
            best_improve[cls_id] = improve
            best_case[cls_id] = dict(
                idx         = i,
                target_cls  = cls_id,
                hr_recall   = hr_recall,
                abl_recall  = abl_recall,
                full_recall = full_recall,
                improvement = improve,
                cls_pixels  = int(cls_pixels),
                cls_ratio   = float(cls_ratio),
                hr_rgb      = to_rgb(hr),
                abl_rgb     = to_rgb(x0_abl),
                full_rgb    = to_rgb(x0_full),
                gt_lbl      = gt_np,
                gt_nuc      = gt_nuc_np.astype(np.float32),
                hr_lbl      = hr_np,
                hr_nuc      = hr_nuc,
                abl_lbl     = abl_np,
                abl_nuc     = abl_nuc,
                full_lbl    = full_np,
                full_nuc    = full_nuc,
            )
            print(f"  更新 {cfg['name']:>14} 最佳案例 | 样本{i:>4} | "
                  f"ratio={cls_ratio:.2f}  "
                  f"HR={hr_recall:.3f}  Abl={abl_recall:.3f}  "
                  f"Full={full_recall:.3f}  Δ={improve:+.3f}")

# ── 汇总找到的案例 ───────────────────────────────────────────────────
found_cases = []
for cls_id in [1, 2, 3, 5]:
    if best_case[cls_id] is not None:
        found_cases.append(best_case[cls_id])
    else:
        print(f"⚠️  未找到 {TARGET_CLASSES[cls_id]['name']} 的案例，"
              f"请尝试降低 min_improve 或 min_pixels 阈值。")

if not found_cases:
    print("未找到任何案例，请调整筛选参数。")
    exit()

print(f"\n共找到 {len(found_cases)} 个典型案例，开始绘图...")

# ── 绘图 ────────────────────────────────────────────────────────────
col_titles = [
    'HR', 'Ablation', 'Full model (ours)', '',
    'GT overlay', 'HR pred', 'Ablation pred', 'Full pred (ours)',
]
n_cols = len(col_titles)
n_rows = len(found_cases)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.2 * n_cols, 3.8 * n_rows),
    squeeze=False,
)
fig.suptitle(
    f'Per-class correction cases — t={INFER_T}, single-step inference\n'
    'Full model improves over ablation (evaluated on GT nucleus region vs GT label)',
    fontsize=13, y=1.01,
)

for ci, title in enumerate(col_titles):
    axes[0, ci].set_title(title, fontsize=10, pad=6)

for ri, case in enumerate(found_cases):
    cls_id    = case['target_cls']
    cls_color = COLOR_MAP[cls_id].tolist()

    axes[ri, 0].text(
        0.02, 0.98, f"[{CLASS_NAMES[cls_id]}]",
        transform=axes[ri, 0].transAxes,
        fontsize=10, fontweight='bold', va='top', ha='left',
        color=cls_color,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'),
    )
    axes[ri, 4].text(
        0.02, 0.98,
        f"HR={case['hr_recall']:.3f}\n"
        f"Abl={case['abl_recall']:.3f}\n"
        f"Full={case['full_recall']:.3f}  Δ={case['improvement']:+.3f}",
        transform=axes[ri, 4].transAxes,
        fontsize=7.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.80, ec='none'),
    )

    # 列 0-2：图像，列 3 空白
    for ci, rgb in enumerate([case['hr_rgb'], case['abl_rgb'], case['full_rgb']]):
        axes[ri, ci].imshow(rgb)
        axes[ri, ci].axis('off')
    axes[ri, 3].axis('off')

    # 列 4：GT overlay
    axes[ri, 4].imshow(
        make_overlay(case['hr_rgb'], case['gt_lbl'], case['gt_nuc'])
    )
    axes[ri, 4].axis('off')

    # 列 5：HR pred
    axes[ri, 5].imshow(
        make_overlay(case['hr_rgb'], case['hr_lbl'], case['hr_nuc'])
    )
    axes[ri, 5].axis('off')

    # 列 6：消融 pred
    axes[ri, 6].imshow(
        make_overlay(case['abl_rgb'], case['abl_lbl'], case['abl_nuc'])
    )
    axes[ri, 6].axis('off')

    # 列 7：本文 pred
    axes[ri, 7].imshow(
        make_overlay(case['full_rgb'], case['full_lbl'], case['full_nuc'])
    )
    axes[ri, 7].axis('off')

    for col in [4, 5, 6, 7]:
        for spine in axes[ri, col].spines.values():
            spine.set_edgecolor(COLOR_MAP[cls_id])
            spine.set_linewidth(3)

legend_patches = [
    mpatches.Patch(color=np.array(CLASS_COLORS[c]) / 255., label=CLASS_NAMES[c])
    for c in [1, 2, 3, 5]
]
fig.legend(
    handles=legend_patches, title='Cell type (focus)',
    loc='lower center', ncol=len(legend_patches),
    fontsize=9, bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
os.makedirs('./logs', exist_ok=True)
save_path = './logs/correction_cases_by_class_ablation.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"按类别典型案例图已保存到: {save_path}")

# ── 打印汇总 ────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"{'类别':>14}  {'样本':>6}  {'GT像素':>8}  {'占比':>6}  "
      f"{'HR召回':>8}  {'消融召回':>9}  {'本文召回':>9}  {'Δ':>8}")
print("-" * 85)
for case in found_cases:
    print(f"{CLASS_NAMES[case['target_cls']]:>14}  "
          f"{case['idx']:>6}  "
          f"{case['cls_pixels']:>8}  "
          f"{case['cls_ratio']:>6.2f}  "
          f"{case['hr_recall']:>8.4f}  "
          f"{case['abl_recall']:>9.4f}  "
          f"{case['full_recall']:>9.4f}  "
          f"{case['improvement']:>+8.4f}")