"""
find_correction_cases_by_class.py
按细胞类别定向筛选典型纠正案例，保证每个目标类别至少出现一个代表性案例。

筛选策略：
  对每个目标类别 c，找满足以下条件的 patch：
    1. GT 核区域内该类别像素数 > MIN_CLASS_PIXELS
    2. LR pred 在该类别上的召回率 < MAX_LR_RECALL（退化导致误分类）
    3. Semantic SR 在该类别召回率上比 Baseline SR 高 > MIN_SM_IMPROVE

目标类别：
    1: Neoplastic（红）
    2: Inflammatory（蓝）
    3: Connective（黄）
    5: Epithelial（品红）

可视化每个案例 8 列：
  HR | LR | Baseline SR | Semantic SR |
  GT overlay | LR pred | Baseline pred | Semantic pred
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from models.segmentation.cell_segmentation.cellvit import CellViT256
from ddpm_dataset import PanNukeDataset
from degradation import degrade
from unet_wrapper import create_model
from ddpm_utils import predict_x0_from_noise_shared
from diffusers import DDPMScheduler

device = 'cuda'

CLASS_NAMES  = ['Background', 'Neoplastic', 'Inflammatory',
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

# 目标类别及其筛选参数
TARGET_CLASSES = {
    1: dict(name='Neoplastic',    min_pixels=300, max_lr_recall=0.75,
            min_improve=0.02, min_cls_ratio=0.35),
    2: dict(name='Inflammatory',  min_pixels=50,  max_lr_recall=0.50,
            min_improve=0.02, min_cls_ratio=0.05),
    3: dict(name='Connective',    min_pixels=300, max_lr_recall=0.80,
            min_improve=0.02, min_cls_ratio=0.35),
    5: dict(name='Epithelial',    min_pixels=300, max_lr_recall=0.80,
            min_improve=0.02, min_cls_ratio=0.35),
}

# ── CellViT 加载 ────────────────────────────────────────────────────
print("加载 CellViT...")
ckpt_cv = torch.load(
    '/home/xuwen/DDPM/CellViT/CellViT-256-x40.pth',
    map_location='cpu',
)
run_conf = ckpt_cv.get('config', {})
cellvit = CellViT256(
    model256_path=None,
    num_nuclei_classes=int(run_conf.get('data.num_nuclei_classes', 6)),
    num_tissue_classes=int(run_conf.get('data.num_tissue_classes', 19)),
).to(device)
cellvit.load_state_dict(ckpt_cv['model_state_dict'], strict=True)
cellvit.eval()
for p in cellvit.parameters():
    p.requires_grad = False


def run_cellvit_full(img_01: torch.Tensor):
    out       = cellvit(img_01)
    type_prob = F.softmax(out['nuclei_type_map'], dim=1)
    type_lbl  = type_prob.argmax(dim=1)
    bin_prob  = F.softmax(out['nuclei_binary_map'], dim=1)
    nuc_prob  = bin_prob[:, 1]
    return type_lbl, type_prob, nuc_prob


# ── UNet 加载 ───────────────────────────────────────────────────────
def load_unet(ckpt_path: str):
    unet = create_model(use_semantic=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    return unet


print("加载 UNet 模型...")
unet_baseline = load_unet("/home/xuwen/DDPM/logs/checkpoints_cellvit/unet_sr_epoch_20.pth")
unet_semantic = load_unet("/home/xuwen/DDPM/logs/checkpoints_cellvit/unet_sr_epoch_180.pth")

scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset   = PanNukeDataset(
    fold_dirs=['/data/xuwen/PanNuke/Fold 3'],
    target_size=256,
)
print(f"测试集大小: {len(dataset)} 张\n")

FIXED_T = 50
torch.manual_seed(42)


# ── 辅助函数 ────────────────────────────────────────────────────────

def class_recall(pred_np, gt_np, cls_id):
    """计算单个类别在 GT 该类像素上的召回率"""
    mask = (gt_np == cls_id)
    if mask.sum() == 0:
        return 0.0
    return (pred_np[mask] == cls_id).mean()


def make_overlay(img_rgb, label_map, nuc_mask, alpha=0.65):
    label_map = np.clip(label_map, 0, len(COLOR_MAP) - 1)
    color  = COLOR_MAP[label_map]
    valid  = (label_map > 0) & (nuc_mask > 0.3)
    a      = (alpha * nuc_mask * valid)[..., None].clip(0, 1)
    return (img_rgb * (1 - a) + color * a).clip(0, 1)


def tensor_to_np(t):
    return t.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def label_to_np(t):
    return t.squeeze(0).detach().cpu().numpy().astype(np.int32)


def nuc_to_np(t):
    return t.squeeze(0).detach().cpu().numpy()


# ── 按类别定向筛选 ──────────────────────────────────────────────────

# best_case[cls_id] = (improve, case_dict)
best_case = {cls_id: None for cls_id in TARGET_CLASSES}
best_improve = {cls_id: -1.0 for cls_id in TARGET_CLASSES}

print("扫描测试集，按类别定向筛选...")

for i in range(len(dataset)):
    # 如果所有类别都找到了，提前退出
    if all(v is not None for v in best_case.values()):
        break

    sample  = dataset[i]
    hr_cpu  = sample['hr']
    gt_lbl  = sample['gt_label_map']
    gt_nuc  = sample['gt_nuc_mask'].bool()

    if gt_nuc.sum() < 100:
        continue

    gt_np = gt_lbl.numpy()

    hr = hr_cpu.unsqueeze(0).to(device)
    lr_cpu = degrade(
        hr_cpu, scale=4,
        blur_sigma_range=(2.0, 3.0),
        noise_std_range=(0.03, 0.08),
        stain_jitter_strength=0.15,
    )
    lr = lr_cpu.unsqueeze(0).to(device)

    noise    = torch.randn_like(hr)
    t        = torch.tensor([FIXED_T], device=device)
    noisy_hr = scheduler.add_noise(hr, noise, t)

    with torch.no_grad():
        hr_lbl, hr_prob, hr_nuc = run_cellvit_full(hr)
        lr_lbl, lr_prob, lr_nuc = run_cellvit_full(lr)

        inp_b  = torch.cat([lr, noisy_hr], dim=1)
        x0_b   = predict_x0_from_noise_shared(
            noisy_hr, unet_baseline(inp_b, t).sample, t, scheduler)
        bl_lbl, bl_prob, bl_nuc = run_cellvit_full(x0_b)

        inp_s  = torch.cat([lr, noisy_hr], dim=1)
        x0_s   = predict_x0_from_noise_shared(
            noisy_hr, unet_semantic(inp_s, t).sample, t, scheduler)
        sm_lbl, sm_prob, sm_nuc = run_cellvit_full(x0_s)

    gt_nuc_np = sample['gt_nuc_mask'].numpy()
    lr_np  = lr_lbl.squeeze(0).cpu().numpy()
    bl_np  = bl_lbl.squeeze(0).cpu().numpy()
    sm_np  = sm_lbl.squeeze(0).cpu().numpy()

    # 只在 GT 核区域内评估
    gt_nuc_mask = gt_nuc.numpy()

    for cls_id, cfg in TARGET_CLASSES.items():
        # 该类在 GT 核区域内的像素数
        cls_pixels = ((gt_np == cls_id) & gt_nuc_mask).sum()
        if cls_pixels < cfg['min_pixels']:
            continue

        # 该类占 GT 核区域的比例（保证视觉上是主体）
        cls_ratio = cls_pixels / max(gt_nuc_mask.sum(), 1)
        if cls_ratio < cfg['min_cls_ratio']:
            continue

        # 在 GT 核区域内计算该类召回率
        nuc_gt = gt_np[gt_nuc_mask]
        nuc_lr = lr_np[gt_nuc_mask]
        nuc_bl = bl_np[gt_nuc_mask]
        nuc_sm = sm_np[gt_nuc_mask]

        lr_recall = class_recall(nuc_lr, nuc_gt, cls_id)
        bl_recall = class_recall(nuc_bl, nuc_gt, cls_id)
        sm_recall = class_recall(nuc_sm, nuc_gt, cls_id)
        improve   = sm_recall - bl_recall

        if lr_recall > cfg['max_lr_recall']:
            continue
        if improve < cfg['min_improve']:
            continue

        if improve > best_improve[cls_id]:
            best_improve[cls_id] = improve
            best_case[cls_id] = dict(
                idx        = i,
                target_cls = cls_id,
                lr_recall  = lr_recall,
                bl_recall  = bl_recall,
                sm_recall  = sm_recall,
                improvement= improve,
                cls_pixels = int(cls_pixels),
                cls_ratio  = float(cls_ratio),
                hr_rgb     = tensor_to_np(hr),
                lr_rgb     = tensor_to_np(lr),
                bl_rgb     = tensor_to_np(x0_b),
                sm_rgb     = tensor_to_np(x0_s),
                gt_lbl     = gt_np,
                gt_nuc     = gt_nuc_np,
                hr_lbl     = label_to_np(hr_lbl),
                hr_nuc     = nuc_to_np(hr_nuc),
                lr_lbl     = lr_np,
                lr_nuc     = nuc_to_np(lr_nuc),
                bl_lbl     = bl_np,
                bl_nuc     = nuc_to_np(bl_nuc),
                sm_lbl     = sm_np,
                sm_nuc     = nuc_to_np(sm_nuc),
            )
            print(f"  更新 {cfg['name']:>14} 最佳案例 | 样本{i:>4} | "
                  f"ratio={cls_ratio:.2f}  "
                  f"LR={lr_recall:.3f}  BL={bl_recall:.3f}  "
                  f"SM={sm_recall:.3f}  Δ={improve:+.3f}")

# ── 汇总找到的案例 ──────────────────────────────────────────────────
found_cases = []
for cls_id in [1, 2, 3, 5]:   # 按类别顺序排列
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
n_rows  = len(found_cases)
col_titles = [
    'HR', 'LR', 'Baseline SR', 'Semantic SR (ours)',
    'GT overlay', 'LR pred', 'Baseline pred', 'Semantic pred (ours)',
]
n_cols = len(col_titles)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.2 * n_cols, 3.8 * n_rows),
    squeeze=False,
)
fig.suptitle(
    'Per-class correction cases: Semantic SR improves cell type prediction\n'
    '(evaluated on GT nucleus region vs CellViT(HR) pseudo-labels)',
    fontsize=13, y=1.01,
)

for ci, title in enumerate(col_titles):
    axes[0, ci].set_title(title, fontsize=10, pad=6)

for ri, case in enumerate(found_cases):
    cls_id   = case['target_cls']
    cls_name = CLASS_NAMES[cls_id]
    # 行标题：在第一列图像左上角叠加文字
    cls_color = COLOR_MAP[cls_id].tolist()
    axes[ri, 0].text(
        0.02, 0.98,
        f"[{cls_name}]",
        transform=axes[ri, 0].transAxes,
        fontsize=10, fontweight='bold',
        va='top', ha='left',
        color=cls_color,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'),
    )
    # 在第4列（GT overlay）左上角显示召回率对比
    axes[ri, 4].text(
        0.02, 0.98,
        f"LR={case['lr_recall']:.3f}\nBL={case['bl_recall']:.3f}\n"
        f"SM={case['sm_recall']:.3f}  Δ={case['improvement']:+.3f}",
        transform=axes[ri, 4].transAxes,
        fontsize=7.5, va='top', ha='left',
        color='black',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.80, ec='none'),
    )

    # 列 0-3：图像
    for ci, rgb in enumerate([
        case['hr_rgb'], case['lr_rgb'],
        case['bl_rgb'], case['sm_rgb'],
    ]):
        axes[ri, ci].imshow(rgb)
        axes[ri, ci].axis('off')

    # 列 4：GT overlay
    gt_ov = make_overlay(case['hr_rgb'], case['gt_lbl'], case['gt_nuc'])
    axes[ri, 4].imshow(gt_ov)
    axes[ri, 4].axis('off')

    # 列 5：LR pred overlay
    lr_ov = make_overlay(case['lr_rgb'], case['lr_lbl'], case['lr_nuc'])
    axes[ri, 5].imshow(lr_ov)
    axes[ri, 5].axis('off')

    # 列 6：Baseline SR pred overlay
    bl_ov = make_overlay(case['bl_rgb'], case['bl_lbl'], case['bl_nuc'])
    axes[ri, 6].imshow(bl_ov)
    axes[ri, 6].axis('off')

    # 列 7：Semantic SR pred overlay
    sm_ov = make_overlay(case['sm_rgb'], case['sm_lbl'], case['sm_nuc'])
    axes[ri, 7].imshow(sm_ov)
    axes[ri, 7].axis('off')

    # 在 GT overlay 上用矩形框高亮目标类别区域
    for ax in [axes[ri, 4], axes[ri, 5], axes[ri, 6], axes[ri, 7]]:
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_MAP[cls_id])
            spine.set_linewidth(3)

# 图例
legend_patches = [
    mpatches.Patch(
        color=np.array(CLASS_COLORS[c]) / 255.,
        label=CLASS_NAMES[c],
    )
    for c in [1, 2, 3, 5]
]
fig.legend(
    handles=legend_patches,
    title='Cell type (focus)',
    loc='lower center',
    ncol=len(legend_patches),
    fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
save_path = './logs/correction_cases_by_class.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"按类别典型案例图已保存到: {save_path}")

# ── 打印汇总 ────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"{'类别':>14}  {'样本':>6}  {'GT像素数':>10}  {'占比':>6}  "
      f"{'LR召回':>8}  {'BL召回':>8}  {'SM召回':>8}  {'Δ':>8}")
print("-" * 75)
for case in found_cases:
    print(f"{CLASS_NAMES[case['target_cls']]:>14}  "
          f"{case['idx']:>6}  "
          f"{case['cls_pixels']:>10}  "
          f"{case.get('cls_ratio', 0):>6.2f}  "
          f"{case['lr_recall']:>8.4f}  "
          f"{case['bl_recall']:>8.4f}  "
          f"{case['sm_recall']:>8.4f}  "
          f"{case['improvement']:>+8.4f}")