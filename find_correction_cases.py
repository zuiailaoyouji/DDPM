"""
find_correction_cases.py
从测试集（Fold 3）中挑选典型的语义标签纠正案例。

筛选标准：
  在 GT 核掩膜区域内，满足以下条件的 patch：
  1. LR pred 和 GT 不一致（存在误分类）
  2. Semantic SR pred 比 Baseline SR pred 更接近 GT
  3. 改善幅度显著（Semantic Dir_Acc - Baseline Dir_Acc > 阈值）

可视化每个案例的 7 列：
  HR | LR | Baseline SR | Semantic SR |
  GT overlay | LR pred overlay | Baseline pred overlay | Semantic pred overlay
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
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
    [0,   0,   0  ],   # 0 Background: 黑
    [255, 0,   0  ],   # 1 Neoplastic:  红
    [0,   0,   255],   # 2 Inflammatory: 蓝
    [255, 255, 0  ],   # 3 Connective:  黄
    [0,   255, 0  ],   # 4 Dead:        绿
    [255, 0,   255],   # 5 Epithelial:  品红
]
COLOR_MAP = np.array(CLASS_COLORS, dtype=np.float32) / 255.0


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
    """返回 (type_label [B,H,W], type_prob [B,6,H,W], nuc_prob [B,H,W])"""
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

FIXED_T         = 50
MIN_IMPROVEMENT = 0.03    # Semantic - Baseline Dir_Acc 最小改善阈值
MIN_LR_ERROR    = 0.15    # LR 必须有足够的误分类
MAX_CASES       = 6       # 最多挑选几个案例
torch.manual_seed(42)

# ── 可视化辅助函数 ──────────────────────────────────────────────────

def make_overlay(img_rgb: np.ndarray,
                 label_map: np.ndarray,
                 nuc_mask: np.ndarray,
                 alpha: float = 0.6) -> np.ndarray:
    """
    在图像上叠加类别颜色 overlay。
    img_rgb  : [H, W, 3] float32 [0,1]
    label_map: [H, W] int
    nuc_mask : [H, W] float [0,1]
    """
    label_map = np.clip(label_map, 0, len(COLOR_MAP) - 1)
    color     = COLOR_MAP[label_map]              # [H, W, 3]
    valid     = (label_map > 0) & (nuc_mask > 0.3)
    a         = (alpha * nuc_mask * valid)[..., None].clip(0, 1)
    return (img_rgb * (1 - a) + color * a).clip(0, 1)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] or [1,C,H,W] → [H,W,3] float32"""
    x = t.squeeze(0).detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def label_to_np(t: torch.Tensor) -> np.ndarray:
    """[H,W] or [1,H,W] → [H,W] int"""
    return t.squeeze(0).detach().cpu().numpy().astype(np.int32)


def nuc_to_np(t: torch.Tensor) -> np.ndarray:
    """[H,W] or [1,H,W] → [H,W] float"""
    return t.squeeze(0).detach().cpu().numpy()


# ── 筛选和绘图 ─────────────────────────────────────────────────────

cases = []   # 存储筛选出的案例信息
print(f"\n扫描测试集，筛选典型纠正案例（最多 {MAX_CASES} 个）...")

for i in range(len(dataset)):
    if len(cases) >= MAX_CASES:
        break

    sample  = dataset[i]
    hr_cpu  = sample['hr']
    gt_lbl  = sample['gt_label_map']       # [H,W] int
    gt_nuc  = sample['gt_nuc_mask'].bool() # [H,W]

    if gt_nuc.sum() < 500:   # 核像素太少的跳过
        continue

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
        # CellViT(HR) 伪标签
        hr_lbl, hr_prob, hr_nuc = run_cellvit_full(hr)

        # CellViT(LR)
        lr_lbl, lr_prob, lr_nuc = run_cellvit_full(lr)

        # Baseline SR
        inp_b  = torch.cat([lr, noisy_hr], dim=1)
        np_b   = unet_baseline(inp_b, t).sample
        x0_b   = predict_x0_from_noise_shared(noisy_hr, np_b, t, scheduler)
        bl_lbl, bl_prob, bl_nuc = run_cellvit_full(x0_b)

        # Semantic SR
        inp_s  = torch.cat([lr, noisy_hr], dim=1)
        np_s   = unet_semantic(inp_s, t).sample
        x0_s   = predict_x0_from_noise_shared(noisy_hr, np_s, t, scheduler)
        sm_lbl, sm_prob, sm_nuc = run_cellvit_full(x0_s)

    # 计算核区域内的指标（对比 GT label_map）
    gt_np  = gt_lbl[gt_nuc].numpy()
    lr_np  = lr_lbl.squeeze(0).cpu()[gt_nuc].numpy()
    bl_np  = bl_lbl.squeeze(0).cpu()[gt_nuc].numpy()
    sm_np  = sm_lbl.squeeze(0).cpu()[gt_nuc].numpy()

    lr_acc = (lr_np == gt_np).mean()
    bl_acc = (bl_np == gt_np).mean()
    sm_acc = (sm_np == gt_np).mean()
    improvement = sm_acc - bl_acc
    lr_error    = 1.0 - lr_acc

    # 筛选条件
    if lr_error < MIN_LR_ERROR:
        continue
    if improvement < MIN_IMPROVEMENT:
        continue

    print(f"  ✓ 案例 {len(cases)+1} | 样本 {i:>4} | "
          f"LR_acc={lr_acc:.3f}  BL_acc={bl_acc:.3f}  "
          f"SM_acc={sm_acc:.3f}  Δ={improvement:+.3f}")

    cases.append(dict(
        idx        = i,
        hr_rgb     = tensor_to_np(hr),
        lr_rgb     = tensor_to_np(lr),
        bl_rgb     = tensor_to_np(x0_b),
        sm_rgb     = tensor_to_np(x0_s),
        gt_lbl     = gt_lbl.numpy(),
        gt_nuc     = sample['gt_nuc_mask'].numpy(),
        hr_lbl     = label_to_np(hr_lbl),
        hr_nuc     = nuc_to_np(hr_nuc),
        lr_lbl     = label_to_np(lr_lbl),
        lr_nuc     = nuc_to_np(lr_nuc),
        bl_lbl     = label_to_np(bl_lbl),
        bl_nuc     = nuc_to_np(bl_nuc),
        sm_lbl     = label_to_np(sm_lbl),
        sm_nuc     = nuc_to_np(sm_nuc),
        lr_acc     = lr_acc,
        bl_acc     = bl_acc,
        sm_acc     = sm_acc,
        improvement= improvement,
    ))

if not cases:
    print("未找到满足条件的案例，请适当降低 MIN_IMPROVEMENT 或 MIN_LR_ERROR 阈值。")
else:
    print(f"\n共找到 {len(cases)} 个典型案例，开始绘图...")

    # ── 绘图 ───────────────────────────────────────────────────────
    n_cases = len(cases)
    col_titles = [
        'HR', 'LR', 'Baseline SR', 'Semantic SR (ours)',
        'GT overlay', 'LR pred', 'Baseline pred', 'Semantic pred (ours)',
    ]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(
        n_cases, n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_cases),
        squeeze=False,
    )
    fig.suptitle(
        'Typical correction cases: Semantic SR improves cell type prediction\n'
        '(GT nucleus region, compared against CellViT(HR) pseudo-labels)',
        fontsize=13, y=1.01,
    )

    for ci, col_title in enumerate(col_titles):
        axes[0, ci].set_title(col_title, fontsize=10, pad=6)

    for ri, case in enumerate(cases):
        acc_str = (f"LR={case['lr_acc']:.3f}  "
                   f"BL={case['bl_acc']:.3f}  "
                   f"SM={case['sm_acc']:.3f}  "
                   f"Δ={case['improvement']:+.3f}")
        axes[ri, 0].set_ylabel(f"Case {ri+1}\n{acc_str}", fontsize=8)

        # 列 0-3：图像
        for ci, rgb in enumerate([
            case['hr_rgb'], case['lr_rgb'],
            case['bl_rgb'], case['sm_rgb'],
        ]):
            axes[ri, ci].imshow(rgb)
            axes[ri, ci].axis('off')

        # 列 4：GT overlay（用 GT label_map 和 GT nuc_mask）
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

    # 图例
    legend_patches = [
        mpatches.Patch(color=np.array(c) / 255., label=n)
        for n, c in zip(CLASS_NAMES[1:], CLASS_COLORS[1:])
    ]
    fig.legend(
        handles=legend_patches,
        title='Cell type',
        loc='lower center',
        ncol=len(legend_patches),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    save_path = './logs/correction_cases.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"典型案例图已保存到: {save_path}")

    # ── 打印案例汇总 ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("案例汇总")
    print(f"{'='*60}")
    print(f"{'案例':>4}  {'样本':>6}  {'LR_acc':>8}  "
          f"{'BL_acc':>8}  {'SM_acc':>8}  {'Δ':>8}")
    for ri, case in enumerate(cases):
        print(f"{ri+1:>4}  {case['idx']:>6}  {case['lr_acc']:>8.4f}  "
              f"{case['bl_acc']:>8.4f}  {case['sm_acc']:>8.4f}  "
              f"{case['improvement']:>+8.4f}")
