"""
confusion_matrix_compare.py
对比 HR 基线、消融模型、本文方法 在测试集（Fold 3）上的混淆矩阵。

三组对比：
  HR 基线    : HR 直接送 CellViT，不经过任何纠正
  消融模型   : 无语义监督（只有重建损失，use_semantic=False）
  本文方法   : 交集 Focal-CE + CellViT 软标签 sem_tensor

推理策略：
  对同一张 HR 用 N_RUNS 组不同随机噪声分别做单步推理，
  将 N_RUNS 次 CellViT 概率图在通道维度取平均后 argmax，
  得到最终分类预测（概率空间集成）。

类别定义：
  0: Background  1: Neoplastic  2: Inflammatory
  3: Connective  4: Dead        5: Epithelial
"""

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from sklearn.metrics import confusion_matrix
from diffusers import DDPMScheduler

from ddpm_dataset import PanNukeDataset
from unet_wrapper import create_model
from ddpm_utils import load_cellvit, predict_x0_from_noise_shared
from semantic_sr_loss import run_cellvit, build_sem_tensor_from_cellvit

device      = 'cuda'
CLASS_NAMES = ['Background', 'Neoplastic', 'Inflammatory',
               'Connective', 'Dead', 'Epithelial']

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
    print(f"  加载 {ckpt_path}  epoch={ckpt.get('epoch', '?')}")
    return unet


print("\n加载 UNet 模型...")
unet_ablation = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction/best_unet_ablation.pth",
    use_semantic=False,
)
unet_full = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction/best_unet_correction.pth",
    use_semantic=True,
)

# ── 数据集 ──────────────────────────────────────────────────────────
scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset   = PanNukeDataset(
    fold_dirs   = ['/data/xuwen/PanNuke/Fold 3'],
    target_size = 256,
)
print(f"\n测试集大小: {len(dataset)} 张")

INFER_T   = 200
N_RUNS    = 5    # 每张图推理次数，概率图取平均后 argmax
N_SAMPLES = 200
torch.manual_seed(42)


# ── 集成推理 ─────────────────────────────────────────────────────────
def infer_ensemble(unet, hr, t, sem=None):
    """
    对同一张 HR 用 N_RUNS 组不同随机噪声做单步推理，
    返回 N_RUNS 次 CellViT 概率图均值 argmax 得到的分类图 (H, W)。
    """
    prob_sum = None
    for _ in range(N_RUNS):
        noise    = torch.randn_like(hr)
        noisy_hr = scheduler.add_noise(hr, noise, t)
        inp      = torch.cat([hr, noisy_hr], dim=1)
        with torch.no_grad():
            x0   = predict_x0_from_noise_shared(
                noisy_hr,
                unet(inp, t, semantic=sem).sample,
                t, scheduler,
            )
            prob = run_cellvit(cellvit, x0)['nuclei_type_prob']   # (1, C, H, W)
        prob_sum = prob if prob_sum is None else prob_sum + prob

    mean_prob = prob_sum / N_RUNS
    return mean_prob.argmax(dim=1).squeeze(0).cpu()               # (H, W)


# ── 推理 ────────────────────────────────────────────────────────────
all_gt, all_hr, all_abl, all_full = [], [], [], []
n_valid = 0

print(f"\n开始推理（INFER_T={INFER_T}，N_RUNS={N_RUNS}，最多 {N_SAMPLES} 个有效样本）...")

for i in range(len(dataset)):
    if n_valid >= N_SAMPLES:
        break

    sample  = dataset[i]
    hr_cpu  = sample['hr']
    gt_lbl  = sample['gt_label_map']
    gt_nuc  = sample['gt_nuc_mask'].bool()

    if gt_nuc.sum() < 10:
        continue

    hr = hr_cpu.unsqueeze(0).to(device)
    t  = torch.tensor([INFER_T], device=device)

    with torch.no_grad():
        # CellViT(HR)：直接推理，无需集成（确定性输出）
        hr_cv  = run_cellvit(cellvit, hr)
        hr_lbl = hr_cv['nuclei_type_label'].squeeze(0).cpu()

        sem = build_sem_tensor_from_cellvit(
            hr_cv['nuclei_type_prob'],
            hr_cv['nuclei_nuc_prob'],
        )

    # 消融模型集成推理（无 sem_tensor）
    abl_lbl  = infer_ensemble(unet_ablation, hr, t, sem=None)

    # 完整模型集成推理（有 sem_tensor）
    full_lbl = infer_ensemble(unet_full, hr, t, sem=sem)

    mask = gt_nuc.numpy()
    all_gt.append(gt_lbl.numpy()[mask])
    all_hr.append(hr_lbl.numpy()[mask])
    all_abl.append(abl_lbl.numpy()[mask])
    all_full.append(full_lbl.numpy()[mask])

    if n_valid % 20 == 0:
        print(f"  [{n_valid:>3}/{N_SAMPLES}] 样本 {i:>4}")
    n_valid += 1

print(f"\n有效样本数: {n_valid}")

all_gt   = np.concatenate(all_gt)
all_hr   = np.concatenate(all_hr)
all_abl  = np.concatenate(all_abl)
all_full = np.concatenate(all_full)

# ── 混淆矩阵 ────────────────────────────────────────────────────────
labels  = list(range(6))
cm_hr   = confusion_matrix(all_gt, all_hr,   labels=labels)
cm_abl  = confusion_matrix(all_gt, all_abl,  labels=labels)
cm_full = confusion_matrix(all_gt, all_full, labels=labels)

recall_hr   = np.diag(cm_hr)   / cm_hr.sum(axis=1).clip(min=1)
recall_abl  = np.diag(cm_abl)  / cm_abl.sum(axis=1).clip(min=1)
recall_full = np.diag(cm_full) / cm_full.sum(axis=1).clip(min=1)

# ── 文字汇总 ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("各类召回率对比（GT 核区域内）")
print(f"{'='*80}")
print(f"{'类别':>14}  {'HR基线':>10}  {'消融模型':>10}  "
      f"{'本文方法':>10}  {'Δ(本文-HR)':>12}")
print("-" * 80)

for i, name in enumerate(CLASS_NAMES):
    n_gt = cm_hr.sum(axis=1)[i]
    if n_gt == 0:
        continue
    delta = recall_full[i] - recall_hr[i]
    arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')
    print(f"{name:>14}  {recall_hr[i]:>10.4f}  {recall_abl[i]:>10.4f}  "
          f"{recall_full[i]:>10.4f}  {delta:>+10.4f} {arrow}  "
          f"(GT={n_gt:>8})")

print(f"\n整体准确率："
      f"HR={( all_hr  ==all_gt).mean():.4f}  "
      f"消融={(all_abl ==all_gt).mean():.4f}  "
      f"本文={(all_full==all_gt).mean():.4f}")

# ── 可视化 ───────────────────────────────────────────────────────────
def plot_cm_normalized(ax, cm, title, class_names):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    valid   = cm.sum(axis=1) > 0
    cm_v    = cm_norm[valid][:, valid]
    names_v = [n for n, v in zip(class_names, valid) if v]

    im = ax.imshow(cm_v, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('GT', fontsize=9)
    ticks = np.arange(len(names_v))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(names_v, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(names_v, fontsize=8)
    for ri in range(len(names_v)):
        for ci in range(len(names_v)):
            val = cm_v[ri, ci]
            ax.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if val > 0.5 else 'black')
    return im

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f'Confusion Matrices — ensemble N={N_RUNS} runs '
    f'(normalized by GT class, nucleus region only)',
    fontsize=13, y=1.02,
)
plot_cm_normalized(axes[0], cm_hr,   'HR (baseline)',        CLASS_NAMES)
plot_cm_normalized(axes[1], cm_abl,  'Ablation (no sem)',    CLASS_NAMES)
im = plot_cm_normalized(axes[2], cm_full, 'Full model (ours)', CLASS_NAMES)
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Recall')
plt.tight_layout()

os.makedirs('./logs', exist_ok=True)
save_path = './logs/confusion_matrix_comparison_ablation_v3.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n混淆矩阵图已保存到: {save_path}")