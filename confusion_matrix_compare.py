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

新增：按 tissue type 分组输出各组的召回率和整体准确率。
"""

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict
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
    "/home/xuwen/DDPM/logs/checkpoints_correction_v3/best_unet_ablation.pth",
    use_semantic=False,
)
unet_full = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction_v3/best_unet_correction.pth",
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
N_RUNS    = 5
N_SAMPLES = 200
torch.manual_seed(42)

# ── 分层采样：按 tissue type 均匀取样，避免顺序偏差 ────────────────
import random, math
random.seed(42)

from collections import defaultdict as _dd
_type_to_idx = _dd(list)
for _i in range(len(dataset)):
    _s = dataset[_i]
    if _s['gt_nuc_mask'].bool().sum() < 10:
        continue
    _type_to_idx[_s['type_name']].append(_i)

_n_types = len(_type_to_idx)
_per_type = max(1, math.ceil(N_SAMPLES / _n_types))

sampled_indices = []
for _tname, _idxs in sorted(_type_to_idx.items()):
    _chosen = random.sample(_idxs, min(_per_type, len(_idxs)))
    sampled_indices.extend(_chosen)

random.shuffle(sampled_indices)
sampled_indices = sampled_indices[:N_SAMPLES]

_type_counts = {t: 0 for t in _type_to_idx}
for _i in sampled_indices:
    _type_counts[dataset[_i]['type_name']] += 1
print(f'\n分层采样结果（共 {len(sampled_indices)} 张，覆盖 {_n_types} 种 tissue）：')
for _t, _c in sorted(_type_counts.items(), key=lambda x: -x[1]):
    print(f'  {_t:<25} {_c:>4} 张')


# ── 集成推理 ─────────────────────────────────────────────────────────
def infer_ensemble(unet, hr, t, sem=None):
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
# 全局像素列表
all_gt, all_hr, all_abl, all_full = [], [], [], []

# 退步分析像素列表
regress_gt, regress_hr, regress_abl, regress_full = [], [], [], []

# Per-tissue 像素列表：tissue_name -> {'gt': [], 'hr': [], 'abl': [], 'full': []}
tissue_pixels = defaultdict(lambda: {'gt': [], 'hr': [], 'abl': [], 'full': []})
tissue_n      = defaultdict(int)

n_valid = 0
print(f"\n开始推理（INFER_T={INFER_T}，N_RUNS={N_RUNS}，共 {len(sampled_indices)} 个分层采样样本）...")

for i in sampled_indices:   # 分层采样索引，已预筛选 gt_nuc >= 10

    sample    = dataset[i]
    hr_cpu    = sample['hr']
    gt_lbl    = sample['gt_label_map']
    gt_nuc    = sample['gt_nuc_mask'].bool()
    type_name = sample['type_name']           # patch 级 tissue type

    hr = hr_cpu.unsqueeze(0).to(device)
    t  = torch.tensor([INFER_T], device=device)

    with torch.no_grad():
        hr_cv  = run_cellvit(cellvit, hr)
        hr_lbl = hr_cv['nuclei_type_label'].squeeze(0).cpu()
        sem    = build_sem_tensor_from_cellvit(
            hr_cv['nuclei_type_prob'],
            hr_cv['nuclei_nuc_prob'],
        )

    abl_lbl  = infer_ensemble(unet_ablation, hr, t, sem=None)
    full_lbl = infer_ensemble(unet_full,     hr, t, sem=sem)

    mask = gt_nuc.numpy()

    # 全局
    all_gt.append(gt_lbl.numpy()[mask])
    all_hr.append(hr_lbl.numpy()[mask])
    all_abl.append(abl_lbl.numpy()[mask])
    all_full.append(full_lbl.numpy()[mask])

    # 退步分析
    regress_gt.append(gt_lbl.numpy()[mask])
    regress_hr.append(hr_lbl.numpy()[mask])
    regress_abl.append(abl_lbl.numpy()[mask])
    regress_full.append(full_lbl.numpy()[mask])

    # Per-tissue
    tissue_pixels[type_name]['gt'].append(gt_lbl.numpy()[mask])
    tissue_pixels[type_name]['hr'].append(hr_lbl.numpy()[mask])
    tissue_pixels[type_name]['abl'].append(abl_lbl.numpy()[mask])
    tissue_pixels[type_name]['full'].append(full_lbl.numpy()[mask])
    tissue_n[type_name] += 1

    if n_valid % 20 == 0:
        print(f"  [{n_valid:>3}/{N_SAMPLES}] 样本 {i:>4}  tissue={type_name}")
    n_valid += 1

print(f"\n有效样本数: {n_valid}")

# ── 拼接 ────────────────────────────────────────────────────────────
all_gt   = np.concatenate(all_gt)
all_hr   = np.concatenate(all_hr)
all_abl  = np.concatenate(all_abl)
all_full = np.concatenate(all_full)

# ── 全局混淆矩阵 ────────────────────────────────────────────────────
labels  = list(range(6))
cm_hr   = confusion_matrix(all_gt, all_hr,   labels=labels)
cm_abl  = confusion_matrix(all_gt, all_abl,  labels=labels)
cm_full = confusion_matrix(all_gt, all_full, labels=labels)

recall_hr   = np.diag(cm_hr)   / cm_hr.sum(axis=1).clip(min=1)
recall_abl  = np.diag(cm_abl)  / cm_abl.sum(axis=1).clip(min=1)
recall_full = np.diag(cm_full) / cm_full.sum(axis=1).clip(min=1)

# ── 全局文字汇总 ────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("各类召回率对比（GT 核区域内）—— 全局")
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

# ── 退步分析 ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("退步分析：HR基线 & 消融模型 均答对，但本文方法答错的像素")
print(f"{'='*80}")

rg    = np.concatenate(regress_gt)
rhr   = np.concatenate(regress_hr)
rabl  = np.concatenate(regress_abl)
rfull = np.concatenate(regress_full)

hr_correct   = (rhr   == rg)
abl_correct  = (rabl  == rg)
full_correct = (rfull == rg)

both_correct_full_wrong = hr_correct & abl_correct & ~full_correct
only_full_correct       = ~hr_correct & ~abl_correct & full_correct

total_nuc = len(rg)
n_regress  = both_correct_full_wrong.sum()
n_improve  = only_full_correct.sum()

print(f"\n  核区域总像素数           : {total_nuc:>10,}")
print(f"  退步像素数 (两者对本文错) : {n_regress:>10,}  ({100*n_regress/total_nuc:.2f}%)")
print(f"  进步像素数 (仅本文对)     : {n_improve:>10,}  ({100*n_improve/total_nuc:.2f}%)")

print(f"\n  按GT类别细分退步情况：")
print(f"  {'类别':>14}  {'GT像素':>8}  {'退步像素':>8}  {'退步率':>8}  "
      f"{'退步时被误分为（Top-3）'}")
print(f"  {'-'*75}")

for cls_id, name in enumerate(CLASS_NAMES):
    gt_cls_mask   = (rg == cls_id)
    regress_mask  = both_correct_full_wrong & gt_cls_mask
    n_gt_cls      = gt_cls_mask.sum()
    n_regress_cls = regress_mask.sum()
    if n_gt_cls == 0 or n_regress_cls == 0:
        continue
    wrong_preds = rfull[regress_mask]
    unique, counts = np.unique(wrong_preds, return_counts=True)
    top3 = sorted(zip(counts, unique), reverse=True)[:3]
    top3_str = "  ".join(f"{CLASS_NAMES[c]}({cnt})" for cnt, c in top3)
    print(f"  {name:>14}  {n_gt_cls:>8,}  {n_regress_cls:>8,}  "
          f"{100*n_regress_cls/n_gt_cls:>7.2f}%  {top3_str}")

print(f"\n  说明：退步 = HR基线正确 AND 消融模型正确 AND 本文方法错误")
print(f"        进步 = HR基线错误 AND 消融模型错误 AND 本文方法正确")

# ── Per-tissue 汇总 ─────────────────────────────────────────────────
print(f"\n{'='*95}")
print("按 Tissue Type 分组 —— 整体准确率 & 各类召回率（GT 核区域内）")
print(f"{'='*95}")

all_tissues = sorted(tissue_pixels.keys())

# 输出每个 tissue 的整体准确率（三组对比）
print(f"\n── 整体准确率（Overall Accuracy）──")
print(f"{'Tissue':<22} {'N':>4}  {'HR基线':>8}  {'消融模型':>9}  {'本文方法':>9}  "
      f"{'Δ(本文-HR)':>12}")
print("-" * 80)

for tname in all_tissues:
    tp = tissue_pixels[tname]
    tgt  = np.concatenate(tp['gt'])
    thr  = np.concatenate(tp['hr'])
    tabl = np.concatenate(tp['abl'])
    tful = np.concatenate(tp['full'])
    n    = tissue_n[tname]

    oa_hr  = (thr  == tgt).mean()
    oa_abl = (tabl == tgt).mean()
    oa_ful = (tful == tgt).mean()
    delta  = oa_ful - oa_hr
    arrow  = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')

    print(f"{tname:<22} {n:>4}  {oa_hr:>8.4f}  {oa_abl:>9.4f}  {oa_ful:>9.4f}  "
          f"{delta:>+10.4f} {arrow}")

# 输出每个 tissue 的各细胞类别召回率（仅展示本文方法，Δ vs HR基线）
print(f"\n── 各细胞类别召回率：本文方法（Δ = 本文 − HR基线）──")

# 表头：每列是一个细胞类别
cell_classes_to_show = [c for c in range(1, 6)]   # 跳过 Background
header_cls = "  ".join(f"{CLASS_NAMES[c]:>16}" for c in cell_classes_to_show)
print(f"\n{'Tissue':<22} {'N':>4}  {header_cls}")
print("-" * (26 + 18 * len(cell_classes_to_show)))

for tname in all_tissues:
    tp   = tissue_pixels[tname]
    tgt  = np.concatenate(tp['gt'])
    thr  = np.concatenate(tp['hr'])
    tful = np.concatenate(tp['full'])
    n    = tissue_n[tname]

    tcm_hr   = confusion_matrix(tgt, thr,  labels=labels)
    tcm_full = confusion_matrix(tgt, tful, labels=labels)
    tr_hr    = np.diag(tcm_hr)   / tcm_hr.sum(axis=1).clip(min=1)
    tr_full  = np.diag(tcm_full) / tcm_full.sum(axis=1).clip(min=1)

    cls_str = ""
    for c in cell_classes_to_show:
        n_gt_c = tcm_hr.sum(axis=1)[c]
        if n_gt_c == 0:
            cls_str += f"{'  N/A':>16}  "
        else:
            delta = tr_full[c] - tr_hr[c]
            arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')
            cls_str += f"{tr_full[c]:>6.4f}({delta:>+.4f}{arrow})  "

    print(f"{tname:<22} {n:>4}  {cls_str}")

print(f"\n{'='*95}")

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

# 全局混淆矩阵（三合一）
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
print(f"\n全局混淆矩阵图已保存到: {save_path}")

# Per-tissue 整体准确率对比条形图
n_tissues = len(all_tissues)
fig, ax = plt.subplots(figsize=(max(10, n_tissues * 1.2), 5))

x      = np.arange(n_tissues)
width  = 0.25
oa_hr_list, oa_abl_list, oa_ful_list = [], [], []

for tname in all_tissues:
    tp   = tissue_pixels[tname]
    tgt  = np.concatenate(tp['gt'])
    oa_hr_list.append((np.concatenate(tp['hr'])  == tgt).mean())
    oa_abl_list.append((np.concatenate(tp['abl']) == tgt).mean())
    oa_ful_list.append((np.concatenate(tp['full'])== tgt).mean())

ax.bar(x - width, oa_hr_list,  width, label='HR baseline', color='#4878D0')
ax.bar(x,         oa_abl_list, width, label='Ablation',    color='#EE854A')
ax.bar(x + width, oa_ful_list, width, label='Full (ours)', color='#6ACC65')

ax.set_xticks(x)
ax.set_xticklabels(all_tissues, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Overall Accuracy (nucleus region)')
ax.set_title(f'Per-tissue Overall Accuracy — ensemble N={N_RUNS}')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

tissue_bar_path = './logs/per_tissue_accuracy_comparison.png'
plt.savefig(tissue_bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Per-tissue 准确率条形图已保存到: {tissue_bar_path}")