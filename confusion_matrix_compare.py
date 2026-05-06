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

【指标层级与统计粒度（v2 改造）】
─────────────────────────────────────────────────────────────────────
  - 全局像素汇总：召回率（per-class，像素本就有 GT 类别归属）
  - 全局像素汇总：错误重定向（error redirection / introduction）
                 这是诊断纠错任务的核心指标，比单看准确率更深
  - Per-tissue：整体准确率（tissue 维度有意义）
              + 提升最大/最差的细胞类（替代之前 N×5 的稀疏表）
  - 不再做：Per-tissue × Per-cell-type 二维表（信息密度过低）

【新增归一化】
  - Improvement Ratio = (full − hr) / (1 − hr)
      含义：HR→上限 1.0 之间的差距覆盖了多少
  - 错误重定向比值 = 改对的错误像素数 / 改错的正确像素数
      含义：每引入 1 个新错误，纠正了几个原有错误
"""

import json
import math
import os
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

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
    model_path        = '/home/xuwen/DDPM/CellViT/CellViT-SAM-H-x40.pth',
    cellvit_repo_path = '/home/xuwen/DDPM/CellViT',
    device            = device,
    variant           = 'sam_h',
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
    "/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_ablation.pth",
    use_semantic=False,
)
unet_full = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_correction.pth",
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
random.seed(42)

# ── 分层采样 ────────────────────────────────────────────────────────
_type_to_idx = defaultdict(list)
for _i in range(len(dataset)):
    _s = dataset[_i]
    if _s['gt_nuc_mask'].bool().sum() < 10:
        continue
    _type_to_idx[_s['type_name']].append(_i)

_n_types  = len(_type_to_idx)
_per_type = max(1, math.ceil(N_SAMPLES / _n_types))

sampled_indices = []
for _tname, _idxs in sorted(_type_to_idx.items()):
    _chosen = random.sample(_idxs, min(_per_type, len(_idxs)))
    sampled_indices.extend(_chosen)
random.shuffle(sampled_indices)
sampled_indices = sampled_indices[:N_SAMPLES]

_type_counts = defaultdict(int)
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
            x0 = predict_x0_from_noise_shared(
                noisy_hr,
                unet(inp, t, semantic=sem).sample,
                t, scheduler,
            )
            prob = run_cellvit(cellvit, x0)['nuclei_type_prob']
        prob_sum = prob if prob_sum is None else prob_sum + prob

    mean_prob = prob_sum / N_RUNS
    return mean_prob.argmax(dim=1).squeeze(0).cpu()


# ─────────────────────────────────────────────────────────────────────
# 推理：合并了之前的 all_*  和 regress_*（重复变量，本质同一份数据）
# ─────────────────────────────────────────────────────────────────────
all_gt, all_hr, all_abl, all_full = [], [], [], []

# Per-tissue 像素列表
tissue_pixels = defaultdict(lambda: {'gt': [], 'hr': [], 'abl': [], 'full': []})
tissue_n      = defaultdict(int)

n_valid = 0
print(f"\n开始推理（INFER_T={INFER_T}，N_RUNS={N_RUNS}，"
      f"共 {len(sampled_indices)} 个分层采样样本）...")

for i in sampled_indices:
    sample    = dataset[i]
    hr_cpu    = sample['hr']
    gt_lbl    = sample['gt_label_map']
    gt_nuc    = sample['gt_nuc_mask'].bool()
    type_name = sample['type_name']

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
    gt_pix   = gt_lbl.numpy()[mask]
    hr_pix   = hr_lbl.numpy()[mask]
    abl_pix  = abl_lbl.numpy()[mask]
    full_pix = full_lbl.numpy()[mask]

    all_gt.append(gt_pix)
    all_hr.append(hr_pix)
    all_abl.append(abl_pix)
    all_full.append(full_pix)

    tissue_pixels[type_name]['gt'].append(gt_pix)
    tissue_pixels[type_name]['hr'].append(hr_pix)
    tissue_pixels[type_name]['abl'].append(abl_pix)
    tissue_pixels[type_name]['full'].append(full_pix)
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


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────
def _improvement_ratio(hr_v, full_v):
    """覆盖了 HR→1.0 多少差距。HR=1 或非 [0,1] 时返回 NaN。"""
    if not (0 <= hr_v <= 1):
        return float('nan')
    if 1.0 - hr_v < 1e-6:
        return float('nan')
    return (full_v - hr_v) / (1.0 - hr_v)


def _arrow(delta, eps=1e-3):
    return '↑' if delta > eps else ('↓' if delta < -eps else '—')


def _fmt_pct(x, width=7):
    """格式化百分比；NaN 用 N/A 占位，宽度统一。"""
    if math.isnan(x):
        return f"{'N/A':>{width}}"
    return f"{x*100:>{width-1}.2f}%"


# ─────────────────────────────────────────────────────────────────────
# 0. 准备：labels 与全局变量
# ─────────────────────────────────────────────────────────────────────
labels   = list(range(6))
W = 95

# 注：全局 Overall_Acc / Intersect_Acc / PSNR / SSIM / Sem_MAE
#   → 见 compare_baseline_with_senmatic.py(本脚本聚焦混淆与重定向分析)


# ─────────────────────────────────────────────────────────────────────
# 1. 全局：各类召回率 + 改善率（核心表）
# ─────────────────────────────────────────────────────────────────────
cm_hr   = confusion_matrix(all_gt, all_hr,   labels=labels)
cm_abl  = confusion_matrix(all_gt, all_abl,  labels=labels)
cm_full = confusion_matrix(all_gt, all_full, labels=labels)

recall_hr   = np.diag(cm_hr)   / cm_hr.sum(axis=1).clip(min=1)
recall_abl  = np.diag(cm_abl)  / cm_abl.sum(axis=1).clip(min=1)
recall_full = np.diag(cm_full) / cm_full.sum(axis=1).clip(min=1)

print(f"\n{'='*W}")
print("【全局各类召回率】（像素级汇总；每行 GT 像素数同列在右）")
print(f"{'='*W}")
print(f"{'类别':<14}{'HR':>10}{'消融':>10}{'本文':>10}"
      f"{'Δ(本文-HR)':>13}{'改善率':>9}{'GT像素':>11}")
print("-" * W)
for i, name in enumerate(CLASS_NAMES):
    n_gt = int(cm_hr.sum(axis=1)[i])
    if n_gt == 0:
        continue
    delta = recall_full[i] - recall_hr[i]
    imp   = _improvement_ratio(recall_hr[i], recall_full[i])
    arrow = _arrow(delta)
    print(f"{name:<14}{recall_hr[i]:>10.4f}{recall_abl[i]:>10.4f}{recall_full[i]:>10.4f}"
          f"{delta:>+11.4f} {arrow}"
          f"{_fmt_pct(imp):>9}"
          f"{n_gt:>11,}")

# 标出提升最大的两类（仅看细胞类，跳过背景）
sorted_deltas = sorted(
    [(c, recall_full[c] - recall_hr[c])
     for c in range(1, 6) if cm_hr.sum(axis=1)[c] > 0],
    key=lambda x: -x[1],
)
if len(sorted_deltas) >= 2:
    a, b = sorted_deltas[0], sorted_deltas[1]
    print(f"\n  提升最大类别：{CLASS_NAMES[a[0]]} (Δ={a[1]:+.4f})、"
          f"{CLASS_NAMES[b[0]]} (Δ={b[1]:+.4f})")


# ─────────────────────────────────────────────────────────────────────
# 2. 错误重定向 / 引入分析（替代单纯的退步分析）
#
#     本文 vs HR 基线：
#       error_corrected  = HR 错 → 本文 对  （DGFR 修复的错误）
#       error_introduced = HR 对 → 本文 错  （DGFR 引入的新错误）
#       net_correction   = corrected − introduced
#       redirect_ratio   = corrected / max(introduced, 1)
#                          每引入 1 个错误，纠正几个？
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*W}")
print("【错误重定向分析】（vs HR 基线;DGFR 是否在做正确的修改)")
print(f"{'='*W}")

hr_correct   = (all_hr   == all_gt)
full_correct = (all_full == all_gt)

# 与 HR 基线对比
hr_wrong_full_right = ~hr_correct &  full_correct   # 修复
hr_right_full_wrong =  hr_correct & ~full_correct   # 引入

n_corrected   = int(hr_wrong_full_right.sum())
n_introduced  = int(hr_right_full_wrong.sum())
n_total_nuc   = len(all_gt)

n_hr_wrong = int((~hr_correct).sum())
n_hr_right = int(hr_correct.sum())

# 比率定义
rate_corrected  = n_corrected  / max(n_hr_wrong, 1)   # HR 错的里面修复了多少
rate_introduced = n_introduced / max(n_hr_right, 1)   # HR 对的里面破坏了多少
redirect_ratio  = n_corrected  / max(n_introduced, 1)

print(f"\n  核区域总像素            : {n_total_nuc:>10,}")
print(f"  HR 答对像素             : {n_hr_right:>10,}  ({100*n_hr_right/n_total_nuc:.2f}%)")
print(f"  HR 答错像素             : {n_hr_wrong:>10,}  ({100*n_hr_wrong/n_total_nuc:.2f}%)")
print(f"\n  ✓ 修复（HR错→本文对）   : {n_corrected:>10,}  "
      f"占 HR 错误的 {100*rate_corrected:.2f}%（错误纠正率）")
print(f"  ✗ 引入（HR对→本文错）   : {n_introduced:>10,}  "
      f"占 HR 正确的 {100*rate_introduced:.2f}%（错误引入率）")
print(f"\n  净纠正像素              : {n_corrected - n_introduced:>+10,}")
print(f"  纠错/引入比             : {redirect_ratio:>10.2f}x   "
      f"（每引入 1 个错误，纠正 {redirect_ratio:.2f} 个；越大越好）")

# 三方均答错但本文答对的"困难纠正"——只有 DGFR 能解的
abl_correct = (all_abl == all_gt)
hard_corrected = ~hr_correct & ~abl_correct &  full_correct   # 仅本文对
both_wrong_introduced = hr_correct & abl_correct & ~full_correct  # 两者都对，本文错
print(f"\n  困难纠正（HR&消融均错→本文对）: {int(hard_corrected.sum()):>8,}  "
      f"({100*hard_corrected.mean():.2f}%)")
print(f"  共识破坏（HR&消融均对→本文错）: {int(both_wrong_introduced.sum()):>8,}  "
      f"({100*both_wrong_introduced.mean():.2f}%)")


# ─────────────────────────────────────────────────────────────────────
# 3. 修复方向 / 引入方向：按 GT 类别
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*W}")
print("【修复 vs 引入 · 按 GT 细胞类别】")
print(f"{'='*W}")
print(f"{'GT类别':<14}{'GT像素':>9}{'修复':>9}{'引入':>9}"
      f"{'净纠正':>9}{'纠错率':>9}{'引入率':>9}")
print("-" * W)
for c in range(1, 6):
    cls_mask = (all_gt == c)
    n_gt_c   = int(cls_mask.sum())
    if n_gt_c == 0:
        continue
    n_hr_wrong_c   = int((cls_mask & ~hr_correct).sum())
    n_hr_right_c   = int((cls_mask &  hr_correct).sum())
    corrected_c    = int((cls_mask & hr_wrong_full_right).sum())
    introduced_c   = int((cls_mask & hr_right_full_wrong).sum())
    rate_corr_c    = corrected_c / max(n_hr_wrong_c, 1)
    rate_intr_c    = introduced_c / max(n_hr_right_c, 1)
    print(f"{CLASS_NAMES[c]:<14}{n_gt_c:>9,}{corrected_c:>9,}{introduced_c:>9,}"
          f"{corrected_c - introduced_c:>+9,}"
          f"{_fmt_pct(rate_corr_c, 9)}"
          f"{_fmt_pct(rate_intr_c, 9)}")


# ─────────────────────────────────────────────────────────────────────
# 4. Per-tissue：整体准确率 + 改善率
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*W}")
print("【按 Tissue Type · 整体准确率】（像素级汇总；提升最大类一并标注）")
print(f"{'='*W}")
print(f"{'Tissue':<22}{'N':>4}  "
      f"{'HR':>9}{'消融':>10}{'本文':>10}{'Δ':>8}{'改善率':>9}  "
      f"{'本文最强提升类(Δ)':<28}")
print("-" * (W + 35))

tissue_names = sorted(tissue_pixels.keys())
per_tissue_summary = []

for tname in tissue_names:
    tp   = tissue_pixels[tname]
    tgt  = np.concatenate(tp['gt'])
    thr  = np.concatenate(tp['hr'])
    tabl = np.concatenate(tp['abl'])
    tful = np.concatenate(tp['full'])
    n    = tissue_n[tname]

    oa_h = float((thr  == tgt).mean())
    oa_a = float((tabl == tgt).mean())
    oa_f = float((tful == tgt).mean())
    delta = oa_f - oa_h
    imp   = _improvement_ratio(oa_h, oa_f)

    # 找出该 tissue 中提升最大的细胞类
    tcm_hr   = confusion_matrix(tgt, thr,  labels=labels)
    tcm_full = confusion_matrix(tgt, tful, labels=labels)
    tr_hr    = np.diag(tcm_hr)   / tcm_hr.sum(axis=1).clip(min=1)
    tr_full  = np.diag(tcm_full) / tcm_full.sum(axis=1).clip(min=1)

    cls_deltas = [
        (c, tr_full[c] - tr_hr[c])
        for c in range(1, 6) if tcm_hr.sum(axis=1)[c] >= 30   # 至少 30 像素才稳定
    ]
    if cls_deltas:
        best_c, best_d = max(cls_deltas, key=lambda x: x[1])
        worst_c, worst_d = min(cls_deltas, key=lambda x: x[1])
        best_str = f"{CLASS_NAMES[best_c]}({best_d:+.3f})"
    else:
        best_str = "—"
        best_c = best_d = worst_c = worst_d = None

    per_tissue_summary.append((tname, n, oa_h, oa_a, oa_f, delta, imp,
                               best_c, best_d, worst_c, worst_d))

    print(f"{tname:<22}{n:>4}  "
          f"{oa_h:>9.4f}{oa_a:>10.4f}{oa_f:>10.4f}"
          f"{delta:>+8.4f}{_fmt_pct(imp):>9}  "
          f"{best_str:<28}")


# ─────────────────────────────────────────────────────────────────────
# 5. Per-tissue 改善率排名
# ─────────────────────────────────────────────────────────────────────
print(f"\n── Per-tissue 改善率排名（HR→1.0 差距覆盖度）──")
ranked = sorted(
    [(t, n, oa_h, oa_f, imp)
     for t, n, oa_h, _, oa_f, _, imp, *_ in per_tissue_summary
     if not math.isnan(imp)],
    key=lambda x: -x[4],
)
for rank, (t, n, oa_h, oa_f, imp) in enumerate(ranked, 1):
    bar = '█' * max(0, int(imp * 100 * 0.4))
    print(f"  {rank:>2}. {t:<22}  改善率={imp*100:>+6.1f}%  "
          f"(HR={oa_h:.4f}→Full={oa_f:.4f})  N={n:>3}  {bar}")


# ─────────────────────────────────────────────────────────────────────
# 6. 可视化（保留）
# ─────────────────────────────────────────────────────────────────────
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

os.makedirs('./logs/downstream_Cellvitsamh', exist_ok=True)
save_path = './logs/downstream_Cellvitsamh/confusion_matrix_comparison_ablation.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n全局混淆矩阵图已保存到: {save_path}")

# Per-tissue 整体准确率对比条形图
n_tissues = len(tissue_names)
fig, ax = plt.subplots(figsize=(max(10, n_tissues * 1.2), 5))
x      = np.arange(n_tissues)
width  = 0.25
oa_hr_list  = [oa_h for _, _, oa_h, _, _, _, _, *_ in per_tissue_summary]
oa_abl_list = [oa_a for _, _, _, oa_a, _, _, _, *_ in per_tissue_summary]
oa_ful_list = [oa_f for _, _, _, _, oa_f, _, _, *_ in per_tissue_summary]

ax.bar(x - width, oa_hr_list,  width, label='HR baseline', color='#4878D0')
ax.bar(x,         oa_abl_list, width, label='Ablation',    color='#EE854A')
ax.bar(x + width, oa_ful_list, width, label='Full (ours)', color='#6ACC65')

ax.set_xticks(x)
ax.set_xticklabels(tissue_names, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Overall Accuracy (nucleus region)')
ax.set_title(f'Per-tissue Overall Accuracy — ensemble N={N_RUNS}')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

tissue_bar_path = './logs/downstream_Cellvitsamh/per_tissue_accuracy_comparison.png'
plt.savefig(tissue_bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Per-tissue 准确率条形图已保存到: {tissue_bar_path}")


# ─────────────────────────────────────────────────────────────────────
# 7. 说明
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*W}")
print("方法学说明：")
print("  - 所有指标均在 GT 核区域内、像素级汇总后计算（不是 patch-mean）")
print("  - 改善率   = (Full − HR) / (1 − HR)，HR→1.0 差距的覆盖度")
print("  - 错误纠正率 = 修复像素数 / HR 错误像素数")
print("  - 错误引入率 = 引入像素数 / HR 正确像素数（应 << 纠正率）")
print("  - 纠错/引入比 > 1 说明 DGFR 在做净正向的修改")
print("  - 全局 Overall_Acc / Intersect_Acc / PSNR / SSIM / Sem_MAE → "
      "见 compare_baseline_with_senmatic.py")
print(f"{'='*W}")


# ═══════════════════════════════════════════════════════════════════════
# JSON 导出(供跨模型评估脚本对比用;若 compare_baseline_with_senmatic.py
# 已产出 JSON 则合并;否则新建)
# ═══════════════════════════════════════════════════════════════════════
def _f(v):
    """安全 NaN/Inf → None。"""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    if hasattr(v, '__float__'):
        try:
            return float(v)
        except Exception:
            return None
    return v

# 各类召回率(全局)
recall_per_class = {}
for i in range(6):
    n_gt = int(cm_hr.sum(axis=1)[i])
    if n_gt == 0:
        continue
    recall_per_class[CLASS_NAMES[i]] = {
        'n_gt_pixels': n_gt,
        'recall_hr':   _f(recall_hr[i]),
        'recall_abl':  _f(recall_abl[i]),
        'recall_full': _f(recall_full[i]),
        'delta_full_minus_hr':  _f(recall_full[i] - recall_hr[i]),
        'improvement_ratio':    _f(_improvement_ratio(recall_hr[i], recall_full[i])),
    }

# Per-tissue 准确率(配 best class)
per_tissue_cm = {}
for tup in per_tissue_summary:
    tname, n, oa_h, oa_a, oa_f, delta, imp, best_c, best_d, worst_c, worst_d = tup
    per_tissue_cm[tname] = {
        'n': int(n),
        'overall_acc': {'hr': _f(oa_h), 'ablation': _f(oa_a), 'full': _f(oa_f)},
        'delta_full_minus_hr': _f(delta),
        'improvement_ratio':   _f(imp),
        'best_improving_class': CLASS_NAMES[best_c] if best_c is not None else None,
        'best_improvement':    _f(best_d) if best_d is not None else None,
        'worst_class':         CLASS_NAMES[worst_c] if worst_c is not None else None,
        'worst_delta':         _f(worst_d) if worst_d is not None else None,
    }

cm_results = {
    'recall_per_class': recall_per_class,
    'error_redirect': {
        'n_corrected':  int(n_corrected),
        'n_introduced': int(n_introduced),
        'redirect_ratio':  _f(redirect_ratio),
        'correction_rate':   _f(rate_corrected),
        'introduction_rate': _f(rate_introduced),
    },
    'per_tissue_cm': per_tissue_cm,
}

# 与 compare_baseline_with_senmatic.py 输出的 JSON 合并
json_path = './logs/downstream_Cellvitsamh/cellvit_results.json'
existing = {}
if os.path.exists(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    except Exception:
        existing = {}

existing['confusion_matrix'] = cm_results

os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(existing, f, ensure_ascii=False, indent=2)
print(f"\n  ✅ 混淆矩阵 JSON 已合并到: {json_path}")