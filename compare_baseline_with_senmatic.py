"""
compare_baseline_with_semantic.py
在相同推理条件下对比三组：
  - HR 基线：HR 直接送 CellViT，不经过任何纠正
  - 消融模型：无语义监督（只有 L_noise+L_rec+L_grad+L_tv，无 sem_tensor）
  - 本文方法：交集 Focal-CE + CellViT 软标签 sem_tensor

消融模型和本文方法使用完全相同的超参数，只有语义监督开关不同。

推理策略：
  对同一张 HR 用 N_RUNS 组不同随机噪声分别做单步推理，
  将 N_RUNS 次 CellViT 概率图在通道维度取平均后 argmax，
  得到最终分类预测（概率空间集成）。
  PSNR/SSIM 使用 N_RUNS 次 x0 的均值图像与 HR 比较。

【指标层级与统计粒度（v2 版改造）】
─────────────────────────────────────────────────────────────────────
  ┌───────────────────┬───────────┬──────────────┬─────────────────┐
  │ 指标              │ 本体层级  │ 主聚合方式   │ 分组维度         │
  ├───────────────────┼───────────┼──────────────┼─────────────────┤
  │ Overall_Acc       │ 像素      │ 像素级汇总   │ tissue ✓ / cell │
  │ (前 Dir_Acc)       │           │ +patch均值辅助│ type ✓(召回)    │
  ├───────────────────┼───────────┼──────────────┼─────────────────┤
  │ Intersect_Acc     │ 像素      │ 像素级汇总   │ tissue ✓        │
  ├───────────────────┼───────────┼──────────────┼─────────────────┤
  │ Sem_MAE           │ 像素      │ 全局平均     │ —(不分组)        │
  ├───────────────────┼───────────┼──────────────┼─────────────────┤
  │ PSNR / SSIM       │ patch     │ patch 平均   │ tissue ✓        │
  └───────────────────┴───────────┴──────────────┴─────────────────┘

【归一化的差异度量（v2 新增）】
  - Improvement Ratio   = (full − hr) / (1 − hr)
      含义：把 HR→上限 1.0 的差距看作 100%，DGFR 覆盖了多少？
      用于：Overall_Acc / Intersect_Acc / 各类召回率
  - Error Reduction     = (abl − full) / abl
      含义：相对于消融模型的误差，DGFR 减少了多少？
      用于：Sem_MAE
"""

import json
import math
import random
import sys
import os
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from diffusers import DDPMScheduler

from ddpm_dataset import PanNukeDataset
from unet_wrapper import create_model
from ddpm_utils import load_cellvit, predict_x0_from_noise_shared
from semantic_sr_loss import run_cellvit, build_sem_tensor_from_cellvit
from metrics import compute_psnr, compute_ssim

device = 'cuda'

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
    print(f"  加载 {ckpt_path}  epoch={ckpt.get('epoch', '?')}  "
          f"use_semantic={use_semantic}")
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

# ── 调度器与数据集 ──────────────────────────────────────────────────
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

# ── 分层采样：按 tissue type 均匀取样 ───────────────────────────────
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
print(f"\n分层采样结果（共 {len(sampled_indices)} 张，覆盖 {_n_types} 种 tissue）：")
for _t, _c in sorted(_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {_t:<25} {_c:>4} 张")


# ── 集成推理 ─────────────────────────────────────────────────────────
def infer_ensemble(unet, hr, sem=None):
    prob_sum = None
    x0_sum   = None
    t = torch.tensor([INFER_T], device=device)

    for _ in range(N_RUNS):
        noise    = torch.randn_like(hr)
        noisy_hr = scheduler.add_noise(hr, noise, t)
        with torch.no_grad():
            noise_pred = unet(torch.cat([hr, noisy_hr], dim=1),
                              t, semantic=sem).sample
            x0   = predict_x0_from_noise_shared(noisy_hr, noise_pred, t, scheduler)
            cv   = run_cellvit(cellvit, x0)
            prob = cv['nuclei_type_prob']

        prob_sum = prob if prob_sum is None else prob_sum + prob
        x0_sum   = x0   if x0_sum  is None else x0_sum  + x0

    mean_prob = prob_sum / N_RUNS
    mean_x0   = x0_sum  / N_RUNS
    lbl       = mean_prob.argmax(dim=1).squeeze(0).cpu()
    return mean_prob, lbl, mean_x0


# ─────────────────────────────────────────────────────────────────────
# 结果容器
# ─────────────────────────────────────────────────────────────────────
keys      = ['hr', 'ablation', 'full']

# ── 像素级容器（核心，用于像素汇总聚合）────────────────────────────
# 把所有有效 patch 的核像素拼起来，在末尾一次性算 confusion matrix /
# 整体准确率 / 各类召回率，避免"先 per-patch 平均再平均"带来的偏差。
all_gt_pixels   = []                                # 每个 patch 的 GT 像素向量
all_pred_pixels = {k: [] for k in keys}             # 三组的预测向量

# 交集区域的像素级容器
inter_gt_pixels   = []
inter_pred_pixels = {k: [] for k in keys}

# Per-tissue 像素列表：tissue_pixels[tname][k] = list of 1D arrays
tissue_pixels_gt   = defaultdict(list)
tissue_pixels_pred = defaultdict(lambda: {k: [] for k in keys})
tissue_pixels_inter_gt   = defaultdict(list)
tissue_pixels_inter_pred = defaultdict(lambda: {k: [] for k in keys})

# ── Patch 级容器（PSNR / SSIM / Sem_MAE / 辅助 patch-mean Acc）──────
patch_psnr = {k: [] for k in ['ablation', 'full']}
patch_ssim = {k: [] for k in ['ablation', 'full']}
patch_sem_mae = {k: [] for k in ['ablation', 'full']}
patch_acc  = {k: [] for k in keys}                   # 用于辅助 patch-mean

tissue_psnr = defaultdict(lambda: {k: [] for k in ['ablation', 'full']})
tissue_ssim = defaultdict(lambda: {k: [] for k in ['ablation', 'full']})

tissue_n = defaultdict(int)


print(f"\n开始评估（INFER_T={INFER_T}，N_RUNS={N_RUNS}，样本数={N_SAMPLES}）...")
print(f"{'idx':>4}  {'Tissue':<18}  {'HR_acc':>8}  {'Abl_acc':>9}  {'Full_acc':>9}  "
      f"{'Abl_PSNR':>10}  {'Full_PSNR':>10}")

n_valid = 0
for i in sampled_indices:

    sample     = dataset[i]
    hr_cpu     = sample['hr']
    gt_lbl     = sample['gt_label_map']
    gt_nuc     = sample['gt_nuc_mask'].bool()
    type_name  = sample['type_name']

    hr = hr_cpu.unsqueeze(0).to(device)

    with torch.no_grad():
        hr_cv   = run_cellvit(cellvit, hr)
        hr_lbl  = hr_cv['nuclei_type_label'].squeeze(0).cpu()
        hr_prob = hr_cv['nuclei_type_prob']
        sem = build_sem_tensor_from_cellvit(
            hr_cv['nuclei_type_prob'],
            hr_cv['nuclei_nuc_prob'],
        )

    abl_prob,  abl_lbl,  abl_x0  = infer_ensemble(unet_ablation, hr, sem=None)
    full_prob, full_lbl, full_x0 = infer_ensemble(unet_full,     hr, sem=sem)

    gt_np     = gt_lbl.numpy()
    gt_nuc_np = gt_nuc.numpy()
    hr_np     = hr_lbl.numpy()
    abl_np    = abl_lbl.numpy()
    full_np   = full_lbl.numpy()

    cell_mask  = gt_nuc_np & (gt_np > 0)
    inter_mask = cell_mask & (hr_np == gt_np)

    if cell_mask.sum() == 0:
        continue

    # ── 收集像素（核心改动：所有像素级指标都从这些数组重算）──────
    gt_cells   = gt_np[cell_mask]
    all_gt_pixels.append(gt_cells)
    all_pred_pixels['hr'].append(hr_np[cell_mask])
    all_pred_pixels['ablation'].append(abl_np[cell_mask])
    all_pred_pixels['full'].append(full_np[cell_mask])

    if inter_mask.sum() > 0:
        gt_inter = gt_np[inter_mask]
        inter_gt_pixels.append(gt_inter)
        inter_pred_pixels['hr'].append(hr_np[inter_mask])
        inter_pred_pixels['ablation'].append(abl_np[inter_mask])
        inter_pred_pixels['full'].append(full_np[inter_mask])

        tissue_pixels_inter_gt[type_name].append(gt_inter)
        tissue_pixels_inter_pred[type_name]['hr'].append(hr_np[inter_mask])
        tissue_pixels_inter_pred[type_name]['ablation'].append(abl_np[inter_mask])
        tissue_pixels_inter_pred[type_name]['full'].append(full_np[inter_mask])

    tissue_pixels_gt[type_name].append(gt_cells)
    tissue_pixels_pred[type_name]['hr'].append(hr_np[cell_mask])
    tissue_pixels_pred[type_name]['ablation'].append(abl_np[cell_mask])
    tissue_pixels_pred[type_name]['full'].append(full_np[cell_mask])

    # ── Patch 级指标 ────────────────────────────────────────────────
    # PSNR / SSIM 是图像重建指标，只对 ablation / full 算（HR 是 GT）
    p_abl = compute_psnr(abl_x0.cpu(),  hr.cpu())
    p_ful = compute_psnr(full_x0.cpu(), hr.cpu())
    s_abl = compute_ssim(abl_x0.cpu(),  hr.cpu())
    s_ful = compute_ssim(full_x0.cpu(), hr.cpu())

    patch_psnr['ablation'].append(p_abl); patch_psnr['full'].append(p_ful)
    patch_ssim['ablation'].append(s_abl); patch_ssim['full'].append(s_ful)
    tissue_psnr[type_name]['ablation'].append(p_abl)
    tissue_psnr[type_name]['full'].append(p_ful)
    tissue_ssim[type_name]['ablation'].append(s_abl)
    tissue_ssim[type_name]['full'].append(s_ful)

    # Sem_MAE：概率分布的距离，本质像素级，但用 patch 平均做监控（不分 tissue）
    mae_abl = (abl_prob.cpu()  - hr_prob.cpu()).abs().mean(dim=1).squeeze(0).numpy()[gt_nuc_np].mean()
    mae_ful = (full_prob.cpu() - hr_prob.cpu()).abs().mean(dim=1).squeeze(0).numpy()[gt_nuc_np].mean()
    patch_sem_mae['ablation'].append(float(mae_abl))
    patch_sem_mae['full'].append(float(mae_ful))

    # Patch-mean accuracy（辅助指标，不是主聚合）
    patch_acc['hr'].append((hr_np[cell_mask]   == gt_np[cell_mask]).mean())
    patch_acc['ablation'].append((abl_np[cell_mask]  == gt_np[cell_mask]).mean())
    patch_acc['full'].append((full_np[cell_mask] == gt_np[cell_mask]).mean())

    tissue_n[type_name] += 1

    if n_valid % 20 == 0:
        print(f"{n_valid:>4}  {type_name:<18}  "
              f"{patch_acc['hr'][-1]:>8.4f}  "
              f"{patch_acc['ablation'][-1]:>9.4f}  "
              f"{patch_acc['full'][-1]:>9.4f}  "
              f"{p_abl:>10.2f}  {p_ful:>10.2f}")
    n_valid += 1


# ─────────────────────────────────────────────────────────────────────
# 拼接像素级数组（一次性，用于全局聚合）
# ─────────────────────────────────────────────────────────────────────
gt_all   = np.concatenate(all_gt_pixels)
pred_all = {k: np.concatenate(all_pred_pixels[k]) for k in keys}

if inter_gt_pixels:
    gt_inter_all   = np.concatenate(inter_gt_pixels)
    pred_inter_all = {k: np.concatenate(inter_pred_pixels[k]) for k in keys}
else:
    gt_inter_all   = np.array([], dtype=np.int64)
    pred_inter_all = {k: np.array([], dtype=np.int64) for k in keys}


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────
def _mean(lst):
    vals = [v for v in lst if not (isinstance(v, float) and v != v)]
    return float(np.mean(vals)) if vals else float('nan')

def _improvement_ratio(hr_v, full_v):
    """覆盖了 HR→1.0 多少差距。HR=1 时返回 NaN。"""
    if not (0 <= hr_v <= 1):
        return float('nan')
    if 1.0 - hr_v < 1e-6:
        return float('nan')
    return (full_v - hr_v) / (1.0 - hr_v)

def _error_reduction(abl_err, full_err):
    """误差减少的百分比。abl_err≈0 时返回 NaN。"""
    if abs(abl_err) < 1e-9:
        return float('nan')
    return (abl_err - full_err) / abl_err

def _arrow(delta, eps=1e-3):
    return '↑' if delta > eps else ('↓' if delta < -eps else '—')


# ═══════════════════════════════════════════════════════════════════════
# 1. 全局指标汇总（主表）
# ═══════════════════════════════════════════════════════════════════════
W = 95
print("\n" + "=" * W)
print("【全局指标】（按指标层级用最合适的聚合方式）")
print("=" * W)

# ── 像素级指标：Overall_Acc ────────────────────────────────────────
oa = {k: float((pred_all[k] == gt_all).mean()) for k in keys}
imp_oa_full = _improvement_ratio(oa['hr'], oa['full'])
imp_oa_abl  = _improvement_ratio(oa['hr'], oa['ablation'])

# 辅助：patch-mean 版本（用于和审稿人解释聚合方式不同的差异）
oa_patch = {k: _mean(patch_acc[k]) for k in keys}

# ── 像素级指标：Intersect_Acc（仅 GT∩CellViT(HR)判对的像素）────────
if len(gt_inter_all) > 0:
    ia = {k: float((pred_inter_all[k] == gt_inter_all).mean()) for k in keys}
    imp_ia_full = _improvement_ratio(ia['hr'], ia['full'])
    imp_ia_abl  = _improvement_ratio(ia['hr'], ia['ablation'])
else:
    ia = {k: float('nan') for k in keys}
    imp_ia_full = imp_ia_abl = float('nan')

# ── Patch 级指标：Sem_MAE（不分 tissue，HR 自身 vs HR 是 0）────────
sm_abl = _mean(patch_sem_mae['ablation'])
sm_ful = _mean(patch_sem_mae['full'])
err_red_sm = _error_reduction(sm_abl, sm_ful)

# ── Patch 级指标：PSNR / SSIM（不和 HR 比；HR 自身 PSNR=∞）─────────
ps_abl = _mean(patch_psnr['ablation']); ps_ful = _mean(patch_psnr['full'])
ss_abl = _mean(patch_ssim['ablation']); ss_ful = _mean(patch_ssim['full'])

# 像素级主表
print(f"\n── 像素级指标（在所有 patch 的核像素汇总后计算）──")
print(f"{'指标':<22}{'HR基线':>10}{'消融':>10}{'本文':>10}"
      f"{'Δ(本文-HR)':>13}{'Δ(本文-消融)':>14}{'改善率':>9}")
print("-" * W)
print(f"{'Overall_Acc':<22}"
      f"{oa['hr']:>10.4f}{oa['ablation']:>10.4f}{oa['full']:>10.4f}"
      f"{oa['full']-oa['hr']:>+13.4f}"
      f"{oa['full']-oa['ablation']:>+14.4f}"
      f"{imp_oa_full*100 if not math.isnan(imp_oa_full) else float('nan'):>8.2f}%")
print(f"{'  (patch-mean 辅助)':<22}"
      f"{oa_patch['hr']:>10.4f}{oa_patch['ablation']:>10.4f}{oa_patch['full']:>10.4f}"
      f"{oa_patch['full']-oa_patch['hr']:>+13.4f}{'—':>14}{'—':>9}")
print(f"{'Intersect_Acc':<22}"
      f"{ia['hr']:>10.4f}{ia['ablation']:>10.4f}{ia['full']:>10.4f}"
      f"{ia['full']-ia['hr']:>+13.4f}"
      f"{ia['full']-ia['ablation']:>+14.4f}"
      f"{imp_ia_full*100 if not math.isnan(imp_ia_full) else float('nan'):>8.2f}%")

# Patch 级主表
print(f"\n── Patch 级指标（patch 平均；HR 自身 PSNR=∞、SSIM=1，无意义）──")
print(f"{'指标':<22}{'HR':>10}{'消融':>10}{'本文':>10}"
      f"{'Δ(本文-消融)':>14}{'误差减少':>10}")
print("-" * W)
print(f"{'Sem_MAE':<22}{'0.0000':>10}"
      f"{sm_abl:>10.4f}{sm_ful:>10.4f}"
      f"{sm_ful-sm_abl:>+14.4f}"
      f"{err_red_sm*100 if not math.isnan(err_red_sm) else float('nan'):>9.2f}%")
print(f"{'PSNR (dB)':<22}{'∞':>10}"
      f"{ps_abl:>10.2f}{ps_ful:>10.2f}"
      f"{ps_ful-ps_abl:>+14.2f}{'—':>10}")
print(f"{'SSIM':<22}{'1.0000':>10}"
      f"{ss_abl:>10.4f}{ss_ful:>10.4f}"
      f"{ss_ful-ss_abl:>+14.4f}{'—':>10}")
print("=" * W)

# 关键提升摘要
print(f"\n关键提升：")
print(f"  Overall_Acc  本文 vs HR基线  绝对差={oa['full']-oa['hr']:+.4f}  "
      f"改善率={imp_oa_full*100:.1f}% (HR→1.0 差距覆盖)")
print(f"  Overall_Acc  本文 vs 消融   绝对差={oa['full']-oa['ablation']:+.4f}")
print(f"  Sem_MAE      本文 vs 消融   误差减少={err_red_sm*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# 2. 按 Tissue Type 分组（仅展示有意义层级的聚合指标）
#    注：各细胞类别召回率 / 混淆矩阵分析 / 改善率排名 → confusion_matrix_compare.py
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * W)
print("【按 Tissue Type 分组】（去掉 Sem_MAE 的 per-tissue 拆分；不必信息）")
print("=" * W)

tissue_names = sorted(tissue_pixels_gt.keys())

# 表头
print(f"\n{'Tissue':<22}{'N':>4}  "
      f"{'Overall_Acc':^32}  {'Intersect_Acc':^32}")
print(f"{'':<22}{'':<4}  "
      f"{'HR':>9}{'消融':>10}{'本文':>10}{'改善%':>7}  "
      f"{'HR':>9}{'消融':>10}{'本文':>10}{'改善%':>7}")
print("-" * (28 + 32 + 32 + 6))

per_tissue_json = {}   # 用于 JSON 导出

for tname in tissue_names:
    n  = tissue_n[tname]
    gt = np.concatenate(tissue_pixels_gt[tname])

    pr = {k: np.concatenate(tissue_pixels_pred[tname][k]) for k in keys}
    oa_t = {k: float((pr[k] == gt).mean()) for k in keys}
    imp_oa = _improvement_ratio(oa_t['hr'], oa_t['full'])

    if tissue_pixels_inter_gt[tname]:
        gt_i = np.concatenate(tissue_pixels_inter_gt[tname])
        pr_i = {k: np.concatenate(tissue_pixels_inter_pred[tname][k]) for k in keys}
        ia_t = {k: float((pr_i[k] == gt_i).mean()) for k in keys}
        imp_ia = _improvement_ratio(ia_t['hr'], ia_t['full'])
    else:
        ia_t = {k: float('nan') for k in keys}
        imp_ia = float('nan')

    # 收集到 JSON 容器
    per_tissue_json[tname] = {
        'n': int(n),
        'overall_acc': {k: oa_t[k] for k in keys},
        'intersect_acc': {k: ia_t[k] for k in keys},
        'improvement_ratio_oa': float(imp_oa) if not math.isnan(imp_oa) else None,
        'improvement_ratio_ia': float(imp_ia) if not math.isnan(imp_ia) else None,
    }

    imp_oa_s = f"{imp_oa*100:>5.1f}%" if not math.isnan(imp_oa) else f"{'N/A':>6}"
    imp_ia_s = f"{imp_ia*100:>5.1f}%" if not math.isnan(imp_ia) else f"{'N/A':>6}"

    print(f"{tname:<22}{n:>4}  "
          f"{oa_t['hr']:>9.4f}{oa_t['ablation']:>10.4f}{oa_t['full']:>10.4f}"
          f"{imp_oa_s:>7}  "
          f"{ia_t['hr']:>9.4f}{ia_t['ablation']:>10.4f}{ia_t['full']:>10.4f}"
          f"{imp_ia_s:>7}")

# Patch 级指标（PSNR / SSIM）单独一表，不和像素级混
print(f"\n{'Tissue':<22}{'N':>4}  "
      f"{'PSNR (dB)':^22}  {'SSIM':^22}")
print(f"{'':<22}{'':<4}  "
      f"{'消融':>10}{'本文':>10}{'Δ':>5}  "
      f"{'消融':>10}{'本文':>10}{'Δ':>5}")
print("-" * (28 + 22 + 22 + 4))
for tname in tissue_names:
    n = tissue_n[tname]
    ps_a = _mean(tissue_psnr[tname]['ablation'])
    ps_f = _mean(tissue_psnr[tname]['full'])
    ss_a = _mean(tissue_ssim[tname]['ablation'])
    ss_f = _mean(tissue_ssim[tname]['full'])
    d_ps = ps_f - ps_a
    d_ss = ss_f - ss_a
    print(f"{tname:<22}{n:>4}  "
          f"{ps_a:>10.2f}{ps_f:>10.2f}{d_ps:>+5.2f}  "
          f"{ss_a:>10.4f}{ss_f:>10.4f}{d_ss:>+5.3f}")

print("=" * W)


print("\n说明：")
print("  - Overall_Acc / Intersect_Acc = 像素级汇总（拼接所有 patch 核像素后计算）")
print("  - PSNR / SSIM / Sem_MAE = patch 平均（图像/分布级指标，不能像素级聚合）")
print("  - 改善率 = (Full − HR) / (1 − HR)，即 HR→1.0 差距覆盖了多少；消除任务难度")
print("  - 误差减少 = (消融 − 本文) / 消融，即 DGFR 相对消融的 Sem_MAE 降低比例")
print("  - 各细胞类别召回率 / 混淆矩阵 / 错误重定向 / 改善率排名 → 见 confusion_matrix_compare.py")


# ═══════════════════════════════════════════════════════════════════════
# JSON 导出(供跨模型评估脚本对比用)
# ═══════════════════════════════════════════════════════════════════════
def _f(v):
    """安全 NaN/Inf → None 转换。"""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return float(v) if hasattr(v, '__float__') else v

results_json = {
    'evaluator': 'CellViT-SAM-H',
    'n_samples': int(n_valid),
    'global': {
        'overall_acc':   {k: _f(oa[k])   for k in keys},
        'intersect_acc': {k: _f(ia[k])   for k in keys},
        'overall_acc_patch_mean': {k: _f(oa_patch[k]) for k in keys},
        'sem_mae': {
            'ablation': _f(sm_abl),
            'full':     _f(sm_ful),
            'error_reduction': _f(err_red_sm),
        },
        'psnr': {'ablation': _f(ps_abl), 'full': _f(ps_ful)},
        'ssim': {'ablation': _f(ss_abl), 'full': _f(ss_ful)},
        'improvement_ratio_oa_full_vs_hr':       _f(imp_oa_full),
        'improvement_ratio_oa_ablation_vs_hr':   _f(imp_oa_abl),
    },
    'per_tissue': per_tissue_json,
}

# 默认输出路径(与图保存目录同根)
json_out_path = './logs/downstream_Cellvitsamh/cellvit_results.json'
os.makedirs(os.path.dirname(json_out_path) or '.', exist_ok=True)
with open(json_out_path, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print(f"\n  ✅ JSON 结果已导出: {json_out_path}")
print(f"     供 cross_model_eval_hovernet.py 通过 --cellvit_results_json 参数读取")