"""
cross_model_eval_hovernet.py
跨模型泛化性验证：用 HoVer-Net 替代 CellViT 作为评估判别器。

【实验目的】
─────────────────────────────────────────────────────────────────────────────
SPM-UNet 训练时使用 CellViT 作为语义教师（判别器），
本脚本用一个训练时从未见过的判别器（HoVer-Net）来评估推理结果，
验证图像修改是否具有跨模型泛化性，即：
  - 修改不是过拟合到 CellViT 的特定决策边界
  - 修改对不同架构的分类器都能带来诊断准确性提升

【对比三组】
  HR 基线    : HR 直接送 HoVer-Net，不经过任何纠正
  消融模型   : 无语义监督（只有重建损失）
  本文方法   : 交集 Focal-CE + CellViT 软标签 sem_tensor

【推理策略】
  对同一张 HR 用 N_RUNS 组不同随机噪声分别做单步推理，
  将 N_RUNS 次 HoVer-Net 概率图在通道维度取平均后 argmax，
  得到最终分类预测（概率空间集成）。

【HoVer-Net 特殊处理】
  PanNuke fast 模式输入 256×256，输出 164×164。
  所有指标在 164×164 的中心裁剪区域内计算（GT 也做相同裁剪）。
  不做任何上采样，避免引入插值伪影影响评估公正性。

【类别定义】（与 CellViT 一致）
  0: Background  1: Neoplastic  2: Inflammatory
  3: Connective  4: Dead        5: Epithelial

【输出指标】
  Dir_Acc       : HoVer-Net(pred) vs GT 的类别一致率
  Intersect_Acc : 仅在交集区域（GT∩HoVer-Net(HR)判对）的一致率
  各类召回率     : 每个细胞类别的 Recall
  整体准确率     : Overall Accuracy
  退步/进步分析  : HR&消融均正确但本文错误 / 仅本文正确
  以上指标按 tissue type 分组输出

【使用方法】
  python cross_model_eval_hovernet.py

  可选参数：
    --hovernet_repo   HoVer-Net 代码仓库路径（默认 /home/xuwen/DDPM/HoVer-net）
    --hovernet_ckpt   HoVer-Net 权重路径
    --n_samples       评估样本数（默认 200）
    --n_runs          集成推理次数（默认 5）
    --infer_t         推理噪声强度（默认 200）
"""

import argparse
import json
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── 路径配置（可通过命令行覆盖） ────────────────────────────────────
HOVERNET_REPO   = '/home/xuwen/DDPM/HoVer-net'
HOVERNET_CKPT   = '/home/xuwen/DDPM/HoVer-net/hovernet_fast_pannuke_type_tf2pytorch.tar'
CELLVIT_PATH    = '/home/xuwen/DDPM/CellViT/CellViT-SAM-H-x40.pth'
CELLVIT_REPO    = '/home/xuwen/DDPM/CellViT'
UNET_ABL_PATH   = '/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_ablation.pth'
UNET_FULL_PATH  = '/home/xuwen/DDPM/logs/checkpoints_correction_samh/best_unet_correction.pth'
PANNUKE_FOLD3   = '/data/xuwen/PanNuke/Fold 3'

CLASS_NAMES = ['Background', 'Neoplastic', 'Inflammatory',
               'Connective', 'Dead', 'Epithelial']

# HoVer-Net fast 模式：输入 256×256，输出 164×164
HOVERNET_OUTPUT_SIZE = 164
HOVERNET_CROP_OFFSET = (256 - 164) // 2   # = 46


# ─────────────────────────────────────────────────────────────────────────────
# HoVer-Net 加载与推理封装
# ─────────────────────────────────────────────────────────────────────────────

def load_hovernet(repo_path, ckpt_path, device='cuda'):
    """
    加载 PanNuke 预训练 HoVer-Net（fast 模式，nr_types=6）。

    使用临时 sys.path 操作 + 导入后清理，避免 HoVer-Net 的 `models` 包
    与 CellViT 的 `models` 包冲突。

    Args:
        repo_path : HoVer-Net 代码仓库根目录
        ckpt_path : 权重 .tar 文件路径
        device    : 设备

    Returns:
        hovernet : 加载好并冻结参数的 HoVer-Net 模型
    """
    import importlib

    # ── 1. 临时把 HoVer-Net 路径加到最前面 ────────────────────────────
    # 记录当前 sys.path 和已导入的 models 模块，以便加载后恢复
    _had_models = 'models' in sys.modules
    _old_models = sys.modules.get('models')
    _old_hovernet_models = {
        k: v for k, v in sys.modules.items()
        if k == 'models' or k.startswith('models.')
    }

    # 清除已有的 models 缓存（可能来自 CellViT）
    for k in list(sys.modules.keys()):
        if k == 'models' or k.startswith('models.'):
            del sys.modules[k]

    sys.path.insert(0, repo_path)

    # ── 2. 导入 HoVer-Net 的 create_model ────────────────────────────
    from models.hovernet.net_desc import create_model as create_hovernet

    # ── 3. 加载权重并推断 nr_types ────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['desc'] if 'desc' in ckpt else ckpt

    # tp 分支最后一层: decoder.tp.u0.conv.weight → shape [nr_types, 64, 1, 1]
    tp_key = 'decoder.tp.u0.conv.weight'
    if tp_key in state_dict:
        nr_types = state_dict[tp_key].shape[0]
    else:
        nr_types = 6
        for k, v in state_dict.items():
            if 'tp' in k and 'conv.weight' in k:
                nr_types = v.shape[0]

    print(f"HoVer-Net: 推断 nr_types={nr_types}")

    hovernet = create_hovernet(mode='fast', nr_types=nr_types)
    hovernet.load_state_dict(state_dict, strict=True)
    hovernet = hovernet.to(device)
    hovernet.eval()

    for p in hovernet.parameters():
        p.requires_grad = False

    # ── 4. 清理 sys.path 和 sys.modules，恢复 CellViT 的 models ─────
    if repo_path in sys.path:
        sys.path.remove(repo_path)

    # 清除 HoVer-Net 导入的 models 缓存
    for k in list(sys.modules.keys()):
        if k == 'models' or k.startswith('models.'):
            del sys.modules[k]

    # 恢复之前的 models 模块缓存（如果有的话）
    if _had_models and _old_models is not None:
        sys.modules['models'] = _old_models
        for k, v in _old_hovernet_models.items():
            if k != 'models':  # 只恢复非 hovernet 的子模块
                pass  # 让后续 import 自然重新导入

    print(f"✓ HoVer-Net 加载完成（fast 模式，nr_types={nr_types}）")
    return hovernet, nr_types


@torch.no_grad()
def run_hovernet(hovernet, img_01, nr_types=6):
    """
    对 [0,1] 范围的图像运行 HoVer-Net 推理。

    Args:
        hovernet : HoVer-Net 模型
        img_01   : [B, 3, H, W] float32, 取值 [0, 1]
        nr_types : 类别数

    Returns:
        dict:
          nuclei_type_prob  : [B, nr_types, h, w]  softmax 概率
          nuclei_type_label : [B, h, w]             argmax 类别
          nuclei_nuc_prob   : [B, h, w]             核概率（np 分支 softmax[:,1]）

    注意：HoVer-Net forward 内部会做 /255，所以这里输入要 *255
    输出尺寸 h,w = 164（fast 模式下比输入小）
    """
    # HoVer-Net 期望输入范围 [0, 255]（内部做 /255）
    imgs_255 = img_01 * 255.0

    out = hovernet(imgs_255)

    # tp: [B, nr_types, h, w] logits → softmax 概率
    tp_logits = out['tp']
    tp_prob   = F.softmax(tp_logits, dim=1)
    tp_label  = tp_prob.argmax(dim=1)          # [B, h, w]

    # np: [B, 2, h, w] logits → softmax，取通道1为核概率
    np_logits = out['np']
    np_prob   = F.softmax(np_logits, dim=1)
    nuc_prob  = np_prob[:, 1]                  # [B, h, w]

    return dict(
        nuclei_type_prob  = tp_prob,
        nuclei_type_label = tp_label,
        nuclei_nuc_prob   = nuc_prob,
    )


def center_crop_2d(tensor, crop_size):
    """
    对 2D tensor (或 batch) 做中心裁剪。

    支持:
      [H, W]       → [crop_size, crop_size]
      [B, H, W]    → [B, crop_size, crop_size]
      [B, C, H, W] → [B, C, crop_size, crop_size]
    """
    if tensor.ndim == 2:
        H, W = tensor.shape
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2
        return tensor[y0:y0+crop_size, x0:x0+crop_size]
    elif tensor.ndim == 3:
        H, W = tensor.shape[-2], tensor.shape[-1]
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2
        return tensor[:, y0:y0+crop_size, x0:x0+crop_size]
    elif tensor.ndim == 4:
        H, W = tensor.shape[-2], tensor.shape[-1]
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2
        return tensor[:, :, y0:y0+crop_size, x0:x0+crop_size]
    else:
        raise ValueError(f"Unsupported tensor ndim={tensor.ndim}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='跨模型泛化性验证：用 HoVer-Net 替代 CellViT 评估')
    parser.add_argument('--hovernet_repo', default=HOVERNET_REPO)
    parser.add_argument('--hovernet_ckpt', default=HOVERNET_CKPT)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_runs',    type=int, default=5)
    parser.add_argument('--infer_t',   type=int, default=200)
    parser.add_argument('--device',    default='cuda')
    parser.add_argument(
        '--cellvit_results_json',
        default='./logs/downstream_Cellvitsamh/cellvit_results.json',
        help='CellViT 评估结果 JSON 路径(由 compare_baseline_with_senmatic.py + '
             'confusion_matrix_compare.py 产出)。若文件不存在,跳过反差列。'
    )
    args = parser.parse_args()

    device    = args.device
    N_SAMPLES = args.n_samples
    N_RUNS    = args.n_runs
    INFER_T   = args.infer_t
    OUT_SIZE  = HOVERNET_OUTPUT_SIZE

    # ── 改善率工具 ──────────────────────────────────────────────────
    def _improvement_ratio(hr_val, full_val, upper=1.0):
        """适用于"越高越好"的指标。Δ / (上限 − HR)。"""
        gap = upper - hr_val
        if gap <= 1e-8 or hr_val != hr_val:
            return float('nan')
        return (full_val - hr_val) / gap

    def _fmt_pct(r):
        if isinstance(r, float) and r != r:
            return f"{'N/A':>7}"
        return f"{r*100:>+6.1f}%"

    # ── 加载 CellViT 评估结果 JSON(可选)─────────────────────────────
    cellvit_json = None
    if os.path.exists(args.cellvit_results_json):
        try:
            with open(args.cellvit_results_json, 'r', encoding='utf-8') as f:
                cellvit_json = json.load(f)
            print(f"\n  ✅ 已加载 CellViT 评估结果: {args.cellvit_results_json}")
        except Exception as e:
            print(f"\n  ⚠️ 无法读取 CellViT JSON ({e}),将跳过反差列")
            cellvit_json = None
    else:
        print(f"\n  ⚠️ 未找到 CellViT JSON ({args.cellvit_results_json}),"
              f"将只显示 HoVer-Net 自己的指标")

    # ── 添加项目路径 ─────────────────────────────────────────────────
    sys.path.insert(0, CELLVIT_REPO)
    sys.path.insert(0, '/home/xuwen/DDPM')

    from diffusers import DDPMScheduler
    from sklearn.metrics import confusion_matrix

    from ddpm_dataset import PanNukeDataset
    from unet_wrapper import create_model
    from ddpm_utils import load_cellvit, predict_x0_from_noise_shared
    from semantic_sr_loss import run_cellvit, build_sem_tensor_from_cellvit

    # ── 加载 HoVer-Net ───────────────────────────────────────────────
    print("=" * 70)
    print("跨模型泛化性验证：HoVer-Net 作为评估判别器")
    print("=" * 70)

    # 注意加载顺序：先加载 CellViT，再加载 HoVer-Net
    # 因为两者都有 `models` 顶层包，需要避免 sys.path 冲突
    # load_hovernet 内部会临时操作 sys.path 并在加载后清理

    # ── 加载 CellViT（仅用于生成 sem_tensor，不用于评估） ────────────
    print("\n加载 CellViT（仅用于生成 sem_tensor）...")
    cellvit = load_cellvit(
        model_path        = CELLVIT_PATH,
        cellvit_repo_path = CELLVIT_REPO,
        device            = device,
        variant           = 'sam_h',
    )

    print("\n加载 HoVer-Net...")
    hovernet, nr_types = load_hovernet(
        args.hovernet_repo, args.hovernet_ckpt, device)

    # ── 加载 UNet 模型 ───────────────────────────────────────────────
    def load_unet(ckpt_path, use_semantic):
        unet = create_model(use_semantic=use_semantic).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        unet.load_state_dict(ckpt['model_state_dict'])
        unet.eval()
        print(f"  加载 {ckpt_path}  epoch={ckpt.get('epoch', '?')}  "
              f"use_semantic={use_semantic}")
        return unet

    print("\n加载 UNet 模型...")
    unet_ablation = load_unet(UNET_ABL_PATH,  use_semantic=False)
    unet_full     = load_unet(UNET_FULL_PATH, use_semantic=True)

    # ── 数据集与调度器 ───────────────────────────────────────────────
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    dataset   = PanNukeDataset(
        fold_dirs   = [PANNUKE_FOLD3],
        target_size = 256,
    )
    print(f"\n测试集大小: {len(dataset)} 张")

    torch.manual_seed(42)
    random.seed(42)

    # ── 分层采样 ─────────────────────────────────────────────────────
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

    _type_counts = {t: 0 for t in _type_to_idx}
    for _i in sampled_indices:
        _type_counts[dataset[_i]['type_name']] += 1
    print(f"\n分层采样结果（共 {len(sampled_indices)} 张，覆盖 {_n_types} 种 tissue）：")
    for _t, _c in sorted(_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {_t:<25} {_c:>4} 张")

    # ── 集成推理 ─────────────────────────────────────────────────────
    def infer_ensemble_hovernet(unet, hr, sem=None):
        """
        对同一张 HR 用 N_RUNS 组不同随机噪声做单步推理，
        将 N_RUNS 次 HoVer-Net 概率图取平均后 argmax。

        Returns:
          mean_prob : [1, nr_types, OUT_SIZE, OUT_SIZE]
          lbl       : [OUT_SIZE, OUT_SIZE] numpy int
        """
        prob_sum = None
        t = torch.tensor([INFER_T], device=device)

        for _ in range(N_RUNS):
            noise    = torch.randn_like(hr)
            noisy_hr = scheduler.add_noise(hr, noise, t)
            with torch.no_grad():
                noise_pred = unet(torch.cat([hr, noisy_hr], dim=1),
                                  t, semantic=sem).sample
                x0 = predict_x0_from_noise_shared(
                    noisy_hr, noise_pred, t, scheduler)
                hv_out = run_hovernet(hovernet, x0, nr_types)
                prob   = hv_out['nuclei_type_prob']   # [1, C, 164, 164]

            prob_sum = prob if prob_sum is None else prob_sum + prob

        mean_prob = prob_sum / N_RUNS
        lbl       = mean_prob.argmax(dim=1).squeeze(0).cpu().numpy()
        return mean_prob, lbl

    # ── 结果容器 ─────────────────────────────────────────────────────
    # 全局像素列表
    all_gt, all_hr, all_abl, all_full = [], [], [], []

    # Per-tissue
    tissue_pixels = defaultdict(
        lambda: {'gt': [], 'hr': [], 'abl': [], 'full': []})
    tissue_n = defaultdict(int)

    # ── 推理循环 ─────────────────────────────────────────────────────
    print(f"\n开始评估（INFER_T={INFER_T}，N_RUNS={N_RUNS}，"
          f"样本数={len(sampled_indices)}）...")
    print(f"评估区域：中心 {OUT_SIZE}×{OUT_SIZE}（HoVer-Net fast 原生输出尺寸）")
    print(f"\n{'idx':>4}  {'Tissue':<18}  {'HR_acc':>8}  {'Abl_acc':>9}  "
          f"{'Full_acc':>9}")

    n_valid = 0
    for i in sampled_indices:

        sample    = dataset[i]
        hr_cpu    = sample['hr']
        gt_lbl    = sample['gt_label_map']      # [256, 256]
        gt_nuc    = sample['gt_nuc_mask'].bool() # [256, 256]
        type_name = sample['type_name']

        # 中心裁剪 GT 到 164×164（与 HoVer-Net 输出对齐）
        gt_lbl_crop = center_crop_2d(gt_lbl, OUT_SIZE)       # [164, 164]
        gt_nuc_crop = center_crop_2d(gt_nuc, OUT_SIZE)       # [164, 164]

        gt_np     = gt_lbl_crop.numpy()
        gt_nuc_np = gt_nuc_crop.numpy()
        cell_mask = gt_nuc_np & (gt_np > 0)

        if cell_mask.sum() < 10:
            continue

        hr = hr_cpu.unsqueeze(0).to(device)

        with torch.no_grad():
            # CellViT(HR) → sem_tensor（供完整模型使用）
            hr_cv = run_cellvit(cellvit, hr)
            sem   = build_sem_tensor_from_cellvit(
                hr_cv['nuclei_type_prob'],
                hr_cv['nuclei_nuc_prob'],
            )

            # HoVer-Net(HR)：评估基准
            hr_hv  = run_hovernet(hovernet, hr, nr_types)
            hr_lbl = hr_hv['nuclei_type_label'].squeeze(0).cpu().numpy()
            # hr_lbl 已经是 164×164

        # 消融模型集成推理
        _, abl_lbl = infer_ensemble_hovernet(unet_ablation, hr, sem=None)

        # 完整模型集成推理
        _, full_lbl = infer_ensemble_hovernet(unet_full, hr, sem=sem)

        # 类别映射检查：HoVer-Net 的 PanNuke 类别编号可能与 GT 不同
        # PanNuke GT: 0=bg, 1=Neoplastic, 2=Inflammatory, 3=Connective,
        #             4=Dead, 5=Epithelial
        # HoVer-Net PanNuke checkpoint 通常也是同样的映射
        # （如果 nr_types=6，则 0=bg, 1-5 对应 5 种细胞类型）

        # 收集全局像素
        all_gt.append(gt_np[cell_mask])
        all_hr.append(hr_lbl[cell_mask])
        all_abl.append(abl_lbl[cell_mask])
        all_full.append(full_lbl[cell_mask])

        # Per-tissue
        tissue_pixels[type_name]['gt'].append(gt_np[cell_mask])
        tissue_pixels[type_name]['hr'].append(hr_lbl[cell_mask])
        tissue_pixels[type_name]['abl'].append(abl_lbl[cell_mask])
        tissue_pixels[type_name]['full'].append(full_lbl[cell_mask])
        tissue_n[type_name] += 1

        # 打印进度
        da_hr  = (hr_lbl[cell_mask]   == gt_np[cell_mask]).mean()
        da_abl = (abl_lbl[cell_mask]  == gt_np[cell_mask]).mean()
        da_ful = (full_lbl[cell_mask] == gt_np[cell_mask]).mean()

        if n_valid % 20 == 0:
            print(f"{n_valid:>4}  {type_name:<18}  "
                  f"{da_hr:>8.4f}  {da_abl:>9.4f}  {da_ful:>9.4f}")
        n_valid += 1

    print(f"\n有效样本数: {n_valid}")

    # ── 拼接 ────────────────────────────────────────────────────────
    all_gt   = np.concatenate(all_gt)
    all_hr   = np.concatenate(all_hr)
    all_abl  = np.concatenate(all_abl)
    all_full = np.concatenate(all_full)

    # ── 全局混淆矩阵与召回率 ─────────────────────────────────────────
    labels  = list(range(nr_types))
    cm_hr   = confusion_matrix(all_gt, all_hr,   labels=labels)
    cm_abl  = confusion_matrix(all_gt, all_abl,  labels=labels)
    cm_full = confusion_matrix(all_gt, all_full, labels=labels)

    recall_hr   = np.diag(cm_hr)   / cm_hr.sum(axis=1).clip(min=1)
    recall_abl  = np.diag(cm_abl)  / cm_abl.sum(axis=1).clip(min=1)
    recall_full = np.diag(cm_full) / cm_full.sum(axis=1).clip(min=1)

    # ── 全局汇总 ────────────────────────────────────────────────────
    W = 85
    print(f"\n{'=' * W}")
    print("跨模型泛化性验证 —— HoVer-Net 作为评估判别器")
    print(f"（训练时使用 CellViT，评估时使用 HoVer-Net，评估区域："
          f"中心 {OUT_SIZE}×{OUT_SIZE}）")
    print(f"{'=' * W}")

    print(f"\n各类召回率对比(GT 核区域内,加跨模型反差列)")
    print(f"{'=' * W}")
    has_cv_recall = cellvit_json is not None and 'confusion_matrix' in (cellvit_json or {}) \
                    and 'recall_per_class' in cellvit_json['confusion_matrix']
    if has_cv_recall:
        print(f"{'类别':>14}  {'HR(HV)':>8}  {'本文(HV)':>9}  "
              f"{'改善率(HV)':>10}  {'改善率(CV)':>10}  {'反差Δ%':>9}")
    else:
        print(f"{'类别':>14}  {'HR基线':>10}  {'消融模型':>10}  "
              f"{'本文方法':>10}  {'Δ(本文-HR)':>12}  {'改善率':>9}")
    print("-" * W)

    for i in range(nr_types):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class_{i}'
        n_gt = cm_hr.sum(axis=1)[i]
        if n_gt == 0:
            continue
        delta = recall_full[i] - recall_hr[i]
        arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')
        imp_hv = _improvement_ratio(recall_hr[i], recall_full[i])

        if has_cv_recall:
            cv_entry = cellvit_json['confusion_matrix']['recall_per_class'].get(name)
            if cv_entry and cv_entry.get('improvement_ratio') is not None:
                imp_cv = cv_entry['improvement_ratio']
                # 反差:HoVer-Net 改善率 vs CellViT 改善率 的差(百分点)
                gap = (imp_hv - imp_cv) * 100 if not (math.isnan(imp_hv) or imp_cv != imp_cv) \
                      else float('nan')
                gap_s = f"{gap:>+7.1f}pp" if not (isinstance(gap, float) and math.isnan(gap)) \
                        else f"{'N/A':>9}"
                imp_cv_s = f"{imp_cv*100:>+7.1f}%"
            else:
                imp_cv_s = f"{'N/A':>9}"; gap_s = f"{'N/A':>9}"
            print(f"{name:>14}  {recall_hr[i]:>8.4f}  {recall_full[i]:>9.4f}  "
                  f"{_fmt_pct(imp_hv):>10}  {imp_cv_s:>10}  {gap_s:>9}")
        else:
            print(f"{name:>14}  {recall_hr[i]:>10.4f}  {recall_abl[i]:>10.4f}  "
                  f"{recall_full[i]:>10.4f}  {delta:>+10.4f} {arrow}  {_fmt_pct(imp_hv):>9}")

    if has_cv_recall:
        print("\n  说明:HV = HoVer-Net 评估,CV = CellViT 评估")
        print("       反差Δ% = 改善率(HV) − 改善率(CV),单位百分点(pp)")
        print("       理想情况:|反差| 较小 → DGFR 修改对两种判别器的提升幅度接近 → 跨模型泛化")

    oa_hr   = (all_hr   == all_gt).mean()
    oa_abl  = (all_abl  == all_gt).mean()
    oa_full = (all_full == all_gt).mean()
    imp_oa_hv = _improvement_ratio(oa_hr, oa_full)
    print(f"\n整体准确率(Overall Accuracy)：")
    print(f"  HR基线  = {oa_hr:.4f}")
    print(f"  消融模型 = {oa_abl:.4f}")
    print(f"  本文方法 = {oa_full:.4f}  Δ(vs HR) = {oa_full - oa_hr:+.4f}  "
          f"Δ(vs 消融) = {oa_full - oa_abl:+.4f}")
    print(f"  改善率(HV)= {_fmt_pct(imp_oa_hv)} (HR→1.0 差距覆盖)")

    # CellViT 反差(若有 JSON)
    if cellvit_json and 'global' in cellvit_json:
        cv_g = cellvit_json['global']
        cv_imp = cv_g.get('improvement_ratio_oa_full_vs_hr')
        cv_oa = cv_g.get('overall_acc', {})
        if cv_imp is not None and cv_oa.get('hr') is not None:
            print(f"\n  ── 跨模型反差(Overall Accuracy)──")
            print(f"  HR  : HoVer-Net = {oa_hr:.4f}    CellViT = {cv_oa['hr']:.4f}")
            print(f"  Full: HoVer-Net = {oa_full:.4f}    CellViT = {cv_oa['full']:.4f}")
            print(f"  改善率: HoVer-Net = {_fmt_pct(imp_oa_hv)}    "
                  f"CellViT = {_fmt_pct(cv_imp)}")
            gap_pp = (imp_oa_hv - cv_imp) * 100
            print(f"  反差 = {gap_pp:+.2f} pp (越接近 0,跨模型泛化越好)")

    # ── 交集区域准确率（HoVer-Net(HR) 判对的区域） ────────────────────
    inter_mask = (all_gt > 0) & (all_hr == all_gt)
    if inter_mask.sum() > 0:
        ia_hr   = (all_hr[inter_mask]   == all_gt[inter_mask]).mean()
        ia_abl  = (all_abl[inter_mask]  == all_gt[inter_mask]).mean()
        ia_full = (all_full[inter_mask] == all_gt[inter_mask]).mean()
        print(f"\n交集区域准确率（GT∩HoVer-Net(HR)判对）：")
        print(f"  HR基线   = {ia_hr:.4f}（定义为 1.0）")
        print(f"  消融模型 = {ia_abl:.4f}")
        print(f"  本文方法 = {ia_full:.4f}")
        print(f"  交集像素数 = {inter_mask.sum():,}  "
              f"占核区域比例 = {inter_mask.mean():.3f}")

    # ── 错误重定向分析(HoVer-Net 视角,与主脚本 confusion_matrix_compare 一致)
    print(f"\n{'=' * W}")
    print("错误重定向分析(HR 错的修复 vs HR 对的引入)")
    print(f"{'=' * W}")

    hr_correct   = (all_hr   == all_gt)
    full_correct = (all_full == all_gt)

    # 修复:HR 错 → Full 对
    error_corrected   = (~hr_correct) & full_correct
    # 引入:HR 对 → Full 错
    error_introduced  =  hr_correct & (~full_correct)

    n_corrected   = int(error_corrected.sum())
    n_introduced  = int(error_introduced.sum())
    n_hr_wrong    = int((~hr_correct).sum())
    n_hr_right    = int(hr_correct.sum())
    rate_corrected   = n_corrected  / max(n_hr_wrong, 1)
    rate_introduced  = n_introduced / max(n_hr_right, 1)
    redirect_ratio   = n_corrected  / max(n_introduced, 1)

    print(f"\n  HR 错误像素总数         : {n_hr_wrong:>10,}")
    print(f"  Full 修复的错误像素     : {n_corrected:>10,}  "
          f"(纠正率 = {rate_corrected*100:>5.2f}%)")
    print(f"  HR 正确像素总数         : {n_hr_right:>10,}")
    print(f"  Full 引入的新错误像素   : {n_introduced:>10,}  "
          f"(引入率 = {rate_introduced*100:>5.2f}%)")
    print(f"  纠错/引入比             : {redirect_ratio:>10.2f}x   "
          f"(每引入 1 个错误,纠正 {redirect_ratio:.2f} 个;越大越好)")

    # 跨模型反差(若有 CellViT JSON)
    if cellvit_json and 'confusion_matrix' in cellvit_json \
            and 'error_redirect' in cellvit_json['confusion_matrix']:
        cv_er = cellvit_json['confusion_matrix']['error_redirect']
        cv_ratio = cv_er.get('redirect_ratio')
        if cv_ratio is not None:
            print(f"\n  ── 跨模型反差(纠错/引入比)──")
            print(f"  HoVer-Net : {redirect_ratio:.2f}x")
            print(f"  CellViT   : {cv_ratio:.2f}x")
            print(f"  反差比例  : {redirect_ratio/cv_ratio:.2f}  "
                  f"(接近 1.0 → 跨模型泛化好)")

    # 按 GT 类别看修复 vs 引入
    print(f"\n  按 GT 类别细分:")
    print(f"  {'类别':>14}  {'GT像素':>8}  {'修复像素':>8}  {'修复率':>7}  "
          f"{'引入像素':>8}  {'引入率':>7}")
    print(f"  {'-' * 70}")
    for cls_id in range(nr_types):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Class_{cls_id}'
        gt_cls_mask = (all_gt == cls_id)
        n_gt_cls = int(gt_cls_mask.sum())
        if n_gt_cls == 0:
            continue
        cls_corrected_n  = int((error_corrected  & gt_cls_mask).sum())
        cls_introduced_n = int((error_introduced & gt_cls_mask).sum())
        cls_wrong_n      = int(((~hr_correct) & gt_cls_mask).sum())
        cls_right_n      = int((hr_correct  & gt_cls_mask).sum())
        cls_corr_rate = cls_corrected_n  / max(cls_wrong_n, 1)
        cls_intr_rate = cls_introduced_n / max(cls_right_n, 1)
        print(f"  {name:>14}  {n_gt_cls:>8,}  {cls_corrected_n:>8,}  "
              f"{cls_corr_rate*100:>6.2f}%  {cls_introduced_n:>8,}  "
              f"{cls_intr_rate*100:>6.2f}%")

    # ── Per-tissue 汇总 ─────────────────────────────────────────────
    all_tissues = sorted(tissue_pixels.keys())

    print(f"\n{'=' * W}")
    print("按 Tissue Type 分组 —— 整体准确率(HoVer-Net 评估,带改善率与提升最大类)")
    print(f"{'=' * W}")
    has_cv_pt = cellvit_json is not None and 'confusion_matrix' in (cellvit_json or {}) \
                and 'per_tissue_cm' in cellvit_json['confusion_matrix']
    if has_cv_pt:
        print(f"{'Tissue':<22}{'N':>4}  {'HR(HV)':>8}  {'Full(HV)':>9}  "
              f"{'改善率(HV)':>10}  {'改善率(CV)':>10}  {'反差pp':>8}  {'提升最大类':<22}")
    else:
        print(f"{'Tissue':<22}{'N':>4}  {'HR':>8}  {'消融':>9}  {'本文':>9}  "
              f"{'Δ':>9}  {'改善率':>9}  {'提升最大类':<22}")
    print("-" * (W + 30))

    # 收集 per-tissue 用于条形图
    pt_oa_hr_list, pt_oa_abl_list, pt_oa_ful_list = [], [], []

    for tname in all_tissues:
        tp   = tissue_pixels[tname]
        tgt  = np.concatenate(tp['gt'])
        thr  = np.concatenate(tp['hr'])
        tabl = np.concatenate(tp['abl'])
        tful = np.concatenate(tp['full'])
        n    = tissue_n[tname]

        t_oa_hr  = float((thr  == tgt).mean())
        t_oa_abl = float((tabl == tgt).mean())
        t_oa_ful = float((tful == tgt).mean())
        delta    = t_oa_ful - t_oa_hr
        imp_hv   = _improvement_ratio(t_oa_hr, t_oa_ful)

        # 找该 tissue 的提升最大类
        tcm_hr   = confusion_matrix(tgt, thr,  labels=labels)
        tcm_full = confusion_matrix(tgt, tful, labels=labels)
        tr_hr    = np.diag(tcm_hr)   / tcm_hr.sum(axis=1).clip(min=1)
        tr_full  = np.diag(tcm_full) / tcm_full.sum(axis=1).clip(min=1)
        cls_deltas = [(c, tr_full[c] - tr_hr[c])
                      for c in range(1, min(nr_types, 6))
                      if tcm_hr.sum(axis=1)[c] >= 30]
        if cls_deltas:
            best_c, best_d = max(cls_deltas, key=lambda x: x[1])
            best_str = f"{CLASS_NAMES[best_c]}({best_d:+.3f})"
        else:
            best_str = "—"

        pt_oa_hr_list.append(t_oa_hr); pt_oa_abl_list.append(t_oa_abl); pt_oa_ful_list.append(t_oa_ful)

        if has_cv_pt:
            cv_pt = cellvit_json['confusion_matrix']['per_tissue_cm'].get(tname, {})
            cv_imp = cv_pt.get('improvement_ratio')
            if cv_imp is not None and not (isinstance(imp_hv, float) and math.isnan(imp_hv)):
                gap_pp = (imp_hv - cv_imp) * 100
                gap_s  = f"{gap_pp:>+6.1f}pp"
                imp_cv_s = f"{cv_imp*100:>+7.1f}%"
            else:
                gap_s = f"{'N/A':>8}"; imp_cv_s = f"{'N/A':>10}"
            print(f"{tname:<22}{n:>4}  {t_oa_hr:>8.4f}  {t_oa_ful:>9.4f}  "
                  f"{_fmt_pct(imp_hv):>10}  {imp_cv_s:>10}  {gap_s:>8}  {best_str:<22}")
        else:
            arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')
            print(f"{tname:<22}{n:>4}  {t_oa_hr:>8.4f}  {t_oa_abl:>9.4f}  "
                  f"{t_oa_ful:>9.4f}  {delta:>+8.4f}{arrow}  "
                  f"{_fmt_pct(imp_hv):>9}  {best_str:<22}")

    # 注:Per-tissue × per-cell-type 二维表已被合并为上表的"提升最大类"列
    # 详细的各类召回率分析见 confusion_matrix_compare.py(CellViT 视角)


    print(f"\n{'=' * W}")

    # ── 可视化：全局混淆矩阵 ─────────────────────────────────────────
    def plot_cm_normalized(ax, cm, title, class_names):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        valid   = cm.sum(axis=1) > 0
        cm_v    = cm_norm[valid][:, valid]
        names_v = [n for n, v in zip(class_names, valid) if v]

        im = ax.imshow(cm_v, interpolation='nearest', cmap='Blues',
                       vmin=0, vmax=1)
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
                        fontsize=7,
                        color='white' if val > 0.5 else 'black')
        return im

    cn = CLASS_NAMES[:nr_types]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Cross-model Generalization — HoVer-Net as evaluator\n'
        f'(ensemble N={N_RUNS}, center {OUT_SIZE}×{OUT_SIZE}, '
        f'normalized by GT class)',
        fontsize=13, y=1.02,
    )
    plot_cm_normalized(axes[0], cm_hr,   'HR baseline (HoVer-Net)', cn)
    plot_cm_normalized(axes[1], cm_abl,  'Ablation (HoVer-Net)',    cn)
    im = plot_cm_normalized(axes[2], cm_full, 'Full model (HoVer-Net)', cn)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Recall')
    plt.tight_layout()

    cm_path = './logs/downstream_Cellvitsamh/cross_model_confusion_matrix_hovernet.png'
    os.makedirs(os.path.dirname(cm_path) or '.', exist_ok=True)
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n混淆矩阵图已保存: {cm_path}")

    # ── 可视化：Per-tissue 准确率对比 ────────────────────────────────
    n_tissues = len(all_tissues)
    fig, ax = plt.subplots(figsize=(max(10, n_tissues * 1.2), 5))

    x     = np.arange(n_tissues)
    width = 0.25
    # 复用上面 per-tissue 主表已计算的数据,避免重复 concatenate
    ax.bar(x - width, pt_oa_hr_list,  width, label='HR baseline', color='#4878D0')
    ax.bar(x,         pt_oa_abl_list, width, label='Ablation',    color='#EE854A')
    ax.bar(x + width, pt_oa_ful_list, width, label='Full (ours)', color='#6ACC65')

    ax.set_xticks(x)
    ax.set_xticklabels(all_tissues, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Overall Accuracy (nucleus region)')
    ax.set_title(f'Cross-model: Per-tissue Accuracy — HoVer-Net evaluator, '
                 f'N={N_RUNS}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    tissue_path = './logs/downstream_Cellvitsamh/cross_model_per_tissue_accuracy_hovernet.png'
    os.makedirs(os.path.dirname(tissue_path) or '.', exist_ok=True)
    plt.savefig(tissue_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-tissue 准确率图已保存: {tissue_path}")

    # ─────────────────────────────────────────────────────────────────────
    # 绝对 Δ 纵向对比表 —— 跨模型泛化性的核心证据
    # 不依赖改善率(分母 1−HR 受 baseline 差距影响),只看 Δ 和方向一致性
    # ─────────────────────────────────────────────────────────────────────
    if cellvit_json:
        print(f"\n{'=' * W}")
        print("绝对 Δ 纵向对比 —— 跨模型方向一致性")
        print("(避开 baseline 差距问题:核心看两评估器 Δ 是否同号且同方向)")
        print(f"{'=' * W}")

        rows = []   # (指标, cv_hr, cv_full, cv_delta, hv_hr, hv_full, hv_delta, 类型)
        # 1. Overall Accuracy
        cv_g = cellvit_json.get('global', {})
        cv_oa = cv_g.get('overall_acc', {}) or {}
        cv_oa_hr   = cv_oa.get('hr')
        cv_oa_full = cv_oa.get('full')
        if cv_oa_hr is not None and cv_oa_full is not None:
            cv_d = cv_oa_full - cv_oa_hr
            hv_d = oa_full - oa_hr
            rows.append(('Overall_Acc', cv_oa_hr, cv_oa_full, cv_d,
                         oa_hr, oa_full, hv_d, 'higher_better'))

        # 2. 各类召回率
        cv_rpc = (cellvit_json.get('confusion_matrix') or {}).get('recall_per_class', {})
        for i in range(nr_types):
            n_gt = cm_hr.sum(axis=1)[i]
            if n_gt == 0:
                continue
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class_{i}'
            if name == 'Background':
                continue
            cv_e = cv_rpc.get(name)
            if cv_e is None or cv_e.get('recall_hr') is None:
                continue
            cv_d = cv_e['recall_full'] - cv_e['recall_hr']
            hv_d = recall_full[i] - recall_hr[i]
            rows.append((f'Recall_{name}',
                         cv_e['recall_hr'], cv_e['recall_full'], cv_d,
                         recall_hr[i], recall_full[i], hv_d, 'higher_better'))

        # 3. 纠错/引入比(越大越好)
        cv_er = (cellvit_json.get('confusion_matrix') or {}).get('error_redirect', {})
        cv_ratio = cv_er.get('redirect_ratio')
        if cv_ratio is not None:
            # 这里的 hr 列填 1.0 作为参考(纠错=引入时为 1.0,无修改时也接近 1.0)
            # 所以"Δ"近似 ratio − 1.0,正值表示净纠错
            rows.append(('纠错/引入比', 1.0, cv_ratio, cv_ratio - 1.0,
                         1.0, redirect_ratio, redirect_ratio - 1.0,
                         'ratio'))

        # 表头
        print(f"{'指标':<22}  {'CV: HR / Full / Δ':^28}  {'HV: HR / Full / Δ':^28}  {'方向':>6}")
        print("-" * (22 + 30 + 30 + 8))

        n_consistent = 0
        n_total = 0
        for r in rows:
            metric, cv_hr_v, cv_full_v, cv_d, hv_hr_v, hv_full_v, hv_d, kind = r
            n_total += 1
            same_sign = (cv_d >= 0 and hv_d >= 0) or (cv_d < 0 and hv_d < 0)
            if same_sign:
                n_consistent += 1
            arrow = '✓ 同向' if same_sign else '✗ 反向'
            if kind == 'ratio':
                cv_str = f"{cv_hr_v:>5.2f}/{cv_full_v:>5.2f}/{cv_d:>+5.2f}"
                hv_str = f"{hv_hr_v:>5.2f}/{hv_full_v:>5.2f}/{hv_d:>+5.2f}"
            else:
                cv_str = f"{cv_hr_v:>6.4f}/{cv_full_v:>6.4f}/{cv_d:>+7.4f}"
                hv_str = f"{hv_hr_v:>6.4f}/{hv_full_v:>6.4f}/{hv_d:>+7.4f}"
            print(f"{metric:<22}  {cv_str:^28}  {hv_str:^28}  {arrow:>6}")

        # 方向一致性总判
        consistency_rate = n_consistent / max(n_total, 1)
        print(f"\n  方向一致性: {n_consistent}/{n_total} = {consistency_rate*100:.0f}%")
        print(f"  说明:'同向' = 两评估器都看到 Δ 同号(都正/都负),不依赖 baseline 大小。")
        print(f"      论点核心:DGFR 修改方向具有跨模型可迁移性。")

    # ── 实验总结(重写:基于方向一致性 + 绝对 Δ + 净纠错,不依赖改善率反差) ──
    print(f"\n{'=' * W}")
    print("实验总结")
    print(f"{'=' * W}")
    print(f"\n本实验使用 HoVer-Net(训练时未见过的判别器)评估 SPM-UNet 的推理结果。")
    print(f"\n关键指标(HoVer-Net 评估):")
    print(f"  Overall Accuracy  Δ(Full − HR)     = {oa_full - oa_hr:+.4f}  "
          f"{'(正向)' if oa_full > oa_hr else '(负向!)'}")
    print(f"  Overall Accuracy  改善率           = {_fmt_pct(imp_oa_hv)}")
    print(f"  纠错/引入比                        = {redirect_ratio:.2f}x  "
          f"{'(净纠错)' if redirect_ratio > 1.0 else '(净退步!)'}")

    if cellvit_json and 'global' in cellvit_json:
        cv_imp = cellvit_json['global'].get('improvement_ratio_oa_full_vs_hr')
        cv_oa_full = (cellvit_json['global'].get('overall_acc') or {}).get('full')
        cv_oa_hr_v = (cellvit_json['global'].get('overall_acc') or {}).get('hr')
        cv_er_ratio = ((cellvit_json.get('confusion_matrix') or {})
                       .get('error_redirect') or {}).get('redirect_ratio')

        # 多维度评判,不再只看改善率反差
        print(f"\n  ── 跨模型一致性多维度判定 ──")
        cv_oa_delta = (cv_oa_full - cv_oa_hr_v) if (cv_oa_full is not None and cv_oa_hr_v is not None) else None
        cond_oa = (oa_full > oa_hr) and (cv_oa_delta is not None and cv_oa_delta > 0)
        cond_redir = (redirect_ratio > 1.0) and (cv_er_ratio is not None and cv_er_ratio > 1.0)
        cond_consistency = (consistency_rate >= 0.7) if 'consistency_rate' in locals() else None

        cv_oa_d_str = f"{cv_oa_delta:+.4f}" if cv_oa_delta is not None else "N/A"
        print(f"  ① 两评估器 Δ(Overall Acc) 同正?  "
              f"{'✅' if cond_oa else '❌'}  "
              f"(HV={oa_full-oa_hr:+.4f}, CV={cv_oa_d_str})")

        if cv_er_ratio is not None:
            print(f"  ② 两评估器纠错/引入比 > 1?       "
                  f"{'✅' if cond_redir else '❌'}  "
                  f"(HV={redirect_ratio:.2f}x, CV={cv_er_ratio:.2f}x)")
        else:
            print(f"  ② 两评估器纠错/引入比 > 1?       N/A")

        if cond_consistency is not None:
            print(f"  ③ 方向一致性 ≥ 70%?             "
                  f"{'✅' if cond_consistency else '❌'}  "
                  f"({n_consistent}/{n_total} = {consistency_rate*100:.0f}%)")

        # 综合评判
        passed = sum([1 for c in [cond_oa, cond_redir, cond_consistency] if c is True])
        total_checks = sum([1 for c in [cond_oa, cond_redir, cond_consistency] if c is not None])
        if passed == total_checks and total_checks >= 2:
            verdict = "✅ 跨模型泛化性成立(三项核心检查全部通过)"
        elif passed >= 2:
            verdict = "⚠️  跨模型部分泛化(主要正向但存在反差)"
        else:
            verdict = "❌ 跨模型泛化性较弱"
        print(f"\n  综合评判: {verdict}")
        if cv_imp is not None and not (isinstance(imp_oa_hv, float) and math.isnan(imp_oa_hv)):
            print(f"\n  附注:改善率反差({(imp_oa_hv - cv_imp)*100:+.1f} pp)受两评估器 baseline 差距影响,")
            print(f"      不作为主判依据;以 Δ 同向 + 净纠错为准。")
    else:
        print(f"\n  注:未加载 CellViT 评估 JSON;请先运行 compare_baseline_with_senmatic.py 与")
        print(f"     confusion_matrix_compare.py 生成 cellvit_results.json,再用本脚本对比")
    print(f"{'=' * W}")

    # ── HoVer-Net 评估结果 JSON 导出 ──────────────────────────────────
    def _f(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if hasattr(v, '__float__'):
            try:
                return float(v)
            except Exception:
                return None
        return v

    hv_recall_per_class = {}
    for i in range(nr_types):
        n_gt = int(cm_hr.sum(axis=1)[i])
        if n_gt == 0:
            continue
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class_{i}'
        hv_recall_per_class[name] = {
            'n_gt_pixels': n_gt,
            'recall_hr':   _f(recall_hr[i]),
            'recall_abl':  _f(recall_abl[i]),
            'recall_full': _f(recall_full[i]),
            'improvement_ratio': _f(_improvement_ratio(recall_hr[i], recall_full[i])),
        }

    hv_per_tissue = {}
    for ti, tname in enumerate(all_tissues):
        hv_per_tissue[tname] = {
            'n': int(tissue_n[tname]),
            'overall_acc': {
                'hr': _f(pt_oa_hr_list[ti]),
                'ablation': _f(pt_oa_abl_list[ti]),
                'full': _f(pt_oa_ful_list[ti]),
            },
            'improvement_ratio': _f(_improvement_ratio(pt_oa_hr_list[ti], pt_oa_ful_list[ti])),
        }

    hv_results = {
        'evaluator': 'HoVer-Net',
        'n_samples': int(n_valid),
        'global': {
            'overall_acc': {'hr': _f(oa_hr), 'ablation': _f(oa_abl), 'full': _f(oa_full)},
            'improvement_ratio_oa': _f(imp_oa_hv),
            'recall_per_class': hv_recall_per_class,
            'error_redirect': {
                'n_corrected':  int(n_corrected),
                'n_introduced': int(n_introduced),
                'redirect_ratio':    _f(redirect_ratio),
                'correction_rate':   _f(rate_corrected),
                'introduction_rate': _f(rate_introduced),
            },
        },
        'per_tissue': hv_per_tissue,
    }

    hv_json_path = './logs/downstream_Cellvitsamh/hovernet_results.json'
    os.makedirs(os.path.dirname(hv_json_path) or '.', exist_ok=True)
    with open(hv_json_path, 'w', encoding='utf-8') as f:
        json.dump(hv_results, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ HoVer-Net 评估 JSON 已导出: {hv_json_path}")
    print(f"     供未来其他跨模型评估脚本(如 Prov-GigaPath / GPFM)对比")


if __name__ == '__main__':
    main()