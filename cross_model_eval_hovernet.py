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
CELLVIT_PATH    = '/home/xuwen/DDPM/CellViT/CellViT-256-x40.pth'
CELLVIT_REPO    = '/home/xuwen/DDPM/CellViT'
UNET_ABL_PATH   = '/home/xuwen/DDPM/logs/checkpoints_correction_v3/best_unet_ablation.pth'
UNET_FULL_PATH  = '/home/xuwen/DDPM/logs/checkpoints_correction_v3/best_unet_correction.pth'
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
    args = parser.parse_args()

    device    = args.device
    N_SAMPLES = args.n_samples
    N_RUNS    = args.n_runs
    INFER_T   = args.infer_t
    OUT_SIZE  = HOVERNET_OUTPUT_SIZE

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

    print(f"\n各类召回率对比（GT 核区域内）")
    print(f"{'=' * W}")
    print(f"{'类别':>14}  {'HR基线':>10}  {'消融模型':>10}  "
          f"{'本文方法':>10}  {'Δ(本文-HR)':>12}")
    print("-" * W)

    for i in range(nr_types):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class_{i}'
        n_gt = cm_hr.sum(axis=1)[i]
        if n_gt == 0:
            continue
        delta = recall_full[i] - recall_hr[i]
        arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')
        print(f"{name:>14}  {recall_hr[i]:>10.4f}  {recall_abl[i]:>10.4f}  "
              f"{recall_full[i]:>10.4f}  {delta:>+10.4f} {arrow}  "
              f"(GT={n_gt:>8})")

    oa_hr   = (all_hr   == all_gt).mean()
    oa_abl  = (all_abl  == all_gt).mean()
    oa_full = (all_full == all_gt).mean()
    print(f"\n整体准确率（Overall Accuracy）：")
    print(f"  HR基线  = {oa_hr:.4f}")
    print(f"  消融模型 = {oa_abl:.4f}")
    print(f"  本文方法 = {oa_full:.4f}  Δ(vs HR) = {oa_full - oa_hr:+.4f}  "
          f"Δ(vs 消融) = {oa_full - oa_abl:+.4f}")

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

    # ── 退步/进步分析 ────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("退步/进步分析（HoVer-Net 视角）")
    print(f"{'=' * W}")

    hr_correct   = (all_hr   == all_gt)
    abl_correct  = (all_abl  == all_gt)
    full_correct = (all_full == all_gt)

    both_correct_full_wrong = hr_correct & abl_correct & ~full_correct
    only_full_correct       = ~hr_correct & ~abl_correct & full_correct

    total_nuc  = len(all_gt)
    n_regress  = both_correct_full_wrong.sum()
    n_improve  = only_full_correct.sum()

    print(f"\n  核区域总像素数            : {total_nuc:>10,}")
    print(f"  退步像素数 (两者对本文错)  : {n_regress:>10,}  "
          f"({100 * n_regress / total_nuc:.2f}%)")
    print(f"  进步像素数 (仅本文对)      : {n_improve:>10,}  "
          f"({100 * n_improve / total_nuc:.2f}%)")
    print(f"  进步/退步比                : "
          f"{n_improve / max(n_regress, 1):.2f}x")

    print(f"\n  按 GT 类别细分退步情况：")
    print(f"  {'类别':>14}  {'GT像素':>8}  {'退步像素':>8}  {'退步率':>8}  "
          f"{'退步时被误分为（Top-3）'}")
    print(f"  {'-' * 75}")

    for cls_id in range(nr_types):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Class_{cls_id}'
        gt_cls_mask   = (all_gt == cls_id)
        regress_mask  = both_correct_full_wrong & gt_cls_mask
        n_gt_cls      = gt_cls_mask.sum()
        n_regress_cls = regress_mask.sum()
        if n_gt_cls == 0 or n_regress_cls == 0:
            continue
        wrong_preds = all_full[regress_mask]
        unique, counts = np.unique(wrong_preds, return_counts=True)
        top3 = sorted(zip(counts, unique), reverse=True)[:3]
        top3_str = "  ".join(
            f"{CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Cls{c}'}({cnt})"
            for cnt, c in top3)
        print(f"  {name:>14}  {n_gt_cls:>8,}  {n_regress_cls:>8,}  "
              f"{100 * n_regress_cls / n_gt_cls:>7.2f}%  {top3_str}")

    # ── Per-tissue 汇总 ─────────────────────────────────────────────
    all_tissues = sorted(tissue_pixels.keys())

    print(f"\n{'=' * W}")
    print("按 Tissue Type 分组 —— 整体准确率（HoVer-Net 评估）")
    print(f"{'=' * W}")
    print(f"{'Tissue':<22} {'N':>4}  {'HR基线':>8}  {'消融模型':>9}  "
          f"{'本文方法':>9}  {'Δ(本文-HR)':>12}")
    print("-" * W)

    for tname in all_tissues:
        tp   = tissue_pixels[tname]
        tgt  = np.concatenate(tp['gt'])
        thr  = np.concatenate(tp['hr'])
        tabl = np.concatenate(tp['abl'])
        tful = np.concatenate(tp['full'])
        n    = tissue_n[tname]

        t_oa_hr  = (thr  == tgt).mean()
        t_oa_abl = (tabl == tgt).mean()
        t_oa_ful = (tful == tgt).mean()
        delta    = t_oa_ful - t_oa_hr
        arrow    = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')

        print(f"{tname:<22} {n:>4}  {t_oa_hr:>8.4f}  {t_oa_abl:>9.4f}  "
              f"{t_oa_ful:>9.4f}  {delta:>+10.4f} {arrow}")

    # ── Per-tissue 各细胞类别召回率 ──────────────────────────────────
    print(f"\n── 各细胞类别召回率：本文方法（Δ = 本文 − HR基线）──")
    cell_cls = [c for c in range(1, min(nr_types, 6))]
    header   = "  ".join(f"{CLASS_NAMES[c]:>16}" for c in cell_cls)
    print(f"\n{'Tissue':<22} {'N':>4}  {header}")
    print("-" * (26 + 18 * len(cell_cls)))

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
        for c in cell_cls:
            n_gt_c = tcm_hr.sum(axis=1)[c]
            if n_gt_c == 0:
                cls_str += f"{'  N/A':>16}  "
            else:
                delta = tr_full[c] - tr_hr[c]
                arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')
                cls_str += f"{tr_full[c]:>6.4f}({delta:>+.4f}{arrow})  "

        print(f"{tname:<22} {n:>4}  {cls_str}")

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

    os.makedirs('./logs', exist_ok=True)
    cm_path = './logs/cross_model_confusion_matrix_hovernet.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n混淆矩阵图已保存: {cm_path}")

    # ── 可视化：Per-tissue 准确率对比 ────────────────────────────────
    n_tissues = len(all_tissues)
    fig, ax = plt.subplots(figsize=(max(10, n_tissues * 1.2), 5))

    x     = np.arange(n_tissues)
    width = 0.25
    oa_hr_list, oa_abl_list, oa_ful_list = [], [], []

    for tname in all_tissues:
        tp   = tissue_pixels[tname]
        tgt  = np.concatenate(tp['gt'])
        oa_hr_list.append((np.concatenate(tp['hr'])   == tgt).mean())
        oa_abl_list.append((np.concatenate(tp['abl']) == tgt).mean())
        oa_ful_list.append((np.concatenate(tp['full'])== tgt).mean())

    ax.bar(x - width, oa_hr_list,  width, label='HR baseline', color='#4878D0')
    ax.bar(x,         oa_abl_list, width, label='Ablation',    color='#EE854A')
    ax.bar(x + width, oa_ful_list, width, label='Full (ours)', color='#6ACC65')

    ax.set_xticks(x)
    ax.set_xticklabels(all_tissues, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Overall Accuracy (nucleus region)')
    ax.set_title(f'Cross-model: Per-tissue Accuracy — HoVer-Net evaluator, '
                 f'N={N_RUNS}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    tissue_path = './logs/cross_model_per_tissue_accuracy_hovernet.png'
    plt.savefig(tissue_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-tissue 准确率图已保存: {tissue_path}")

    # ── 与 CellViT 评估结果的对比提示 ────────────────────────────────
    print(f"\n{'=' * W}")
    print("实验总结")
    print(f"{'=' * W}")
    print(f"\n本实验使用 HoVer-Net（训练时未见过的判别器）评估 SPM-UNet 的推理结果。")
    print(f"若本文方法在 HoVer-Net 评估下仍优于消融模型和 HR 基线，")
    print(f"则说明图像修改具有跨模型泛化性，不是过拟合到 CellViT 的决策边界。")
    print(f"\n关键指标：")
    print(f"  Overall Accuracy  Δ(本文 vs HR)  = {oa_full - oa_hr:+.4f}")
    print(f"  Overall Accuracy  Δ(本文 vs 消融) = {oa_full - oa_abl:+.4f}")
    print(f"  进步/退步比                       = "
          f"{n_improve / max(n_regress, 1):.2f}x")
    print(f"\n请将上述结果与 CellViT 评估结果（compare_baseline_with_semantic.py）对比，")
    print(f"以验证跨模型泛化性。")
    print(f"{'=' * W}")


if __name__ == '__main__':
    main()