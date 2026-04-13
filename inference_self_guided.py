"""
inference_self_guided.py
判别器标签纠正模型的推理与评估脚本。

【推理逻辑】
─────────────────────────────────────────────────────────────────────────────
输入：原始 HR 病理切片
输出：微调后的 HR（CellViT 对其预测更接近 GT）

单步推理，无迭代：
  1. CellViT(HR) → 软标签 sem_tensor（与训练一致）
  2. [HR, noisy_HR] 送入模型 → 预测噪声 → 还原 x0
  3. 评估 CellViT(x0) vs GT 的各类召回率
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from diffusers import DDPMScheduler
from sklearn.metrics import confusion_matrix

from ddpm_dataset import PanNukeDataset
from unet_wrapper import create_model
from ddpm_utils import load_cellvit, predict_x0_from_noise_shared
from semantic_sr_loss import run_cellvit, build_sem_tensor_from_cellvit

device = 'cuda'

CLASS_NAMES = ['Background', 'Neoplastic', 'Inflammatory',
               'Connective', 'Dead', 'Epithelial']

# ── 加载 CellViT ────────────────────────────────────────────────────
print("加载 CellViT...")
cellvit = load_cellvit(
    model_path        = '/home/xuwen/DDPM/CellViT/CellViT-256-x40.pth',
    cellvit_repo_path = '/home/xuwen/DDPM/CellViT',
    device            = device,
)

# ── 加载纠正模型 ────────────────────────────────────────────────────
print("加载纠正模型...")
unet = create_model(use_semantic=True).to(device)
ckpt = torch.load(
    '/home/xuwen/DDPM/logs/checkpoints_correction/best_unet_correction.pth',
    map_location=device,
)
unet.load_state_dict(ckpt['model_state_dict'])
unet.eval()
print(f"  epoch={ckpt.get('epoch','?')}  "
      f"val_dir_acc={ckpt.get('val_dir_acc','?'):.4f}")

# ── 调度器与数据集 ──────────────────────────────────────────────────
scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset   = PanNukeDataset(
    fold_dirs    = ['/data/xuwen/PanNuke/Fold 3'],
    target_size  = 256,
)
print(f"测试集大小: {len(dataset)} 张")

INFER_T   = 100   # 推理噪声强度，与验证时保持一致
N_SAMPLES = 200
torch.manual_seed(42)


def correct_image(hr: torch.Tensor) -> torch.Tensor:
    """
    单步推理：对单张 HR 图像进行一次纠正。

    Args:
        hr: [1, 3, H, W]  原始 HR 图像

    Returns:
        x0: [1, 3, H, W]  纠正后的图像
    """
    # Step 1: CellViT(HR) 软标签 → sem_tensor
    with torch.no_grad():
        cv_out = run_cellvit(cellvit, hr)
    sem = build_sem_tensor_from_cellvit(
        cv_out['nuclei_type_prob'],   # [1, 6, H, W]
        cv_out['nuclei_nuc_prob'],    # [1, H, W]
    )  # [1, 7, H, W]

    # Step 2: 加噪 + 模型前向 → x0
    noise    = torch.randn_like(hr)
    t        = torch.tensor([INFER_T], device=device)
    noisy_hr = scheduler.add_noise(hr, noise, t)

    with torch.no_grad():
        model_input = torch.cat([hr, noisy_hr], dim=1)   # [1, 6, H, W]
        noise_pred  = unet(model_input, t, semantic=sem).sample
        x0          = predict_x0_from_noise_shared(noisy_hr, noise_pred, t, scheduler)

    return x0


# ── 评估 ────────────────────────────────────────────────────────────
print(f"\n开始评估（INFER_T={INFER_T}，样本数={N_SAMPLES}）...")

all_gt        = []
all_hr_pred   = []
all_corr_pred = []
n_valid = 0

for i in range(len(dataset)):
    if n_valid >= N_SAMPLES:
        break

    sample  = dataset[i]
    hr_cpu  = sample['hr']
    gt_lbl  = sample['gt_label_map']    # [H, W]  int
    gt_nuc  = sample['gt_nuc_mask'].bool()  # [H, W]

    if gt_nuc.sum() < 10:
        continue

    hr = hr_cpu.unsqueeze(0).to(device)

    # 原始 HR 的 CellViT 预测
    with torch.no_grad():
        hr_cv  = run_cellvit(cellvit, hr)
        hr_lbl = hr_cv['nuclei_type_label'].squeeze(0).cpu()

    # 纠正后的预测
    x0 = correct_image(hr)
    with torch.no_grad():
        x0_cv  = run_cellvit(cellvit, x0)
        x0_lbl = x0_cv['nuclei_type_label'].squeeze(0).cpu()

    # 仅在 GT 核区域内收集结果
    all_gt.append(gt_lbl[gt_nuc].numpy())
    all_hr_pred.append(hr_lbl[gt_nuc].numpy())
    all_corr_pred.append(x0_lbl[gt_nuc].numpy())

    if n_valid % 20 == 0:
        print(f"  [{n_valid:>3}/{N_SAMPLES}] 样本 {i:>4}")
    n_valid += 1

# ── 汇总 ────────────────────────────────────────────────────────────
all_gt        = np.concatenate(all_gt)
all_hr_pred   = np.concatenate(all_hr_pred)
all_corr_pred = np.concatenate(all_corr_pred)

labels = list(range(6))
cm_hr  = confusion_matrix(all_gt, all_hr_pred,   labels=labels)
cm_cor = confusion_matrix(all_gt, all_corr_pred, labels=labels)

recall_hr  = np.diag(cm_hr)  / cm_hr.sum(axis=1).clip(min=1)
recall_cor = np.diag(cm_cor) / cm_cor.sum(axis=1).clip(min=1)

print(f"\n{'='*65}")
print("各类召回率对比（GT 核区域内）")
print(f"{'='*65}")
print(f"{'类别':>14}  {'HR 原始':>10}  {'纠正后':>10}  {'Δ':>10}")
print("-" * 65)
for i, name in enumerate(CLASS_NAMES):
    n_gt = cm_hr.sum(axis=1)[i]
    if n_gt == 0:
        continue
    delta = recall_cor[i] - recall_hr[i]
    arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '—')
    print(f"{name:>14}  {recall_hr[i]:>10.4f}  "
          f"{recall_cor[i]:>10.4f}  {delta:>+10.4f} {arrow}  "
          f"(GT={n_gt:>8})")

overall_hr  = (all_hr_pred   == all_gt).mean()
overall_cor = (all_corr_pred == all_gt).mean()
print(f"\n整体准确率：HR={overall_hr:.4f}  纠正后={overall_cor:.4f}  "
      f"Δ={overall_cor - overall_hr:+.4f}")

# 交集区域（CellViT(HR) 判对且 GT 有标注）的准确率
inter_mask = (all_gt > 0) & (all_hr_pred == all_gt)
if inter_mask.sum() > 0:
    inter_acc_hr  = (all_hr_pred[inter_mask]   == all_gt[inter_mask]).mean()
    inter_acc_cor = (all_corr_pred[inter_mask] == all_gt[inter_mask]).mean()
    print(f"\n交集区域准确率（GT∩CellViT(HR) 判对）：")
    print(f"  HR={inter_acc_hr:.4f}  纠正后={inter_acc_cor:.4f}  "
          f"Δ={inter_acc_cor - inter_acc_hr:+.4f}")
    print(f"  交集像素数={inter_mask.sum()}  "
          f"占 GT 核区域比例={inter_mask.mean():.3f}")