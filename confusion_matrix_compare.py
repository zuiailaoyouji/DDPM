"""
confusion_matrix_compare.py
对比 LR、Baseline SR、Semantic SR 三种输入在测试集（Fold 3）上的混淆矩阵。

评估方式：
  - 对每张图分别跑 CellViT，得到类别预测图
  - 在 GT 核掩膜区域内，与 GT label_map 对比
  - 输出三个混淆矩阵及每类的精确率/召回率/F1

类别定义：
  0: Background
  1: Neoplastic
  2: Inflammatory
  3: Connective
  4: Dead
  5: Epithelial
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from models.segmentation.cell_segmentation.cellvit import CellViT256
from ddpm_dataset import PanNukeDataset
from degradation import degrade
from unet_wrapper import create_model
from ddpm_utils import predict_x0_from_noise_shared
from diffusers import DDPMScheduler

device = 'cuda'

CLASS_NAMES = ['Background', 'Neoplastic', 'Inflammatory',
               'Connective', 'Dead', 'Epithelial']

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
print("✓ CellViT 加载完成")


def run_cellvit_label(img_01: torch.Tensor) -> torch.Tensor:
    """返回 [B, H, W] 类别索引图"""
    out  = cellvit(img_01)
    prob = F.softmax(out['nuclei_type_map'], dim=1)
    return prob.argmax(dim=1)


# ── UNet 加载 ───────────────────────────────────────────────────────
def load_unet(ckpt_path: str):
    unet = create_model(use_semantic=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"  加载 {ckpt_path}  epoch={epoch}")
    return unet


print("\n加载 UNet 模型...")
unet_baseline = load_unet("/home/xuwen/DDPM/logs/checkpoints_cellvit/unet_sr_epoch_20.pth")
unet_semantic = load_unet("/home/xuwen/DDPM/logs/checkpoints_cellvit/unet_sr_epoch_180.pth")

# ── 调度器与数据集 ──────────────────────────────────────────────────
scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset   = PanNukeDataset(
    fold_dirs=['/data/xuwen/PanNuke/Fold 3'],
    target_size=256,
)
print(f"\n测试集大小: {len(dataset)} 张")

FIXED_T   = 50
N_SAMPLES = 200
torch.manual_seed(42)

# ── 收集预测结果 ────────────────────────────────────────────────────
all_gt       = []
all_lr       = []
all_baseline = []
all_semantic = []

print(f"\n开始推理（固定 t={FIXED_T}，最多 {N_SAMPLES} 个有效样本）...")
n_valid = 0

for i in range(len(dataset)):
    if n_valid >= N_SAMPLES:
        break

    sample  = dataset[i]
    hr_cpu  = sample['hr']
    gt_lbl  = sample['gt_label_map']    # [H, W] int，0-5
    gt_nuc  = sample['gt_nuc_mask'].bool()  # [H, W]

    if gt_nuc.sum() < 10:
        continue

    hr = hr_cpu.unsqueeze(0).to(device)

    lr_cpu = degrade(
        hr_cpu,
        scale=4,
        blur_sigma_range=(2.0, 3.0),
        noise_std_range=(0.03, 0.08),
        stain_jitter_strength=0.15,
    )
    lr = lr_cpu.unsqueeze(0).to(device)

    noise    = torch.randn_like(hr)
    t        = torch.tensor([FIXED_T], device=device)
    noisy_hr = scheduler.add_noise(hr, noise, t)

    with torch.no_grad():
        # LR 预测
        lr_lbl = run_cellvit_label(lr).squeeze(0).cpu()

        # Baseline SR 预测
        inp_b  = torch.cat([lr, noisy_hr], dim=1)
        np_b   = unet_baseline(inp_b, t).sample
        x0_b   = predict_x0_from_noise_shared(noisy_hr, np_b, t, scheduler)
        bl_lbl = run_cellvit_label(x0_b).squeeze(0).cpu()

        # Semantic SR 预测
        inp_s  = torch.cat([lr, noisy_hr], dim=1)
        np_s   = unet_semantic(inp_s, t).sample
        x0_s   = predict_x0_from_noise_shared(noisy_hr, np_s, t, scheduler)
        sm_lbl = run_cellvit_label(x0_s).squeeze(0).cpu()

    # 只取 GT 核掩膜区域内的像素
    all_gt.append(gt_lbl[gt_nuc].numpy())
    all_lr.append(lr_lbl[gt_nuc].numpy())
    all_baseline.append(bl_lbl[gt_nuc].numpy())
    all_semantic.append(sm_lbl[gt_nuc].numpy())

    if n_valid % 20 == 0:
        print(f"  [{n_valid:>3}/{N_SAMPLES}] 样本 {i:>4} 处理完成")
    n_valid += 1

print(f"\n有效样本数: {n_valid}")

all_gt       = np.concatenate(all_gt)
all_lr       = np.concatenate(all_lr)
all_baseline = np.concatenate(all_baseline)
all_semantic = np.concatenate(all_semantic)

# ── 计算混淆矩阵 ────────────────────────────────────────────────────
labels = list(range(6))

cm_lr  = confusion_matrix(all_gt, all_lr,       labels=labels)
cm_bl  = confusion_matrix(all_gt, all_baseline, labels=labels)
cm_sm  = confusion_matrix(all_gt, all_semantic, labels=labels)

# ── 打印文字结果 ────────────────────────────────────────────────────
def print_cm_and_report(cm, pred_arr, title):
    print(f"\n{'='*65}")
    print(f"{title}")
    print(f"{'='*65}")
    print(f"混淆矩阵 (行=GT, 列=Pred):")
    header = f"{'':>14}" + "".join(f"{n:>14}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>14}" + "".join(f"{v:>14}" for v in row))

    print(f"\n各类 GT 像素数: {cm.sum(axis=1)}")
    print(f"各类 Pred 像素数: {cm.sum(axis=0)}")

    # 每类准确率（对角线 / 该类GT总数）
    per_class_acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    print(f"\n每类召回率（预测正确/GT总数）:")
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class_acc)):
        gt_count = cm.sum(axis=1)[i]
        print(f"  {name:>14}: {acc:.4f}  (GT={gt_count:>8})")

    overall = (pred_arr == all_gt).mean()
    print(f"\n整体准确率: {overall:.4f}")

print_cm_and_report(cm_lr,  all_lr,       "LR 直接推理")
print_cm_and_report(cm_bl,  all_baseline, "Baseline SR（无语义监督）")
print_cm_and_report(cm_sm,  all_semantic, "Semantic SR（语义引导，本文）")

# ── 各类召回率对比表 ────────────────────────────────────────────────
print(f"\n{'='*65}")
print("各类召回率对比（GT核区域内）")
print(f"{'='*65}")
print(f"{'类别':>14}  {'LR':>8}  {'Baseline':>10}  {'Semantic':>10}  {'Δ(Sem-BL)':>12}")
print("-" * 65)

recall_lr = np.diag(cm_lr) / cm_lr.sum(axis=1).clip(min=1)
recall_bl = np.diag(cm_bl) / cm_bl.sum(axis=1).clip(min=1)
recall_sm = np.diag(cm_sm) / cm_sm.sum(axis=1).clip(min=1)

for i, name in enumerate(CLASS_NAMES):
    gt_count = cm_bl.sum(axis=1)[i]
    if gt_count == 0:
        continue
    delta = recall_sm[i] - recall_bl[i]
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '—')
    print(f"{name:>14}  {recall_lr[i]:>8.4f}  {recall_bl[i]:>10.4f}  "
          f"{recall_sm[i]:>10.4f}  {delta:>+10.4f} {arrow}")

# ── 可视化混淆矩阵 ──────────────────────────────────────────────────
def plot_cm_normalized(ax, cm, title, class_names):
    """绘制归一化混淆矩阵（按行归一化，即召回率矩阵）"""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    # 只显示 GT 数量 > 0 的类
    valid = cm.sum(axis=1) > 0
    cm_norm_v = cm_norm[valid][:, valid]
    names_v   = [n for n, v in zip(class_names, valid) if v]

    im = ax.imshow(cm_norm_v, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('GT', fontsize=9)

    ticks = np.arange(len(names_v))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names_v, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(names_v, fontsize=8)

    thresh = 0.5
    for ri in range(len(names_v)):
        for ci in range(len(names_v)):
            val = cm_norm_v[ri, ci]
            color = 'white' if val > thresh else 'black'
            ax.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)
    return im

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices (normalized by GT class, nucleus region only)',
             fontsize=13, y=1.02)

plot_cm_normalized(axes[0], cm_lr,  'LR direct',           CLASS_NAMES)
plot_cm_normalized(axes[1], cm_bl,  'Baseline SR',          CLASS_NAMES)
im = plot_cm_normalized(axes[2], cm_sm, 'Semantic SR (ours)', CLASS_NAMES)

plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Recall')
plt.tight_layout()
save_path = './logs/confusion_matrix_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n混淆矩阵图已保存到: {save_path}")
