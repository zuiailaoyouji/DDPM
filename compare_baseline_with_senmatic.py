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

输出指标（均在 GT 核区域内计算）：
  Dir_Acc       : CellViT(pred) vs GT 的类别一致率
  Intersect_Acc : 仅在交集区域（GT∩CellViT(HR)判对）的一致率
  Sem_MAE       : CellViT(pred) 和 CellViT(HR) 概率分布的 MAE
  PSNR / SSIM   : pred vs HR（衡量图像修改幅度）
"""

import numpy as np
import torch
import sys
sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from diffusers import DDPMScheduler

from ddpm_dataset import PanNukeDataset
from unet_wrapper import create_model
from ddpm_utils import load_cellvit, predict_x0_from_noise_shared
from semantic_sr_loss import run_cellvit, build_sem_tensor_from_cellvit
from metrics import compute_psnr, compute_ssim

device = 'cuda'

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
    print(f"  加载 {ckpt_path}  epoch={ckpt.get('epoch', '?')}  "
          f"use_semantic={use_semantic}")
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

# ── 调度器与数据集 ──────────────────────────────────────────────────
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
def infer_ensemble(unet, hr, sem=None):
    """
    对同一张 HR 用 N_RUNS 组不同随机噪声做单步推理，
    返回：
      mean_prob  : (1, C, H, W)  N_RUNS 次 CellViT 概率图的均值
      lbl        : (H, W)        mean_prob argmax 得到的分类图
      mean_x0    : (1, 3, H, W)  N_RUNS 次 x0 的均值（用于 PSNR/SSIM）
    """
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
            prob = cv['nuclei_type_prob']          # (1, C, H, W)

        prob_sum = prob if prob_sum is None else prob_sum + prob
        x0_sum   = x0   if x0_sum  is None else x0_sum  + x0

    mean_prob = prob_sum / N_RUNS                  # (1, C, H, W)
    mean_x0   = x0_sum  / N_RUNS                  # (1, 3, H, W)
    lbl       = mean_prob.argmax(dim=1).squeeze(0).cpu()   # (H, W)
    return mean_prob, lbl, mean_x0


# ── 结果容器 ────────────────────────────────────────────────────────
keys      = ['hr', 'ablation', 'full']
dir_acc   = {k: [] for k in keys}
inter_acc = {k: [] for k in keys}
sem_mae   = {k: [] for k in ['ablation', 'full']}
psnr_d    = {k: [] for k in ['ablation', 'full']}
ssim_d    = {k: [] for k in ['ablation', 'full']}

print(f"\n开始评估（INFER_T={INFER_T}，N_RUNS={N_RUNS}，样本数={N_SAMPLES}）...")
print(f"{'idx':>4}  {'HR_acc':>8}  {'Abl_acc':>9}  {'Full_acc':>9}  "
      f"{'Abl_PSNR':>10}  {'Full_PSNR':>10}")

n_valid = 0
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

    with torch.no_grad():
        # CellViT(HR)：评估基准 + 完整模型的 sem_tensor
        hr_cv   = run_cellvit(cellvit, hr)
        hr_lbl  = hr_cv['nuclei_type_label'].squeeze(0).cpu()
        hr_prob = hr_cv['nuclei_type_prob']             # (1, C, H, W)

        sem = build_sem_tensor_from_cellvit(
            hr_cv['nuclei_type_prob'],
            hr_cv['nuclei_nuc_prob'],
        )

    # 消融模型集成推理（无 sem_tensor）
    abl_prob, abl_lbl, abl_x0 = infer_ensemble(unet_ablation, hr, sem=None)

    # 完整模型集成推理（有 sem_tensor）
    full_prob, full_lbl, full_x0 = infer_ensemble(unet_full, hr, sem=sem)

    gt_np     = gt_lbl.numpy()
    gt_nuc_np = gt_nuc.numpy()
    hr_np     = hr_lbl.numpy()

    cell_mask  = gt_nuc_np & (gt_np > 0)
    inter_mask = cell_mask & (hr_np == gt_np)

    def cls_acc(pred_np, mask):
        if mask.sum() == 0:
            return float('nan')
        return (pred_np[mask] == gt_np[mask]).mean()

    def prob_mae_fn(pred_prob_gpu, mask):
        # pred_prob_gpu: (1, C, H, W)，与 hr_prob 比较 MAE
        mae = (pred_prob_gpu.cpu() - hr_prob.cpu()).abs().mean(dim=1).squeeze(0).numpy()
        if mask.sum() == 0:
            return float('nan')
        return mae[mask].mean()

    dir_acc['hr'].append(cls_acc(hr_np, cell_mask))
    dir_acc['ablation'].append(cls_acc(abl_lbl.numpy(), cell_mask))
    dir_acc['full'].append(cls_acc(full_lbl.numpy(), cell_mask))

    inter_acc['hr'].append(cls_acc(hr_np, inter_mask))
    inter_acc['ablation'].append(cls_acc(abl_lbl.numpy(), inter_mask))
    inter_acc['full'].append(cls_acc(full_lbl.numpy(), inter_mask))

    sem_mae['ablation'].append(prob_mae_fn(abl_prob, gt_nuc_np))
    sem_mae['full'].append(prob_mae_fn(full_prob, gt_nuc_np))

    # PSNR/SSIM 用均值图像计算
    psnr_d['ablation'].append(compute_psnr(abl_x0.cpu(), hr.cpu()))
    psnr_d['full'].append(compute_psnr(full_x0.cpu(), hr.cpu()))
    ssim_d['ablation'].append(compute_ssim(abl_x0.cpu(), hr.cpu()))
    ssim_d['full'].append(compute_ssim(full_x0.cpu(), hr.cpu()))

    if n_valid % 20 == 0:
        print(f"{n_valid:>4}  "
              f"{dir_acc['hr'][-1]:>8.4f}  "
              f"{dir_acc['ablation'][-1]:>9.4f}  "
              f"{dir_acc['full'][-1]:>9.4f}  "
              f"{psnr_d['ablation'][-1]:>10.2f}  "
              f"{psnr_d['full'][-1]:>10.2f}")
    n_valid += 1

# ── 汇总 ────────────────────────────────────────────────────────────
def _mean(lst):
    vals = [v for v in lst if not (isinstance(v, float) and v != v)]
    return np.mean(vals) if vals else float('nan')

print("\n" + "=" * 75)
print(f"{'指标':<22} {'HR基线':>12} {'消融模型':>12} {'本文方法':>12}")
print("-" * 75)
print(f"{'Dir_Acc (vs GT)':<22} "
      f"{_mean(dir_acc['hr']):>12.4f} "
      f"{_mean(dir_acc['ablation']):>12.4f} "
      f"{_mean(dir_acc['full']):>12.4f}")
print(f"{'Intersect_Acc':<22} "
      f"{_mean(inter_acc['hr']):>12.4f} "
      f"{_mean(inter_acc['ablation']):>12.4f} "
      f"{_mean(inter_acc['full']):>12.4f}")
print(f"{'Sem_MAE (vs HR prob)':<22} "
      f"{'0.0000':>12} "
      f"{_mean(sem_mae['ablation']):>12.4f} "
      f"{_mean(sem_mae['full']):>12.4f}")
print(f"{'PSNR (dB)':<22} "
      f"{'—':>12} "
      f"{_mean(psnr_d['ablation']):>12.2f} "
      f"{_mean(psnr_d['full']):>12.2f}")
print(f"{'SSIM':<22} "
      f"{'—':>12} "
      f"{_mean(ssim_d['ablation']):>12.4f} "
      f"{_mean(ssim_d['full']):>12.4f}")
print("=" * 75)

print(f"\n本文 vs HR基线   Dir_Acc   提升: "
      f"{_mean(dir_acc['full']) - _mean(dir_acc['hr']):+.4f}")
print(f"本文 vs 消融模型 Dir_Acc   提升: "
      f"{_mean(dir_acc['full']) - _mean(dir_acc['ablation']):+.4f}")
print(f"本文 vs HR基线   Intersect 提升: "
      f"{_mean(inter_acc['full']) - _mean(inter_acc['hr']):+.4f}")
print(f"本文 vs 消融模型 Sem_MAE   降低: "
      f"{_mean(sem_mae['ablation']) - _mean(sem_mae['full']):+.4f}")