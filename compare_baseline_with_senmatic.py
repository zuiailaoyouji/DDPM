"""
compare_baseline_vs_semantic.py
在相同推理条件下对比：
  - LR 直接送 CellViT
  - 纯像素 SR（Stage 1 best，无语义监督）
  - 语义引导 SR（Stage 2 best）

输出指标：
  Dir_Acc  : CellViT(pred) vs CellViT(HR) 的类别一致率（GT核区域内）
  PSNR     : pred vs HR
  SSIM     : pred vs HR
  Sem_MAE  : CellViT(pred) 和 CellViT(HR) 概率分布的 MAE
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/xuwen/DDPM/CellViT')
sys.path.insert(0, '/home/xuwen/DDPM')

from models.segmentation.cell_segmentation.cellvit import CellViT256
from ddpm_dataset import PanNukeDataset
from degradation import degrade
from unet_wrapper import create_model
from ddpm_utils import predict_x0_from_noise_shared
from metrics import compute_psnr, compute_ssim
from diffusers import DDPMScheduler

device = 'cuda'

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


def run_cellvit(img_01: torch.Tensor):
    """返回 (type_label [B,H,W], type_prob [B,6,H,W], nuc_prob [B,H,W])"""
    out       = cellvit(img_01)
    type_prob = F.softmax(out['nuclei_type_map'], dim=1)   # [B,6,H,W]
    type_lbl  = type_prob.argmax(dim=1)                    # [B,H,W]
    return type_lbl, type_prob


# ── UNet 加载 ───────────────────────────────────────────────────────
def load_unet(ckpt_path: str, use_semantic: bool = True):
    unet = create_model(use_semantic=use_semantic).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()
    epoch = ckpt.get('epoch', '?')
    psnr  = ckpt.get('val_psnr', '?')
    print(f"  加载 {ckpt_path}  epoch={epoch}  val_psnr={psnr}")
    return unet


print("\n加载 UNet 模型...")
# Stage 1 best（纯像素基线，无语义监督）
unet_baseline = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_cellvit/unet_sr_epoch_20.pth",
    use_semantic=True,
)
# 语义监督 best
unet_semantic = load_unet(
    "/home/xuwen/DDPM/logs/checkpoints_cellvit/unet_sr_epoch_180.pth",
    use_semantic=True,
)

# ── 调度器与数据集 ──────────────────────────────────────────────────
scheduler = DDPMScheduler(num_train_timesteps=1000)

dataset = PanNukeDataset(
    fold_dirs=['/data/xuwen/PanNuke/Fold 3'],
    target_size=256,
)
print(f"\n验证集大小: {len(dataset)} 张")

# ── 固定推理参数 ────────────────────────────────────────────────────
FIXED_T   = 50      # 固定时间步，保证两个模型推理条件一致
N_SAMPLES = 200     # 评估样本数（建议至少100）
torch.manual_seed(42)

# ── 结果容器 ────────────────────────────────────────────────────────
keys = ['lr', 'baseline', 'semantic']
dir_acc  = {k: [] for k in keys}
sem_mae  = {k: [] for k in keys}
psnr_d   = {k: [] for k in ['baseline', 'semantic']}
ssim_d   = {k: [] for k in ['baseline', 'semantic']}

print(f"\n开始评估（固定 t={FIXED_T}，样本数={N_SAMPLES}）...")
print(f"{'idx':>4}  {'LR_acc':>7}  {'BL_acc':>7}  {'SM_acc':>7}  "
      f"{'BL_PSNR':>8}  {'SM_PSNR':>8}")

n_valid = 0
for i in range(len(dataset)):
    if n_valid >= N_SAMPLES:
        break

    sample  = dataset[i]
    hr_cpu  = sample['hr']                    # [3,256,256]
    gt_nuc  = sample['gt_nuc_mask'].bool()    # [256,256]

    if gt_nuc.sum() < 10:
        continue

    hr = hr_cpu.unsqueeze(0).to(device)       # [1,3,256,256]

    # 退化生成 LR（与训练时完全相同的参数）
    lr_cpu = degrade(
        hr_cpu,
        scale=4,
        blur_sigma_range=(2.0, 3.0),
        noise_std_range=(0.03, 0.08),
        stain_jitter_strength=0.15,
    )
    lr = lr_cpu.unsqueeze(0).to(device)       # [1,3,256,256]

    # 固定噪声和时间步
    noise    = torch.randn_like(hr)
    t        = torch.tensor([FIXED_T], device=device)
    noisy_hr = scheduler.add_noise(hr, noise, t)

    with torch.no_grad():
        # ── CellViT(HR) 伪标签 ─────────────────────────────────────
        hr_lbl, hr_prob = run_cellvit(hr)
        hr_lbl_cpu = hr_lbl.squeeze(0).cpu()   # [256,256]

        # ── CellViT(LR) ───────────────────────────────────────────
        lr_lbl, lr_prob = run_cellvit(lr)
        lr_lbl_cpu = lr_lbl.squeeze(0).cpu()

        # ── 基线 SR ────────────────────────────────────────────────
        inp_b  = torch.cat([lr, noisy_hr], dim=1)
        np_b   = unet_baseline(inp_b, t).sample
        x0_b   = predict_x0_from_noise_shared(noisy_hr, np_b, t, scheduler)
        bl_lbl, bl_prob = run_cellvit(x0_b)
        bl_lbl_cpu = bl_lbl.squeeze(0).cpu()

        # ── 语义 SR ────────────────────────────────────────────────
        inp_s  = torch.cat([lr, noisy_hr], dim=1)
        np_s   = unet_semantic(inp_s, t).sample
        x0_s   = predict_x0_from_noise_shared(noisy_hr, np_s, t, scheduler)
        sm_lbl, sm_prob = run_cellvit(x0_s)
        sm_lbl_cpu = sm_lbl.squeeze(0).cpu()

    # ── 计算指标 ───────────────────────────────────────────────────
    def cls_acc(pred_lbl, ref_lbl, mask):
        return (pred_lbl[mask] == ref_lbl[mask]).float().mean().item()

    def prob_mae(pred_prob, ref_prob, mask):
        # pred_prob: [1,6,H,W], ref_prob: [1,6,H,W]
        mae_map = (pred_prob.squeeze(0) - ref_prob.squeeze(0)).abs().mean(dim=0)  # [H,W]
        return mae_map[mask].mean().item()

    dir_acc['lr'].append(cls_acc(lr_lbl_cpu, hr_lbl_cpu, gt_nuc))
    dir_acc['baseline'].append(cls_acc(bl_lbl_cpu, hr_lbl_cpu, gt_nuc))
    dir_acc['semantic'].append(cls_acc(sm_lbl_cpu, hr_lbl_cpu, gt_nuc))

    sem_mae['lr'].append(prob_mae(lr_prob.cpu(), hr_prob.cpu(), gt_nuc))
    sem_mae['baseline'].append(prob_mae(bl_prob.cpu(), hr_prob.cpu(), gt_nuc))
    sem_mae['semantic'].append(prob_mae(sm_prob.cpu(), hr_prob.cpu(), gt_nuc))

    psnr_d['baseline'].append(compute_psnr(x0_b.cpu(), hr.cpu()))
    psnr_d['semantic'].append(compute_psnr(x0_s.cpu(), hr.cpu()))
    ssim_d['baseline'].append(compute_ssim(x0_b.cpu(), hr.cpu()))
    ssim_d['semantic'].append(compute_ssim(x0_s.cpu(), hr.cpu()))

    if n_valid % 20 == 0:
        print(f"{n_valid:>4}  "
              f"{dir_acc['lr'][-1]:>7.4f}  "
              f"{dir_acc['baseline'][-1]:>7.4f}  "
              f"{dir_acc['semantic'][-1]:>7.4f}  "
              f"{psnr_d['baseline'][-1]:>8.2f}  "
              f"{psnr_d['semantic'][-1]:>8.2f}")

    n_valid += 1

# ── 汇总结果 ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'指标':<20} {'LR':>10} {'Baseline':>10} {'Semantic':>10}")
print("-" * 65)
print(f"{'Dir_Acc (vs HR)':<20} "
      f"{np.mean(dir_acc['lr']):>10.4f} "
      f"{np.mean(dir_acc['baseline']):>10.4f} "
      f"{np.mean(dir_acc['semantic']):>10.4f}")
print(f"{'Sem_MAE (vs HR)':<20} "
      f"{np.mean(sem_mae['lr']):>10.4f} "
      f"{np.mean(sem_mae['baseline']):>10.4f} "
      f"{np.mean(sem_mae['semantic']):>10.4f}")
print(f"{'PSNR (dB)':<20} "
      f"{'—':>10} "
      f"{np.mean(psnr_d['baseline']):>10.2f} "
      f"{np.mean(psnr_d['semantic']):>10.2f}")
print(f"{'SSIM':<20} "
      f"{'—':>10} "
      f"{np.mean(ssim_d['baseline']):>10.4f} "
      f"{np.mean(ssim_d['semantic']):>10.4f}")
print("=" * 65)

print(f"\n语义 vs 基线  Dir_Acc 提升: "
      f"{np.mean(dir_acc['semantic']) - np.mean(dir_acc['baseline']):+.4f}")
print(f"语义 vs 基线  Sem_MAE 降低: "
      f"{np.mean(sem_mae['baseline']) - np.mean(sem_mae['semantic']):+.4f}")
print(f"语义 vs 基线  PSNR   变化: "
      f"{np.mean(psnr_d['semantic']) - np.mean(psnr_d['baseline']):+.2f} dB")
print(f"语义 vs 基线  SSIM   变化: "
      f"{np.mean(ssim_d['semantic']) - np.mean(ssim_d['baseline']):+.4f}")
print(f"\n语义 vs LR   Dir_Acc 提升: "
      f"{np.mean(dir_acc['semantic']) - np.mean(dir_acc['lr']):+.4f}")