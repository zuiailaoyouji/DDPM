"""
inference_self_guided.py
用于语义引导 DDPM 的 SR 推理脚本 (已修复数据泄露、退化对齐及批量处理问题)。

核心更新：
  1. 修复 Baseline 逻辑：直接使用 degrade() 输出的同尺寸 LR 作为 bicubic_up 基线和保真度参照。
  2. 修复插值误差：保真度 Fidelity Loss 直接计算 L1(pred, lr)，不再进行多余的插值。
  3. 修复随机状态：使用 get_rng_state/set_rng_state 真正隔离并恢复退化过程的随机性。
  4. 补全批量推理：完整实现了 run_batch_inference 及其闭环评分逻辑。
  5. 扩展文件支持：支持 png, jpg, jpeg, tif, tiff。
  6. 增加 Stage 3 提示：对是否开启 semantic_injection 给出阶段匹配警告。
  7. 输出三联图 comparison.png：LR | SR | HR，上方标注 Baseline / Final 的 PSNR、SSIM。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
import csv
import random
from typing import Optional, Union

from tqdm import tqdm
from diffusers import DDPMScheduler
from torch.utils.data import Dataset, DataLoader

from unet_wrapper import create_model
from ddpm_utils import load_hovernet, predict_x0_from_noise_shared, get_device
from degradation import degrade, apply_degradation
from metrics import (compute_psnr, compute_ssim, compute_artifact_penalty)
from hovernet_input_preprocess import run_hovernet_semantics_aligned


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_image(img_path, target_size=256, device='cuda'):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    t   = torch.from_numpy(img).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)   # [1,3,H,W]


def save_tensor(t, path):
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _tensor_to_rgb_u8(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] 或 [1,C,H,W] → uint8 RGB [H,W,3]"""
    if t.dim() == 4:
        t = t.squeeze(0)
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_lr_sr_hr_triptych(
    lr_t: torch.Tensor,
    sr_t: torch.Tensor,
    hr_t: torch.Tensor,
    path: str,
    baseline_psnr: float,
    baseline_ssim: float,
    final_psnr: float,
    final_ssim: float,
    caption_h: int = 80,
    font_scale: float = 0.45,
    text_margin: int = 8,
) -> None:
    """
    横向拼接 LR | SR | HR，上方留出条带绘制指标（BGR 保存）。
    Baseline：LR 相对 HR；Final：SR（BEST_SR）相对 HR。
    """
    lr_u8 = _tensor_to_rgb_u8(lr_t)
    sr_u8 = _tensor_to_rgb_u8(sr_t)
    hr_u8 = _tensor_to_rgb_u8(hr_t)
    h, w, _ = lr_u8.shape
    strip = np.hstack([lr_u8, sr_u8, hr_u8])
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)

    W = strip_bgr.shape[1]
    banner = np.full((caption_h, W, 3), 255, dtype=np.uint8)
    line1 = (
        f"Baseline (LR vs HR)  PSNR={baseline_psnr:.2f} dB   SSIM={baseline_ssim:.4f}"
    )
    line2 = (
        f"Final (SR vs HR)      PSNR={final_psnr:.2f} dB   SSIM={final_ssim:.4f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 1
    color = (20, 20, 20)
    y1 = text_margin + 18
    y2 = text_margin + 42
    cv2.putText(banner, line1, (text_margin, y1), font, font_scale, color, thick, cv2.LINE_AA)
    cv2.putText(banner, line2, (text_margin, y2), font, font_scale, color, thick, cv2.LINE_AA)

    # 子图标题（LR / SR / HR）
    sub_w = w
    for i, lab in enumerate(['LR input', 'SR (best)', 'HR ref']):
        x0 = i * sub_w + text_margin
        cv2.putText(strip_bgr, lab, (x0, h - 8), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    out = np.vstack([banner, strip_bgr])
    cv2.imwrite(path, out)


def get_hovernet_maps_multiclass(hovernet, img_01, hovernet_upsample_factor: float = 2.0):
    """提取多类别语义特征：tp_prob, tp_conf, nuc_mask"""
    out = run_hovernet_semantics_aligned(
        hovernet,
        img_01,
        upsample_factor=hovernet_upsample_factor,
    )
    return out["tp_prob"].cpu(), out["tp_conf"].unsqueeze(1).cpu(), out["nuc_mask"].unsqueeze(1).cpu()


def compute_new_semantic_mae(p_pred, p_clean, conf_clean, mask_clean, tau_nuc=0.4, tau_conf=0.6):
    """计算多类别 Semantic MAE"""
    valid_mask = ((mask_clean > tau_nuc) & (conf_clean > tau_conf)).float()
    denom = valid_mask.sum().clamp(min=1.0)
    mae_map = (p_pred - p_clean).abs().mean(dim=1, keepdim=True)
    return ((mae_map * valid_mask).sum() / denom).item()


def build_semantic_tensor(hovernet, img_01: torch.Tensor,
                          device: Optional[Union[str, torch.device]] = None,
                          hovernet_upsample_factor: float = 2.0) -> torch.Tensor:
    """输出: [B,8,H,W]"""
    out = run_hovernet_semantics_aligned(
        hovernet,
        img_01,
        upsample_factor=hovernet_upsample_factor,
    )
    tp_prob = out["tp_prob"]
    nuc_mask = out["nuc_mask"].unsqueeze(1)  # [B,1,H,W]
    tp_conf = out["tp_conf"].unsqueeze(1)    # [B,1,H,W]
    sem = torch.cat([tp_prob, nuc_mask, tp_conf], dim=1)
    if device is None:
        device = img_01.device
    return sem.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 批量推理闭环逻辑 (补齐)
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_inference(dataloader, unet, hovernet, scheduler, args):
    csv_path   = os.path.join(args.output_dir, 'sr_results.csv')
    fieldnames = ['Filename', 'Baseline_PSNR', 'Baseline_SSIM',
                  'Final_PSNR', 'Final_SSIM', 'Final_Fidelity_L1',
                  'Final_Artifact', 'Final_Semantic_Shift',
                  'Final_Penalty_Score', 'Stop_Reason', 'Best_Iter']

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    device = args.device

    for batch_idx, (hr_batch, lr_batch, deg_params_batch, filenames) in enumerate(tqdm(dataloader, desc='SR Batch')):
        valid = [i for i, f in enumerate(filenames) if f != 'ERROR']
        if not valid:
            continue

        hr_batch = hr_batch[valid].to(device)
        lr_batch = lr_batch[valid].to(device)
        deg_params_batch = {
            k: (v[valid].to(device) if torch.is_tensor(v) else v)
            for k, v in deg_params_batch.items()
        }
        filenames = [filenames[i] for i in valid]
        B = hr_batch.shape[0]

        # [修复1] lr 本身已经是同尺寸退化图，直接作为 bicubic_up 基线
        bicubic_up = lr_batch.clone()
        base_psnr = [compute_psnr(bicubic_up[i:i+1].cpu(), hr_batch[i:i+1].cpu()) for i in range(B)]
        base_ssim = [compute_ssim(bicubic_up[i:i+1].cpu(), hr_batch[i:i+1].cpu()) for i in range(B)]

        # 提取语义先验（从 lr/bicubic_up 提取，杜绝数据泄露）
        sem_tensor = None
        p_base, conf_base, mask_base = None, None, None
        if args.use_semantic_injection and (hovernet is not None) and getattr(unet, "use_semantic", False):
            sem_tensor = build_semantic_tensor(
                hovernet,
                bicubic_up,
                device=device,
                hovernet_upsample_factor=args.hovernet_upsample_factor,
            )
            p_base, conf_base, mask_base = get_hovernet_maps_multiclass(
                hovernet,
                bicubic_up,
                hovernet_upsample_factor=args.hovernet_upsample_factor,
            )
        elif hovernet is not None:
            # 未开注入时仍可为 Semantic MAE 准备参照（与 bicubic 一致）
            p_base, conf_base, mask_base = get_hovernet_maps_multiclass(
                hovernet,
                bicubic_up,
                hovernet_upsample_factor=args.hovernet_upsample_factor,
            )

        current      = bicubic_up.clone()
        best_tensor  = bicubic_up.clone()
        best_scores  = [float('inf')] * B  # 惩罚分数，越小越好
        best_iters   = [0] * B
        stop_reasons = ['Max_Iters'] * B
        active       = [True] * B

        for i in range(args.iters):
            if not any(active):
                break

            iter_t = max(20, int(args.noise_t * (0.75 ** i)))
            t_ten  = torch.full((B,), iter_t, device=device).long()
            noise  = torch.randn_like(current)
            x_t    = scheduler.add_noise(current, noise, t_ten)

            model_input = torch.cat([lr_batch, x_t], dim=1)
            with torch.no_grad():
                noise_pred = unet(model_input, t_ten, semantic=sem_tensor).sample
            pred = predict_x0_from_noise_shared(x_t, noise_pred, t_ten, scheduler)

            for idx in range(B):
                if not active[idx]:
                    continue

                p = pred[idx:idx+1]
                l = lr_batch[idx:idx+1]
                b = bicubic_up[idx:idx+1].cpu()

                # 使用每个样本原始退化参数执行一致性校验
                curr_sigma = deg_params_batch["sigma"][idx].item()
                curr_stain = deg_params_batch["stain_scales"][idx]  # [3,1,1]
                if args.use_noise_in_fidelity:
                    curr_noise_std = deg_params_batch["noise_std"][idx].item()
                    curr_noise_tensor = deg_params_batch["noise_tensor"][idx]
                else:
                    curr_noise_std = 0.0
                    curr_noise_tensor = None
                deg_p = apply_degradation(
                    p.squeeze(0),  # [C,H,W]
                    scale=args.scale,
                    sigma=curr_sigma,
                    stain_scales=curr_stain,
                    noise_std=curr_noise_std,
                    noise_tensor=curr_noise_tensor,
                ).unsqueeze(0)  # [1,C,H,W]
                fidelity_loss = F.l1_loss(deg_p, l).item()
                art = compute_artifact_penalty(p.cpu(), b)

                sem_shift = 0.0
                if hovernet is not None and p_base is not None:
                    p_p, _, _ = get_hovernet_maps_multiclass(
                        hovernet,
                        p,
                        hovernet_upsample_factor=args.hovernet_upsample_factor,
                    )
                    sem_shift = compute_new_semantic_mae(
                        p_p,
                        p_base[idx:idx+1], conf_base[idx:idx+1], mask_base[idx:idx+1],
                        args.tau_nuc, args.tau_conf
                    )

                penalty = fidelity_loss + 0.1 * sem_shift + 0.01 * art

                if penalty < best_scores[idx]:
                    best_scores[idx] = penalty
                    best_tensor[idx] = pred[idx]
                    best_iters[idx]  = i + 1

                if fidelity_loss > args.max_fidelity_loss:
                    active[idx] = False
                    stop_reasons[idx] = 'Fidelity_Collapse'

            current = pred.clone()

        # 保存结果并写入 CSV
        rows = []
        for idx in range(B):
            fname = filenames[idx]
            d = os.path.join(args.output_dir, os.path.splitext(fname)[0])
            os.makedirs(d, exist_ok=True)

            save_tensor(hr_batch[idx:idx+1], os.path.join(d, 'hr_reference.png'))
            save_tensor(lr_batch[idx:idx+1], os.path.join(d, 'lr_input.png'))
            save_tensor(best_tensor[idx:idx+1], os.path.join(d, 'BEST_SR.png'))

            p_best = best_tensor[idx:idx+1]
            final_psnr = compute_psnr(p_best.cpu(), hr_batch[idx:idx+1].cpu())
            final_ssim = compute_ssim(p_best.cpu(), hr_batch[idx:idx+1].cpu())

            save_lr_sr_hr_triptych(
                lr_batch[idx:idx+1],
                p_best,
                hr_batch[idx:idx+1],
                os.path.join(d, 'comparison.png'),
                baseline_psnr=base_psnr[idx],
                baseline_ssim=base_ssim[idx],
                final_psnr=final_psnr,
                final_ssim=final_ssim,
            )

            curr_sigma = deg_params_batch["sigma"][idx].item()
            curr_stain = deg_params_batch["stain_scales"][idx]
            if args.use_noise_in_fidelity:
                curr_noise_std = deg_params_batch["noise_std"][idx].item()
                curr_noise_tensor = deg_params_batch["noise_tensor"][idx]
            else:
                curr_noise_std = 0.0
                curr_noise_tensor = None
            deg_best = apply_degradation(
                p_best.squeeze(0),
                scale=args.scale,
                sigma=curr_sigma,
                stain_scales=curr_stain,
                noise_std=curr_noise_std,
                noise_tensor=curr_noise_tensor,
            ).unsqueeze(0)
            fid_best = F.l1_loss(deg_best, lr_batch[idx:idx+1]).item()
            art_best = compute_artifact_penalty(p_best.cpu(), bicubic_up[idx:idx+1].cpu())

            sem_best = 0.0
            if hovernet and p_base is not None:
                p_p, _, _ = get_hovernet_maps_multiclass(
                    hovernet,
                    p_best,
                    hovernet_upsample_factor=args.hovernet_upsample_factor,
                )
                sem_best = compute_new_semantic_mae(
                    p_p, p_base[idx:idx+1], conf_base[idx:idx+1], mask_base[idx:idx+1],
                    args.tau_nuc, args.tau_conf
                )

            rows.append(dict(
                Filename=fname,
                Baseline_PSNR=f"{base_psnr[idx]:.4f}", Baseline_SSIM=f"{base_ssim[idx]:.4f}",
                Final_PSNR=f"{final_psnr:.4f}", Final_SSIM=f"{final_ssim:.4f}",
                Final_Fidelity_L1=f"{fid_best:.6f}", Final_Artifact=f"{art_best:.4f}",
                Final_Semantic_Shift=f"{sem_best:.6f}", Final_Penalty_Score=f"{best_scores[idx]:.4f}",
                Stop_Reason=stop_reasons[idx], Best_Iter=best_iters[idx]
            ))

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 数据集定义
# ─────────────────────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, file_paths, args):
        self.file_paths  = file_paths
        self.args        = args

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            hr  = torch.from_numpy(img).float() / 255.0
            hr  = hr.permute(2, 0, 1)

            # [修复4] 真正的保存和恢复随机状态，避免干扰外部随机性
            torch_rng_state = torch.random.get_rng_state()
            py_rng_state = random.getstate()
            torch.manual_seed(idx)
            random.seed(idx)
            lr, deg_params = degrade(
                hr,
                scale=self.args.scale,
                blur_sigma_range=tuple(self.args.blur_sigma_range),
                noise_std_range=tuple(self.args.noise_std_range),
                stain_jitter_strength=self.args.stain_jitter,
                return_params=True,
            )
            torch.random.set_rng_state(torch_rng_state)
            random.setstate(py_rng_state)

            return hr, lr, deg_params, os.path.basename(path)
        except Exception:
            zero_img = torch.zeros(3, 256, 256)
            zero_params = {
                "sigma": torch.tensor(0.0, dtype=torch.float32),
                "noise_std": torch.tensor(0.0, dtype=torch.float32),
                "stain_scales": torch.ones(3, 1, 1),
                "noise_tensor": torch.zeros(3, 256, 256),
            }
            return zero_img, zero_img, zero_params, "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='SPM-UNet 语义注入 SR 批量推理')
    p.add_argument('--input_path',    required=True)
    p.add_argument('--output_dir',    default='./results/sr')
    p.add_argument('--unet_path',     required=True)
    p.add_argument('--hovernet_path', default=None)
    p.add_argument('--hovernet_upsample_factor', type=float, default=2.0,
                   help='HoVer-Net 语义提取前上采样倍率（例如 20x→40x 用 2.0；1.0 关闭）')
    p.add_argument('--iters',         type=int,   default=5)
    p.add_argument('--noise_t',       type=int,   default=200, help='起始噪声步(建议200-300)')
    p.add_argument('--scale',         type=int,   default=2)
    p.add_argument('--max_fidelity_loss', type=float, default=0.05, help='保真度崩塌阈值')
    p.add_argument('--batch_size',    type=int,   default=4)
    p.add_argument('--device',        default=None)
    p.add_argument('--gpu_id',        type=int,   default=None)
    p.add_argument('--use_semantic_injection', action='store_true')

    # 严格对齐训练的退化参数
    p.add_argument('--blur_sigma_range', type=float, nargs=2, default=[1.0, 1.0])
    p.add_argument('--noise_std_range',  type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument('--stain_jitter',     type=float, default=0.0)

    # 语义阈值参数
    p.add_argument('--tau_nuc',  type=float, default=0.4)
    p.add_argument('--tau_conf', type=float, default=0.6)
    p.add_argument('--use_noise_in_fidelity', action='store_true',
                   help='在 fidelity 校验中复用退化噪声 realization（默认关闭以减少抖动）')

    args = p.parse_args()
    args.device = (get_device(gpu_id=args.gpu_id) if args.gpu_id is not None
                   else args.device or get_device())

    os.makedirs(args.output_dir, exist_ok=True)

    # 为兼容带语义权重的 checkpoint，这里按 hovernet_path 决定是否构建语义分支；
    # 实际是否启用 modulation 由 --use_semantic_injection 控制。
    unet = create_model(use_semantic=(args.hovernet_path is not None)).to(args.device)
    ckpt = torch.load(args.unet_path, map_location=args.device)

    # [修复3] 增加 Stage 判断提示
    epoch = ckpt.get('epoch', -1)
    if args.use_semantic_injection:
        print(f"⚠️  提示: 已开启 --use_semantic_injection。如加载的 Checkpoint (Epoch {epoch}) 属于 Stage 3 (纯像素阶段)，强行开启可能与训练末态不符。建议仅对 Stage 2 模型开启。")
    else:
        print(f"ℹ️  提示: 未开启 --use_semantic_injection。若当前为 Stage 2 模型，建议开启以发挥架构性能；若为 Stage 3 模型则保持关闭。")

    unet.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    unet.eval()

    if not args.use_semantic_injection and hasattr(unet, "disable_semantic_modulation"):
        unet.disable_semantic_modulation()
    elif args.use_semantic_injection and hasattr(unet, "enable_semantic_modulation"):
        unet.enable_semantic_modulation()

    hovernet = None
    if args.hovernet_path:
        hovernet = load_hovernet(args.hovernet_path, device=args.device)
        print("✓ HoVer-Net loaded")

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # [修复5] 扩展文件后缀支持
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
             if f.lower().endswith(valid_ext)]

    print(f"\n找到 {len(files)} 张待处理图像...")

    # [修复6] 正式启用 Dataset 和 DataLoader 进行批量推理
    ds = InferenceDataset(files, args)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    run_batch_inference(dl, unet, hovernet, scheduler, args)

    print(f"\n✅ 批量推理完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
