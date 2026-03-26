from __future__ import annotations

"""
inference_self_guided.py
用于语义引导 DDPM 的 SR 推理脚本。

本版新增：
  1. 支持直接从 PanNuke fold / root 读取 .npy patch。
  2. 兼容真实目录结构：
       Fold 1/images/fold1/images.npy
       Fold 1/images/fold1/types.npy
       Fold 1/masks/fold1/masks.npy
  3. 保留原有“普通图片文件夹推理”路径。
  4. 保留 target_size 兼容逻辑：默认 256；若输入本来就是 256x256，则不会重复插值。
"""

import argparse
import csv
import os
import random
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ddpm_dataset import _find_pannuke_fold_files, _is_pannuke_fold_dir, _normalize_fold_dirs
from ddpm_utils import get_device, load_hovernet, predict_x0_from_noise_shared
from degradation import apply_degradation, degrade
from hovernet_input_preprocess import run_hovernet_semantics_aligned
from metrics import compute_artifact_penalty, compute_psnr, compute_ssim
from unet_wrapper import create_model


_VALID_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def save_tensor(t, path):
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _tensor_to_rgb_u8(t: torch.Tensor) -> np.ndarray:
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
    lr_u8 = _tensor_to_rgb_u8(lr_t)
    sr_u8 = _tensor_to_rgb_u8(sr_t)
    hr_u8 = _tensor_to_rgb_u8(hr_t)
    h, w, _ = lr_u8.shape
    strip = np.hstack([lr_u8, sr_u8, hr_u8])
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)

    W = strip_bgr.shape[1]
    banner = np.full((caption_h, W, 3), 255, dtype=np.uint8)
    line1 = f"Baseline (LR vs HR)  PSNR={baseline_psnr:.2f} dB   SSIM={baseline_ssim:.4f}"
    line2 = f"Final (SR vs HR)      PSNR={final_psnr:.2f} dB   SSIM={final_ssim:.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (20, 20, 20)
    y1 = text_margin + 18
    y2 = text_margin + 42
    cv2.putText(banner, line1, (text_margin, y1), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(banner, line2, (text_margin, y2), font, font_scale, color, 1, cv2.LINE_AA)

    for i, lab in enumerate(['LR input', 'SR (best)', 'HR ref']):
        x0 = i * w + text_margin
        cv2.putText(strip_bgr, lab, (x0, h - 8), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    out = np.vstack([banner, strip_bgr])
    cv2.imwrite(path, out)


def get_hovernet_maps_multiclass(hovernet, img_01, hovernet_upsample_factor: float = 2.0):
    out = run_hovernet_semantics_aligned(
        hovernet,
        img_01,
        upsample_factor=hovernet_upsample_factor,
    )
    return out["tp_prob"].cpu(), out["tp_conf"].unsqueeze(1).cpu(), out["nuc_mask"].unsqueeze(1).cpu()


def compute_new_semantic_mae(p_pred, p_clean, conf_clean, mask_clean, tau_nuc=0.4, tau_conf=0.6):
    valid_mask = ((mask_clean > tau_nuc) & (conf_clean > tau_conf)).float()
    denom = valid_mask.sum().clamp(min=1.0)
    mae_map = (p_pred - p_clean).abs().mean(dim=1, keepdim=True)
    return ((mae_map * valid_mask).sum() / denom).item()


def build_semantic_tensor(hovernet, img_01: torch.Tensor,
                          device: Optional[Union[str, torch.device]] = None,
                          hovernet_upsample_factor: float = 2.0) -> torch.Tensor:
    out = run_hovernet_semantics_aligned(
        hovernet,
        img_01,
        upsample_factor=hovernet_upsample_factor,
    )
    tp_prob = out["tp_prob"]
    nuc_mask = out["nuc_mask"].unsqueeze(1)
    tp_conf = out["tp_conf"].unsqueeze(1)
    sem = torch.cat([tp_prob, nuc_mask, tp_conf], dim=1)
    if device is None:
        device = img_01.device
    return sem.to(device)


def _to_hwc3_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f'Expected 3D image array, got shape={img.shape}')
    if img.shape[-1] == 3:
        out = img
    elif img.shape[0] == 3:
        out = np.transpose(img, (1, 2, 0))
    else:
        raise ValueError(f'Cannot infer channel axis for shape={img.shape}')

    if out.dtype != np.uint8:
        if np.issubdtype(out.dtype, np.floating):
            vmax = float(out.max()) if out.size > 0 else 1.0
            out = (out * 255.0).clip(0, 255).astype(np.uint8) if vmax <= 1.0 else out.clip(0, 255).astype(np.uint8)
        else:
            out = out.clip(0, 255).astype(np.uint8)
    return out


def _maybe_resize_rgb(img: np.ndarray, target_size: Optional[int]) -> np.ndarray:
    if target_size is None:
        return img
    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        return img
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def _sanitize_filename(name: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in str(name))
    return safe[:180] if len(safe) > 180 else safe


def infer_input_mode(input_path: str) -> str:
    if os.path.isdir(input_path):
        if _is_pannuke_fold_dir(input_path):
            return 'pannuke'
        for name in os.listdir(input_path):
            cand = os.path.join(input_path, name)
            if os.path.isdir(cand) and _is_pannuke_fold_dir(cand):
                return 'pannuke'
    return 'image_folder'


# ─────────────────────────────────────────────────────────────────────────────
# 批量推理闭环逻辑
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_inference(dataloader, unet, hovernet, scheduler, args):
    csv_path = os.path.join(args.output_dir, 'sr_results.csv')
    fieldnames = [
        'Filename', 'TypeName', 'Baseline_PSNR', 'Baseline_SSIM',
        'Final_PSNR', 'Final_SSIM', 'Final_Fidelity_L1',
        'Final_Artifact', 'Final_Semantic_Shift',
        'Final_Penalty_Score', 'Stop_Reason', 'Best_Iter'
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    device = args.device

    for batch in tqdm(dataloader, desc='SR Batch'):
        hr_batch = batch['hr'].to(device)
        lr_batch = batch['lr'].to(device)
        filenames = list(batch['filename'])
        type_names = list(batch.get('type_name', ['NA'] * len(filenames)))
        deg_params_batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch['deg_params'].items()
        }
        B = hr_batch.shape[0]

        bicubic_up = lr_batch.clone()
        base_psnr = [compute_psnr(bicubic_up[i:i+1].cpu(), hr_batch[i:i+1].cpu()) for i in range(B)]
        base_ssim = [compute_ssim(bicubic_up[i:i+1].cpu(), hr_batch[i:i+1].cpu()) for i in range(B)]

        sem_tensor = None
        p_base, conf_base, mask_base = None, None, None
        if args.use_semantic_injection and (hovernet is not None) and getattr(unet, 'use_semantic', False):
            sem_tensor = build_semantic_tensor(
                hovernet, bicubic_up, device=device,
                hovernet_upsample_factor=args.hovernet_upsample_factor,
            )
            p_base, conf_base, mask_base = get_hovernet_maps_multiclass(
                hovernet, bicubic_up,
                hovernet_upsample_factor=args.hovernet_upsample_factor,
            )
        elif hovernet is not None:
            p_base, conf_base, mask_base = get_hovernet_maps_multiclass(
                hovernet, bicubic_up,
                hovernet_upsample_factor=args.hovernet_upsample_factor,
            )

        current = bicubic_up.clone()
        best_tensor = bicubic_up.clone()
        best_scores = [float('inf')] * B
        best_iters = [0] * B
        stop_reasons = ['Max_Iters'] * B
        active = [True] * B

        for i in range(args.iters):
            if not any(active):
                break

            iter_t = max(20, int(args.noise_t * (0.75 ** i)))
            t_ten = torch.full((B,), iter_t, device=device).long()
            noise = torch.randn_like(current)
            x_t = scheduler.add_noise(current, noise, t_ten)

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

                curr_sigma = deg_params_batch['sigma'][idx].item()
                curr_stain = deg_params_batch['stain_scales'][idx]
                if args.use_noise_in_fidelity:
                    curr_noise_std = deg_params_batch['noise_std'][idx].item()
                    curr_noise_tensor = deg_params_batch['noise_tensor'][idx]
                else:
                    curr_noise_std = 0.0
                    curr_noise_tensor = None

                deg_p = apply_degradation(
                    p.squeeze(0),
                    scale=args.scale,
                    sigma=curr_sigma,
                    stain_scales=curr_stain,
                    noise_std=curr_noise_std,
                    noise_tensor=curr_noise_tensor,
                ).unsqueeze(0)
                fidelity_loss = F.l1_loss(deg_p, l).item()
                art = compute_artifact_penalty(p.cpu(), b)

                sem_shift = 0.0
                if hovernet is not None and p_base is not None:
                    p_p, _, _ = get_hovernet_maps_multiclass(
                        hovernet, p,
                        hovernet_upsample_factor=args.hovernet_upsample_factor,
                    )
                    sem_shift = compute_new_semantic_mae(
                        p_p,
                        p_base[idx:idx+1], conf_base[idx:idx+1], mask_base[idx:idx+1],
                        args.tau_nuc, args.tau_conf,
                    )

                penalty = fidelity_loss + 0.1 * sem_shift + 0.01 * art
                if penalty < best_scores[idx]:
                    best_scores[idx] = penalty
                    best_tensor[idx] = pred[idx]
                    best_iters[idx] = i + 1

                if fidelity_loss > args.max_fidelity_loss:
                    active[idx] = False
                    stop_reasons[idx] = 'Fidelity_Collapse'

            current = pred.clone()

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
                lr_batch[idx:idx+1], p_best, hr_batch[idx:idx+1],
                os.path.join(d, 'comparison.png'),
                baseline_psnr=base_psnr[idx], baseline_ssim=base_ssim[idx],
                final_psnr=final_psnr, final_ssim=final_ssim,
            )

            curr_sigma = deg_params_batch['sigma'][idx].item()
            curr_stain = deg_params_batch['stain_scales'][idx]
            if args.use_noise_in_fidelity:
                curr_noise_std = deg_params_batch['noise_std'][idx].item()
                curr_noise_tensor = deg_params_batch['noise_tensor'][idx]
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
            if hovernet is not None and p_base is not None:
                p_p, _, _ = get_hovernet_maps_multiclass(
                    hovernet, p_best,
                    hovernet_upsample_factor=args.hovernet_upsample_factor,
                )
                sem_best = compute_new_semantic_mae(
                    p_p, p_base[idx:idx+1], conf_base[idx:idx+1], mask_base[idx:idx+1],
                    args.tau_nuc, args.tau_conf,
                )

            rows.append(dict(
                Filename=fname,
                TypeName=type_names[idx],
                Baseline_PSNR=f'{base_psnr[idx]:.4f}',
                Baseline_SSIM=f'{base_ssim[idx]:.4f}',
                Final_PSNR=f'{final_psnr:.4f}',
                Final_SSIM=f'{final_ssim:.4f}',
                Final_Fidelity_L1=f'{fid_best:.6f}',
                Final_Artifact=f'{art_best:.4f}',
                Final_Semantic_Shift=f'{sem_best:.6f}',
                Final_Penalty_Score=f'{best_scores[idx]:.4f}',
                Stop_Reason=stop_reasons[idx],
                Best_Iter=best_iters[idx],
            ))

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 数据集定义
# ─────────────────────────────────────────────────────────────────────────────

class ImageFolderInferenceDataset(Dataset):
    def __init__(self, file_paths: Sequence[str], args):
        self.file_paths = list(file_paths)
        self.args = args

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f'Cannot read image: {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _maybe_resize_rgb(img, self.args.target_size)
        hr = torch.from_numpy(img).float() / 255.0
        hr = hr.permute(2, 0, 1).contiguous()

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

        return {
            'hr': hr,
            'lr': lr,
            'deg_params': deg_params,
            'filename': _sanitize_filename(os.path.basename(path)),
            'type_name': 'NA',
        }


class PanNukeInferenceDataset(Dataset):
    def __init__(self, input_path: str, args):
        self.args = args
        self.fold_dirs = _normalize_fold_dirs(root_dir=input_path)
        if _is_pannuke_fold_dir(input_path):
            self.fold_dirs = _normalize_fold_dirs(fold_dirs=[input_path])
        if not self.fold_dirs:
            raise ValueError(f'Cannot find valid PanNuke fold under: {input_path}')

        self.images = []
        self.types = []
        self.index = []

        for fold_id, fold_dir in enumerate(self.fold_dirs):
            fold_files = _find_pannuke_fold_files(fold_dir)
            if fold_files is None:
                raise ValueError(f'Cannot locate PanNuke npy files under: {fold_dir}')
            images_path, types_path, _ = fold_files
            images = np.load(images_path, mmap_mode='r')
            types = np.load(types_path, mmap_mode='r')
            if len(images) != len(types):
                raise ValueError(f'images/types length mismatch in {fold_dir}: {len(images)} vs {len(types)}')
            self.images.append(images)
            self.types.append(types)
            self.index.extend((fold_id, i) for i in range(len(images)))

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _normalize_type_name(type_value) -> str:
        t = np.asarray(type_value)
        t = np.squeeze(t)
        if t.ndim == 0:
            value = t.item()
        else:
            flat = t.reshape(-1)
            if flat.size == 0:
                value = 'Unknown'
            elif flat.size == 1:
                value = flat[0].item() if hasattr(flat[0], 'item') else flat[0]
            else:
                if np.issubdtype(flat.dtype, np.number):
                    value = f'class_{int(np.argmax(flat))}'
                else:
                    value = flat[0].item() if hasattr(flat[0], 'item') else flat[0]
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        value = str(value).strip()
        return value if value else 'Unknown'

    def __getitem__(self, idx):
        fold_id, local_idx = self.index[idx]
        img = _to_hwc3_uint8(self.images[fold_id][local_idx])
        img = _maybe_resize_rgb(img, self.args.target_size)
        hr = torch.from_numpy(img).float() / 255.0
        hr = hr.permute(2, 0, 1).contiguous()

        type_name = self._normalize_type_name(self.types[fold_id][local_idx])
        filename = _sanitize_filename(f'{type_name}_{idx:05d}.png')

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

        return {
            'hr': hr,
            'lr': lr,
            'deg_params': deg_params,
            'filename': filename,
            'type_name': type_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='SPM-UNet 语义注入 SR 批量推理')
    p.add_argument('--input_path', required=True, help='普通图片文件夹，或 PanNuke 的 fold/root 路径')
    p.add_argument('--input_mode', default='auto', choices=['auto', 'image_folder', 'pannuke'])
    p.add_argument('--output_dir', default='./results/sr')
    p.add_argument('--unet_path', required=True)
    p.add_argument('--hovernet_path', default=None)
    p.add_argument('--hovernet_upsample_factor', type=float, default=2.0,
                   help='HoVer-Net 语义提取前上采样倍率（例如 20x→40x 用 2.0；1.0 关闭）')
    p.add_argument('--iters', type=int, default=5)
    p.add_argument('--noise_t', type=int, default=200, help='起始噪声步(建议200-300)')
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--target_size', type=int, default=256,
                   help='输入 patch 目标尺寸；设为 256 可兼容 PanNuke，设为 -1 表示不 resize')
    p.add_argument('--max_fidelity_loss', type=float, default=0.05, help='保真度崩塌阈值')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default=None)
    p.add_argument('--gpu_id', type=int, default=None)
    p.add_argument('--use_semantic_injection', action='store_true')
    p.add_argument('--blur_sigma_range', type=float, nargs=2, default=[1.0, 1.0])
    p.add_argument('--noise_std_range', type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument('--stain_jitter', type=float, default=0.0)
    p.add_argument('--tau_nuc', type=float, default=0.4)
    p.add_argument('--tau_conf', type=float, default=0.6)
    p.add_argument('--use_noise_in_fidelity', action='store_true',
                   help='在 fidelity 校验中复用退化噪声 realization（默认关闭以减少抖动）')

    args = p.parse_args()
    args.device = get_device(gpu_id=args.gpu_id) if args.gpu_id is not None else (args.device or get_device())
    if args.target_size is not None and args.target_size <= 0:
        args.target_size = None

    os.makedirs(args.output_dir, exist_ok=True)

    unet = create_model(use_semantic=(args.hovernet_path is not None)).to(args.device)
    ckpt = torch.load(args.unet_path, map_location=args.device)
    epoch = ckpt.get('epoch', -1) if isinstance(ckpt, dict) else -1
    if args.use_semantic_injection:
        print(f'⚠️  提示: 已开启 --use_semantic_injection。若 Checkpoint (Epoch {epoch}) 属于 Stage 3，强行开启可能与训练末态不符。')
    else:
        print(f'ℹ️  提示: 未开启 --use_semantic_injection。若当前为 Stage 2 模型，建议开启；若为 Stage 3 模型则保持关闭。')

    unet.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    unet.eval()

    if not args.use_semantic_injection and hasattr(unet, 'disable_semantic_modulation'):
        unet.disable_semantic_modulation()
    elif args.use_semantic_injection and hasattr(unet, 'enable_semantic_modulation'):
        unet.enable_semantic_modulation()

    hovernet = None
    if args.hovernet_path:
        hovernet = load_hovernet(args.hovernet_path, device=args.device)
        print('✓ HoVer-Net loaded')

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    input_mode = infer_input_mode(args.input_path) if args.input_mode == 'auto' else args.input_mode
    print(f'Input mode: {input_mode}')

    if input_mode == 'pannuke':
        ds = PanNukeInferenceDataset(args.input_path, args)
        print(f'找到 {len(ds)} 个 PanNuke patch 待处理...')
    else:
        files = [
            os.path.join(args.input_path, f)
            for f in sorted(os.listdir(args.input_path))
            if f.lower().endswith(_VALID_EXT)
        ]
        print(f'找到 {len(files)} 张待处理图像...')
        ds = ImageFolderInferenceDataset(files, args)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    run_batch_inference(dl, unet, hovernet, scheduler, args)

    print(f'\n✅ 批量推理完成！结果保存在: {args.output_dir}')


if __name__ == '__main__':
    main()
