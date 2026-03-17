"""
inference_self_guided.py
用于语义引导 DDPM 的 SR 推理脚本。

与旧的极化推理脚本的区别
--------------------------
旧流程：像素级方向锁定（最大化 0.98 / 最小化 0.02），
       监控宏观分布，并基于 Conf_Gap 提前停止。

新流程：LR → 重建 HR → 与 LR 上采样基线比较 PSNR/SSIM/L1，
       早停条件：
         - L1 > max_l1_distortion （出现结构性幻觉）
         - artifact_penalty > max_artifact_ratio （出现明显伪纹 / 伪纹理）
       最优结果：以 composite_score（PSNR/SSIM/Semantic_MAE/Artifact）最高为准。
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
import csv
from tqdm import tqdm
from diffusers import DDPMScheduler
from torch.utils.data import Dataset, DataLoader

from unet_wrapper import create_model
from ddpm_utils import (load_hovernet, predict_x0_from_noise_shared,
                         get_device, print_gpu_info)
from degradation import degrade
from metrics import (compute_psnr, compute_ssim,
                      compute_artifact_penalty, compute_masked_semantic_mae,
                      compute_composite_score)


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
    arr = (t.detach().cpu().clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def get_hovernet_maps(hovernet, img_01):
    """img_01: [B,3,H,W] → (p_neo [B,H,W], cell_mask [B,H,W])"""
    with torch.no_grad():
        dev = next(hovernet.parameters()).device
        out = hovernet(img_01.to(dev) * 255.0)
        p   = torch.softmax(out['tp'], dim=1)[:, 1, :, :]
        m   = torch.softmax(out['np'], dim=1)[:, 1, :, :]
    return p.cpu(), m.cpu()


def build_semantic_tensor(hovernet, img_01: torch.Tensor,
                          tau_pos: float = 0.65, tau_neg: float = 0.35,
                          device: str | torch.device | None = None) -> torch.Tensor:
    """
    构造 SPM-UNet 所需语义先验张量 S = [p_clean, nuc_mask, conf_mask]。

    - img_01 : [B,3,H,W]，取值范围 [0,1]
    - 输出  : [B,3,H,W]，在 `device` 上（若 device=None 则保持在 img_01.device）
    """
    with torch.no_grad():
        dev = next(hovernet.parameters()).device
        out = hovernet(img_01.to(dev) * 255.0)
        p_clean  = torch.softmax(out['tp'], dim=1)[:, 1:2, :, :]  # [B,1,H,W]
        nuc_mask = torch.softmax(out['np'], dim=1)[:, 1:2, :, :]  # [B,1,H,W]
        conf_mask = ((p_clean >= tau_pos) | (p_clean <= tau_neg)).float()
        sem = torch.cat([p_clean, nuc_mask, conf_mask], dim=1)    # [B,3,H,W]
        if device is None:
            device = img_01.device
        return sem.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 单张图像推理
# ─────────────────────────────────────────────────────────────────────────────

def run_single_inference(
    img_path, unet, hovernet, scheduler,
    output_dir, device='cuda',
    num_iters=5, noise_t=100,
    max_l1_distortion=0.08,
    max_artifact_ratio=1.5,
    scale=2,
    use_semantic_injection: bool = False,
    tau_pos: float = 0.65,
    tau_neg: float = 0.35,
):
    """
    对单张图像进行 SR 推理。

    工作流程：
      1. 加载 HR（或将输入视为 HR 参考）
      2. 在线合成 LR（退化方式与训练阶段保持一致）
      3. 使用退火噪声进行多次 DDPM 迭代细化
      4. 依据 composite score 选择最佳结果；若违反保真度约束则提前停止
    """
    fname       = os.path.basename(img_path)
    name_no_ext = os.path.splitext(fname)[0]
    img_dir     = os.path.join(output_dir, name_no_ext)
    os.makedirs(img_dir, exist_ok=True)

    hr = load_image(img_path, device=device)
    if hr is None:
        print(f"⚠️  Cannot read {img_path}")
        return None

    # 合成 LR（单样本，使用固定随机种子以保证可复现）
    torch.manual_seed(0)
    lr = degrade(hr.squeeze(0).cpu(), scale=scale).unsqueeze(0).to(device)
    torch.manual_seed(torch.initial_seed())

    save_tensor(hr, os.path.join(img_dir, 'hr_reference.png'))
    save_tensor(lr, os.path.join(img_dir, 'lr_input.png'))

    # 双三次插值基线（上采样参考）
    bicubic_up = F.interpolate(
        F.interpolate(lr, scale_factor=1/scale, mode='bicubic', align_corners=False),
        size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False).clamp(0,1)
    baseline_psnr = compute_psnr(bicubic_up.cpu(), hr.cpu())
    baseline_ssim = compute_ssim(bicubic_up.cpu(), hr.cpu())
    print(f"\n{fname}  |  Bicubic baseline: PSNR={baseline_psnr:.2f}dB  SSIM={baseline_ssim:.4f}")

    # 语义先验（可选）：推理阶段没有真实 HR 时应使用输入/基线作为 proxy。
    # 本脚本把输入图当作参考 HR，因此这里直接用 hr 构造 p_clean。
    sem_tensor = None
    if use_semantic_injection and (hovernet is not None) and getattr(unet, "use_semantic", False):
        sem_tensor = build_semantic_tensor(
            hovernet, hr, tau_pos=tau_pos, tau_neg=tau_neg, device=device)

    current = lr.clone()
    best_tensor   = lr.clone()
    best_score    = -float('inf')
    best_iter     = 0
    stop_reason   = 'Max_Iters'
    score_history = []

    for i in range(num_iters):
        iter_t = max(20, int(noise_t * (0.75 ** i)))
        t_ten  = torch.tensor([iter_t], device=device).long()
        noise  = torch.randn_like(hr)
        x_t    = scheduler.add_noise(current, noise, t_ten)

        model_input = torch.cat([lr, x_t], dim=1)
        with torch.no_grad():
            noise_pred = unet(model_input, t_ten, semantic=sem_tensor).sample
        pred = predict_x0_from_noise_shared(x_t, noise_pred, t_ten, scheduler)

        # 相对于 HR 参考的评价指标
        psnr = compute_psnr(pred.cpu(), hr.cpu())
        ssim = compute_ssim(pred.cpu(), hr.cpu())
        l1   = F.l1_loss(pred, hr).item()
        art  = compute_artifact_penalty(pred.cpu(), hr.cpu())

        # 语义指标（可选）
        sem_mae = 0.0
        if hovernet is not None:
            p_c, cm = get_hovernet_maps(hovernet, hr)
            p_p, _  = get_hovernet_maps(hovernet, pred)
            # 与当前训练目标一致：希望 p_pred ≈ p_clean
            sem_mae = compute_masked_semantic_mae(p_p, p_c, cm)

        score = compute_composite_score(psnr, ssim, sem_mae, art)
        score_history.append(score)

        print(f"  Iter {i+1} | t={iter_t:3d} | "
              f"PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  "
              f"L1={l1:.4f}  Art={art:.3f}  Score={score:.4f}", end="")

        save_tensor(pred, os.path.join(img_dir,
            f"iter{i+1}_PSNR{psnr:.1f}_score{score:.3f}.png"))

        if score > best_score:
            best_score  = score
            best_tensor = pred.clone()
            best_iter   = i + 1
            print(" ★")
        else:
            print("")

        current = pred

        # 早停判断
        if l1 > max_l1_distortion:
            stop_reason = 'Artifact_L1'
            print(f"  [Stop] L1={l1:.4f} > {max_l1_distortion}")
            break
        if art > max_artifact_ratio:
            stop_reason = 'Artifact_TV'
            print(f"  [Stop] ArtifactRatio={art:.3f} > {max_artifact_ratio}")
            break

    save_tensor(best_tensor, os.path.join(img_dir,
        f"BEST_iter{best_iter}_score{best_score:.3f}.png"))
    print(f"  >> Best: iter={best_iter}  score={best_score:.4f}  "
          f"(baseline PSNR was {baseline_psnr:.2f}dB)")

    return dict(
        Filename=fname,
        Baseline_PSNR=baseline_psnr, Baseline_SSIM=baseline_ssim,
        Best_Score=best_score, Best_Iter=best_iter,
        Stop_Reason=stop_reason,
        Score_History=score_history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 批量推理数据集
# ─────────────────────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, file_paths, target_size=256, scale=2):
        self.file_paths  = file_paths
        self.target_size = target_size
        self.scale       = scale

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = cv2.imread(path)
            if img is None:
                return torch.zeros(3, self.target_size, self.target_size), \
                       torch.zeros(3, self.target_size, self.target_size), "ERROR"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LANCZOS4)
            hr  = torch.from_numpy(img).float() / 255.0
            hr  = hr.permute(2, 0, 1)
            torch.manual_seed(idx)   # 为每个样本的 LR 生成过程固定随机种子，保证可复现
            lr  = degrade(hr, scale=self.scale)
            return hr, lr, os.path.basename(path)
        except Exception:
            return (torch.zeros(3, self.target_size, self.target_size),
                    torch.zeros(3, self.target_size, self.target_size), "ERROR")


def run_batch_inference(dataloader, unet, hovernet, scheduler, args):
    csv_path   = os.path.join(args.output_dir, 'sr_results.csv')
    fieldnames = ['Filename', 'Baseline_PSNR', 'Baseline_SSIM',
                  'Final_PSNR', 'Final_SSIM', 'Final_L1',
                  'Final_Artifact', 'Final_Semantic_MAE',
                  'Final_Score', 'Stop_Reason', 'Total_Iters']
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    device = args.device

    for batch_idx, (hr_batch, lr_batch, filenames) in enumerate(
            tqdm(dataloader, desc='SR Batch')):

        valid = [i for i, f in enumerate(filenames) if f != 'ERROR']
        if not valid:
            continue
        hr_batch = hr_batch[valid].to(device)
        lr_batch = lr_batch[valid].to(device)
        filenames = [filenames[i] for i in valid]
        B = hr_batch.shape[0]

        # Save originals
        for i, fname in enumerate(filenames):
            d = os.path.join(args.output_dir, os.path.splitext(fname)[0])
            os.makedirs(d, exist_ok=True)
            save_tensor(hr_batch[i:i+1], os.path.join(d, 'hr_reference.png'))
            save_tensor(lr_batch[i:i+1], os.path.join(d, 'lr_input.png'))

        # Bicubic baseline
        bup = F.interpolate(
            F.interpolate(lr_batch, scale_factor=1/args.scale,
                          mode='bicubic', align_corners=False),
            size=(hr_batch.shape[2], hr_batch.shape[3]),
            mode='bicubic', align_corners=False).clamp(0,1)
        base_psnr = [compute_psnr(bup[i:i+1].cpu(), hr_batch[i:i+1].cpu()) for i in range(B)]
        base_ssim = [compute_ssim(bup[i:i+1].cpu(), hr_batch[i:i+1].cpu()) for i in range(B)]

        current     = lr_batch.clone()
        best_tensor = lr_batch.clone()
        best_scores = [-float('inf')] * B
        stop_reasons = ['Max_Iters'] * B
        total_iters  = 1
        active       = [True] * B

        # 可选：批量语义先验（来自 HR 参考图）
        sem_tensor = None
        if args.use_semantic_injection and (hovernet is not None) and getattr(unet, "use_semantic", False):
            sem_tensor = build_semantic_tensor(
                hovernet, hr_batch, tau_pos=args.tau_pos, tau_neg=args.tau_neg, device=device)

        for i in range(args.iters):
            if not any(active):
                break
            total_iters = i + 1

            iter_t = max(20, int(args.noise_t * (0.75 ** i)))
            t_ten  = torch.full((B,), iter_t, device=device).long()
            noise  = torch.randn_like(hr_batch)
            x_t    = scheduler.add_noise(current, noise, t_ten)

            model_input = torch.cat([lr_batch, x_t], dim=1)
            with torch.no_grad():
                noise_pred = unet(model_input, t_ten, semantic=sem_tensor).sample
            pred = predict_x0_from_noise_shared(x_t, noise_pred, t_ten, scheduler)

            for idx in range(B):
                if not active[idx]:
                    continue
                p  = pred[idx:idx+1].cpu()
                h  = hr_batch[idx:idx+1].cpu()
                psnr = compute_psnr(p, h)
                ssim = compute_ssim(p, h)
                l1   = F.l1_loss(p, h).item()
                art  = compute_artifact_penalty(p, h)

                sem = 0.0
                if hovernet is not None:
                    pc, cm = get_hovernet_maps(hovernet, hr_batch[idx:idx+1])
                    pp, _  = get_hovernet_maps(hovernet, pred[idx:idx+1])
                    sem    = compute_masked_semantic_mae(pp, pc, cm)

                score = compute_composite_score(psnr, ssim, sem, art)
                if score > best_scores[idx]:
                    best_scores[idx]     = score
                    best_tensor[idx]     = pred[idx]

                if l1 > args.max_l1_distortion:
                    active[idx] = False; stop_reasons[idx] = 'Artifact_L1'
                elif art > args.max_artifact_ratio:
                    active[idx] = False; stop_reasons[idx] = 'Artifact_TV'

            current = pred.clone()

        # Save best and write CSV
        rows = []
        for idx in range(B):
            fname = filenames[idx]
            d = os.path.join(args.output_dir, os.path.splitext(fname)[0])
            save_tensor(best_tensor[idx:idx+1], os.path.join(d, 'BEST_SR.png'))

            p = best_tensor[idx:idx+1].cpu()
            h = hr_batch[idx:idx+1].cpu()
            fp = compute_psnr(p, h);  fs = compute_ssim(p, h)
            fl = F.l1_loss(p, h).item()
            fa = compute_artifact_penalty(p, h)
            fsm = 0.0
            if hovernet:
                pc,cm = get_hovernet_maps(hovernet, hr_batch[idx:idx+1])
                pp,_  = get_hovernet_maps(hovernet, best_tensor[idx:idx+1])
                fsm   = compute_masked_semantic_mae(pp, pc, cm)
            fc = compute_composite_score(fp, fs, fsm, fa)

            rows.append(dict(
                Filename=fname,
                Baseline_PSNR=f"{base_psnr[idx]:.4f}",
                Baseline_SSIM=f"{base_ssim[idx]:.4f}",
                Final_PSNR=f"{fp:.4f}", Final_SSIM=f"{fs:.4f}",
                Final_L1=f"{fl:.6f}", Final_Artifact=f"{fa:.4f}",
                Final_Semantic_MAE=f"{fsm:.6f}",
                Final_Score=f"{fc:.4f}",
                Stop_Reason=stop_reasons[idx],
                Total_Iters=total_iters,
            ))

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

        avg_gain = sum(compute_psnr(best_tensor[i:i+1].cpu(), hr_batch[i:i+1].cpu()) - base_psnr[i]
                       for i in range(B)) / B
        print(f"  Batch {batch_idx+1} done | ΔPSNRavg={avg_gain:+.2f}dB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='SPM-UNet 语义注入 SR 推理')
    p.add_argument('--input_path',    required=True)
    p.add_argument('--output_dir',    default='./results/sr')
    p.add_argument('--unet_path',     required=True)
    p.add_argument('--hovernet_path', default=None)
    p.add_argument('--iters',         type=int,   default=5)
    p.add_argument('--noise_t',       type=int,   default=100)
    p.add_argument('--scale',         type=int,   default=2,
                   help='LR downscale factor (must match training)')
    p.add_argument('--max_l1_distortion', type=float, default=0.08)
    p.add_argument('--max_artifact_ratio',type=float, default=1.5)
    p.add_argument('--batch_size',    type=int,   default=None)
    p.add_argument('--device',        default=None)
    p.add_argument('--gpu_id',        type=int,   default=None)
    p.add_argument('--use_semantic_injection', action='store_true',
                   help='启用 SPM-UNet 架构层语义注入（需要提供 hovernet_path）')
    p.add_argument('--tau_pos', type=float, default=0.65,
                   help='conf_mask 正类高置信阈值（与训练保持一致）')
    p.add_argument('--tau_neg', type=float, default=0.35,
                   help='conf_mask 负类高置信阈值（与训练保持一致）')
    args = p.parse_args()

    print_gpu_info()
    args.device = (get_device(gpu_id=args.gpu_id) if args.gpu_id is not None
                   else args.device or get_device())
    print(f"Device: {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 只有在启用语义注入且 hovernet 可用时才创建带语义分支的模型
    unet = create_model(use_semantic=(args.use_semantic_injection and args.hovernet_path is not None)).to(args.device)
    ckpt = torch.load(args.unet_path, map_location=args.device)
    unet.load_state_dict(ckpt.get('model_state_dict', ckpt))
    unet.eval()
    print("✓ U-Net loaded")

    hovernet = None
    if args.hovernet_path:
        hovernet = load_hovernet(args.hovernet_path, device=args.device)
        print("✓ HoVer-Net loaded")
    if args.use_semantic_injection and hovernet is None:
        print("⚠️  已指定 --use_semantic_injection 但未加载 HoVer-Net，语义注入将被忽略。")
    if args.use_semantic_injection and hasattr(unet, "enable_semantic_modulation") and (hovernet is not None):
        # 推理阶段确保 hooks 启用
        unet.enable_semantic_modulation()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    exts  = ('.png','.jpg','.jpeg','.tif','.tiff','.bmp')
    files = ([os.path.join(args.input_path, f)
              for f in os.listdir(args.input_path) if f.lower().endswith(exts)]
             if os.path.isdir(args.input_path) else [args.input_path])
    print(f"\n{len(files)} image(s) to process")

    bs = args.batch_size or 1
    ds = InferenceDataset(files, scale=args.scale)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    run_batch_inference(dl, unet, hovernet, scheduler, args)

    print(f"\n✅ Done. Results: {args.output_dir}")


if __name__ == '__main__':
    main()
