"""
train.py
SPM-UNet（语义先验调制 U-Net）的三阶段训练循环。

v2 升级：GT mask 驱动的语义监督与注入
──────────────────────────────────────
语义先验张量 sem_tensor 来源于 Dataset 的 GT 像素级标注（masks.npy 转换而来）：

  sem_tensor = build_gt_sem_tensor(gt_label_map, gt_nuc_mask)
             = [gt_tp_onehot(6), gt_nuc_mask(1)]   共 7 通道
             其中 gt_tp_onehot 在 build_gt_sem_tensor 内部由 gt_label_map 用 F.one_hot 生成

  - gt_label_map : 像素级整数类别索引（0-5，HoVer-Net tp 类别空间）
  - gt_nuc_mask  : GT 细胞核二值掩膜

语义损失 clean 侧使用 GT，不再对 HR 跑 HoVer-Net（节省推理开销）；
pred 侧（x0_hat）仍用 HoVer-Net 实时预测，损失为 CE(pred_prob, gt_label_map)。

三阶段训练策略（不变）
──────────────────────
阶段 1（epoch < semantic_start_epoch）：
    仅骨干重建预训练；semantic=None；无 L_sem

阶段 2（semantic_start_epoch <= epoch < semantic_end_epoch）：
    L_sem（GT mask CE 驱动）；sem_tensor 架构注入

阶段 3（epoch >= semantic_end_epoch）：
    关闭语义损失与注入；纯像素收尾（L_noise + L_rec + L_grad + L_tv）
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os
import argparse
import csv
import datetime
from tqdm import tqdm

from ddpm_dataset import build_dataset
from unet_wrapper import create_model, count_parameters
from semantic_sr_loss import SemanticSRLoss, build_gt_sem_tensor
from ddpm_utils import load_hovernet, get_device, print_gpu_info, predict_x0_from_noise_shared
from logger import ExperimentLogger
from validation import (ValidationSet, create_val_dataloader,
                         save_validation_debug_images)
from metrics import (compute_psnr, compute_ssim,
                      compute_artifact_penalty,
                      compute_composite_score)
from ddpm_config import get_default_config


# ─────────────────────────────────────────────────────────────────────────────
# 语义损失预热调度器
# ─────────────────────────────────────────────────────────────────────────────

def semantic_weight_scale(epoch, start, warmup):
    """
    返回 [0, 1] 标量，在 start epoch 之后经过 warmup 个 epoch 线性升到 1。
    """
    if epoch < start:
        return 0.0
    return min((epoch - start) / max(warmup, 1), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────────────────────

def train(
    hovernet,
    dataset_type='pannuke',
    tum_dir=None, norm_dir=None,
    pannuke_root=None, pannuke_train_fold_dir=None, pannuke_val_fold_dir=None, pannuke_test_fold_dir=None,
    epochs=100, batch_size=8, lr=1e-4,
    device='cuda', save_dir='./checkpoints_sr',
    # 各项损失权重
    lambda_noise=1.0, lambda_rec=1.0, lambda_grad=0.1,
    lambda_sem=0.05,  lambda_tv=0.001,
    # 语义监督阈值
    tau_nuc=0.5,
    # 语义子项内部权重
    lambda_sem_cls=0.3,
    # 三阶段训练日程
    semantic_start_epoch=5, semantic_end_epoch=None, semantic_warmup_epochs=5,
    # 在线退化配置
    scale=2,
    blur_sigma_range=(0.5, 1.5),
    noise_std_range=(0.0, 0.02),
    stain_jitter=0.05,
    # 其他参数
    accumulation_steps=1, resume_path=None,
    logger=None, val_vis_dir=None,
    num_train_timesteps=1000, t_max=400,
    create_semantic_branch: bool = False,
    hovernet_upsample_factor: float = 1.0,
    # 优化器与 DataLoader
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    oversample: bool = True,
    train_drop_last: bool = True,
):
    if semantic_end_epoch is None:
        semantic_end_epoch = epochs

    base_lr = lr

    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # ── CSV 日志 ────────────────────────────────────────────────────
    log_path = os.path.join(save_dir, 'training_log.csv')
    csv_mode = 'a' if os.path.exists(log_path) else 'w'
    with open(log_path, csv_mode, newline='') as f:
        w = csv.writer(f)
        if csv_mode == 'w':
            w.writerow([
                'Epoch', 'Stage',
                'Total_Loss', 'L_noise', 'L_rec', 'L_grad',
                'L_sem', 'L_tv',
                'L_sem_cls',
                'Sem_MAE', 'Dir_Acc',
                'Sem_Scale',
            ])

    # ── 模型与调度器 ────────────────────────────────────────────────
    print("初始化 SPM-UNet...")
    unet      = create_model(use_semantic=(create_semantic_branch or (hovernet is not None))).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    param_info = count_parameters(unet)
    bb  = param_info.get('backbone', param_info['total'])
    sem = param_info.get('semantic', 0)
    print(f"  参数量：bone={bb:,}  semantic={sem:,}  total={param_info['total']:,}")

    # ── 损失函数 ────────────────────────────────────────────────────
    # GT mask 模式下 hovernet 仅用于 pred 侧（对 x0_hat 推理），clean 侧不再调用
    loss_fn = None
    if hovernet is not None:
        loss_fn = SemanticSRLoss(
            hovernet=hovernet,
            noise_scheduler=scheduler,
            tau_nuc=tau_nuc,
            lambda_sem_cls=lambda_sem_cls,
            lambda_noise=lambda_noise,
            lambda_rec=lambda_rec,
            lambda_grad=lambda_grad,
            lambda_sem=lambda_sem,
            lambda_tv=lambda_tv,
            t_max=t_max,
            hovernet_upsample_factor=hovernet_upsample_factor,
        ).to(device)
        print(
            "模式：GT mask 驱动语义监督（clean 侧使用 masks.npy GT，pred 侧 HoVer-Net 推理）\n"
            f"  Stage 1（epoch < {semantic_start_epoch}）：仅主干损失\n"
            f"  Stage 2（{semantic_start_epoch} ≤ epoch < {semantic_end_epoch}）："
            "GT sem_tensor 注入 + 语义损失\n"
            f"  Stage 3（epoch ≥ {semantic_end_epoch}）：纯像素收尾"
        )
    else:
        print("模式：仅重建 SR（hovernet=None；语义损失与注入均关闭）")

    # ── 优化器 ──────────────────────────────────────────────────────
    def _make_adamw(params):
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    def _transfer_adamw_state(old_opt, new_opt):
        old_state = old_opt.state
        for group in new_opt.param_groups:
            for p in group["params"]:
                if p in old_state:
                    new_opt.state[p] = old_state[p]

    def _backbone_params():
        return (list(unet.backbone_parameters())
                if hasattr(unet, "backbone_parameters")
                else list(unet.parameters()))

    # ── 断点续训 ────────────────────────────────────────────────────
    start_epoch = 0
    ckpt_opt_loaded = False
    ckpt_optimizer_state = None
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        unet.load_state_dict(ckpt['model_state_dict'])
        ckpt_optimizer_state = ckpt.get('optimizer_state_dict', None)
        start_epoch = ckpt.get('epoch', 0)

    start_in_stage2 = (
        (loss_fn is not None)
        and (start_epoch >= semantic_start_epoch)
        and (start_epoch < semantic_end_epoch)
    )
    start_in_stage3 = (loss_fn is not None) and (start_epoch >= semantic_end_epoch)

    optimizer_full = _make_adamw(unet.parameters())
    if ckpt_optimizer_state is not None:
        try:
            optimizer_full.load_state_dict(ckpt_optimizer_state)
            ckpt_opt_loaded = True
        except Exception as e:
            print(f"⚠️  优化器状态加载失败，将从头初始化：{e}")

    if start_in_stage2 or start_in_stage3:
        optimizer = optimizer_full
        optimizer_is_full = True
    else:
        optimizer = _make_adamw(_backbone_params())
        optimizer_is_full = False
        if ckpt_opt_loaded:
            _transfer_adamw_state(optimizer_full, optimizer)

    # ── 验证集（可视化）────────────────────────────────────────────
    val_set = None
    if val_vis_dir:
        val_set = ValidationSet(
            val_vis_dir, scheduler, device,
            scale=scale, fixed_timestep=100,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
        )
        if not val_set.load():
            val_set = None

    _pin = pin_memory and ('cuda' in str(device))
    best_composite = -float('inf')

    # ── 数据集 ─────────────────────────────────────────────────────
    print("正在加载数据集...")
    print(f"  dataset_type={dataset_type}")

    dataset_type = str(dataset_type).lower()
    val_dl = None

    if dataset_type == 'pannuke':
        if pannuke_train_fold_dir is None and pannuke_root is None:
            raise ValueError('dataset_type=pannuke 时，至少需要提供 pannuke_train_fold_dir 或 pannuke_root。')
        train_folds = [pannuke_train_fold_dir] if pannuke_train_fold_dir else None
        val_folds   = [pannuke_val_fold_dir]   if pannuke_val_fold_dir   else None

        dataset = build_dataset(
            dataset_type='pannuke',
            pannuke_root=None if train_folds else pannuke_root,
            pannuke_folds=train_folds,
            scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
            target_size=256,
        )
        if pannuke_val_fold_dir:
            val_ds = build_dataset(
                dataset_type='pannuke',
                pannuke_folds=val_folds,
                scale=scale,
                blur_sigma_range=blur_sigma_range,
                noise_std_range=noise_std_range,
                stain_jitter=stain_jitter,
                target_size=256,
            )
            val_dl = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=_pin,
                drop_last=False,
            )
            print(f"定量验证集（PanNuke）：{len(val_ds)} patch | fold={pannuke_val_fold_dir}")
        else:
            print('⚠️  未提供 pannuke_val_fold_dir —— 跳过定量验证。')
    else:
        dataset = build_dataset(
            dataset_type='nct',
            tum_dir=tum_dir,
            norm_dir=norm_dir,
            oversample=oversample,
            scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
            target_size=256,
        )
        val_dl = create_val_dataloader(
            val_vis_dir, batch_size, device, scale=scale,
            blur_sigma_range=blur_sigma_range,
            noise_std_range=noise_std_range,
            stain_jitter=stain_jitter,
            num_workers=num_workers,
            pin_memory=_pin,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=train_drop_last,
    )

    print(f"总训练轮数：{epochs} 个 epoch...")
    global_step = 0

    # Stage 1 或 Stage 3 续训：关闭语义注入 hooks
    if (
        loss_fn is not None
        and hasattr(unet, "disable_semantic_modulation")
        and semantic_start_epoch > 0
        and (not start_in_stage2)
    ):
        unet.disable_semantic_modulation()

    for epoch in range(start_epoch, epochs):
        unet.train()

        sem_scale   = semantic_weight_scale(epoch, semantic_start_epoch, semantic_warmup_epochs)
        in_stage2   = (epoch >= semantic_start_epoch) and (epoch < semantic_end_epoch) and (loss_fn is not None)
        in_stage3   = epoch >= semantic_end_epoch
        semantic_on = in_stage2

        stage_name     = '1' if epoch < semantic_start_epoch else ('2' if epoch < semantic_end_epoch else '3')
        sem_scale_log  = sem_scale if in_stage2 else 0.0
        stage_label    = f"Stage{stage_name} sem={sem_scale_log:.2f}"

        # Stage 2：开启语义调制 + 扩展优化器到全参数
        if in_stage2:
            if hasattr(unet, "enable_semantic_modulation") and (not getattr(unet, "use_semantic", True)):
                unet.enable_semantic_modulation()
            if not optimizer_is_full:
                new_opt = _make_adamw(unet.parameters())
                _transfer_adamw_state(optimizer, new_opt)
                optimizer = new_opt
                optimizer_is_full = True

        # Stage 3：关闭语义调制
        if in_stage3:
            if hasattr(unet, "disable_semantic_modulation") and getattr(unet, "use_semantic", False):
                unet.disable_semantic_modulation()

        # 每 epoch 累计器
        acc = {k: 0.0 for k in (
            'total', 'noise', 'rec', 'grad', 'sem', 'tv',
            'sem_cls',
            'sem_mae', 'dir_acc', 'sem_mae_n', 'dir_acc_n',
        )}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} | {stage_label}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            hr     = batch['hr'].to(device)
            lr_img = batch['lr'].to(device)
            bs     = hr.shape[0]

            noise     = torch.randn_like(hr)
            timesteps = torch.randint(0, t_max, (bs,), device=device).long()
            noisy_hr  = scheduler.add_noise(hr, noise, timesteps)

            # ── GT 字段（来自 PanNuke Dataset，NCT 时不存在）────────
            gt_nuc_mask  = batch.get('gt_nuc_mask')    # [B, H, W]    or None
            gt_label_map = batch.get('gt_label_map')   # [B, H, W]    or None

            # ── Stage 2：构造 GT sem_tensor 用于架构注入 ──────────
            # sem_tensor = [gt_tp_onehot(6), gt_nuc_mask(1)]，共 7 通道
            # gt_tp_onehot 由 build_gt_sem_tensor 内部用 F.one_hot 生成
            sem_tensor = None
            if semantic_on and (gt_label_map is not None) and (gt_nuc_mask is not None):
                sem_tensor = build_gt_sem_tensor(
                    gt_label_map, gt_nuc_mask, device=device,
                )   # [B, 7, H, W]

            # ── U-Net 前向 ─────────────────────────────────────────
            model_input = torch.cat([lr_img, noisy_hr], dim=1)   # [B, 6, H, W]
            noise_pred  = unet(
                model_input,
                timesteps,
                semantic=(sem_tensor if semantic_on else None),
            ).sample

            # ── 损失计算 ───────────────────────────────────────────
            if loss_fn is not None:
                if semantic_on and (gt_label_map is not None):
                    # Stage 2：GT mask 驱动的完整语义损失
                    total_loss, bd = loss_fn(
                        noise_pred   = noise_pred,
                        noise        = noise,
                        noisy_hr     = noisy_hr,
                        hr           = hr,
                        t            = timesteps,
                        gt_nuc_mask  = gt_nuc_mask.to(device),
                        gt_label_map = gt_label_map.to(device),
                        lambda_sem   = lambda_sem * sem_scale,
                        semantic_on  = True,
                    )
                else:
                    # Stage 1 / Stage 3：仅主干损失（semantic_on=False）
                    _dummy_nuc = (gt_nuc_mask.to(device)
                                  if gt_nuc_mask is not None
                                  else torch.zeros(bs, hr.shape[2], hr.shape[3], device=device))
                    _dummy_lbl = (gt_label_map.to(device)
                                  if gt_label_map is not None
                                  else torch.zeros(bs, hr.shape[2], hr.shape[3], device=device, dtype=torch.long))
                    total_loss, bd = loss_fn(
                        noise_pred   = noise_pred,
                        noise        = noise,
                        noisy_hr     = noisy_hr,
                        hr           = hr,
                        t            = timesteps,
                        gt_nuc_mask  = _dummy_nuc,
                        gt_label_map = _dummy_lbl,
                        lambda_sem   = 0.0,
                        semantic_on  = False,
                    )
            else:
                # 无 HoVer-Net 的纯重建回退分支
                x0_hat  = predict_x0_from_noise_shared(noisy_hr, noise_pred, timesteps, scheduler)
                l_noise = F.mse_loss(noise_pred, noise)
                l_rec   = F.l1_loss(x0_hat, hr)
                def _grad(x):
                    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
                    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
                    return dh, dw
                ph, pw = _grad(x0_hat)
                th, tw = _grad(hr)
                l_grad = (ph - th).abs().mean() + (pw - tw).abs().mean()
                total_loss = lambda_noise * l_noise + lambda_rec * l_rec + lambda_grad * l_grad
                bd = dict(
                    l_noise=l_noise.detach(), l_rec=l_rec.detach(),
                    l_grad=l_grad.detach(),
                    l_sem=torch.zeros(()), l_tv=torch.zeros(()),
                    l_sem_cls=torch.zeros(()),
                    sem_mae=torch.tensor(-1.0), dir_acc=torch.tensor(-1.0),
                )

            # ── 梯度累积 ───────────────────────────────────────────
            loss_val = total_loss.item()
            (total_loss / accumulation_steps).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # ── 损失累计 ───────────────────────────────────────────
            acc['total'] += loss_val
            for k in ('noise', 'rec', 'grad', 'sem', 'tv'):
                acc[k] += bd[f'l_{k}'].item()
            acc['sem_cls'] += bd['l_sem_cls'].item()
            sm = bd['sem_mae'].item()
            da = bd['dir_acc'].item()
            if sm >= 0:
                acc['sem_mae'] += sm;  acc['sem_mae_n'] += 1
            if da >= 0:
                acc['dir_acc'] += da;  acc['dir_acc_n'] += 1

            # ── TensorBoard 步级日志 ────────────────────────────────
            if logger and global_step % 10 == 0:
                logger.log_metrics({
                    'Total_Loss': loss_val,
                    'L_noise':    bd['l_noise'].item(),
                    'L_rec':      bd['l_rec'].item(),
                    'L_grad':     bd['l_grad'].item(),
                    'L_sem':      bd['l_sem'].item(),
                    'L_tv':       bd['l_tv'].item(),
                    'L_sem_cls':  bd['l_sem_cls'].item(),
                    'Sem_Scale':  sem_scale_log,
                }, step=global_step, prefix='Train')
                logger.flush()

            # ── 每 epoch 首批可视化 ────────────────────────────────
            if batch_idx == 0 and val_set and val_set.is_available():
                result = val_set.generate_reconstructions(
                    unet, loss_fn, use_semantic_injection=semantic_on)
                if result:
                    col_titles = None
                    try:
                        if hasattr(val_set, "col_titles") and val_set.col_titles:
                            col_titles = [str(x) for x in val_set.col_titles[:min(8, val_set.hr.shape[0])]]
                    except Exception:
                        col_titles = [f"Sample {i+1}" for i in range(min(8, val_set.hr.shape[0]))]

                    try:
                        v_psnr = compute_psnr(result['reconstructed'].cpu(), val_set.hr.cpu())
                        v_ssim = compute_ssim(result['reconstructed'].cpu(), val_set.hr.cpu())
                        v_l1   = F.l1_loss(result['reconstructed'], val_set.hr).item()
                        v_art  = compute_artifact_penalty(result['reconstructed'].cpu(), val_set.hr.cpu())
                        suptitle = (
                            f"Epoch {epoch+1} | {stage_label} | "
                            f"PSNR={v_psnr:.2f}dB  SSIM={v_ssim:.4f}  "
                            f"L1={v_l1:.4f}  Art={v_art:.3f}"
                        )
                    except Exception:
                        suptitle = f"Epoch {epoch+1} | {stage_label}"

                    grid = save_validation_debug_images(
                        hr=val_set.hr, lr=val_set.lr,
                        reconstructed=result['reconstructed'],
                        diff_vis=result['diff_vis'],
                        cls_clean=result['cls_clean'],
                        cls_pred=result['cls_pred'],
                        conf_pred=result['conf_pred'],
                        nuc_mask_clean=result['nuc_mask_clean'],
                        nuc_mask_pred=result['nuc_mask_pred'],
                        nr_types=result.get('nr_types', 6),
                        epoch=epoch + 1, save_dir=vis_dir,
                        num_vis=8, return_tensor=True,
                        col_titles=col_titles,
                        suptitle=suptitle,
                        cls_pred_lr=result.get('cls_pred_lr'),
                        conf_pred_lr=result.get('conf_pred_lr'),
                        nuc_mask_pred_lr=result.get('nuc_mask_pred_lr'),
                    )
                    if logger and grid is not None:
                        logger.log_images('Validation/SR_Comparison',
                                          grid, step=global_step, max_images=1)

            pbar.set_postfix({'L': f"{loss_val:.4f}", 'stage': stage_label[:8]})
            global_step += 1

        # ── 每 epoch 平均指标 ─────────────────────────────────────
        n = len(dataloader)
        avg = {k: acc[k] / n for k in ('total', 'noise', 'rec', 'grad', 'sem', 'tv')}
        avg_sem_cls = acc['sem_cls'] / n
        avg_sem_mae = acc['sem_mae'] / acc['sem_mae_n'] if acc['sem_mae_n'] else -1.0
        avg_dir_acc = acc['dir_acc'] / acc['dir_acc_n'] if acc['dir_acc_n'] else -1.0

        print(f"\nEpoch {epoch+1} | {stage_label}")
        print(f"  Total={avg['total']:.4f}  noise={avg['noise']:.4f}  "
              f"rec={avg['rec']:.4f}  grad={avg['grad']:.4f}")
        print(f"  sem={avg['sem']:.4f}  tv={avg['tv']:.4f}  sem_cls={avg_sem_cls:.4f}")
        if avg_sem_mae >= 0:
            print(f"  Sem_MAE={avg_sem_mae:.4f}  Dir_Acc={avg_dir_acc:.4f}")

        if logger:
            logger.log_metrics({
                **{f'L_{k}': v for k, v in avg.items()},
                'L_sem_cls':  avg_sem_cls,
                'Sem_MAE': avg_sem_mae, 'Dir_Acc': avg_dir_acc,
                'Sem_Scale': sem_scale_log,
            }, step=epoch + 1, prefix='Epoch')
            logger.flush()

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, stage_name,
                avg['total'], avg['noise'], avg['rec'], avg['grad'],
                avg['sem'], avg['tv'],
                avg_sem_cls,
                avg_sem_mae, avg_dir_acc, sem_scale_log,
            ])

        # ── 定量验证（PanNuke val fold）──────────────────────────
        if val_dl is not None:
            unet.eval()
            v_psnr = v_ssim = v_l1 = v_art = v_sem_mae = 0.0
            v_n = 0

            with torch.no_grad():
                for vb in val_dl:
                    vhr    = vb['hr'].to(device)
                    vlr    = vb['lr'].to(device)
                    vbs    = vhr.shape[0]

                    vnoise    = torch.randn_like(vhr)
                    vts       = torch.randint(0, t_max, (vbs,), device=device).long()
                    vnoisy_hr = scheduler.add_noise(vhr, vnoise, vts)
                    vinput    = torch.cat([vlr, vnoisy_hr], dim=1)
                    vnoise_p  = unet(vinput, vts).sample

                    vx0 = predict_x0_from_noise_shared(vnoisy_hr, vnoise_p, vts, scheduler)

                    v_psnr += compute_psnr(vx0, vhr)
                    v_ssim += compute_ssim(vx0, vhr)
                    v_l1   += F.l1_loss(vx0, vhr).item()
                    v_art  += compute_artifact_penalty(vx0, vhr)

                    # 语义 MAE：pred_prob vs GT one-hot（由 gt_label_map 按需生成）
                    if loss_fn is not None and semantic_on:
                        vgt_lbl = vb.get('gt_label_map')
                        vgt_nuc = vb.get('gt_nuc_mask')
                        if vgt_lbl is not None and vgt_nuc is not None:
                            vgt_lbl = vgt_lbl.to(device)
                            vgt_nuc = vgt_nuc.to(device)
                            valid_mask = (vgt_nuc > loss_fn.tau_nuc).float()
                            denom      = valid_mask.sum().clamp(min=1.0)

                            pred_out  = loss_fn._run_hovernet(vx0)
                            pred_prob = pred_out['tp_prob']   # [B, 6, H, W]
                            H, W      = vhr.shape[2], vhr.shape[3]
                            if pred_prob.shape[-2:] != (H, W):
                                pred_prob = F.interpolate(
                                    pred_prob, size=(H, W), mode='bilinear', align_corners=False
                                )
                            # GT one-hot 按需生成，不存储
                            vgt_onehot = F.one_hot(vgt_lbl.long(), num_classes=6) \
                                          .permute(0, 3, 1, 2).float()
                            mae_map   = (pred_prob - vgt_onehot).abs().mean(dim=1)
                            v_sem_mae += ((mae_map * valid_mask).sum() / denom).item()

                    v_n += 1

            if v_n > 0:
                vp  = v_psnr  / v_n
                vs  = v_ssim  / v_n
                vl  = v_l1    / v_n
                va  = v_art   / v_n
                vsm = v_sem_mae / v_n if semantic_on else -1.0

                comp = compute_composite_score(
                    psnr=vp, ssim=vs,
                    semantic_mae=vsm if vsm >= 0 else 0.0,
                    artifact_penalty=va,
                )

                print(f"  [Val] PSNR={vp:.2f}dB  SSIM={vs:.4f}  "
                      f"L1={vl:.4f}  ArtifactRatio={va:.3f}")
                print(f"  [Val] Semantic_MAE={vsm:.4f}  Composite={comp:.4f}")

                if logger:
                    logger.log_metrics(dict(
                        PSNR=vp, SSIM=vs, L1=vl,
                        Artifact_Penalty=va,
                        Semantic_MAE=vsm,
                        Composite_Score=comp,
                    ), step=epoch + 1, prefix='Val')

                if comp > best_composite:
                    best_composite = comp
                    best_path = os.path.join(save_dir, 'best_unet_sr.pth')
                    torch.save(dict(
                        epoch=epoch + 1,
                        model_state_dict=unet.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        val_psnr=vp, val_ssim=vs,
                        val_semantic_mae=vsm,
                        val_composite=comp,
                    ), best_path)
                    print(f"  🔥 New best model saved → {best_path}  "
                          f"(composite={comp:.4f})")

            unet.train()

        # 周期性保存 checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(save_dir, f'unet_sr_epoch_{epoch+1}.pth')
            torch.save(dict(
                epoch=epoch + 1,
                model_state_dict=unet.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                loss=avg['total'],
            ), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")
    if logger:
        logger.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _add_bool_mutex(parser, dest: str, default: bool, opt_name: str, help_on: str = None):
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        f'--{opt_name}', dest=dest, action='store_true',
        help=help_on or f'启用 {dest}',
    )
    g.add_argument(
        f'--no-{opt_name}', dest=dest, action='store_false',
        help=f'关闭 {dest}（覆盖默认 {default}）',
    )
    parser.set_defaults(**{dest: default})


def main():
    cfg = get_default_config()
    p = argparse.ArgumentParser(description='SPM-UNet 语义引导 SR DDPM 训练（GT mask 驱动）')
    p.add_argument('--dataset_type', type=str, default=getattr(cfg, 'dataset_type', 'pannuke'),
                   choices=['nct', 'pannuke'])
    p.add_argument('--tum_dir',  default=getattr(cfg, 'tum_dir', None))
    p.add_argument('--norm_dir', default=getattr(cfg, 'norm_dir', None))
    p.add_argument('--pannuke_root',            default=getattr(cfg, 'pannuke_root', None))
    p.add_argument('--pannuke_train_fold_dir',  default=getattr(cfg, 'pannuke_train_fold_dir', None))
    p.add_argument('--pannuke_val_fold_dir',    default=getattr(cfg, 'pannuke_val_fold_dir', None))
    p.add_argument('--pannuke_test_fold_dir',   default=getattr(cfg, 'pannuke_test_fold_dir', None))
    p.add_argument('--hovernet_path',           default=cfg.hovernet_path)
    p.add_argument('--hovernet_upsample_factor', type=float,
                   default=getattr(cfg, "hovernet_upsample_factor", 1.0),
                   help='对 x0_hat 跑 HoVer-Net 前的上采样倍率（pred 侧；PanNuke 256 通常为 1.0）')
    p.add_argument('--epochs',     type=int,   default=cfg.epochs)
    p.add_argument('--batch_size', type=int,   default=cfg.batch_size)
    p.add_argument('--lr',         type=float, default=cfg.lr)
    p.add_argument('--device',     type=str,   default=cfg.device)
    p.add_argument('--gpu_id',     type=int,   default=None)
    p.add_argument('--save_dir',               default=cfg.save_dir)
    # Loss weights
    p.add_argument('--lambda_noise', type=float, default=cfg.lambda_noise)
    p.add_argument('--lambda_rec',   type=float, default=cfg.lambda_rec)
    p.add_argument('--lambda_grad',  type=float, default=cfg.lambda_grad)
    p.add_argument('--lambda_sem',   type=float, default=cfg.lambda_sem)
    p.add_argument('--lambda_tv',    type=float, default=cfg.lambda_tv)
    # Semantic
    p.add_argument('--tau_nuc',            type=float, default=cfg.tau_nuc)
    p.add_argument('--lambda_sem_cls',     type=float, default=cfg.lambda_sem_cls)
    p.add_argument('--semantic_start_epoch',   type=int, default=cfg.semantic_start_epoch)
    p.add_argument('--semantic_end_epoch',     type=int, default=cfg.semantic_end_epoch)
    p.add_argument('--semantic_warmup_epochs', type=int, default=cfg.semantic_warmup_epochs)
    # Degradation
    p.add_argument('--scale', type=int, default=cfg.scale)
    p.add_argument('--blur_sigma_range',  type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=list(cfg.blur_sigma_range))
    p.add_argument('--noise_std_range',   type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=list(cfg.noise_std_range))
    p.add_argument('--stain_jitter',      type=float, default=cfg.stain_jitter)
    # Optimizer & DataLoader
    p.add_argument('--weight_decay',   type=float, default=cfg.weight_decay)
    p.add_argument('--max_grad_norm',  type=float, default=cfg.max_grad_norm)
    p.add_argument('--num_workers',    type=int,   default=cfg.num_workers)
    _add_bool_mutex(p, 'pin_memory',      cfg.pin_memory,      'pin-memory')
    _add_bool_mutex(p, 'oversample',      cfg.oversample,      'oversample')
    _add_bool_mutex(p, 'train_drop_last', cfg.train_drop_last, 'train-drop-last')
    # Misc
    p.add_argument('--accumulation_steps', type=int, default=cfg.accumulation_steps)
    p.add_argument('--resume_path',        default=None)
    p.add_argument('--no_tb',             action='store_true')
    p.add_argument('--log_dir',            default='./logs')
    p.add_argument('--exp_name',           default=None)
    p.add_argument('--val_vis_dir',        default=cfg.val_vis_dir)
    p.add_argument('--t_max',             type=int, default=cfg.t_max)
    p.add_argument('--create_semantic_branch', action='store_true')

    args = p.parse_args()
    print_gpu_info()

    if args.exp_name is None:
        args.exp_name = 'SR_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    logger = ExperimentLogger(
        use_tensorboard=not args.no_tb,
        log_dir=args.log_dir,
        experiment_name=args.exp_name,
    )

    device = (get_device(gpu_id=args.gpu_id) if args.gpu_id is not None
              else args.device or get_device())

    hovernet = None
    if args.hovernet_path:
        hovernet = load_hovernet(args.hovernet_path, device=device)

    train(
        hovernet=hovernet,
        dataset_type=args.dataset_type,
        tum_dir=args.tum_dir, norm_dir=args.norm_dir,
        pannuke_root=args.pannuke_root,
        pannuke_train_fold_dir=args.pannuke_train_fold_dir,
        pannuke_val_fold_dir=args.pannuke_val_fold_dir,
        pannuke_test_fold_dir=args.pannuke_test_fold_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, save_dir=args.save_dir,
        lambda_noise=args.lambda_noise, lambda_rec=args.lambda_rec,
        lambda_grad=args.lambda_grad,   lambda_sem=args.lambda_sem,
        lambda_tv=args.lambda_tv,
        tau_nuc=args.tau_nuc,
        lambda_sem_cls=args.lambda_sem_cls,
        semantic_start_epoch=args.semantic_start_epoch,
        semantic_end_epoch=args.semantic_end_epoch,
        semantic_warmup_epochs=args.semantic_warmup_epochs,
        scale=args.scale,
        blur_sigma_range=tuple(args.blur_sigma_range),
        noise_std_range=tuple(args.noise_std_range),
        stain_jitter=args.stain_jitter,
        hovernet_upsample_factor=args.hovernet_upsample_factor,
        accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        oversample=args.oversample,
        train_drop_last=args.train_drop_last,
        resume_path=args.resume_path,
        logger=logger, val_vis_dir=args.val_vis_dir,
        t_max=args.t_max,
        create_semantic_branch=args.create_semantic_branch,
    )


if __name__ == '__main__':
    main()