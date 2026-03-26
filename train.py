"""
train.py
SPM-UNet（语义先验调制 U-Net）的三阶段训练循环。

架构层语义注入（相较旧版训练代码的新增点）
------------------------------------------
SPMUNet 在扩散骨干 UNet2DModel 之上新增了 SemanticEncoder + SemanticModBlock，
把 `[tp_onehot(6), nuc_mask, tp_conf]` 注入到解码器高分辨率层（128×128）。
每个 batch 都会基于 HR 真值图通过 HoVer-Net 构造语义先验张量 `S`，并作为
`semantic=S` 传入模型前向（仅 Stage 2 启用）。

阶段 1（epoch < semantic_start_epoch）：
    - 骨干重建预训练；semantic=None；无 L_sem / L_dir

阶段 2（semantic_start_epoch <= epoch < semantic_end_epoch）：
    - L_sem + L_dir，warmup 升权；semantic=S 架构注入

阶段 3（epoch >= semantic_end_epoch）：
    - 再次关闭语义损失与注入，纯像素收尾（L_noise + L_rec + L_grad + L_tv）

若 train(..., semantic_end_epoch=None)，则 semantic_end_epoch 自动设为 epochs，退化为两阶段。
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
from semantic_sr_loss import SemanticSRLoss
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
    返回一个位于 [0, 1] 的标量，在 `start` epoch 之后，
    将 lambda_sem / lambda_dir 在线性预热的 `warmup` 个 epoch 内
    从 0 逐步提升到 1。
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
    lambda_sem=0.05,  lambda_dir=0.02, lambda_tv=0.001,
    # 多类别语义监督阈值
    tau_nuc=0.5, tau_conf=0.7,
    # 语义子项内部权重
    lambda_sem_dist=1.0, lambda_sem_cls=0.3, lambda_sem_conf=0.1,
    # 三阶段训练日程（semantic_end_epoch=None → 训练结束时仍为 Stage 2，即原两阶段）
    semantic_start_epoch=5, semantic_end_epoch=None, semantic_warmup_epochs=5,
    # 在线退化配置（与 ddpm_config / degradation.degrade 一致）
    scale=2,
    blur_sigma_range=(0.5, 1.5),
    noise_std_range=(0.0, 0.02),
    stain_jitter=0.05,
    # 其他参数
    accumulation_steps=1, resume_path=None,
    logger=None, val_vis_dir=None,
    num_train_timesteps=1000, t_max=400,
    create_semantic_branch: bool = False,
    hovernet_upsample_factor: float = 2.0,
    # 优化器与 DataLoader（与 ddpm_config 对齐）
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    oversample: bool = True,
    train_drop_last: bool = True,
):
    if semantic_end_epoch is None:
        semantic_end_epoch = epochs

    # 学习率超参；循环内低分辨率图用 lr_img，避免覆盖本参数导致 AdamW 把 Tensor 当 lr
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
                'L_sem', 'L_dir', 'L_tv',
                'L_sem_dist', 'L_sem_cls', 'L_sem_conf',
                'Sem_MAE', 'Dir_Acc',
                'Sem_Scale',
            ])

    # ── 模型与调度器 ────────────────────────────────────────────────
    print("初始化 SPM-UNet...")
    # 允许 Stage-1 也创建语义分支但不使用：
    # - create_semantic_branch=True：即使 hovernet=None 也创建 sem_encoder/mod_B
    # - Stage-1 会关闭 hooks 且传 semantic=None，因此语义分支不会参与前向/训练
    unet      = create_model(use_semantic=(create_semantic_branch or (hovernet is not None))).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    # 参数量拆分（便于确认语义分支是否足够轻量）
    param_info = count_parameters(unet)
    bb = param_info.get('backbone', param_info['total'])
    sem = param_info.get('semantic', 0)
    print(f"  参数量：bone={bb:,}  semantic={sem:,}  total={param_info['total']:,}")

    loss_fn = None
    if hovernet is not None:
        loss_fn = SemanticSRLoss(
            hovernet=hovernet,
            noise_scheduler=scheduler,
            tau_nuc=tau_nuc, tau_conf=tau_conf,
            lambda_sem_dist=lambda_sem_dist,
            lambda_sem_cls=lambda_sem_cls,
            lambda_sem_conf=lambda_sem_conf,
            lambda_noise=lambda_noise,
            lambda_rec=lambda_rec,
            lambda_grad=lambda_grad,
            lambda_sem=lambda_sem,
            lambda_dir=lambda_dir,
            lambda_tv=lambda_tv,
            t_max=t_max,
            hovernet_upsample_factor=hovernet_upsample_factor,
        ).to(device)
        # 注意：只要加载了 HoVer-Net 就会构造 SemanticSRLoss，但真正是否启用语义
        # 由 epoch 与 semantic_start_epoch / semantic_end_epoch 决定（semantic_on）。
        print(
            "模式：已加载 HoVer-Net（冻结）并构建 SemanticSRLoss；"
            f"Stage 1（epoch < {semantic_start_epoch}）：仅主干损失，关闭注入；"
            f"Stage 2（{semantic_start_epoch} ≤ epoch < {semantic_end_epoch}）："
            "语义损失 + 架构注入；"
            f"Stage 3（epoch ≥ {semantic_end_epoch}）：再次纯像素收尾。"
        )
    else:
        if create_semantic_branch:
            print("模式：仅重建 SR（未使用 HoVer-Net；已创建语义分支但阶段一不启用）")
        else:
            print("模式：仅重建 SR（不使用 HoVer-Net，语义分支关闭）")

    # 优化器策略：
    # - Stage 1：仅 backbone 参数
    # - Stage 2 / 3：全参数（Stage 3 仍 fine-tune 全量权重，仅关掉语义损失与注入）
    def _make_adamw(params):
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    def _transfer_adamw_state(old_opt, new_opt):
        """
        将 old_opt 中与 new_opt 共享的参数 state（如 Adam 的动量）迁移到 new_opt。
        依赖参数对象 identity（同一 Parameter 对象在新旧优化器中一致）。
        """
        old_state = old_opt.state
        for group in new_opt.param_groups:
            for p in group["params"]:
                if p in old_state:
                    new_opt.state[p] = old_state[p]

    def _backbone_params():
        return list(unet.backbone_parameters()) if hasattr(unet, "backbone_parameters") else list(unet.parameters())

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

    # 根据 start_epoch 决定从哪一阶段开始（续训不误判 Stage 3 为 Stage 2）
    start_in_stage2 = (
        (loss_fn is not None)
        and (start_epoch >= semantic_start_epoch)
        and (start_epoch < semantic_end_epoch)
    )
    start_in_stage3 = (loss_fn is not None) and (start_epoch >= semantic_end_epoch)

    # 先构建一个“全参数”优化器用于兼容加载 Stage 2/3 的 checkpoint state（若有）
    optimizer_full = _make_adamw(unet.parameters())
    if ckpt_optimizer_state is not None:
        try:
            optimizer_full.load_state_dict(ckpt_optimizer_state)
            ckpt_opt_loaded = True
        except Exception as e:
            print(f"⚠️  优化器状态加载失败，将从头初始化优化器 state：{e}")

    # Stage 1：只用 backbone；Stage 2 / 3：全参数
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
    print(
        f"  在线退化: scale={scale}, blur_sigma_range={blur_sigma_range}, "
        f"noise_std_range={noise_std_range}, stain_jitter={stain_jitter}"
    )

    dataset_type = str(dataset_type).lower()
    if dataset_type == 'pannuke':
        if pannuke_train_fold_dir is None and pannuke_root is None:
            raise ValueError('dataset_type=pannuke 时，至少需要提供 pannuke_train_fold_dir 或 pannuke_root。')
        train_folds = [pannuke_train_fold_dir] if pannuke_train_fold_dir else None
        val_folds = [pannuke_val_fold_dir] if pannuke_val_fold_dir else None

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
        val_dl = None
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
            print(f"定量验证集（PanNuke）：{len(val_ds)} 张 patch | fold={pannuke_val_fold_dir}")
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

    # Stage 1 或 Stage 3 续训：关闭语义注入 hooks；Stage 2 续训保持开启由循环内 in_stage2 处理
    if (
        loss_fn is not None
        and hasattr(unet, "disable_semantic_modulation")
        and semantic_start_epoch > 0
        and (not start_in_stage2)
    ):
        unet.disable_semantic_modulation()

    for epoch in range(start_epoch, epochs):
        unet.train()

        sem_scale = semantic_weight_scale(epoch, semantic_start_epoch,
                                          semantic_warmup_epochs)

        in_stage2 = (
            (epoch >= semantic_start_epoch)
            and (epoch < semantic_end_epoch)
            and (loss_fn is not None)
        )
        in_stage3 = epoch >= semantic_end_epoch
        semantic_on = in_stage2

        if epoch < semantic_start_epoch:
            stage_name = '1'
        elif epoch < semantic_end_epoch:
            stage_name = '2'
        else:
            stage_name = '3'

        sem_scale_log = sem_scale if in_stage2 else 0.0
        stage_label = f"Stage{stage_name} sem={sem_scale_log:.2f}"

        # Stage 2：开启语义调制（注册 hooks）并扩展优化器到“全参数”
        if in_stage2:
            if hasattr(unet, "enable_semantic_modulation") and (getattr(unet, "use_semantic", False) is False):
                unet.enable_semantic_modulation()
            if not optimizer_is_full:
                new_opt = _make_adamw(unet.parameters())
                _transfer_adamw_state(optimizer, new_opt)
                optimizer = new_opt
                optimizer_is_full = True

        # Stage 3：关闭语义调制（纯像素路径）
        if in_stage3:
            if hasattr(unet, "disable_semantic_modulation") and (getattr(unet, "use_semantic", False) is True):
                unet.disable_semantic_modulation()

        # 每个 epoch 的累计器
        acc = {k: 0.0 for k in
               ('total','noise','rec','grad','sem','dir','tv',
                'sem_dist','sem_cls','sem_conf',
                'sem_mae','dir_acc','sem_mae_n','dir_acc_n')}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} | {stage_label}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            hr = batch['hr'].to(device)
            lr_img = batch['lr'].to(device)
            bs = hr.shape[0]

            noise     = torch.randn_like(hr)
            timesteps = torch.randint(0, t_max, (bs,), device=device).long()
            noisy_hr  = scheduler.add_noise(hr, noise, timesteps)

            # ── 构造语义先验张量 S：
            # S = [tp_prob(6), nuc_mask(1), tp_conf(1)]（8 通道语义先验）
            # 重要：统一走 loss_fn._run_hovernet()，内部已包含“上采样→HoVer-Net→缩回原尺寸”的对齐逻辑。
            # 仅在 Stage-2（semantic_on=True）时需要（用于架构注入）。
            sem_tensor = None
            if semantic_on and loss_fn is not None and getattr(unet, "use_semantic", False):
                with torch.no_grad():
                    c = loss_fn._run_hovernet(hr)  # tp_prob/tp_conf/nuc_mask 已对齐回 hr 尺寸
                    sem_tensor = torch.cat(
                        [
                            c["tp_prob"],
                            c["nuc_mask"].unsqueeze(1),
                            c["tp_conf"].unsqueeze(1),
                        ],
                        dim=1,
                    ).to(device)  # [B,8,H,W]

            # U-Net：以 LR 为条件，对 noisy_hr 去噪（Stage-2 才启用语义注入）
            model_input = torch.cat([lr_img, noisy_hr], dim=1)   # [B, 6, H, W]
            noise_pred  = unet(
                model_input,
                timesteps,
                semantic=(sem_tensor if semantic_on else None),
            ).sample

            # ── 损失计算 ───────────────────────────────────────────
            if loss_fn is not None:
                total_loss, bd = loss_fn(
                    noise_pred=noise_pred,
                    noise=noise,
                    noisy_hr=noisy_hr,
                    hr=hr,
                    t=timesteps,
                    lambda_sem=lambda_sem * sem_scale,
                    lambda_dir=lambda_dir * sem_scale,
                    semantic_on=semantic_on,
                )
            else:
                # 仅重建的回退分支（不使用 HoVer-Net）
                x0_hat     = predict_x0_from_noise_shared(
                    noisy_hr, noise_pred, timesteps, scheduler)
                l_noise    = F.mse_loss(noise_pred, noise)
                l_rec      = F.l1_loss(x0_hat, hr)
                def _grad(x):
                    dh = x[:,:,1:,:] - x[:,:,:-1,:]
                    dw = x[:,:,:,1:] - x[:,:,:,:-1]
                    return dh, dw
                ph,pw = _grad(x0_hat); th,tw = _grad(hr)
                l_grad = (ph-th).abs().mean() + (pw-tw).abs().mean()
                total_loss = lambda_noise*l_noise + lambda_rec*l_rec + lambda_grad*l_grad
                bd = dict(l_noise=l_noise.detach(), l_rec=l_rec.detach(),
                          l_grad=l_grad.detach(),
                          l_sem=torch.zeros(()), l_dir=torch.zeros(()),
                          l_tv=torch.zeros(()),
                          l_sem_dist=torch.zeros(()), l_sem_cls=torch.zeros(()), l_sem_conf=torch.zeros(()),
                          sem_mae=torch.tensor(-1.0),
                          dir_acc=torch.tensor(-1.0),
                          p_hat_mean=torch.tensor(-1.0),
                          p_tgt_mean=torch.tensor(-1.0))

            # 梯度累积
            loss_val = total_loss.item()
            (total_loss / accumulation_steps).backward()
            if (batch_idx+1) % accumulation_steps == 0 or (batch_idx+1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # 损失累计
            acc['total'] += loss_val
            for k in ('noise','rec','grad','sem','dir','tv'):
                acc[k] += bd[f'l_{k}'].item()
            acc['sem_dist'] += bd['l_sem_dist'].item()
            acc['sem_cls']  += bd['l_sem_cls'].item()
            acc['sem_conf'] += bd['l_sem_conf'].item()
            sm = bd['sem_mae'].item(); da = bd['dir_acc'].item()
            if sm >= 0:
                acc['sem_mae'] += sm;  acc['sem_mae_n'] += 1
            if da >= 0:
                acc['dir_acc'] += da;  acc['dir_acc_n'] += 1

            # TensorBoard 步级日志
            if logger and global_step % 10 == 0:
                logger.log_metrics({
                    'Total_Loss': loss_val,
                    'L_noise': bd['l_noise'].item(),
                    'L_rec':   bd['l_rec'].item(),
                    'L_grad':  bd['l_grad'].item(),
                    'L_sem':   bd['l_sem'].item(),
                    'L_dir':   bd['l_dir'].item(),
                    'L_tv':    bd['l_tv'].item(),
                    'L_sem_dist': bd['l_sem_dist'].item(),
                    'L_sem_cls':  bd['l_sem_cls'].item(),
                    'L_sem_conf': bd['l_sem_conf'].item(),
                    'Sem_Scale': sem_scale_log,
                }, step=global_step, prefix='Train')
                logger.flush()

            # 每个 epoch 的第一批样本可视化
            if batch_idx == 0 and val_set and val_set.is_available():
                # Stage-2 才启用结构注入的验证可视化；Stage-1 默认走 plain 路径
                result = val_set.generate_reconstructions(
                    unet, loss_fn, use_semantic_injection=semantic_on)
                if result:
                    # 生成列标题（TUM-1 / NORM-1 / Sample-i）
                    col_titles = None
                    try:
                        if val_set.labels is not None:
                            t_i = n_i = 0
                            col_titles = []
                            for lbl in val_set.labels[:min(8, val_set.labels.shape[0])].detach().cpu().tolist():
                                if int(lbl) == 1:
                                    t_i += 1
                                    col_titles.append(f"TUM-{t_i}")
                                else:
                                    n_i += 1
                                    col_titles.append(f"NORM-{n_i}")
                    except Exception:
                        col_titles = [f"Sample {i+1}" for i in range(min(8, val_set.hr.shape[0]))]

                    # 计算并拼接总标题指标（整图顶部）
                    try:
                        v_psnr = compute_psnr(result['reconstructed'].cpu(), val_set.hr.cpu())
                        v_ssim = compute_ssim(result['reconstructed'].cpu(), val_set.hr.cpu())
                        v_l1   = F.l1_loss(result['reconstructed'], val_set.hr).item()
                        v_art  = compute_artifact_penalty(result['reconstructed'].cpu(), val_set.hr.cpu())
                        v_sem  = None
                        if loss_fn is not None and semantic_on:
                            c = loss_fn._run_hovernet(val_set.hr)
                            p = loss_fn._run_hovernet(result['reconstructed'])
                            cm = c['nuc_mask']
                            valid_mask = ((cm > loss_fn.tau_nuc) & (c['tp_conf'] > loss_fn.tau_conf)).float()
                            denom = valid_mask.sum().clamp(min=1.0)
                            mae_map = (p['tp_prob'] - c['tp_prob']).abs().mean(dim=1)
                            v_sem = ((mae_map * valid_mask).sum() / denom).item()

                        suptitle = (f"Epoch {epoch+1} | {stage_label} | "
                                    f"PSNR={v_psnr:.2f}dB  SSIM={v_ssim:.4f}  L1={v_l1:.4f}  Art={v_art:.3f}")
                        if v_sem is not None:
                            suptitle += f"  SemMAE={v_sem:.4f}"
                    except Exception:
                        suptitle = f"Epoch {epoch+1} | {stage_label}"

                    grid = save_validation_debug_images(
                        hr=val_set.hr, lr=val_set.lr,
                        reconstructed=result['reconstructed'],
                        diff_vis=result['diff_vis'],
                        cls_clean=result['cls_clean'],
                        cls_pred=result['cls_pred'],
                        conf_clean=result['conf_clean'],
                        conf_pred=result['conf_pred'],
                        nuc_mask_clean=result['nuc_mask_clean'],
                        nuc_mask_pred=result['nuc_mask_pred'],
                        nr_types=result.get('nr_types', 6),
                        epoch=epoch+1, save_dir=vis_dir,
                        num_vis=8, return_tensor=True,
                        col_titles=col_titles,
                        suptitle=suptitle,
                    )
                    if logger and grid is not None:
                        logger.log_images('Validation/SR_Comparison',
                                          grid, step=global_step, max_images=1)

            pbar.set_postfix({'L': f"{loss_val:.4f}", 'stage': stage_label[:8]})
            global_step += 1

        # ── 每个 epoch 的平均指标 ─────────────────────────────────
        n = len(dataloader)
        avg = {k: acc[k]/n for k in ('total','noise','rec','grad','sem','dir','tv')}
        avg_sem_dist = acc['sem_dist'] / n
        avg_sem_cls  = acc['sem_cls'] / n
        avg_sem_conf = acc['sem_conf'] / n
        avg_sem_mae = acc['sem_mae']/acc['sem_mae_n'] if acc['sem_mae_n'] else -1.0
        avg_dir_acc = acc['dir_acc']/acc['dir_acc_n'] if acc['dir_acc_n'] else -1.0

        print(f"\nEpoch {epoch+1} | {stage_label}")
        print(f"  Total={avg['total']:.4f}  noise={avg['noise']:.4f}  "
              f"rec={avg['rec']:.4f}  grad={avg['grad']:.4f}")
        print(f"  sem={avg['sem']:.4f}  dir={avg['dir']:.4f}  tv={avg['tv']:.4f}")
        print(f"  sem_dist={avg_sem_dist:.4f}  sem_cls={avg_sem_cls:.4f}  sem_conf={avg_sem_conf:.4f}")
        if avg_sem_mae >= 0:
            print(f"  Sem_MAE={avg_sem_mae:.4f}  Dir_Acc={avg_dir_acc:.4f}")

        if logger:
            logger.log_metrics({
                **{f'L_{k}': v for k,v in avg.items()},
                'L_sem_dist': avg_sem_dist,
                'L_sem_cls': avg_sem_cls,
                'L_sem_conf': avg_sem_conf,
                'Sem_MAE': avg_sem_mae, 'Dir_Acc': avg_dir_acc,
                'Sem_Scale': sem_scale_log,
            }, step=epoch+1, prefix='Epoch')
            logger.flush()

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1,
                stage_name,
                avg['total'], avg['noise'], avg['rec'], avg['grad'],
                avg['sem'], avg['dir'], avg['tv'],
                avg_sem_dist, avg_sem_cls, avg_sem_conf,
                avg_sem_mae, avg_dir_acc, sem_scale_log,
            ])

        # ── 定量验证 ───────────────────────────────────────────────
        if val_dl is not None:
            unet.eval()
            v_psnr = v_ssim = v_l1 = v_art = v_sem_mae = 0.0
            v_n = 0

            with torch.no_grad():
                for vb in val_dl:
                    vhr = vb['hr'].to(device)
                    vlr = vb['lr'].to(device)
                    bs  = vhr.shape[0]

                    vnoise    = torch.randn_like(vhr)
                    vts       = torch.randint(0, t_max, (bs,), device=device).long()
                    vnoisy_hr = scheduler.add_noise(vhr, vnoise, vts)
                    vinput    = torch.cat([vlr, vnoisy_hr], dim=1)
                    vnoise_p  = unet(vinput, vts).sample

                    vx0 = predict_x0_from_noise_shared(vnoisy_hr, vnoise_p, vts, scheduler)

                    v_psnr    += compute_psnr(vx0, vhr)
                    v_ssim    += compute_ssim(vx0, vhr)
                    v_l1      += F.l1_loss(vx0, vhr).item()
                    v_art     += compute_artifact_penalty(vx0, vhr)

                    # 语义 MAE（仅在 HoVer-Net 可用时计算）
                    if loss_fn is not None and semantic_on:
                        c = loss_fn._run_hovernet(vhr)
                        p = loss_fn._run_hovernet(vx0)
                        valid_mask = ((c['nuc_mask'] > loss_fn.tau_nuc) & (c['tp_conf'] > loss_fn.tau_conf)).float()
                        denom = valid_mask.sum().clamp(min=1.0)
                        mae_map = (p['tp_prob'] - c['tp_prob']).abs().mean(dim=1)
                        v_sem_mae += ((mae_map * valid_mask).sum() / denom).item()
                    v_n += 1

            if v_n > 0:
                vp  = v_psnr    / v_n
                vs  = v_ssim    / v_n
                vl  = v_l1      / v_n
                va  = v_art     / v_n
                vsm = v_sem_mae / v_n if semantic_on else -1.0

                comp = compute_composite_score(
                    psnr=vp, ssim=vs,
                    semantic_mae=vsm if vsm >= 0 else 0.0,
                    artifact_penalty=va)

                print(f"  [Val] PSNR={vp:.2f}dB  SSIM={vs:.4f}  "
                      f"L1={vl:.4f}  ArtifactRatio={va:.3f}")
                print(f"  [Val] Semantic_MAE={vsm:.4f}  Composite={comp:.4f}")

                if logger:
                    logger.log_metrics(dict(
                        PSNR=vp, SSIM=vs, L1=vl,
                        Artifact_Penalty=va,
                        Semantic_MAE=vsm,
                        Composite_Score=comp,
                    ), step=epoch+1, prefix='Val')

                # 最佳模型：依据最高 composite score 选择
                if comp > best_composite:
                    best_composite = comp
                    best_path = os.path.join(save_dir, 'best_unet_sr.pth')
                    torch.save(dict(
                        epoch=epoch+1,
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
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            ckpt_path = os.path.join(save_dir, f'unet_sr_epoch_{epoch+1}.pth')
            torch.save(dict(
                epoch=epoch+1,
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
    """
    生成 --<opt_name> / --no-<opt_name>，行为类似 Py3.9+ 的 BooleanOptionalAction，
    以便在 Python 3.8 及以下环境使用。
    """
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
    p = argparse.ArgumentParser(description='SPM-UNet 语义引导 SR DDPM 训练')
    p.add_argument('--dataset_type', type=str, default=getattr(cfg, 'dataset_type', 'pannuke'), choices=['nct', 'pannuke'])
    p.add_argument('--tum_dir',  default=getattr(cfg, 'tum_dir', None))
    p.add_argument('--norm_dir', default=getattr(cfg, 'norm_dir', None))
    p.add_argument('--pannuke_root', default=getattr(cfg, 'pannuke_root', None))
    p.add_argument('--pannuke_train_fold_dir', default=getattr(cfg, 'pannuke_train_fold_dir', None))
    p.add_argument('--pannuke_val_fold_dir', default=getattr(cfg, 'pannuke_val_fold_dir', None))
    p.add_argument('--pannuke_test_fold_dir', default=getattr(cfg, 'pannuke_test_fold_dir', None))
    p.add_argument('--hovernet_path', default=cfg.hovernet_path)
    p.add_argument('--hovernet_upsample_factor', type=float, default=getattr(cfg, "hovernet_upsample_factor", 2.0),
                   help='HoVer-Net 语义提取前上采样倍率（例如 20x→40x 用 2.0；1.0 关闭）')
    p.add_argument('--epochs',     type=int,   default=cfg.epochs)
    p.add_argument('--batch_size', type=int,   default=cfg.batch_size)
    p.add_argument('--lr',         type=float, default=cfg.lr)
    p.add_argument('--device',     type=str,   default=cfg.device)
    p.add_argument('--gpu_id',     type=int,   default=None)
    p.add_argument('--save_dir',   default=cfg.save_dir)
    # Loss weights
    p.add_argument('--lambda_noise', type=float, default=cfg.lambda_noise)
    p.add_argument('--lambda_rec',   type=float, default=cfg.lambda_rec)
    p.add_argument('--lambda_grad',  type=float, default=cfg.lambda_grad)
    p.add_argument('--lambda_sem',   type=float, default=cfg.lambda_sem)
    p.add_argument('--lambda_dir',   type=float, default=cfg.lambda_dir)
    p.add_argument('--lambda_tv',    type=float, default=cfg.lambda_tv)
    # Semantic
    p.add_argument('--tau_nuc',   type=float, default=cfg.tau_nuc)
    p.add_argument('--tau_conf',  type=float, default=cfg.tau_conf)
    p.add_argument('--lambda_sem_dist', type=float, default=cfg.lambda_sem_dist)
    p.add_argument('--lambda_sem_cls',  type=float, default=cfg.lambda_sem_cls)
    p.add_argument('--lambda_sem_conf', type=float, default=cfg.lambda_sem_conf)
    p.add_argument('--semantic_start_epoch',  type=int, default=cfg.semantic_start_epoch)
    p.add_argument('--semantic_end_epoch',    type=int, default=cfg.semantic_end_epoch)
    p.add_argument('--semantic_warmup_epochs',type=int, default=cfg.semantic_warmup_epochs)
    # Degradation（与 degradation.degrade / NCTDataset 对齐）
    p.add_argument('--scale', type=int, default=cfg.scale)
    p.add_argument('--blur_sigma_range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=list(cfg.blur_sigma_range),
                   help='高斯模糊 sigma 采样区间 [min, max]')
    p.add_argument('--noise_std_range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   default=list(cfg.noise_std_range),
                   help='加性高斯噪声标准差采样区间 [min, max]')
    p.add_argument('--stain_jitter', type=float, default=cfg.stain_jitter,
                   help='H&E 染色扰动强度（0 关闭）')
    # Optimizer & DataLoader
    p.add_argument('--weight_decay', type=float, default=cfg.weight_decay)
    p.add_argument('--max_grad_norm', type=float, default=cfg.max_grad_norm)
    p.add_argument('--num_workers', type=int, default=cfg.num_workers)
    _add_bool_mutex(
        p, 'pin_memory', cfg.pin_memory, 'pin-memory',
        help_on='DataLoader pin_memory（非 CUDA 时 train/val 内仍会关闭）',
    )
    _add_bool_mutex(
        p, 'oversample', cfg.oversample, 'oversample',
        help_on='训练集 NCTDataset 过采样平衡类别（PanNuke 下忽略）',
    )
    _add_bool_mutex(
        p, 'train_drop_last', cfg.train_drop_last, 'train-drop-last',
        help_on='训练 DataLoader 丢弃最后不完整 batch',
    )
    # Misc
    p.add_argument('--accumulation_steps', type=int, default=cfg.accumulation_steps)
    p.add_argument('--resume_path', default=None)
    p.add_argument('--no_tb',     action='store_true')
    p.add_argument('--log_dir',   default='./logs')
    p.add_argument('--exp_name',  default=None)
    p.add_argument('--val_vis_dir', default=cfg.val_vis_dir)
    p.add_argument('--t_max',     type=int, default=cfg.t_max)
    p.add_argument('--create_semantic_branch', action='store_true',
                   help='即使不传 hovernet_path，也创建 SPM-UNet 语义分支（Stage-1 默认不启用注入，用于与 Stage-2 架构对齐）')

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
        from ddpm_utils import load_hovernet
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
        lambda_grad=args.lambda_grad, lambda_sem=args.lambda_sem,
        lambda_dir=args.lambda_dir, lambda_tv=args.lambda_tv,
        tau_nuc=args.tau_nuc, tau_conf=args.tau_conf,
        lambda_sem_dist=args.lambda_sem_dist,
        lambda_sem_cls=args.lambda_sem_cls,
        lambda_sem_conf=args.lambda_sem_conf,
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
