"""
train.py
SPM-UNet（语义先验调制 U-Net）的双阶段训练循环。

架构层语义注入（相较旧版训练代码的新增点）
------------------------------------------
SPMUNet 在扩散骨干 UNet2DModel 之上新增了 SemanticEncoder + SemanticModBlock，
把 `[tp_prob(6), nuc_mask, tp_conf]` 注入到解码器高分辨率层（128×128）。
每个 batch 都会基于 HR 真值图通过 HoVer-Net 构造语义先验张量 `S`，并作为
`semantic=S` 传入模型前向（Stage-2 才启用）。

阶段 1（epoch < semantic_start_epoch）：
    - 骨干重建预训练（Backbone Reconstruction Pretraining）
    - 不启用语义调制（semantic=None）
    - 不使用 L_sem / L_dir
    - 损失：L_noise + L_rec + L_grad + L_tv
    - 目标：先学到稳定的保真 SR 重建主干

阶段 2（epoch >= semantic_start_epoch）：
    - 解冻全模型
    - 加入 L_sem + L_dir，并在 semantic_warmup_epochs 内线性升权
    - 传入 semantic=S → 启用结构注入（架构层语义调制）

这实现了：
  - 损失层语义引导（SemanticSRLoss）
  - 架构层语义调制（SPMUNet 解码器注入）
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

from ddpm_dataset import NCTDataset
from unet_wrapper import create_model, count_parameters
from semantic_sr_loss import SemanticSRLoss
from ddpm_utils import load_hovernet, get_device, print_gpu_info, predict_x0_from_noise_shared
from logger import ExperimentLogger
from validation import (ValidationSet, create_val_dataloader,
                         save_validation_debug_images)
from metrics import (compute_psnr, compute_ssim,
                      compute_artifact_penalty,
                      compute_composite_score)


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
    tum_dir, norm_dir, hovernet,
    epochs=100, batch_size=8, lr=1e-4,
    device='cuda', save_dir='./checkpoints_sr',
    # 各项损失权重
    lambda_noise=1.0, lambda_rec=1.0, lambda_grad=0.1,
    lambda_sem=0.05,  lambda_dir=0.02, lambda_tv=0.001,
    # 多类别语义监督阈值
    tau_nuc=0.5, tau_conf=0.7,
    # 语义子项内部权重
    lambda_sem_dist=1.0, lambda_sem_cls=0.3, lambda_sem_conf=0.1,
    # 双阶段训练日程
    semantic_start_epoch=5, semantic_warmup_epochs=5,
    # 在线退化配置
    scale=2,
    # 其他参数
    accumulation_steps=1, resume_path=None,
    logger=None, val_vis_dir=None,
    num_train_timesteps=1000, t_max=400,
    create_semantic_branch: bool = False,
):
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
        ).to(device)
        print("模式：SPM-UNet 语义引导 SR（架构注入 + 损失约束）")
    else:
        if create_semantic_branch:
            print("模式：仅重建 SR（未使用 HoVer-Net；已创建语义分支但阶段一不启用）")
        else:
            print("模式：仅重建 SR（不使用 HoVer-Net，语义分支关闭）")

    # 优化器策略（更严格两阶段）：
    # - Stage-1：优化器只包含 backbone 参数（更省 optimizer state 内存）
    # - Stage-2：重建优化器加入全部参数（backbone + 语义分支），并把 backbone 的 state 迁移过去
    wd = 0.01
    def _make_adamw(params):
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

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

    # 根据 start_epoch 决定从哪一阶段开始
    start_in_stage2 = (loss_fn is not None) and (start_epoch >= semantic_start_epoch)

    # 先构建一个“全参数”优化器用于兼容加载 Stage-2 的 checkpoint state（若有）
    optimizer_full = _make_adamw(unet.parameters())
    if ckpt_optimizer_state is not None:
        try:
            optimizer_full.load_state_dict(ckpt_optimizer_state)
            ckpt_opt_loaded = True
        except Exception as e:
            print(f"⚠️  优化器状态加载失败，将从头初始化优化器 state：{e}")

    # Stage-1 严格：只用 backbone 参数；Stage-2：用全参数
    if start_in_stage2:
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
        val_set = ValidationSet(val_vis_dir, scheduler, device,
                                 scale=scale, fixed_timestep=100)
        if not val_set.load():
            val_set = None

    val_dl = create_val_dataloader(val_vis_dir, batch_size, device, scale=scale)
    best_composite = -float('inf')

    # ── 数据集 ─────────────────────────────────────────────────────
    print("正在加载数据集...")
    dataset = NCTDataset(tum_dir, norm_dir, oversample=True, scale=scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4,
                            pin_memory=(device == 'cuda'),
                            drop_last=True)

    print(f"总训练轮数：{epochs} 个 epoch...")
    global_step = 0

    # Stage-1 更严格：完全关闭语义注入 hooks，避免任何 hook 开销
    # 到 Stage-2 再重新启用。
    if loss_fn is not None and hasattr(unet, "disable_semantic_modulation") and semantic_start_epoch > 0 and (not start_in_stage2):
        unet.disable_semantic_modulation()

    for epoch in range(start_epoch, epochs):
        unet.train()

        sem_scale   = semantic_weight_scale(epoch, semantic_start_epoch,
                                            semantic_warmup_epochs)
        semantic_on = (epoch >= semantic_start_epoch) and (loss_fn is not None)
        stage_label = f"Stage{'2' if semantic_on else '1'} sem={sem_scale:.2f}"

        # 进入 Stage-2：开启语义调制（注册 hooks）并扩展优化器到“全参数”
        if semantic_on:
            if hasattr(unet, "enable_semantic_modulation") and (getattr(unet, "use_semantic", False) is False):
                unet.enable_semantic_modulation()
            if not optimizer_is_full:
                new_opt = _make_adamw(unet.parameters())
                _transfer_adamw_state(optimizer, new_opt)
                optimizer = new_opt
                optimizer_is_full = True

        # 每个 epoch 的累计器
        acc = {k: 0.0 for k in
               ('total','noise','rec','grad','sem','dir','tv',
                'sem_dist','sem_cls','sem_conf',
                'sem_mae','dir_acc','sem_mae_n','dir_acc_n')}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} | {stage_label}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            hr = batch['hr'].to(device)
            lr = batch['lr'].to(device)
            bs = hr.shape[0]

            noise     = torch.randn_like(hr)
            timesteps = torch.randint(0, t_max, (bs,), device=device).long()
            noisy_hr  = scheduler.add_noise(hr, noise, timesteps)

            # ── 构造语义先验张量 S = [tp_prob(6), nuc_mask, tp_conf]
            # 仅在 Stage-2（semantic_on=True）时需要（用于架构注入）。
            sem_tensor = None
            if semantic_on and loss_fn is not None and getattr(unet, "use_semantic", False):
                with torch.no_grad():
                    hn_dev = next(loss_fn.hovernet.parameters()).device
                    hn_out = loss_fn.hovernet(hr.to(hn_dev) * 255.0)
                    tp_prob  = torch.softmax(hn_out['tp'], dim=1)                 # [B,6,H,W]
                    nuc_mask = torch.softmax(hn_out['np'], dim=1)[:, 1:2, :, :]   # [B,1,H,W]
                    tp_conf, _ = torch.max(tp_prob, dim=1, keepdim=True)          # [B,1,H,W]
                    sem_tensor = torch.cat([tp_prob, nuc_mask, tp_conf], dim=1).to(device)  # [B,8,H,W]

            # U-Net：以 LR 为条件，对 noisy_hr 去噪（Stage-2 才启用语义注入）
            model_input = torch.cat([lr, noisy_hr], dim=1)   # [B, 6, H, W]
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
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
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
                    'Sem_Scale': sem_scale,
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
                        nuc_mask=result['nuc_mask'],
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
                'Sem_Scale': sem_scale,
            }, step=epoch+1, prefix='Epoch')
            logger.flush()

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1,
                '2' if semantic_on else '1',
                avg['total'], avg['noise'], avg['rec'], avg['grad'],
                avg['sem'], avg['dir'], avg['tv'],
                avg_sem_dist, avg_sem_cls, avg_sem_conf,
                avg_sem_mae, avg_dir_acc, sem_scale,
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

def main():
    p = argparse.ArgumentParser(description='SPM-UNet 语义引导 SR DDPM 训练')
    p.add_argument('--tum_dir',  required=True)
    p.add_argument('--norm_dir', required=True)
    p.add_argument('--hovernet_path', default=None)
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--device',     type=str,   default=None)
    p.add_argument('--gpu_id',     type=int,   default=None)
    p.add_argument('--save_dir',   default='./checkpoints_sr')
    # Loss weights
    p.add_argument('--lambda_noise', type=float, default=1.0)
    p.add_argument('--lambda_rec',   type=float, default=1.0)
    p.add_argument('--lambda_grad',  type=float, default=0.1)
    p.add_argument('--lambda_sem',   type=float, default=0.05)
    p.add_argument('--lambda_dir',   type=float, default=0.02)
    p.add_argument('--lambda_tv',    type=float, default=0.001)
    # Semantic
    p.add_argument('--tau_nuc',   type=float, default=0.5)
    p.add_argument('--tau_conf',  type=float, default=0.7)
    p.add_argument('--lambda_sem_dist', type=float, default=1.0)
    p.add_argument('--lambda_sem_cls',  type=float, default=0.3)
    p.add_argument('--lambda_sem_conf', type=float, default=0.1)
    p.add_argument('--semantic_start_epoch',  type=int, default=5)
    p.add_argument('--semantic_warmup_epochs',type=int, default=5)
    # Degradation
    p.add_argument('--scale', type=int, default=2)
    # Misc
    p.add_argument('--accumulation_steps', type=int, default=1)
    p.add_argument('--resume_path', default=None)
    p.add_argument('--no_tb',     action='store_true')
    p.add_argument('--log_dir',   default='./logs')
    p.add_argument('--exp_name',  default=None)
    p.add_argument('--val_vis_dir', default=None)
    p.add_argument('--t_max',     type=int, default=400)
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
        tum_dir=args.tum_dir, norm_dir=args.norm_dir,
        hovernet=hovernet,
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
        semantic_warmup_epochs=args.semantic_warmup_epochs,
        scale=args.scale,
        accumulation_steps=args.accumulation_steps,
        resume_path=args.resume_path,
        logger=logger, val_vis_dir=args.val_vis_dir,
        t_max=args.t_max,
        create_semantic_branch=args.create_semantic_branch,
    )


if __name__ == '__main__':
    main()
