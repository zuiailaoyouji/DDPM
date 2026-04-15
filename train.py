"""
train.py
SPM-UNet 训练循环（判别器标签纠正版，单阶段）。

【训练模式】
─────────────────────────────────────────────────────────────────────────────
完整模式（默认）：
  - CellViT(HR) 软标签 sem_tensor 架构注入
  - 交集 Focal-CE 语义损失
  - 保存为 best_unet_correction.pth

消融模式（--no_semantic）：
  - 关闭 sem_tensor 注入（use_semantic=False）
  - 关闭语义损失，只保留 L_noise + L_rec + L_grad + L_tv
  - 不调用 CellViT（节省显存和时间）
  - 保存为 best_unet_ablation.pth

两种模式超参数完全一致，只有语义监督开关不同。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os
import argparse
import csv
import datetime
from typing import Optional
from tqdm import tqdm

from ddpm_dataset import (build_dataset, PanNukeDataset,
                          split_train_val, oversample_minority_classes)
from unet_wrapper import create_model, count_parameters
from semantic_sr_loss import (SemanticSRLoss, build_sem_tensor_from_cellvit,
                               run_cellvit)
from ddpm_utils import (load_cellvit, get_device, print_gpu_info,
                        predict_x0_from_noise_shared)
from logger import ExperimentLogger
from metrics import (compute_psnr, compute_ssim,
                     compute_artifact_penalty, compute_composite_score)
from ddpm_config import get_default_config

_DEFAULT_CFG = get_default_config()


# ─────────────────────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cellvit,
    use_semantic           = True,    # False → 消融模式
    dataset_type           = _DEFAULT_CFG.dataset_type,
    tum_dir                = _DEFAULT_CFG.tum_dir,
    norm_dir               = _DEFAULT_CFG.norm_dir,
    pannuke_root           = _DEFAULT_CFG.pannuke_root,
    pannuke_fold_dirs      = None,    # 训练折列表，如 [Fold1, Fold2]
    pannuke_test_fold_dir  = _DEFAULT_CFG.pannuke_test_fold_dir,
    n_val                  = 100,     # 分层抽样验证集数量
    val_seed               = 42,      # 验证集抽样随机种子
    epochs                 = _DEFAULT_CFG.epochs,
    batch_size             = _DEFAULT_CFG.batch_size,
    lr                     = _DEFAULT_CFG.lr,
    device                 = _DEFAULT_CFG.device,
    save_dir               = _DEFAULT_CFG.save_dir,
    lambda_noise           = _DEFAULT_CFG.lambda_noise,
    lambda_rec             = _DEFAULT_CFG.lambda_rec,
    lambda_grad            = _DEFAULT_CFG.lambda_grad,
    lambda_sem             = _DEFAULT_CFG.lambda_sem,
    lambda_tv              = _DEFAULT_CFG.lambda_tv,
    lambda_sem_cls         = _DEFAULT_CFG.lambda_sem_cls,
    correction_boost       = _DEFAULT_CFG.correction_boost,
    focal_gamma            = _DEFAULT_CFG.focal_gamma,
    accumulation_steps     = _DEFAULT_CFG.accumulation_steps,
    resume_path            = None,
    logger                 = None,
    num_train_timesteps    = _DEFAULT_CFG.num_train_timesteps,
    t_max                  = _DEFAULT_CFG.t_max,
    target_size: Optional[int] = None,
    weight_decay: float    = _DEFAULT_CFG.weight_decay,
    max_grad_norm: float   = _DEFAULT_CFG.max_grad_norm,
    num_workers: int       = _DEFAULT_CFG.num_workers,
    pin_memory: bool       = _DEFAULT_CFG.pin_memory,
    oversample: bool       = _DEFAULT_CFG.oversample,
    train_drop_last: bool  = _DEFAULT_CFG.train_drop_last,
    oversample_minority:    bool  = False,  # 是否对少数类过采样
    oversample_ratio:       float = 3.0,    # 过采样倍率
    oversample_classes:     list  = None,   # 默认 [2] 即 Inflammatory
    oversample_min_pixels:  int   = 500,    # 目标类别像素数阈值
):
    if target_size is None:
        target_size = _DEFAULT_CFG.target_size or 256

    # 消融模式下训练不需要 CellViT，但验证时仍需要评估 Dir_Acc
    # 用 _cellvit_val 保存一个只用于验证评估的 CellViT 实例
    _cellvit_val = cellvit   # 完整模式：直接复用
    if not use_semantic:
        # 消融模式：训练不用 CellViT，但验证仍需要
        # 从 cellvit_path 加载一个专用于验证的实例
        cfg_cellvit_path = _DEFAULT_CFG.cellvit_path
        cfg_cellvit_repo = _DEFAULT_CFG.cellvit_repo
        if cfg_cellvit_path and os.path.exists(cfg_cellvit_path):
            print("消融模式：加载 CellViT 用于验证评估...")
            _cellvit_val = load_cellvit(
                model_path        = cfg_cellvit_path,
                cellvit_repo_path = cfg_cellvit_repo,
                device            = device,
            )
        else:
            print("⚠️  消融模式：未找到 CellViT 权重，验证时跳过 Dir_Acc 计算")
            _cellvit_val = None
        cellvit = None   # 训练时不使用

    os.makedirs(save_dir, exist_ok=True)

    # 根据模式决定保存文件名前缀
    mode_name = 'correction' if use_semantic else 'ablation'
    best_ckpt_name = f'best_unet_{mode_name}.pth'

    # ── CSV 日志 ─────────────────────────────────────────────────────
    log_path = os.path.join(save_dir, f'training_log_{mode_name}.csv')
    csv_mode = 'a' if os.path.exists(log_path) else 'w'
    with open(log_path, csv_mode, newline='') as f:
        w = csv.writer(f)
        if csv_mode == 'w':
            if use_semantic:
                w.writerow([
                    'Epoch',
                    'Total_Loss', 'L_noise', 'L_rec', 'L_grad',
                    'L_sem', 'L_tv', 'L_sem_cls',
                    'Sem_MAE', 'Dir_Acc',
                    'Intersect_Ratio', 'Correction_Ratio',
                ])
            else:
                w.writerow([
                    'Epoch',
                    'Total_Loss', 'L_noise', 'L_rec', 'L_grad', 'L_tv',
                ])

    # ── 模型与调度器 ─────────────────────────────────────────────────
    mode_str = '完整模式（语义监督）' if use_semantic else '消融模式（无语义监督）'
    print(f"初始化 SPM-UNet [{mode_str}]...")

    # 消融模式关闭 SemanticEncoder 和 ModBlock
    unet      = create_model(use_semantic=use_semantic).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    param_info = count_parameters(unet)
    print(f"  参数量：backbone={param_info.get('backbone', param_info['total']):,}  "
          f"semantic={param_info.get('semantic', 0):,}  "
          f"total={param_info['total']:,}")

    # ── 损失函数 ─────────────────────────────────────────────────────
    if use_semantic:
        if cellvit is None:
            raise ValueError("完整模式下 cellvit 不能为 None。")
        loss_fn = SemanticSRLoss(
            cellvit          = cellvit,
            noise_scheduler  = scheduler,
            lambda_sem_cls   = lambda_sem_cls,
            correction_boost = correction_boost,
            focal_gamma      = focal_gamma,
            lambda_noise     = lambda_noise,
            lambda_rec       = lambda_rec,
            lambda_grad      = lambda_grad,
            lambda_sem       = lambda_sem,
            lambda_tv        = lambda_tv,
            t_max            = t_max,
        ).to(device)
        print(f"  语义损失：sem_cls={lambda_sem_cls}  "
              f"correction_boost={correction_boost}  focal_gamma={focal_gamma}")
    else:
        # 消融模式：只保留重建损失，不实例化 SemanticSRLoss
        loss_fn = None
        print(f"  消融模式：只使用 L_noise + L_rec + L_grad + L_tv")

    print(f"  损失权重：noise={lambda_noise}  rec={lambda_rec}  "
          f"grad={lambda_grad}  tv={lambda_tv}")

    # ── 优化器 ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ── 断点续训 ─────────────────────────────────────────────────────
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        unet.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️  优化器状态加载失败：{e}")

    _pin = pin_memory and ('cuda' in str(device))
    best_metric = -float('inf')   # 完整模式用 dir_acc，消融模式用 -val_l1

    # ── 数据集 ───────────────────────────────────────────────────────
    print("正在加载数据集...")
    dataset_type = str(dataset_type).lower()
    val_dl = None

    if dataset_type == 'pannuke':
        if not pannuke_fold_dirs and pannuke_root is None:
            raise ValueError('请提供 --pannuke_fold_dirs 或 --pannuke_root。')

        # 加载全部训练折（如 Fold1 + Fold2）
        full_dataset = PanNukeDataset(
            fold_dirs   = pannuke_fold_dirs,
            root_dir    = pannuke_root if not pannuke_fold_dirs else None,
            target_size = target_size,
            verbose     = True,
        )

        # 分层抽样：从合并数据中抽出验证集，其余作为训练集
        train_ds, val_ds = split_train_val(
            dataset = full_dataset,
            n_val   = n_val,
            seed    = val_seed,
            verbose = True,
        )

        val_dl = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=_pin, drop_last=False,
        )
        print(f"训练集：{len(train_ds)} 张  验证集：{len(val_ds)} 张  "
              f"（分层抽样，seed={val_seed}）")

        # 少数类过采样（可选）
        if oversample_minority:
            _cls = oversample_classes if oversample_classes else [2, 3]
            train_ds = oversample_minority_classes(
                train_subset    = train_ds,
                dataset         = full_dataset,
                target_classes  = _cls,
                oversample_ratio= oversample_ratio,
                min_pixels      = oversample_min_pixels,
                verbose         = True,
            )

        dataset = train_ds   # 后续 dataloader 用 train_ds

    else:
        dataset = build_dataset(
            dataset_type = 'nct',
            tum_dir=tum_dir, norm_dir=norm_dir,
            oversample=oversample, target_size=target_size,
        )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=_pin, drop_last=train_drop_last,
    )
    print(f"训练集共 {len(dataset)} 张，共 {epochs} 个 epoch")

    # ── 训练循环 ─────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(start_epoch, epochs):
        unet.train()

        acc = {k: 0.0 for k in (
            'total', 'noise', 'rec', 'grad', 'tv',
            'sem', 'sem_cls', 'sem_mae', 'dir_acc',
            'intersect_ratio', 'correction_ratio', 'n_sem',
        )}

        pbar = tqdm(dataloader,
                    desc=f"Epoch {epoch+1}/{epochs} [{mode_name}]")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            hr  = batch['hr'].to(device)
            bs  = hr.shape[0]

            noise     = torch.randn_like(hr)
            timesteps = torch.randint(0, t_max, (bs,), device=device).long()
            noisy_hr  = scheduler.add_noise(hr, noise, timesteps)

            if use_semantic:
                # ── 完整模式：CellViT(HR) 软标签 sem_tensor ──────────
                gt_nuc_mask  = batch.get('gt_nuc_mask')
                gt_label_map = batch.get('gt_label_map')

                with torch.no_grad():
                    hr_cv = run_cellvit(cellvit, hr)
                sem_tensor = build_sem_tensor_from_cellvit(
                    hr_cv['nuclei_type_prob'],
                    hr_cv['nuclei_nuc_prob'],
                )

                model_input = torch.cat([hr, noisy_hr], dim=1)
                noise_pred  = unet(model_input, timesteps,
                                   semantic=sem_tensor).sample

                _nuc = gt_nuc_mask.to(device) if gt_nuc_mask is not None else \
                       torch.zeros(bs, hr.shape[2], hr.shape[3], device=device)
                _lbl = gt_label_map.to(device) if gt_label_map is not None else \
                       torch.zeros(bs, hr.shape[2], hr.shape[3],
                                   device=device, dtype=torch.long)

                total_loss, bd = loss_fn(
                    noise_pred   = noise_pred,
                    noise        = noise,
                    noisy_hr     = noisy_hr,
                    hr           = hr,
                    t            = timesteps,
                    gt_nuc_mask  = _nuc,
                    gt_label_map = _lbl,
                    semantic_on  = True,
                )

            else:
                # ── 消融模式：无 sem_tensor，无语义损失 ───────────────
                model_input = torch.cat([hr, noisy_hr], dim=1)
                noise_pred  = unet(model_input, timesteps,
                                   semantic=None).sample

                # 手动计算重建损失
                x0_hat = _predict_x0(noisy_hr, noise_pred, timesteps, scheduler)
                l_noise = F.mse_loss(noise_pred, noise)
                l_rec   = F.l1_loss(x0_hat, hr)
                l_grad  = _gradient_loss(x0_hat, hr)
                l_tv    = _tv_loss(x0_hat, hr)

                total_loss = (lambda_noise * l_noise
                              + lambda_rec  * l_rec
                              + lambda_grad * l_grad
                              + lambda_tv   * l_tv)

                bd = dict(
                    l_noise   = l_noise.detach(),
                    l_rec     = l_rec.detach(),
                    l_grad    = l_grad.detach(),
                    l_tv      = l_tv.detach(),
                    l_sem     = torch.tensor(0.0),
                    l_sem_cls = torch.tensor(0.0),
                    sem_mae   = torch.tensor(-1.0),
                    dir_acc   = torch.tensor(-1.0),
                    intersect_ratio  = torch.tensor(-1.0),
                    correction_ratio = torch.tensor(-1.0),
                )

            # ── 梯度累积 ─────────────────────────────────────────────
            (total_loss / accumulation_steps).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # ── 累计 ─────────────────────────────────────────────────
            acc['total'] += total_loss.item()
            acc['noise'] += bd['l_noise'].item()
            acc['rec']   += bd['l_rec'].item()
            acc['grad']  += bd['l_grad'].item()
            acc['tv']    += bd['l_tv'].item()
            acc['sem']   += bd['l_sem'].item()
            acc['sem_cls'] += bd['l_sem_cls'].item()

            if bd['dir_acc'].item() >= 0:
                acc['sem_mae']          += bd['sem_mae'].item()
                acc['dir_acc']          += bd['dir_acc'].item()
                acc['intersect_ratio']  += bd['intersect_ratio'].item()
                acc['correction_ratio'] += bd['correction_ratio'].item()
                acc['n_sem']            += 1

            pbar.set_postfix(
                loss = f"{total_loss.item():.4f}",
                rec  = f"{bd['l_rec'].item():.4f}",
                sem  = f"{bd['l_sem'].item():.4f}" if use_semantic else "—",
                acc  = f"{bd['dir_acc'].item():.3f}"
                       if bd['dir_acc'].item() >= 0 else "—",
            )

            if logger:
                log_d = dict(Total=total_loss.item(),
                             L_noise=bd['l_noise'].item(),
                             L_rec=bd['l_rec'].item())
                if use_semantic:
                    log_d['L_sem']   = bd['l_sem'].item()
                    log_d['Dir_Acc'] = bd['dir_acc'].item()
                logger.log_metrics(log_d, step=global_step,
                                   prefix=f'Train_{mode_name}')

        # ── Epoch 汇总 ───────────────────────────────────────────────
        nb  = len(dataloader)
        ns  = max(acc['n_sem'], 1)
        avg = {k: acc[k] / nb for k in
               ('total', 'noise', 'rec', 'grad', 'tv', 'sem', 'sem_cls')}
        avg.update({k: acc[k] / ns for k in
                    ('sem_mae', 'dir_acc', 'intersect_ratio', 'correction_ratio')})

        if use_semantic:
            print(
                f"\nEpoch {epoch+1}/{epochs} [{mode_name}]\n"
                f"  Loss={avg['total']:.4f}  noise={avg['noise']:.4f}  "
                f"rec={avg['rec']:.4f}  grad={avg['grad']:.4f}\n"
                f"  sem={avg['sem']:.4f}  tv={avg['tv']:.4f}\n"
                f"  Dir_Acc={avg['dir_acc']:.4f}  Sem_MAE={avg['sem_mae']:.4f}  "
                f"intersect={avg['intersect_ratio']:.3f}  "
                f"correction={avg['correction_ratio']:.3f}"
            )
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch+1,
                    avg['total'], avg['noise'], avg['rec'], avg['grad'],
                    avg['sem'], avg['tv'], avg['sem_cls'],
                    avg['sem_mae'], avg['dir_acc'],
                    avg['intersect_ratio'], avg['correction_ratio'],
                ])
        else:
            print(
                f"\nEpoch {epoch+1}/{epochs} [{mode_name}]\n"
                f"  Loss={avg['total']:.4f}  noise={avg['noise']:.4f}  "
                f"rec={avg['rec']:.4f}  grad={avg['grad']:.4f}  "
                f"tv={avg['tv']:.4f}"
            )
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch+1,
                    avg['total'], avg['noise'], avg['rec'],
                    avg['grad'], avg['tv'],
                ])

        if logger:
            log_d = dict(Total=avg['total'], L_rec=avg['rec'])
            if use_semantic:
                log_d.update(dict(
                    L_sem=avg['sem'], Dir_Acc=avg['dir_acc'],
                    Sem_MAE=avg['sem_mae'],
                    Intersect_Ratio=avg['intersect_ratio'],
                    Correction_Ratio=avg['correction_ratio'],
                ))
            logger.log_metrics(log_d, step=epoch+1,
                               prefix=f'Train_Epoch_{mode_name}')

        # ── 定量验证（每 5 epoch）────────────────────────────────────
        if val_dl is not None and (epoch + 1) % 5 == 0:
            unet.eval()
            vp_l, vs_l, vl_l, va_l, vda_l, vsm_l = [], [], [], [], [], []
            # overall_acc：整个 GT 核区域内的准确率（正确的模型选择指标）
            all_gt_val, all_pred_val = [], []

            with torch.no_grad():
                for vb in val_dl:
                    vhr    = vb['hr'].to(device)
                    vbs    = vhr.shape[0]
                    vt     = torch.tensor([t_max // 2] * vbs, device=device).long()
                    vn     = torch.randn_like(vhr)
                    vnoisy = scheduler.add_noise(vhr, vn, vt)

                    if use_semantic:
                        vhr_cv = run_cellvit(cellvit, vhr)
                        vsem   = build_sem_tensor_from_cellvit(
                            vhr_cv['nuclei_type_prob'],
                            vhr_cv['nuclei_nuc_prob'],
                        )
                    else:
                        vsem = None
                    # 消融模式下 cellvit 为 None，用 cellvit_for_val 替代
                    cellvit_for_val = cellvit if use_semantic else _cellvit_val

                    vnp  = unet(torch.cat([vhr, vnoisy], dim=1),
                                vt, semantic=vsem).sample
                    vx0  = predict_x0_from_noise_shared(vnoisy, vnp, vt, scheduler)

                    vp_l.append(compute_psnr(vx0.cpu(), vhr.cpu()))
                    vs_l.append(compute_ssim(vx0.cpu(), vhr.cpu()))
                    vl_l.append(F.l1_loss(vx0, vhr).item())
                    va_l.append(compute_artifact_penalty(vx0.cpu(), vhr.cpu()))

                    # 语义指标：两种模式都计算，消融模式也需要评估 Dir_Acc
                    # 完整模式：CellViT 已在上方调用过，直接复用 vhr_cv
                    # 消融模式：需要临时加载 CellViT 进行评估
                    # 注意：消融模式训练时无 CellViT，但验证时需要外部传入
                    if cellvit_for_val is not None:
                        vgt_lbl = vb.get('gt_label_map')
                        vgt_nuc = vb.get('gt_nuc_mask')
                        if vgt_lbl is not None and vgt_nuc is not None:
                            vgt_lbl  = vgt_lbl.to(device)
                            vgt_nuc  = vgt_nuc.to(device).bool()
                            vhr_cv2  = run_cellvit(cellvit_for_val, vhr)
                            vpred_cv = run_cellvit(cellvit_for_val, vx0)
                            hr_lbl   = vhr_cv2['nuclei_type_label']
                            pred_lbl = vpred_cv['nuclei_type_label']

                            # Dir_Acc：交集区域（GT∩CellViT(HR)判对）
                            inter    = (vgt_lbl > 0) & (hr_lbl == vgt_lbl)
                            if inter.sum() > 0:
                                vda_l.append(
                                    (pred_lbl[inter] == vgt_lbl[inter])
                                    .float().mean().item()
                                )
                                vsm_l.append(
                                    (vpred_cv['nuclei_type_prob']
                                     - vhr_cv2['nuclei_type_prob'])
                                    .abs().mean(dim=1)[inter].mean().item()
                                )

                            # Overall_Acc：整个 GT 核区域内（正确的模型选择指标）
                            cell_mask = vgt_nuc & (vgt_lbl > 0)
                            for b in range(vhr.shape[0]):
                                m = cell_mask[b].cpu().numpy()
                                if m.sum() > 0:
                                    all_gt_val.append(
                                        vgt_lbl[b].cpu().numpy()[m]
                                    )
                                    all_pred_val.append(
                                        pred_lbl[b].cpu().numpy()[m]
                                    )

            vp   = sum(vp_l)  / max(len(vp_l),  1)
            vs   = sum(vs_l)  / max(len(vs_l),  1)
            vl   = sum(vl_l)  / max(len(vl_l),  1)
            va   = sum(va_l)  / max(len(va_l),  1)
            vda  = sum(vda_l) / max(len(vda_l), 1) if vda_l else -1.0
            vsm  = sum(vsm_l) / max(len(vsm_l), 1) if vsm_l else -1.0
            comp = compute_composite_score(vp, vs, max(vsm, 0.0), va)

            # Overall_Acc：整个 GT 核区域内的准确率
            if all_gt_val:
                gt_cat   = np.concatenate(all_gt_val)
                pred_cat = np.concatenate(all_pred_val)
                v_overall_acc = float((gt_cat == pred_cat).mean())
            else:
                v_overall_acc = -1.0

            print(f"  [Val] PSNR={vp:.2f}  SSIM={vs:.4f}  "
                  f"L1={vl:.4f}  Artifact={va:.3f}")
            if v_overall_acc >= 0:
                print(f"  [Val] Overall_Acc={v_overall_acc:.4f}")
            if vda >= 0:
                print(f"  [Val] Dir_Acc(intersect)={vda:.4f}  "
                      f"Sem_MAE(intersect)={vsm:.4f}  Composite={comp:.4f}")

            if logger:
                log_d = dict(PSNR=vp, SSIM=vs, L1=vl, Artifact=va)
                if v_overall_acc >= 0:
                    log_d['Overall_Acc'] = v_overall_acc
                if vda >= 0:
                    log_d.update(dict(
                        Dir_Acc_intersect=vda,
                        Sem_MAE_intersect=vsm,
                        Composite=comp,
                    ))
                logger.log_metrics(log_d, step=epoch+1,
                                   prefix=f'Val_{mode_name}')

            # ── 模型选择指标 ─────────────────────────────────────────
            # 两种模式统一用 overall_acc 作为主要指标（公平对比）
            # overall_acc 衡量整个 GT 核区域内的准确率，比 dir_acc 更合理
            # 回退：若 overall_acc 不可用则用 -L1
            cur_metric = v_overall_acc if v_overall_acc >= 0 else -vl
            if cur_metric > best_metric:
                best_metric = cur_metric
                save_dict = dict(
                    epoch                = epoch + 1,
                    model_state_dict     = unet.state_dict(),
                    optimizer_state_dict = optimizer.state_dict(),
                    val_psnr             = vp,
                    val_ssim             = vs,
                    val_l1               = vl,
                    val_overall_acc      = v_overall_acc,
                    val_dir_acc          = vda,
                    val_sem_mae          = vsm,
                    val_composite        = comp,
                    use_semantic         = use_semantic,
                )
                torch.save(save_dict,
                           os.path.join(save_dir, best_ckpt_name))
                print(f"  🔥 New best → {best_ckpt_name}  "
                      f"(overall_acc={v_overall_acc:.4f})")

            unet.train()

        # 周期性 checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(
                save_dir, f'unet_{mode_name}_epoch_{epoch+1}.pth'
            )
            torch.save(dict(
                epoch                = epoch + 1,
                model_state_dict     = unet.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                loss                 = avg['total'],
                use_semantic         = use_semantic,
            ), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")
    if logger:
        logger.close()


# ─────────────────────────────────────────────────────────────────────────────
# 消融模式用的独立损失函数（不依赖 SemanticSRLoss）
# ─────────────────────────────────────────────────────────────────────────────

def _predict_x0(x_t, noise_pred, t, scheduler):
    dev   = x_t.device
    dtype = x_t.dtype
    alpha = scheduler.alphas_cumprod.to(dev)[t].to(dtype).view(-1, 1, 1, 1)
    beta  = 1.0 - alpha
    return ((x_t - beta ** 0.5 * noise_pred) / (alpha ** 0.5 + 1e-8)).clamp(0.0, 1.0)


def _gradient_loss(pred, target):
    def _grad(x):
        return (x[:, :, 1:, :] - x[:, :, :-1, :],
                x[:, :, :, 1:] - x[:, :, :, :-1])
    ph, pw = _grad(pred)
    th, tw = _grad(target)
    return (ph - th).abs().mean() + (pw - tw).abs().mean()


def _tv_loss(x0_hat, hr,
             tv_margin_factor=1.05, tv_leaky_alpha=0.10):
    def _tv(img):
        dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs()
        dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs()
        return dh.mean(dim=(1, 2, 3)) + dw.mean(dim=(1, 2, 3))

    tv_hat = _tv(x0_hat * 255.0)
    with torch.no_grad():
        tv_ref = _tv(hr * 255.0)
    margin     = tv_ref * tv_margin_factor
    loss_batch = torch.where(
        tv_hat <= margin,
        tv_leaky_alpha * tv_hat,
        tv_leaky_alpha * margin + (tv_hat - margin),
    )
    return loss_batch.mean()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _add_bool_mutex(parser, dest, default, opt_name):
    g = parser.add_mutually_exclusive_group()
    g.add_argument(f'--{opt_name}',    dest=dest, action='store_true')
    g.add_argument(f'--no-{opt_name}', dest=dest, action='store_false')
    parser.set_defaults(**{dest: default})


def main():
    cfg = get_default_config()
    p   = argparse.ArgumentParser(
        description='SPM-UNet 判别器标签纠正训练（单阶段）\n'
                    '加 --no_semantic 训练消融基线模型。'
    )

    # 模式开关
    p.add_argument('--no_semantic', action='store_true',
                   help='消融模式：关闭语义监督和 sem_tensor 注入')

    p.add_argument('--dataset_type', default=cfg.dataset_type,
                   choices=['nct', 'pannuke'])
    p.add_argument('--tum_dir',  default=cfg.tum_dir)
    p.add_argument('--norm_dir', default=cfg.norm_dir)
    p.add_argument('--pannuke_root',       default=cfg.pannuke_root)
    p.add_argument('--pannuke_fold_dirs',  nargs='+', default=None,
                   metavar='FOLD_DIR',
                   help='训练用的折目录列表，如 Fold1 Fold2（支持多个）')
    p.add_argument('--pannuke_test_fold_dir', default=cfg.pannuke_test_fold_dir)
    p.add_argument('--n_val',   type=int, default=100,
                   help='从训练数据中分层抽取的验证集样本数（默认100）')
    p.add_argument('--val_seed', type=int, default=42,
                   help='验证集抽样随机种子（保证可复现，默认42）')
    p.add_argument('--cellvit_path', default=cfg.cellvit_path)
    p.add_argument('--cellvit_repo', default=cfg.cellvit_repo)
    p.add_argument('--epochs',       type=int,   default=cfg.epochs)
    p.add_argument('--batch_size',   type=int,   default=cfg.batch_size)
    p.add_argument('--lr',           type=float, default=cfg.lr)
    p.add_argument('--device',       default=cfg.device)
    p.add_argument('--gpu_id',       type=int,   default=None)
    p.add_argument('--save_dir',     default=cfg.save_dir)
    p.add_argument('--lambda_noise', type=float, default=cfg.lambda_noise)
    p.add_argument('--lambda_rec',   type=float, default=cfg.lambda_rec)
    p.add_argument('--lambda_grad',  type=float, default=cfg.lambda_grad)
    p.add_argument('--lambda_sem',   type=float, default=cfg.lambda_sem)
    p.add_argument('--lambda_tv',    type=float, default=cfg.lambda_tv)
    p.add_argument('--lambda_sem_cls',   type=float, default=cfg.lambda_sem_cls)
    p.add_argument('--correction_boost', type=float, default=cfg.correction_boost)
    p.add_argument('--focal_gamma',      type=float, default=cfg.focal_gamma)
    p.add_argument('--weight_decay',     type=float, default=cfg.weight_decay)
    p.add_argument('--max_grad_norm',    type=float, default=cfg.max_grad_norm)
    p.add_argument('--num_workers',      type=int,   default=cfg.num_workers)
    p.add_argument('--accumulation_steps', type=int, default=cfg.accumulation_steps)
    p.add_argument('--resume_path',   default=None)
    p.add_argument('--no_tb',         action='store_true')
    p.add_argument('--log_dir',       default='./logs')
    p.add_argument('--exp_name',      default=None)
    p.add_argument('--t_max',         type=int, default=cfg.t_max)
    p.add_argument('--num_train_timesteps', type=int,
                   default=cfg.num_train_timesteps)
    p.add_argument('--target_size',   type=int, default=cfg.target_size or 256)
    _add_bool_mutex(p, 'pin_memory',      cfg.pin_memory,      'pin-memory')
    _add_bool_mutex(p, 'oversample',      cfg.oversample,      'oversample')
    _add_bool_mutex(p, 'train_drop_last', cfg.train_drop_last, 'train-drop-last')
    _add_bool_mutex(p, 'oversample_minority', False, 'oversample-minority')
    p.add_argument('--oversample_ratio',   type=float, default=2.0,
                   help='少数类过采样倍率（默认2.0）')
    p.add_argument('--oversample_classes', type=int, nargs='+', default=None,
                   help='过采样的类别索引，默认[2]即Inflammatory')
    p.add_argument('--oversample_min_pixels', type=int, default=500,
                   help='目标类别像素数阈值，超过此值才纳入过采样（默认500）')

    args         = p.parse_args()
    use_semantic = not args.no_semantic
    print_gpu_info()

    mode_str = '完整模式（语义监督）' if use_semantic else '消融模式（无语义监督）'
    if args.exp_name is None:
        tag = 'Correction' if use_semantic else 'Ablation'
        args.exp_name = f'{tag}_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    logger = ExperimentLogger(
        use_tensorboard = not args.no_tb,
        log_dir         = args.log_dir,
        experiment_name = args.exp_name,
    )
    device = (get_device(gpu_id=args.gpu_id) if args.gpu_id is not None
              else args.device or get_device())

    # 消融模式不需要加载 CellViT
    cellvit = None
    if use_semantic:
        if not args.cellvit_path:
            raise ValueError("完整模式下必须提供 --cellvit_path")
        cellvit = load_cellvit(
            model_path        = args.cellvit_path,
            cellvit_repo_path = args.cellvit_repo,
            device            = device,
        )
    else:
        print(f"消融模式：跳过 CellViT 加载")

    train(
        cellvit                = cellvit,
        use_semantic           = use_semantic,
        dataset_type           = args.dataset_type,
        tum_dir                = args.tum_dir,
        norm_dir               = args.norm_dir,
        pannuke_root           = args.pannuke_root,
        pannuke_fold_dirs      = args.pannuke_fold_dirs,
        pannuke_test_fold_dir  = args.pannuke_test_fold_dir,
        n_val                  = args.n_val,
        val_seed               = args.val_seed,
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        device                 = device,
        save_dir               = args.save_dir,
        lambda_noise           = args.lambda_noise,
        lambda_rec             = args.lambda_rec,
        lambda_grad            = args.lambda_grad,
        lambda_sem             = args.lambda_sem,
        lambda_tv              = args.lambda_tv,
        lambda_sem_cls         = args.lambda_sem_cls,
        correction_boost       = args.correction_boost,
        focal_gamma            = args.focal_gamma,
        accumulation_steps     = args.accumulation_steps,
        weight_decay           = args.weight_decay,
        max_grad_norm          = args.max_grad_norm,
        num_workers            = args.num_workers,
        pin_memory             = args.pin_memory,
        oversample             = args.oversample,
        train_drop_last        = args.train_drop_last,
        oversample_minority    = args.oversample_minority,
        oversample_ratio       = args.oversample_ratio,
        oversample_classes     = args.oversample_classes,
        oversample_min_pixels  = args.oversample_min_pixels,
        resume_path            = args.resume_path,
        logger                 = logger,
        t_max                  = args.t_max,
        num_train_timesteps    = args.num_train_timesteps,
        target_size            = args.target_size,
    )


if __name__ == '__main__':
    main()