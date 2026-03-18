"""
validation.py
语义引导型 SR DDPM 的验证集管理。

与旧版本的差异
----------------
旧流程：固定 TUM/NORM 图像 → 加噪 → 增强 → 对比肿瘤热力图，
       指标为：tumer_conf_before / after / gap。

新流程：固定 HR 图像 → 合成 LR → 重建 → 与 HR 对比，
       可视化内容：HR / LR / Reconstructed / Residual / p_clean / p_pred，
       度量指标：PSNR、SSIM、Masked_Semantic_MAE、Artifact_Penalty、
                 Composite_Score（用于选择最佳模型）。
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from ddpm_dataset import NCTDataset
from ddpm_utils import predict_x0_from_noise_shared
from degradation import degrade
from metrics import (compute_psnr, compute_ssim,
                     compute_masked_semantic_mae, compute_artifact_penalty,
                     compute_composite_score)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：加载一小批固定验证样本用于可视化
# ─────────────────────────────────────────────────────────────────────────────

def load_fixed_validation_batch(val_dir, sample_size=256, device='cuda',
                                 scale=2, n_per_class=4):
    """
    从 TUM 和 NORM 中各加载 n_per_class 张图像作为 HR tensor。
    同时合成一份固定的 LR 版本（每个 epoch 使用相同随机种子，方便公平对比）。

    返回
    -------
    hr_tensor  : [B, 3, H, W]
    lr_tensor  : [B, 3, H, W]
    labels     : [B]
    """
    images, labels = [], []
    for subdir, lbl in [('TUM', 1), ('NORM', 0)]:
        path = os.path.join(val_dir, subdir)
        if not os.path.exists(path):
            continue
        files = sorted([f for f in os.listdir(path)
                         if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))])
        for f in files[:n_per_class]:
            img = cv2.imread(os.path.join(path, f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (sample_size, sample_size),
                             interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
            labels.append(lbl)

    if not images:
        return None, None, None

    hr_np   = np.stack(images, axis=0).astype(np.float32) / 255.0
    hr      = torch.from_numpy(hr_np).permute(0, 3, 1, 2).to(device)  # [B,3,H,W]

    # 固定的 LR（使用固定种子 42，保证可复现）
    torch.manual_seed(42)
    lr = torch.stack([degrade(hr[i].cpu(), scale=scale) for i in range(len(hr))]).to(device)
    torch.manual_seed(torch.initial_seed())   # 恢复原始随机状态

    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    return hr, lr, labels_t


# ─────────────────────────────────────────────────────────────────────────────
# ValidationSet：用于可视化的一小批固定验证样本
# ─────────────────────────────────────────────────────────────────────────────

class ValidationSet:
    """
    管理一小批固定的验证样本，用于主观可视化。
    在 load() 之后固定 LR、噪声和时间步，以保证不同 epoch 之间的公平对比。
    """

    def __init__(self, val_dir, scheduler, device='cuda',
                 sample_size=256, fixed_timestep=100, scale=2):
        self.val_dir        = val_dir
        self.scheduler      = scheduler
        self.device         = device
        self.sample_size    = sample_size
        self.fixed_timestep = fixed_timestep
        self.scale          = scale

        self.hr = self.lr = self.noisy_hr = None
        self.labels = self.noise = self.timesteps = None

    def load(self) -> bool:
        if not self.val_dir or not os.path.exists(self.val_dir):
            return False
        print(f"Loading fixed validation batch: {self.val_dir}")
        self.hr, self.lr, self.labels = load_fixed_validation_batch(
            self.val_dir, self.sample_size, self.device, self.scale)
        if self.hr is None:
            return False
        print(f"  ✓ {self.hr.shape[0]} validation images loaded")
        self._prepare_noise()
        return True

    def _prepare_noise(self):
        torch.manual_seed(42)
        self.noise = torch.randn_like(self.hr)
        self.timesteps = torch.full(
            (self.hr.shape[0],), self.fixed_timestep,
            device=self.device, dtype=torch.long)
        self.noisy_hr = self.scheduler.add_noise(self.hr, self.noise, self.timesteps)
        torch.manual_seed(torch.initial_seed())

    def is_available(self):
        return self.hr is not None

    def generate_reconstructions(self, unet, loss_module=None, use_semantic_injection: bool = False):
        """
        在固定的验证 LR / noisy-HR 上运行一次 DDPM 前向。

        返回的字典包含：
          reconstructed : [B,3,H,W]  重建结果
          diff_vis      : [B,3,H,W]  |reconstructed - hr| × 5，可视化残差
          p_clean       : [B,1,H,W]  HoVer-Net 在 HR 上的概率图
          p_pred        : [B,1,H,W]  HoVer-Net 在重建结果上的概率图
        """
        if self.hr is None:
            return None
        with torch.no_grad():
            model_input = torch.cat([self.lr, self.noisy_hr], dim=1)

            # 可选：SPM-UNet 架构层语义注入（semantic modulation）
            # 语义先验 S = [p_clean, nuc_mask, conf_mask]，由 HR 真值图通过 HoVer-Net 得到
            sem_tensor = None
            if use_semantic_injection and (loss_module is not None) and hasattr(loss_module, '_run_hovernet'):
                p_clean_map, nuc_mask = loss_module._run_hovernet(self.hr)  # [B,H,W], [B,H,W]
                tau_pos = getattr(loss_module, 'tau_pos', 0.65)
                tau_neg = getattr(loss_module, 'tau_neg', 0.35)
                conf_mask = ((p_clean_map >= tau_pos) | (p_clean_map <= tau_neg)).float()
                sem_tensor = torch.stack([p_clean_map, nuc_mask, conf_mask], dim=1)  # [B,3,H,W]

            # 若 unet 是 SPMUNet 且当前阶段 hooks 已关闭，则 semantic 会被忽略/走直通路径
            noise_pred = unet(model_input, self.timesteps, semantic=sem_tensor).sample
            recon = predict_x0_from_noise_shared(
                self.noisy_hr, noise_pred, self.timesteps, self.scheduler)

            diff_vis = (recon - self.hr).abs().clamp(0, 1) * 5

            p_clean = p_pred = torch.zeros(
                self.hr.shape[0], 1, self.hr.shape[2], self.hr.shape[3],
                device=self.device)

            if loss_module is not None and hasattr(loss_module, '_run_hovernet'):
                p_c_map, _  = loss_module._run_hovernet(self.hr)
                p_p_map, _  = loss_module._run_hovernet(recon)
                p_clean = p_c_map.unsqueeze(1)
                p_pred  = p_p_map.unsqueeze(1)

        return dict(reconstructed=recon, diff_vis=diff_vis,
                    p_clean=p_clean, p_pred=p_pred)


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def save_validation_debug_images(
    hr, lr, reconstructed, diff_vis, p_clean, p_pred,
    epoch, save_dir, num_vis=8, return_tensor=False,
    col_titles=None,
    suptitle: str = None,
):
    """
    保存 6 行图像的网格：
      第 1 行：HR（真值）
      第 2 行：LR 输入
      第 3 行：重建结果
      第 4 行：绝对残差 ×5
      第 5 行：p_clean 热力图
      第 6 行：p_pred 热力图
    """
    os.makedirs(save_dir, exist_ok=True)
    num_vis = min(num_vis, hr.shape[0])

    def _rgb(t):
        return t[:num_vis].detach().cpu().clamp(0,1).permute(0,2,3,1).numpy()

    def _gray(t):
        return t[:num_vis, 0].detach().cpu().clamp(0,1).numpy()

    # 行标签（左侧）
    rows_data = [
        (_rgb(hr),            'HR'),
        (_rgb(lr),            'LR'),
        (_rgb(reconstructed), 'Recon'),
        (_rgb(diff_vis),      'Residual ×5'),
        (_gray(p_clean),      'p_clean'),
        (_gray(p_pred),       'p_pred'),
    ]

    n_rows, n_cols = len(rows_data), num_vis
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.8 * n_cols, 2.6 * n_rows),
                              squeeze=False)

    # 总标题：用于展示 epoch / checkpoint 指标等
    if suptitle:
        # 标题字体更大，并贴近图像区域
        fig.suptitle(suptitle, fontsize=16, y=0.99)

    # 列标题：Sample 1 / TUM-1 / NORM-1 等
    if col_titles is not None:
        for c in range(min(n_cols, len(col_titles))):
            axes[0, c].set_title(str(col_titles[c]), fontsize=10, pad=8)

    for r, (data, title) in enumerate(rows_data):
        for c in range(n_cols):
            ax = axes[r, c]
            if data.ndim == 3:             # RGB 图像
                ax.imshow(data[c])
            else:                          # 灰度热力图
                ax.imshow(data[c], cmap='jet', vmin=0, vmax=1)
            ax.axis('off')

        # 用文本绘制行标题（比 set_ylabel 更不容易被 tight_layout/axis off 吃掉）
        axes[r, 0].text(
            -0.06, 0.5, title,
            transform=axes[r, 0].transAxes,
            ha='right', va='center',
            fontsize=12,
        )

    # 调整边距：给左侧行标题留空间，并给 suptitle 留更小的顶部空隙
    if suptitle:
        plt.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.02,
                            wspace=0.02, hspace=0.02)
    else:
        plt.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.02,
                            wspace=0.02, hspace=0.02)
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_val.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if not return_tensor:
        return None

    grid = make_grid(
        torch.cat([hr[:num_vis], lr[:num_vis],
                   reconstructed[:num_vis], diff_vis[:num_vis]]).detach().cpu(),
        nrow=num_vis, padding=2)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# 定量验证用 DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def create_val_dataloader(val_vis_dir, batch_size, device='cuda',
                           scale=2, oversample=False):
    """
    基于 TUM/NORM 验证划分构建一个 DataLoader。
    如果 val_vis_dir 缺失或不完整，则返回 None。
    """
    if not val_vis_dir or not os.path.exists(val_vis_dir):
        return None
    tum_dir  = os.path.join(val_vis_dir, 'TUM')
    norm_dir = os.path.join(val_vis_dir, 'NORM')
    if not os.path.exists(tum_dir) or not os.path.exists(norm_dir):
        print("⚠️  验证目录缺少 TUM/NORM 子目录 —— 跳过定量验证。")
        return None

    val_ds = NCTDataset(tum_dir, norm_dir, oversample=oversample, scale=scale)
    dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=(device == 'cuda'))
    print(f"定量验证集：{len(val_ds)} 张图像")
    return dl
