"""
反馈损失模块：cell-level 自适应 directional enhancement。
不依赖 patch label；每个 cell 按分数自适应增强方向（高分→tumor，低分→normal），高置信 cell 保持。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label


class FeedbackLoss(nn.Module):
    """
    纯 cell-level 自适应 directional enhancement 版本
    - 不依赖 patch label
    - 每个 cell 自适应增强方向：高分往 tumor 方向，低分往 normal 方向
    - 高置信 cell 保持不动，delta 控制增强幅度
    """
    def __init__(self, hovernet, scheduler, np_thresh=0.5, min_cell_pixels=8, delta=0.08, low=0.3, high=0.7):
        super().__init__()
        self.hovernet = hovernet
        self.scheduler = scheduler
        self.np_thresh = np_thresh
        self.min_cell_pixels = min_cell_pixels
        self.delta = delta
        self.low = low
        self.high = high

        # 冻结 HoVer-Net 参数
        for p in self.hovernet.parameters():
            p.requires_grad = False
        self.hovernet.eval()

    def predict_x0_from_noise(self, x_t, noise_pred, t):
        """Tweedie公式预测x0"""
        device = x_t.device
        dtype = x_t.dtype

        alpha_bar = self.scheduler.alphas_cumprod.to(device)[t].to(dtype).view(-1, 1, 1, 1)
        beta_bar = 1.0 - alpha_bar

        pred_x0 = (x_t - torch.sqrt(beta_bar) * noise_pred) / (torch.sqrt(alpha_bar) + 1e-8)
        pred_x0 = torch.clamp(pred_x0, 0.0, 1.0)
        return pred_x0

    def build_instance_map_from_np(self, np_prob_2d):
        """
        根据 HoVer-Net np 前景概率图做一个简化版 instance map
        这里只用连通域，控制变量，不引入 hv 后处理

        Args:
            np_prob_2d: numpy array [H, W]

        Returns:
            instance_map [H, W]，背景为 0
        """
        binary = (np_prob_2d > self.np_thresh).astype(np.uint8)
        inst_map = label(binary, connectivity=1)
        return inst_map

    def aggregate_cell_scores(self, prob_map_2d, inst_map, min_pixels=None):
        """
        按 cell instance 聚合分数

        Args:
            prob_map_2d: torch.Tensor [H, W]
            inst_map: numpy.ndarray [H, W] 或 torch.Tensor [H, W]

        Returns:
            cell_ids: list[int]
            cell_scores: torch.Tensor [N_cells] 或 None
        """
        if min_pixels is None:
            min_pixels = self.min_cell_pixels

        device = prob_map_2d.device
        if not torch.is_tensor(inst_map):
            inst_map = torch.from_numpy(inst_map).to(device)

        cell_ids = torch.unique(inst_map)
        cell_ids = cell_ids[cell_ids > 0]

        kept_ids = []
        scores = []

        for cid in cell_ids:
            mask = (inst_map == cid)
            if mask.sum() < min_pixels:
                continue
            kept_ids.append(int(cid.item()))
            scores.append(prob_map_2d[mask].mean())

        if len(scores) == 0:
            return [], None

        return kept_ids, torch.stack(scores, dim=0)

    def build_cell_targets(self, clean_scores):
        """
        基于 cell-level clean score 自适应 directional target
        - 高分 → tumor 强化
        - 低分 → normal 强化
        - 高置信 cell 保持
        """
        target_scores = clean_scores.clone()
        enhance_mask = torch.zeros_like(clean_scores, dtype=torch.bool)
        preserve_mask = torch.zeros_like(clean_scores, dtype=torch.bool)

        # tumor-like cells
        tumor_like = clean_scores >= self.high
        # normal-like cells
        normal_like = clean_scores <= self.low
        # uncertain cells
        uncertain = (~tumor_like) & (~normal_like)

        # 高分 cell 往 tumor 方向微增强
        target_scores[tumor_like] = torch.clamp(clean_scores[tumor_like] + self.delta, 0.0, 1.0)
        enhance_mask[tumor_like] = True

        # 低分 cell 往 normal 方向微增强
        target_scores[normal_like] = torch.clamp(clean_scores[normal_like] - self.delta, 0.0, 1.0)
        enhance_mask[normal_like] = True

        # 保持 uncertain cell
        preserve_mask[uncertain] = True

        return target_scores, enhance_mask, preserve_mask

    def forward(self, x_t, noise_pred, t, clean_images):
        """
        Args:
            x_t: [B,C,H,W] noisy input
            noise_pred: [B,C,H,W] predicted noise
            t: [B] timestep
            clean_images: [B,C,H,W] clean reference
        """
        x0_hat = self.predict_x0_from_noise(x_t, noise_pred, t)

        hovernet_device = next(self.hovernet.parameters()).device
        x0_input = x0_hat.to(hovernet_device)
        clean_input = clean_images.to(hovernet_device)
        x0_input = x0_input * 255.0
        clean_input = clean_input * 255.0

        # 1) clean侧生成 cell-level pseudo target
        with torch.no_grad():
            clean_output = self.hovernet(clean_input)
            clean_tp_probs = torch.softmax(clean_output["tp"], dim=1)   # [B, C_tp, H, W]
            clean_np_probs = torch.softmax(clean_output["np"], dim=1)   # [B, 2, H, W]

            clean_p_neo = clean_tp_probs[:, 1, :, :]   # [B, H, W]
            clean_np_fg = clean_np_probs[:, 1, :, :]   # [B, H, W]

            instance_maps = []
            cell_targets_list = []
            cell_scores_list = []

            B = clean_p_neo.shape[0]
            for b in range(B):
                np_map_np = clean_np_fg[b].detach().cpu().numpy()
                inst_map = self.build_instance_map_from_np(np_map_np)
                instance_maps.append(inst_map)

                cell_ids, clean_scores = self.aggregate_cell_scores(clean_p_neo[b], inst_map)
                if clean_scores is None or len(cell_ids) == 0:
                    cell_targets_list.append(None)
                    cell_scores_list.append(None)
                    continue

                target_scores, enhance_mask, preserve_mask = self.build_cell_targets(clean_scores)
                cell_targets_list.append({
                    "cell_ids": cell_ids,
                    "target_scores": target_scores,
                    "enhance_mask": enhance_mask,
                    "preserve_mask": preserve_mask,
                })
                cell_scores_list.append(clean_scores)

        # 2) pred侧
        output = self.hovernet(x0_input)
        probs = torch.softmax(output["tp"], dim=1)
        p_neo = probs[:, 1, :, :]   # [B, H, W]

        total_loss = 0.0
        valid_count = 0
        tumor_conf_list = []
        normal_conf_list = []

        for b in range(B):
            info = cell_targets_list[b]
            if info is None:
                continue

            inst_map = instance_maps[b]
            cell_ids_pred, pred_scores = self.aggregate_cell_scores(p_neo[b], inst_map)
            if pred_scores is None or len(cell_ids_pred) == 0:
                continue
            if len(cell_ids_pred) != len(info["cell_ids"]):
                continue

            # cell-level directional enhancement loss
            target_scores = info["target_scores"].to(pred_scores.device)
            enhance_mask = info["enhance_mask"].to(pred_scores.device)
            preserve_mask = info["preserve_mask"].to(pred_scores.device)

            loss_b = 0.0
            count = 0

            if enhance_mask.sum() > 0:
                loss_enh = F.smooth_l1_loss(pred_scores[enhance_mask], target_scores[enhance_mask], reduction='mean')
                loss_b += loss_enh
                count += 1
            if preserve_mask.sum() > 0:
                loss_pres = F.smooth_l1_loss(pred_scores[preserve_mask], target_scores[preserve_mask], reduction='mean')
                loss_b += 0.5 * loss_pres
                count += 1
            if count > 0:
                loss_b = loss_b / count
                total_loss += loss_b
                valid_count += 1

            # 监控 tumor/normal confidence
            tumor_mask = pred_scores > 0.5
            normal_mask = ~tumor_mask
            if tumor_mask.sum() > 0:
                tumor_conf_list.append(pred_scores[tumor_mask].mean())
            if normal_mask.sum() > 0:
                normal_conf_list.append(pred_scores[normal_mask].mean())

        if valid_count == 0:
            zero = torch.tensor(0.0, device=x_t.device)
            neg_one = torch.tensor(-1.0, device=x_t.device)
            return zero, zero, neg_one, neg_one

        loss_prob = total_loss / valid_count
        loss_entropy = torch.tensor(0.0, device=x_t.device)

        avg_tumor_conf = torch.stack(tumor_conf_list).mean() if len(tumor_conf_list) > 0 else torch.tensor(-1.0, device=x_t.device)
        avg_normal_conf = torch.stack(normal_conf_list).mean() if len(normal_conf_list) > 0 else torch.tensor(-1.0, device=x_t.device)

        return loss_prob, loss_entropy, avg_tumor_conf, avg_normal_conf
