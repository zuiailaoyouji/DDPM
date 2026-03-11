"""
反馈损失模块：像素级概率转为细胞级概率，避免单个 noisy pixel 主导训练。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label


class FeedbackLoss(nn.Module):
    def __init__(self, hovernet, scheduler, np_thresh=0.5, min_cell_pixels=8):
        super().__init__()
        self.hovernet = hovernet
        self.scheduler = scheduler
        self.np_thresh = np_thresh
        self.min_cell_pixels = min_cell_pixels

        # 冻结 HoVer-Net 参数
        for p in self.hovernet.parameters():
            p.requires_grad = False
        self.hovernet.eval()

    def predict_x0_from_noise(self, x_t, noise_pred, t):
        """
        用 Tweedie 公式从 x_t 和预测噪声恢复 x0
        """
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

    def forward(self, x_t, noise_pred, t, clean_images):
        """
        像素级 -> 细胞级 的控制变量版本。
        在原图上用 HoVer-Net + 连通域得到 cell instance + cell-level pseudo target，
        在预测图上再按 cell 聚合概率，做 cell-level BCE。

        Returns:
            loss_prob: 细胞级 BCE 损失
            loss_entropy: 目前固定返回 0（占位）
            avg_tumor_conf: 癌细胞平均置信度（无该类时为 -1.0）
            avg_normal_conf: 正常细胞平均置信度（无该类时为 -1.0）
        """
        # 1) 预测 x0_hat
        x0_hat = self.predict_x0_from_noise(x_t, noise_pred, t)
        hovernet_device = next(self.hovernet.parameters()).device

        x0_input = x0_hat.to(hovernet_device)
        clean_input = clean_images.to(hovernet_device)

        # HoVer-Net 输入 0~255
        x0_input = x0_input * 255.0
        clean_input = clean_input * 255.0

        # 2) clean 侧：构造 cell instance + cell-level pseudo target
        with torch.no_grad():
            clean_output = self.hovernet(clean_input)
            clean_tp_probs = torch.softmax(clean_output["tp"], dim=1)   # [B, C_tp, H, W]
            clean_np_probs = torch.softmax(clean_output["np"], dim=1)   # [B, 2, H, W]

            clean_p_neo = clean_tp_probs[:, 1, :, :]   # [B, H, W]
            clean_np_fg = clean_np_probs[:, 1, :, :]   # [B, H, W]

            instance_maps = []
            clean_cell_targets = []

            B = clean_p_neo.shape[0]
            for b in range(B):
                np_prob_np = clean_np_fg[b].detach().cpu().numpy()
                inst_map = self.build_instance_map_from_np(np_prob_np)
                instance_maps.append(inst_map)

                cell_ids, clean_scores = self.aggregate_cell_scores(clean_p_neo[b], inst_map)
                if clean_scores is None or len(cell_ids) == 0:
                    clean_cell_targets.append(None)
                    continue

                # 控制变量：沿用原本 0.5 阈值，只是从 pixel 改成 cell
                cell_targets = clean_scores.detach()

                clean_cell_targets.append({
                    "cell_ids": cell_ids,
                    "targets": cell_targets,
                })

        # 3) pred 侧：得到 pred_p_neo
        output = self.hovernet(x0_input)
        probs = torch.softmax(output["tp"], dim=1)
        p_neo = probs[:, 1, :, :]   # [B, H, W]

        # 4) 细胞级 loss_prob
        total_loss = 0.0
        valid_count = 0

        tumor_conf_list = []
        normal_conf_list = []

        B = p_neo.shape[0]
        for b in range(B):
            info = clean_cell_targets[b]
            if info is None:
                continue

            inst_map = instance_maps[b]
            cell_ids_pred, pred_scores = self.aggregate_cell_scores(p_neo[b], inst_map)
            if pred_scores is None or len(cell_ids_pred) == 0:
                continue

            # 同一个 inst_map 下，cell_ids 顺序理论上应一致
            if len(cell_ids_pred) != len(info["cell_ids"]):
                continue

            cell_targets = info["targets"].to(pred_scores.device)

            # cell-level BCE
            loss_b = F.binary_cross_entropy(
                pred_scores.clamp(1e-6, 1 - 1e-6),
                cell_targets,
                reduction="mean",
            )

            total_loss = total_loss + loss_b
            valid_count += 1

            # 监控：仍然按 cell-level pseudo target 分成 tumor/normal cells 统计
            tumor_mask = (cell_targets > 0.5)
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

        # entropy 先保持接口不变，返回 0（控制变量）
        loss_entropy = torch.tensor(0.0, device=x_t.device)

        if len(tumor_conf_list) > 0:
            avg_tumor_conf = torch.stack(tumor_conf_list).mean()
        else:
            avg_tumor_conf = torch.tensor(-1.0, device=x_t.device)

        if len(normal_conf_list) > 0:
            avg_normal_conf = torch.stack(normal_conf_list).mean()
        else:
            avg_normal_conf = torch.tensor(-1.0, device=x_t.device)

        return loss_prob, loss_entropy, avg_tumor_conf, avg_normal_conf

