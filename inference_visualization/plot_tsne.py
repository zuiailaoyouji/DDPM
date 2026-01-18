"""
t-SNE 特征空间可视化 (最终修正版 v5)
修改点：
1. 【优化】混杂区域分离轨迹的颜色：
   - 不再使用灰色实线。
   - 改为使用与【初始点】一致的颜色（TUM为浅红，NORM为浅蓝）。
   - 略微提高透明度和线宽，确保浅色线条清晰可见。
2. 保持其他所有逻辑不变（误判虚线、无大箭头等）。
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import random
from glob import glob

# ==========================================
# 1. 环境设置与导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ddpm_utils import load_hovernet, get_device

# ==========================================
# 2. 数据集与特征提取 (保持不变)
# ==========================================
class ImageDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list) * 2
    
    def __getitem__(self, idx):
        sample_idx = idx // 2
        is_enhanced = (idx % 2 == 1)
        sample = self.data_list[sample_idx]
        path = sample['enh_path'] if is_enhanced else sample['orig_path']
        
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = self.transform(img)
        return img, sample_idx, 1 if is_enhanced else 0

class HoVerNetFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.captured_features = None
        self.hook_handle = None
        
        target_module = None
        if hasattr(model, 'module') and hasattr(model.module, 'd3'):
            target_module = model.module.d3
        elif hasattr(model, 'd3'):
            target_module = model.d3
        else:
            if hasattr(model, 'encoder'):
                target_module = model.encoder
            elif hasattr(model, 'module') and hasattr(model.module, 'encoder'):
                target_module = model.module.encoder
            else:
                for name, module in model.named_modules():
                    if 'layer4' in name:
                        target_module = module
                        break

        if target_module is None:
            raise RuntimeError("无法找到特征层(d3/encoder/layer4)")

        self.hook_handle = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        target_tensor = output
        if isinstance(output, (list, tuple)):
            target_tensor = output[-1]
        
        if target_tensor is not None:
            if target_tensor.dim() == 4:
                pooled = torch.mean(target_tensor, dim=(2, 3))
                self.captured_features = pooled.detach().cpu()
            elif target_tensor.dim() == 2:
                self.captured_features = target_tensor.detach().cpu()

    def extract(self, img_tensor):
        model_input = img_tensor * 255.0
        try:
            _ = self.model(model_input)
        except Exception:
            pass 
        return self.captured_features

    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()

def extract_features_hovernet(loader, model, device):
    extractor = HoVerNetFeatureExtractor(model)
    features_dict = {}
    with torch.no_grad():
        for imgs, sample_idxs, type_ids in tqdm(loader, desc="提取特征"):
            imgs = imgs.to(device)
            feats = extractor.extract(imgs)
            if feats is None: continue
            feats_np = feats.numpy()
            sample_idxs = sample_idxs.numpy()
            type_ids = type_ids.numpy()
            for i in range(len(imgs)):
                key = (sample_idxs[i], type_ids[i])
                features_dict[key] = feats_np[i]
    extractor.close()
    return features_dict

# ==========================================
# 3. 核心绘图逻辑 (Modified)
# ==========================================
def plot_tsne_graph_final(df_merged, output_path, misclassified_count, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    
    # -------------------------------------------------------
    # 1. 配色定义
    # -------------------------------------------------------
    c_tum_orig = '#fbb4b9'  # 浅红 (TUM原图)
    c_tum_enh  = '#de2d26'  # 深红 (TUM增强目标)
    c_norm_orig = '#c6dbef' # 浅蓝 (NORM原图)
    c_norm_enh  = '#3182bd' # 深蓝 (NORM增强目标)
    
    # -------------------------------------------------------
    # 2. 绘制【原图】(Original Points)
    # -------------------------------------------------------
    df_tum = df_merged[df_merged['Label'] == 'TUM']
    df_norm = df_merged[df_merged['Label'] == 'NORM']
    
    ax.scatter(df_tum['orig_x'], df_tum['orig_y'], c=c_tum_orig, 
               label='TUM (Original)', alpha=0.4, s=50, edgecolors='none', zorder=1)
    ax.scatter(df_norm['orig_x'], df_norm['orig_y'], c=c_norm_orig, 
               label='NORM (Original)', alpha=0.4, s=50, edgecolors='none', zorder=1)
    
    # -------------------------------------------------------
    # 3. 绘制【增强图】(Enhanced Points)
    # -------------------------------------------------------
    # 颜色由 Mode 决定
    df_max = df_merged[df_merged['Mode'] == 'maximize']
    df_min = df_merged[df_merged['Mode'] == 'minimize']
    
    ax.scatter(df_max['enh_x'], df_max['enh_y'], c=c_tum_enh, 
               label='Enhanced to TUM', alpha=0.85, s=70, edgecolors='white', linewidth=0.5, zorder=2)
    ax.scatter(df_min['enh_x'], df_min['enh_y'], c=c_norm_enh, 
               label='Enhanced to NORM', alpha=0.85, s=70, edgecolors='white', linewidth=0.5, zorder=2)

    # -------------------------------------------------------
    # 4. 计算混杂区域 (Mixed Region) 并绘制分离轨迹
    # -------------------------------------------------------
    
    # 准备数据
    orig_coords = df_merged[['orig_x', 'orig_y']].values
    labels = df_merged['Label'].map({'TUM': 1, 'NORM': 0}).values
    
    # 4.1 KNN 拟合
    k_neighbors = 20
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(orig_coords)
    distances, indices = nbrs.kneighbors(orig_coords)
    
    # 4.2 计算混杂度
    mixing_scores = []
    for i in range(len(orig_coords)):
        curr_label = labels[i]
        neighbor_indices = indices[i]
        neighbor_labels = labels[neighbor_indices]
        diff_count = np.sum(neighbor_labels != curr_label)
        score = diff_count / k_neighbors
        mixing_scores.append(score)
    
    df_merged['mixing_score'] = mixing_scores
    
    # 4.3 筛选
    mixed_threshold = 0.3
    separation_df = df_merged[
        (df_merged['mixing_score'] > mixed_threshold) & 
        (~df_merged['is_misclassified'])
    ]
    
    print(f"  识别到 {len(separation_df)} 个位于混杂区域且正确增强的样本，正在绘制分离轨迹...")
    
    # 4.4 绘制分离轨迹 (实线)
    # 修改逻辑：颜色跟随初始点类型 (Label)
    if len(separation_df) > 0:
        if len(separation_df) > 300:
            separation_df = separation_df.sample(300, random_state=42)
            
        for _, row in separation_df.iterrows():
            # 颜色判别逻辑
            line_color = c_tum_orig if row['Label'] == 'TUM' else c_norm_orig
            
            # 绘制实线
            # 略微提高 alpha (0.6) 和 linewidth (1.0) 保证浅色线可见
            ax.plot([row['orig_x'], row['enh_x']], [row['orig_y'], row['enh_y']], 
                    color=line_color, linestyle='-', alpha=0.6, linewidth=1.0, zorder=1.5)

    # -------------------------------------------------------
    # 5. 重点标记误判样本 (优先级最高，虚线)
    # -------------------------------------------------------
    misc_df = df_merged[df_merged['is_misclassified']]
    
    if len(misc_df) > 0:
        print(f"  标记 {len(misc_df)} 个误判样本轨迹...")
        for _, row in misc_df.iterrows():
            # 虚线连接
            ax.plot([row['orig_x'], row['enh_x']], [row['orig_y'], row['enh_y']], 
                    color='gray', linestyle='--', alpha=0.7, linewidth=1.2, zorder=4)
            # 增强点加黑圈
            ax.scatter(row['enh_x'], row['enh_y'], s=80, facecolors='none', 
                       edgecolors='black', linewidth=1.5, zorder=5)

    # -------------------------------------------------------
    # 6. 图例与装饰
    # -------------------------------------------------------
    legend_elements = [
        # 原图
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_tum_orig, label='TUM Original', markersize=8, alpha=0.6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_norm_orig, label='NORM Original', markersize=8, alpha=0.6),
        # 增强后
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_tum_enh, label='Enhanced to TUM', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c_norm_enh, label='Enhanced to NORM', markersize=10),
        # 轨迹
        Line2D([0], [0], color=c_tum_orig, lw=1.5, linestyle='-', alpha=0.6, label='Separation (Mixed Region)'),
        Line2D([0], [0], color='gray', lw=1.2, linestyle='--', alpha=0.7, label='Misclassified Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5, markersize=10, label='Misclassified Marker')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95, fancybox=True)
    
    ax.set_title("t-SNE Visualization: Separation of Mixed Samples", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. 主函数 (保持不变)
# ==========================================
def find_original_image(filename, enh_root_dir):
    name_no_ext = os.path.splitext(filename)[0]
    for pat in ["original*.png", "original.png"]:
        matches = glob(os.path.join(enh_root_dir, name_no_ext, pat))
        if matches: return matches[0]
    return None

def find_best_enhanced_image(filename, enh_root_dir):
    name_no_ext = os.path.splitext(filename)[0]
    for pat in ["BEST*.png", "BEST.png"]:
        matches = glob(os.path.join(enh_root_dir, name_no_ext, pat))
        if matches: return matches[0]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--csv_tum_path', type=str, default=None)
    parser.add_argument('--csv_norm_path', type=str, default=None)
    parser.add_argument('--enh_tum_dir', type=str, required=True)
    parser.add_argument('--enh_norm_dir', type=str, required=True)
    parser.add_argument('--hovernet_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./inference_visualization')
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 10])
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device() if hasattr(get_device, '__call__') else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 读取 CSV
    if args.csv_path:
        df = pd.read_csv(args.csv_path)
    else:
        df_list = []
        if args.csv_tum_path: df_list.append(pd.read_csv(args.csv_tum_path))
        if args.csv_norm_path: df_list.append(pd.read_csv(args.csv_norm_path))
        if not df_list: return
        df = pd.concat(df_list, ignore_index=True)
        
    if len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    
    # 2. 匹配文件
    data_list = []
    misclassified_count = 0
    print("匹配文件中...")
    for idx, row in df.iterrows():
        filename = row['Filename']
        mode = row['Mode']
        
        label = 'TUM' if 'TUM' in filename else ('NORM' if 'NORM' in filename else None)
        if not label: continue
        
        is_misclassified = False
        if (label == 'TUM' and mode == 'minimize') or (label == 'NORM' and mode == 'maximize'):
            is_misclassified = True
            misclassified_count += 1
            
        enh_dir = args.enh_tum_dir if label == 'TUM' else args.enh_norm_dir
        orig_path = find_original_image(filename, enh_dir)
        enh_path = find_best_enhanced_image(filename, enh_dir)
        
        if orig_path and enh_path:
            data_list.append({
                'filename': filename, 'orig_path': orig_path, 'enh_path': enh_path, 
                'Label': label, 'Mode': mode, 'is_misclassified': is_misclassified
            })
            
    print(f"匹配成功: {len(data_list)} (误判: {misclassified_count})")
    
    # 3. 特征提取
    model = load_hovernet(args.hovernet_path, device=device)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = ImageDataset(data_list, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    features_dict = extract_features_hovernet(loader, model, device)
    
    # 4. 整理矩阵
    feature_matrix = []
    meta_info = []
    for i in range(len(data_list)):
        if (i, 0) in features_dict and (i, 1) in features_dict:
            feature_matrix.append(features_dict[(i, 0)])
            meta_info.append({'idx': i, 'type': 'orig'})
            feature_matrix.append(features_dict[(i, 1)])
            meta_info.append({'idx': i, 'type': 'enh'})
            
    feature_matrix = np.array(feature_matrix)
    
    # 5. t-SNE
    print("运行 t-SNE...")
    n_samples, n_features = feature_matrix.shape
    n_components = min(50, n_samples, n_features)
    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=args.seed)
        feature_matrix_pca = pca.fit_transform(feature_matrix)
    else:
        feature_matrix_pca = feature_matrix
        
    curr_perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    tsne = TSNE(n_components=2, random_state=args.seed, init='pca', learning_rate='auto', perplexity=curr_perplexity)
    tsne_res = tsne.fit_transform(feature_matrix_pca)
    
    # 6. 坐标回填
    tsne_map = {}
    for k, coords in enumerate(tsne_res):
        info = meta_info[k]
        tsne_map[(info['idx'], info['type'])] = coords
        
    plot_data = []
    for i, item in enumerate(data_list):
        if (i, 'orig') in tsne_map and (i, 'enh') in tsne_map:
            item['orig_x'], item['orig_y'] = tsne_map[(i, 'orig')]
            item['enh_x'], item['enh_y'] = tsne_map[(i, 'enh')]
            plot_data.append(item)
            
    # 7. 绘图
    output_path = os.path.join(args.output_dir, 'tsne_final.png')
    plot_tsne_graph_final(pd.DataFrame(plot_data), output_path, misclassified_count, args.figsize)
    print(f"可视化完成: {output_path}")

if __name__ == "__main__":
    main()