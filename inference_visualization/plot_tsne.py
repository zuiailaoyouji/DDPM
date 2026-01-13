"""
t-SNE 特征空间可视化 - 对比原始图像与增强图像的特征分布
适配文件结构：
- 原始图: {orig_dir}/*.png
- 增强图: {enh_dir}/{subfolder}/BEST_*.png
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import random


class SimpleDataset(Dataset):
    """简单的图像数据集"""
    def __init__(self, file_paths, labels, transform):
        self.files = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        if img is None:
            # 如果读取失败，返回黑色图像
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, self.labels[idx]


def extract_features(loader, model, device):
    """从数据加载器中提取特征"""
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="提取特征"):
            imgs = imgs.to(device)
            # 提取特征 (ResNet50 fc层之前的输出)
            feats = model(imgs).squeeze()
            if len(feats.shape) == 1:
                feats = feats.unsqueeze(0)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.concatenate(features), np.array(labels)


def collect_original_images(orig_dir, sample_size=500):
    """
    从原始图像目录收集图片
    支持直接包含图片文件的目录
    """
    all_files = []
    if os.path.isdir(orig_dir):
        # 查找所有图片文件
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        for pattern in patterns:
            all_files.extend(glob(os.path.join(orig_dir, pattern)))
            all_files.extend(glob(os.path.join(orig_dir, '**', pattern), recursive=True))
    
    # 去重并随机采样
    all_files = list(set(all_files))
    random.shuffle(all_files)
    return all_files[:sample_size]


def collect_enhanced_images(enh_dir, sample_size=500):
    """
    从增强图像目录收集 BEST 图片
    支持结构: {enh_dir}/{subfolder}/BEST_*.png
    """
    all_files = []
    
    if not os.path.exists(enh_dir):
        return all_files
    
    # 获取所有子文件夹
    subfolders = [d for d in os.listdir(enh_dir) 
                  if os.path.isdir(os.path.join(enh_dir, d))]
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(enh_dir, subfolder)
        # 查找 BEST_*.png 文件
        best_files = glob(os.path.join(subfolder_path, "BEST_*.png"))
        all_files.extend(best_files)
    
    # 随机采样
    random.shuffle(all_files)
    return all_files[:sample_size]


def run_tsne_analysis(name, tum_files, norm_files, device, sample_size=500):
    """
    运行 t-SNE 分析
    
    Args:
        name: 数据集名称 (如 'Original' 或 'Enhanced')
        tum_files: TUM 图片文件路径列表
        norm_files: NORM 图片文件路径列表
        device: 计算设备
        sample_size: 每类采样数量
    """
    print(f"\n正在处理 {name} 数据...")
    
    # 随机采样
    random.shuffle(tum_files)
    random.shuffle(norm_files)
    tum_files = tum_files[:sample_size]
    norm_files = norm_files[:sample_size]
    
    if len(tum_files) == 0 and len(norm_files) == 0:
        print(f"⚠️ {name} 数据为空，跳过")
        return None
    
    print(f"  TUM 样本数: {len(tum_files)}")
    print(f"  NORM 样本数: {len(norm_files)}")
    
    all_files = tum_files + norm_files
    # Label: 0 for NORM, 1 for TUM
    all_labels = [0] * len(norm_files) + [1] * len(tum_files)
    
    # 定义模型 (使用预训练的 ResNet50)
    # 兼容不同版本的 PyTorch
    try:
        resnet = models.resnet50(weights='IMAGENET1K_V2')
    except TypeError:
        # 旧版本使用 pretrained=True
        resnet = models.resnet50(pretrained=True)
    # 把 fc 层替换为 Identity，这样输出就是 avgpool 之后的 2048 维向量
    resnet.fc = nn.Identity()
    resnet.to(device)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleDataset(all_files, all_labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 提取特征
    print("  提取特征中...")
    features, labels = extract_features(loader, resnet, device)
    
    # t-SNE 降维
    print("  运行 t-SNE (这可能需要几分钟)...")
    # 兼容不同版本的 sklearn
    try:
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', max_iter=1000)
    except TypeError:
        # 旧版本可能使用 n_iter
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', n_iter=1000)
    tsne_results = tsne.fit_transform(features)
    
    # 返回 DataFrame，包含文件路径信息用于配对
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'Label': ['NORM' if l == 0 else 'TUM' for l in labels],
        'Type': name,
        'File_Path': all_files  # 保存文件路径用于配对
    })
    
    return df


def match_original_enhanced_pairs(df_orig, df_enh, max_pairs=50):
    """
    匹配原始图像和增强图像的配对
    
    Args:
        df_orig: 原始图像的 DataFrame（已重置索引）
        df_enh: 增强图像的 DataFrame（已重置索引）
        max_pairs: 最大配对数量
    
    Returns:
        pairs: [(orig_pos, enh_pos, label), ...] 配对列表（位置索引）
    """
    pairs = []
    
    # 按标签分组
    for label in ['TUM', 'NORM']:
        orig_subset = df_orig[df_orig['Label'] == label].copy()
        enh_subset = df_enh[df_enh['Label'] == label].copy()
        
        if len(orig_subset) == 0 or len(enh_subset) == 0:
            continue
        
        # 随机采样配对（简化处理：随机匹配）
        # 实际应用中，可以根据文件名或特征相似度进行更精确的匹配
        n_pairs = min(len(orig_subset), len(enh_subset), max_pairs // 2)
        
        # 获取位置索引（reset_index 后的位置）
        orig_positions = orig_subset.index.tolist()
        enh_positions = enh_subset.index.tolist()
        
        random.shuffle(orig_positions)
        random.shuffle(enh_positions)
        
        for i in range(n_pairs):
            pairs.append((orig_positions[i], enh_positions[i], label))
    
    return pairs


def plot_tsne_with_trajectories(df_orig, df_enh, pairs, output_path, figsize=(14, 12)):
    """
    绘制带配对连线的 t-SNE 图
    
    Args:
        df_orig: 原始图像 DataFrame
        df_enh: 增强图像 DataFrame
        pairs: 配对列表 [(orig_idx, enh_idx, label), ...]
        output_path: 输出路径
        figsize: 图片尺寸
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # ============================================
    # 1. 绘制原始图像（背景层：半透明、空心圆、颜色较浅）
    # ============================================
    for label in ['TUM', 'NORM']:
        orig_data = df_orig[df_orig['Label'] == label]
        if len(orig_data) > 0:
            color = '#ff9999' if label == 'TUM' else '#99ccff'  # 较浅的颜色
            ax.scatter(
                orig_data['x'], orig_data['y'],
                c=color, marker='o', s=40, alpha=0.3,
                edgecolors=color, linewidths=1.5, facecolors='none',
                label=f'{label} (Original)',
                zorder=1
            )
    
    # ============================================
    # 2. 绘制配对连线（箭头）
    # ============================================
    arrow_alpha = 0.6
    arrow_width = 0.001
    
    for orig_pos, enh_pos, label in pairs:
        if orig_pos >= len(df_orig) or enh_pos >= len(df_enh):
            continue
        
        orig_point = df_orig.iloc[orig_pos]
        enh_point = df_enh.iloc[enh_pos]
        
        # 根据标签选择颜色
        if label == 'TUM':
            arrow_color = '#e74c3c'  # 红色
        else:
            arrow_color = '#2ecc71'  # 绿色
        
        # 绘制箭头
        ax.annotate(
            '', xy=(enh_point['x'], enh_point['y']),
            xytext=(orig_point['x'], orig_point['y']),
            arrowprops=dict(
                arrowstyle='->', color=arrow_color, lw=1.5,
                alpha=arrow_alpha, zorder=2
            )
        )
    
    # ============================================
    # 3. 绘制增强图像（前景层：不透明、实心星形、颜色较深）
    # ============================================
    for label in ['TUM', 'NORM']:
        enh_data = df_enh[df_enh['Label'] == label]
        if len(enh_data) > 0:
            color = '#e74c3c' if label == 'TUM' else '#2ecc71'  # 较深的颜色
            ax.scatter(
                enh_data['x'], enh_data['y'],
                c=color, marker='X', s=150, alpha=0.9,
                edgecolors='white', linewidths=1,
                label=f'{label} (Enhanced)',
                zorder=3
            )
    
    # ============================================
    # 4. 设置标题和标签
    # ============================================
    ax.set_title("t-SNE Feature Space with Trajectory Arrows", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 图例
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已绘制 {len(pairs)} 条配对连线")


def main():
    parser = argparse.ArgumentParser(description='t-SNE 特征空间可视化')
    parser.add_argument('--orig_tum_dir', type=str,
                       default='/data/xuwen/NCT-CRC-HE-100K/TUM',
                       help='原始 TUM 图片目录')
    parser.add_argument('--orig_norm_dir', type=str,
                       default='/data/xuwen/NCT-CRC-HE-100K/NORM',
                       help='原始 NORM 图片目录')
    parser.add_argument('--enh_tum_dir', type=str,
                       default='/data/xuwen/ddpm_inference_results/TUM',
                       help='增强 TUM 图片目录（包含子文件夹）')
    parser.add_argument('--enh_norm_dir', type=str,
                       default='/data/xuwen/ddpm_inference_results/NORM',
                       help='增强 NORM 图片目录（包含子文件夹）')
    parser.add_argument('--sample_size', type=int, default=500,
                       help='每类采样数量')
    parser.add_argument('--output_dir', type=str,
                       default='./inference_visualization/tsne_results',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--only_original', action='store_true',
                       help='仅分析原始图像，不分析增强图像')
    parser.add_argument('--trajectory_pairs', type=int, default=50,
                       help='配对连线数量（随机抽取）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 收集原始图像
    print("\n" + "="*50)
    print("收集原始图像...")
    orig_tum_files = collect_original_images(args.orig_tum_dir, args.sample_size)
    orig_norm_files = collect_original_images(args.orig_norm_dir, args.sample_size)
    print(f"  原始 TUM: {len(orig_tum_files)} 张")
    print(f"  原始 NORM: {len(orig_norm_files)} 张")
    
    # 运行原始数据的 t-SNE
    df_orig = run_tsne_analysis('Original', orig_tum_files, orig_norm_files, device, args.sample_size)
    
    # 保存原始 DataFrame 的引用（用于配对）
    df_orig_for_pairing = df_orig.copy() if df_orig is not None else None
    df_all = df_orig.copy() if df_orig is not None else None
    
    # 收集增强图像（如果不需要可以跳过）
    if not args.only_original:
        print("\n" + "="*50)
        print("收集增强图像...")
        enh_tum_files = collect_enhanced_images(args.enh_tum_dir, args.sample_size)
        enh_norm_files = collect_enhanced_images(args.enh_norm_dir, args.sample_size)
        print(f"  增强 TUM: {len(enh_tum_files)} 张")
        print(f"  增强 NORM: {len(enh_norm_files)} 张")
        
        if len(enh_tum_files) > 0 or len(enh_norm_files) > 0:
            # 运行增强数据的 t-SNE
            df_enh = run_tsne_analysis('Enhanced', enh_tum_files, enh_norm_files, device, args.sample_size)
            
            if df_enh is not None and df_orig is not None:
                df_all = pd.concat([df_orig, df_enh], ignore_index=True)
            elif df_enh is not None:
                df_all = df_enh
        else:
            print("⚠️ 未找到增强图像，仅使用原始图像")
    
    if df_all is None or len(df_all) == 0:
        print("❌ 没有可用的数据进行分析")
        return
    
    # 画图
    print("\n" + "="*50)
    print("生成可视化图表...")
    
    # 如果有原始和增强两种数据，使用配对连线模式
    if (df_orig_for_pairing is not None and not args.only_original and 
        'Enhanced' in df_all['Type'].values):
        df_enh = df_all[df_all['Type'] == 'Enhanced'].copy()
        df_orig_plot = df_all[df_all['Type'] == 'Original'].copy()
        
        # 重置索引以便配对匹配
        df_orig_plot = df_orig_plot.reset_index(drop=True)
        df_enh = df_enh.reset_index(drop=True)
        
        # 匹配配对
        print(f"  匹配配对（最多 {args.trajectory_pairs} 对）...")
        pairs = match_original_enhanced_pairs(df_orig_plot, df_enh, args.trajectory_pairs)
        
        if len(pairs) > 0:
            # 使用配对连线模式
            output_path = os.path.join(args.output_dir, 'tsne_plot_with_trajectories.png')
            plot_tsne_with_trajectories(df_orig_plot, df_enh, pairs, output_path)
        else:
            print("  ⚠️ 无法匹配配对，使用标准模式")
            # 回退到标准模式
            output_path = os.path.join(args.output_dir, 'tsne_plot.png')
            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                data=df_all, x='x', y='y', hue='Label', style='Type',
                palette={'TUM': '#e74c3c', 'NORM': '#2ecc71'},
                alpha=0.6, s=50, edgecolors='w', linewidth=0.5
            )
            plt.title("t-SNE Feature Space Visualization", fontsize=16, fontweight='bold')
            plt.xlabel("t-SNE Component 1", fontsize=12)
            plt.ylabel("t-SNE Component 2", fontsize=12)
            plt.legend(title='Label', title_fontsize=12, fontsize=10)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    else:
        # 标准模式（只有一种类型或只有原始数据）
        output_path = os.path.join(args.output_dir, 'tsne_plot.png')
        plt.figure(figsize=(12, 10))
        
        if 'Type' in df_all.columns and df_all['Type'].nunique() > 1:
            sns.scatterplot(
                data=df_all, x='x', y='y', hue='Label', style='Type',
                palette={'TUM': '#e74c3c', 'NORM': '#2ecc71'},
                alpha=0.6, s=50, edgecolors='w', linewidth=0.5
            )
        else:
            sns.scatterplot(
                data=df_all, x='x', y='y', hue='Label',
                palette={'TUM': '#e74c3c', 'NORM': '#2ecc71'},
                alpha=0.6, s=50, edgecolors='w', linewidth=0.5
            )
        
        plt.title("t-SNE Feature Space Visualization", fontsize=16, fontweight='bold')
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.legend(title='Label', title_fontsize=12, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 保存数据
    csv_path = os.path.join(args.output_dir, 'tsne_data.csv')
    df_all.to_csv(csv_path, index=False)
    
    print(f"\n✓ 完成！")
    print(f"  图表已保存至: {output_path}")
    print(f"  数据已保存至: {csv_path}")
    print("="*50)


if __name__ == "__main__":
    main()

