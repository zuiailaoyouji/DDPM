"""
Figure 2: 散点图 - 去歧义分析（初始分数 vs 最终分数）- 优化版
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse


def plot_scatter(csv_path, output_path, figsize=(8, 8), csv_path_tum=None, csv_path_norm=None):
    """
    绘制散点图：展示增强前后的分数变化
    
    优化点：
    1. 移除原本有歧义的单一箭头。
    2. 根据 Label (TUM/NORM) 动态绘制增强方向箭头：
       - TUM: 红色箭头向上 (Maximize)
       - NORM: 绿色箭头向下 (Minimize)
    """
    # ==========================
    # 1. 数据读取与合并
    # ==========================
    if csv_path_tum is not None and csv_path_norm is not None:
        # 双文件模式
        try:
            df_tum = pd.read_csv(csv_path_tum)
            df_norm = pd.read_csv(csv_path_norm)
        except FileNotFoundError as e:
            print(f"❌ 找不到文件: {e}")
            return False
        
        df_tum.columns = [c.strip() for c in df_tum.columns]
        df_norm.columns = [c.strip() for c in df_norm.columns]
        
        df_tum['Label'] = 'TUM'
        df_norm['Label'] = 'NORM'
        
        df = pd.concat([df_tum, df_norm], ignore_index=True)
    else:
        # 单文件模式
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ 找不到文件: {csv_path}")
            return False
        
        df.columns = [c.strip() for c in df.columns]
        
        if 'Label' not in df.columns:
            df['Label'] = df['Detected_Mode'].apply(
                lambda x: 'TUM' if 'maximize' in str(x).lower() else 'NORM'
            )
    
    # ==========================
    # 2. 绘图设置
    # ==========================
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    plt.figure(figsize=figsize)
    
    # 绘制对角线 (No Change Line)
    plt.plot([0, 1], [0, 1], ls='--', c='gray', alpha=0.5, label='No Change')
    
    # 绘制散点
    sns.scatterplot(
        x='Original_Score', 
        y='Best_Score', 
        hue='Label', 
        data=df, 
        palette={'TUM': '#e74c3c', 'NORM': '#2ecc71'}, 
        style='Label', 
        s=80, 
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.title("Disambiguation Effect: Initial vs Final Score", fontweight='bold')
    plt.xlabel("Original Score (Initial)")
    plt.ylabel("Enhanced Score (Final)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # ==========================
    # 3. 动态添加无歧义的箭头
    # ==========================
    unique_labels = df['Label'].unique()
    
    # 箭头参数
    arrow_props = dict(head_width=0.03, width=0.005, alpha=0.9, zorder=10, length_includes_head=True)
    
    # --- TUM 箭头 (Maximize: Upward) ---
    # 放置在左侧区域 (x=0.15)，表示从低分向高分
    if 'TUM' in unique_labels:
        plt.arrow(x=0.15, y=0.30, dx=0.0, dy=0.30, color='#e74c3c', **arrow_props)
        plt.text(0.19, 0.45, "TUM Goal\n(Maximize)", 
                 color='#e74c3c', fontsize=11, fontweight='bold', va='center', ha='left')

    # --- NORM 箭头 (Minimize: Downward) ---
    # 放置在右侧区域 (x=0.85)，表示从高分向低分
    if 'NORM' in unique_labels:
        plt.arrow(x=0.85, y=0.70, dx=0.0, dy=-0.30, color='#2ecc71', **arrow_props)
        plt.text(0.81, 0.55, "NORM Goal\n(Minimize)", 
                 color='#2ecc71', fontsize=11, fontweight='bold', va='center', ha='right')
    
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 散点图已保存 (优化箭头版): {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='绘制散点图：去歧义分析')
    parser.add_argument('--csv_path', type=str, 
                       default='./results/guided_inference/guided_results.csv',
                       help='CSV 文件路径')
    parser.add_argument('--output_path', type=str,
                       default='./inference_visualization/Figure_2_Scatter.png',
                       help='输出图片路径')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 8],
                       help='图片尺寸 (宽 高)')
    # 双文件模式参数
    parser.add_argument('--csv_path_tum', type=str, default=None)
    parser.add_argument('--csv_path_norm', type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 自动识别模式
    if args.csv_path_tum and args.csv_path_norm:
        plot_scatter(None, args.output_path, tuple(args.figsize), 
                    csv_path_tum=args.csv_path_tum, 
                    csv_path_norm=args.csv_path_norm)
    else:
        plot_scatter(args.csv_path, args.output_path, tuple(args.figsize))


if __name__ == "__main__":
    main()