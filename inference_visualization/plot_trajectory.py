"""
Figure 3: 收敛轨迹图 - 展示增强过程中的分数变化 (动态范围修复版)
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse


def plot_trajectory(csv_path, output_path, label='TUM', figsize=(14, 6), csv_path_tum=None, csv_path_norm=None):
    """
    绘制收敛轨迹图：展示增强过程中的分数变化
    
    改进点：
    1. 动态计算有效步数：先剔除 NaN 数据，再计算 max_step。
       即便 CSV 表头有 Iter_0 到 Iter_10，如果数据只到 Iter_4，X 轴也只会画到 4。
    2. 保持左右分栏和独立 X 轴。
    """
    
    # ==========================
    # 场景 A: 双文件模式 (TUM & NORM) -> 左右分栏绘制
    # ==========================
    if csv_path_tum is not None and csv_path_norm is not None:
        try:
            df_tum = pd.read_csv(csv_path_tum)
            df_norm = pd.read_csv(csv_path_norm)
        except FileNotFoundError as e:
            print(f"❌ 找不到文件: {e}")
            return False
        
        # 数据清洗与准备
        for df, name in [(df_tum, 'TUM'), (df_norm, 'NORM')]:
            df.columns = [c.strip() for c in df.columns]
            df['Label'] = name
            
        # 设置绘图风格
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 创建画布：1行2列
        # sharey=True: 共享 Y 轴
        # sharex=False: X 轴独立
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # 定义辅助函数：处理单个 dataframe 并画图
        def process_and_plot(df_in, ax, color, title):
            # 提取 Iter 列
            iter_cols = [c for c in df_in.columns if c.startswith('Iter_')]
            if not iter_cols:
                return
            
            # 转 Numeric
            for col in iter_cols:
                df_in[col] = pd.to_numeric(df_in[col], errors='coerce')
                
            # Melt
            df_melt = df_in.melt(
                value_vars=iter_cols,
                var_name='Iteration',
                value_name='Score'
            )
            
            # 关键修复步骤 1：剔除 Score 为 NaN 的行
            # 这样全是空的列（如 Iter_5 到 Iter_10）就会被丢弃，不参与步数计算
            df_melt = df_melt.dropna(subset=['Score'])
            
            # 如果剔除后为空，直接返回
            if df_melt.empty:
                return

            # 提取数字 'Iter_0' -> 0
            df_melt['Iter_Num'] = df_melt['Iteration'].apply(lambda x: int(x.split('_')[1]))
            
            # 关键修复步骤 2：基于有效数据计算最大步数
            max_step = df_melt['Iter_Num'].max()
            
            # 绘图
            sns.lineplot(
                x='Iter_Num',
                y='Score',
                data=df_melt,
                ax=ax,
                color=color,
                linewidth=3,
                marker='o',
                markersize=9,
                errorbar='sd'
            )
            
            ax.set_title(title, fontweight='bold', pad=15)
            ax.set_xlabel("Iteration Step")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 设置 X 轴范围 (动态适配)
            # 例如 NORM 数据清理后 max_step=4，则范围是 0-4.2
            ax.set_xlim(left=0, right=max_step + 0.2) 
            
            # 强制整数刻度
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        # 绘制左图 (TUM)
        process_and_plot(df_tum, axes[0], '#e74c3c', "TUM Samples (Maximization)")
        axes[0].set_ylabel("Average Confidence")
        
        # 绘制右图 (NORM)
        process_and_plot(df_norm, axes[1], '#2ecc71', "NORM Samples (Minimization)")
        axes[1].set_ylabel("") 
        
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    # ==========================
    # 场景 B: 单文件模式 -> 单张图绘制
    # ==========================
    else:
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
        
        iter_cols = [c for c in df.columns if c.startswith('Iter_')]
        df_filtered = df[df['Label'] == label].copy()
        
        if df_filtered.empty or not iter_cols:
            print(f"⚠️ 数据缺失或无 Iter 列")
            return False
            
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.rcParams['font.family'] = 'sans-serif'
        
        for col in iter_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
            
        df_melt = df_filtered.melt(
            value_vars=iter_cols, 
            var_name='Iteration', 
            value_name='Score'
        )
        
        # 同样应用修复逻辑
        df_melt = df_melt.dropna(subset=['Score'])
        
        if df_melt.empty:
            print("⚠️ 有效数据为空")
            return False
            
        df_melt['Iter_Num'] = df_melt['Iteration'].apply(lambda x: int(x.split('_')[1]))
        max_step = df_melt['Iter_Num'].max()
        
        plt.figure(figsize=figsize)
        color = '#e74c3c' if label == 'TUM' else '#2ecc71'
        
        ax = sns.lineplot(
            x='Iter_Num', 
            y='Score', 
            data=df_melt, 
            color=color, 
            linewidth=3,
            marker='o',
            markersize=9,
            errorbar='sd'
        )
        
        plt.title(f"Enhancement Trajectory ({label} Samples)", fontweight='bold')
        plt.xlabel("Iteration Step")
        plt.ylabel("Average Confidence")
        plt.ylim(0, 1.05)
        
        plt.xlim(left=0, right=max_step + 0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ 轨迹图已保存 (动态X轴修复版): {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='绘制收敛轨迹图')
    parser.add_argument('--csv_path', type=str, default='./results/guided_results.csv')
    parser.add_argument('--output_path', type=str, default='./inference_visualization/Figure_3_Trajectory.png')
    parser.add_argument('--label', type=str, default='TUM', choices=['TUM', 'NORM'])
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 6])
    parser.add_argument('--csv_path_tum', type=str, default=None)
    parser.add_argument('--csv_path_norm', type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.csv_path_tum and args.csv_path_norm:
        plot_trajectory(
            csv_path=None, 
            output_path=args.output_path, 
            csv_path_tum=args.csv_path_tum, 
            csv_path_norm=args.csv_path_norm,
            figsize=tuple(args.figsize)
        )
    else:
        plot_trajectory(args.csv_path, args.output_path, args.label, tuple(args.figsize))


if __name__ == "__main__":
    main()