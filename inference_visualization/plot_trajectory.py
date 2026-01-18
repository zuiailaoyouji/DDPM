"""
Figure 3: 收敛轨迹图 - 简化版 (基于起点和终点)
适配新版 CSV 格式：
- 输入: 包含 Original_Score, Best_Score, Best_Iter 的 CSV
- 输出: 连接 (0, Original) -> (Best_Iter, Best) 的射线图
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse
import numpy as np


def plot_trajectory(csv_path, output_path, label='TUM', figsize=(14, 6), csv_path_tum=None, csv_path_norm=None):
    """
    绘制简化的收敛轨迹图
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
        
        # 数据预处理
        df_tum.columns = [c.strip() for c in df_tum.columns]
        df_norm.columns = [c.strip() for c in df_norm.columns]

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # 定义绘图函数 (内部复用)
        def draw_subplot(ax, df, title, color_line, goal_text, goal_y):
            # 1. 绘制轨迹线
            # 逻辑：每个样本是一条线段，起点 (0, Original)，终点 (Best_Iter, Best)
            
            # 为了性能，如果样本太多，只采样前 500 个
            if len(df) > 500:
                df_plot = df.sample(500, random_state=42)
            else:
                df_plot = df

            max_iter_found = 0
            
            for _, row in df_plot.iterrows():
                start_y = row['Original_Score']
                end_y = row['Best_Score']
                end_x = int(row['Best_Iter'])
                
                max_iter_found = max(max_iter_found, end_x)
                
                # 绘制线段
                ax.plot([0, end_x], [start_y, end_y], 
                        color=color_line, alpha=0.15, linewidth=1.0)
                
                # 绘制终点
                ax.scatter(end_x, end_y, s=10, color=color_line, alpha=0.3)

            # 2. 装饰
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Iteration Steps")
            ax.set_xlim(left=0, right=max_iter_found + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # 强制整数刻度
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 3. 绘制目标区域 (Goal Zone)
            if goal_y > 0.5: # TUM Goal (Top)
                ax.axhspan(0.95, 1.0, color='red', alpha=0.1, label='Target Zone (>0.95)')
            else: # NORM Goal (Bottom)
                ax.axhspan(0.0, 0.05, color='green', alpha=0.1, label='Target Zone (<0.05)')

        # 左图：TUM
        draw_subplot(axes[0], df_tum, "TUM Enhancement Trajectory\n(Maximize Tumor Feature)", 
                    '#e74c3c', "Target: >0.95", 0.98)
        axes[0].set_ylabel("Tumor Confidence Score")
        
        # 右图：NORM
        draw_subplot(axes[1], df_norm, "NORM Enhancement Trajectory\n(Minimize Tumor Feature)", 
                    '#2ecc71', "Target: <0.05", 0.02)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    # ==========================
    # 场景 B: 单文件模式 -> 单图绘制
    # ==========================
    else:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ 找不到文件: {csv_path}")
            return False
            
        df.columns = [c.strip() for c in df.columns]
        
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # 根据 label 决定颜色
        color = '#e74c3c' if label == 'TUM' else '#2ecc71'
        goal_y = 0.98 if label == 'TUM' else 0.02
        title = f"{label} Enhancement Trajectory"
        
        # 采样
        if len(df) > 500:
            df_plot = df.sample(500, random_state=42)
        else:
            df_plot = df
            
        max_iter_found = 0
        
        for _, row in df_plot.iterrows():
            start_y = row['Original_Score']
            end_y = row['Best_Score']
            end_x = int(row['Best_Iter'])
            max_iter_found = max(max_iter_found, end_x)
            
            # 线段
            ax.plot([0, end_x], [start_y, end_y], color=color, alpha=0.15, linewidth=1.0)
            # 终点
            ax.scatter(end_x, end_y, s=15, color=color, alpha=0.4)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Iteration Steps")
        ax.set_ylabel("Tumor Confidence Score")
        ax.set_xlim(left=0, right=max_iter_found + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if goal_y > 0.5:
             ax.axhspan(0.95, 1.0, color='red', alpha=0.1)
        else:
             ax.axhspan(0.0, 0.05, color='green', alpha=0.1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ 简化版轨迹图已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='绘制收敛轨迹图 (简化版)')
    parser.add_argument('--csv_path', type=str, default='./results/guided_results.csv')
    parser.add_argument('--output_path', type=str, default='./inference_visualization/Figure_3_Trajectory.png')
    parser.add_argument('--label', type=str, default='TUM', choices=['TUM', 'NORM'])
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 6])
    parser.add_argument('--csv_path_tum', type=str, default=None)
    parser.add_argument('--csv_path_norm', type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.csv_path_tum and args.csv_path_norm:
        plot_trajectory(None, args.output_path, label=args.label, figsize=tuple(args.figsize),
                                  csv_path_tum=args.csv_path_tum, 
                                  csv_path_norm=args.csv_path_norm)
    else:
        plot_trajectory(args.csv_path, args.output_path, label=args.label, figsize=tuple(args.figsize))


if __name__ == "__main__":
    main()