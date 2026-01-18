"""
Figure 2: 散点图 - 模型中心视角 (Model-Centric View)
逻辑调整：
1. 假设：模型判断 (Mode) 是正确的 Ground Truth。
2. 颜色：由 Mode 决定 (Maximize=红, Minimize=绿)。
3. 高亮：标记出"模型与人工标签不一致"的样本，视为【AI 纠正的标签错误】。
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np


def plot_scatter(csv_path, output_path, figsize=(8, 8), csv_path_tum=None, csv_path_norm=None):
    """
    绘制散点图：以模型判断为准
    """
    # ==========================
    # 1. 数据读取与合并
    # ==========================
    if csv_path_tum is not None and csv_path_norm is not None:
        try:
            df_tum = pd.read_csv(csv_path_tum)
            df_norm = pd.read_csv(csv_path_norm)
        except FileNotFoundError as e:
            print(f"❌ 找不到文件: {e}")
            return False
        
        df_tum.columns = [c.strip() for c in df_tum.columns]
        df_norm.columns = [c.strip() for c in df_norm.columns]
        
        # 这里的 Label 依然代表"人为标签" (Human Label)
        df_tum['Human_Label'] = 'TUM'
        df_norm['Human_Label'] = 'NORM'
        
        df = pd.concat([df_tum, df_norm], ignore_index=True)
    else:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ 找不到文件: {csv_path}")
            return False
        
        df.columns = [c.strip() for c in df.columns]
        
        # 提取人为标签
        if 'Filename' in df.columns:
            def get_label(fname):
                if 'TUM' in str(fname): return 'TUM'
                if 'NORM' in str(fname): return 'NORM'
                return 'Unknown'
            df['Human_Label'] = df['Filename'].apply(get_label)
            df = df[df['Human_Label'] != 'Unknown']
        else:
            # 如果没有 Filename，这个逻辑可能失效，需谨慎
            print("⚠️ 警告：无法从文件名提取 Human Label，跳过高亮逻辑")
            df['Human_Label'] = 'Unknown'

    # ==========================
    # 2. 确定模型判定 (Model Decision)
    # ==========================
    if 'Mode' not in df.columns:
        print("❌ 错误：CSV 中必须包含 'Mode' 列才能使用模型中心视角")
        return False

    # 标准化 Mode
    df['Mode'] = df['Mode'].astype(str).str.lower()
    
    # 定义 Model_Decision 列用于绘图颜色
    # maximize -> Predicted TUM
    # minimize -> Predicted NORM
    df['Model_Decision'] = df['Mode'].apply(
        lambda x: 'Predicted TUM' if 'maximize' in x else 'Predicted NORM'
    )
    
    # ==========================
    # 3. 检测标签冲突 (Label Mismatch)
    # ==========================
    # 冲突定义：Human Label 与 Model Decision 不符
    # Case 1: Human=TUM, Model=Minimize (Predicted NORM) -> AI 认为人类标错了 (假阳性)
    # Case 2: Human=NORM, Model=Maximize (Predicted TUM) -> AI 认为人类标错了 (假阴性)
    
    condition_1 = (df['Human_Label'] == 'TUM') & (df['Model_Decision'] == 'Predicted NORM')
    condition_2 = (df['Human_Label'] == 'NORM') & (df['Model_Decision'] == 'Predicted TUM')
    
    df['is_correction'] = condition_1 | condition_2
    num_corrections = df['is_correction'].sum()
    print(f"检测到 {num_corrections} 个纠正样本 (Label Mismatch)")

    # ==========================
    # 4. 绘图设置
    # ==========================
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    plt.figure(figsize=figsize)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], ls='--', c='gray', alpha=0.5, label='No Change', zorder=1)
    
    # 定义颜色：基于模型的判定
    # Predicted TUM -> Red, Predicted NORM -> Green
    palette = {'Predicted TUM': '#e74c3c', 'Predicted NORM': '#2ecc71'}
    
    # A. 绘制基础散点 (颜色由模型决定)
    # 注意：这里不再区分形状，统一用圆形，强调"模型眼中的世界是纯净的"
    sns.scatterplot(
        x='Original_Score', 
        y='Best_Score', 
        hue='Model_Decision',  # 关键改变：颜色由模型决定
        data=df, 
        palette=palette, 
        style='Model_Decision',
        markers={'Predicted TUM': 'o', 'Predicted NORM': 'o'}, 
        s=80, 
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5,
        zorder=2
    )
    
    # B. 绘制 AI 纠正标记 (Highlights)
    # 对那些有人类标签冲突的样本，画上黑圈
    correction_df = df[df['is_correction']]
    if len(correction_df) > 0:
        plt.scatter(
            x=correction_df['Original_Score'],
            y=correction_df['Best_Score'],
            s=90,               
            facecolors='none',   
            edgecolors='black',  # 黑圈
            linewidth=1.5,
            label='Label Mismatch', # 图例改名
            zorder=3
        )
    
    plt.title("AI-Driven Refinement: Model Decision vs Initial Score", fontweight='bold')
    plt.xlabel("Original Score (Initial Confidence)")
    plt.ylabel("Enhanced Score (Model Confidence)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # ==========================
    # 5. 动态添加箭头
    # ==========================
    unique_decisions = df['Model_Decision'].unique()
    arrow_props = dict(head_width=0.03, width=0.005, alpha=0.9, zorder=10, length_includes_head=True)
    
    if 'Predicted TUM' in unique_decisions:
        plt.arrow(x=0.15, y=0.30, dx=0.0, dy=0.30, color='#e74c3c', **arrow_props)
        plt.text(0.19, 0.45, "Enhance to TUM", 
                 color='#e74c3c', fontsize=11, fontweight='bold', va='center', ha='left')

    if 'Predicted NORM' in unique_decisions:
        plt.arrow(x=0.85, y=0.70, dx=0.0, dy=-0.30, color='#2ecc71', **arrow_props)
        plt.text(0.81, 0.55, "Enhance to NORM", 
                 color='#2ecc71', fontsize=11, fontweight='bold', va='center', ha='right')
    
    plt.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 散点图已保存 (模型中心版): {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='绘制散点图：模型中心视角')
    parser.add_argument('--csv_path', type=str, 
                       default='./results/guided_inference/guided_results.csv',
                       help='CSV 文件路径')
    parser.add_argument('--output_path', type=str,
                       default='./inference_visualization/Figure_2_Scatter_ModelCentric.png',
                       help='输出图片路径')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 8],
                       help='图片尺寸 (宽 高)')
    parser.add_argument('--csv_path_tum', type=str, default=None)
    parser.add_argument('--csv_path_norm', type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.csv_path_tum and args.csv_path_norm:
        plot_scatter(None, args.output_path, tuple(args.figsize), 
                                  csv_path_tum=args.csv_path_tum, 
                                  csv_path_norm=args.csv_path_norm)
    else:
        plot_scatter(args.csv_path, args.output_path, tuple(args.figsize))


if __name__ == "__main__":
    main()