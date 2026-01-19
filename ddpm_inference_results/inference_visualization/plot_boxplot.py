"""
Figure 1: 箱线图 - 对比增强前后的置信度分数
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse


def plot_boxplot(csv_path, output_path, figsize=(8, 6), csv_path_tum=None, csv_path_norm=None):
    """
    绘制箱线图：对比增强前后的置信度分数
    
    Args:
        csv_path: CSV 文件路径（单个文件模式，可选）
        output_path: 输出图片路径
        figsize: 图片尺寸
        csv_path_tum: TUM 数据 CSV 文件路径（双文件模式，可选）
        csv_path_norm: NORM 数据 CSV 文件路径（双文件模式，可选）
    """
    # 读取数据
    if csv_path_tum is not None and csv_path_norm is not None:
        # 双文件模式
        try:
            df_tum = pd.read_csv(csv_path_tum)
            df_norm = pd.read_csv(csv_path_norm)
        except FileNotFoundError as e:
            print(f"❌ 找不到文件: {e}")
            return False
        
        # 简单的列名清洗
        df_tum.columns = [c.strip() for c in df_tum.columns]
        df_norm.columns = [c.strip() for c in df_norm.columns]
        
        # 添加标签
        df_tum['Label'] = 'TUM'
        df_norm['Label'] = 'NORM'
        
        # 合并数据
        df = pd.concat([df_tum, df_norm], ignore_index=True)
    else:
        # 单文件模式
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"❌ 找不到文件: {csv_path}")
            return False
        
        # 简单的列名清洗（防止CSV有空格）
        df.columns = [c.strip() for c in df.columns]
        
        # 自动推断 Label (如果 CSV 里没有 Label 列，根据 Mode 推断)
        if 'Label' not in df.columns:
            # 如果 Mode是 maximize -> TUM, minimize -> NORM
            df['Label'] = df['Detected_Mode'].apply(
                lambda x: 'TUM' if 'maximize' in str(x).lower() else 'NORM'
            )
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 构建绘图数据 (Long Format)
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({
            'Label': row['Label'], 
            'Stage': 'Original', 
            'Score': row['Original_Score']
        })
        plot_data.append({
            'Label': row['Label'], 
            'Stage': 'Enhanced', 
            'Score': row['Best_Score']
        })
    df_plot = pd.DataFrame(plot_data)
    
    # 绘图
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        x='Label', 
        y='Score', 
        hue='Stage', 
        data=df_plot, 
        palette=['#bdc3c7', '#e74c3c'], 
        width=0.5, 
        fliersize=3
    )
    
    plt.title("Confidence Score Improvement (TUM vs NORM)", fontweight='bold')
    plt.ylim(0.00, 1.00)
    plt.xlabel("")
    plt.ylabel("HoVer-Net Confidence")
    plt.legend(title="Stage", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 箱线图已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='绘制箱线图：对比增强前后的置信度分数')
    parser.add_argument('--csv_path', type=str, 
                       default='./results/guided_inference/guided_results.csv',
                       help='CSV 文件路径')
    parser.add_argument('--output_path', type=str,
                       default='./inference_visualization/Figure_1_Boxplot.png',
                       help='输出图片路径')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 6],
                       help='图片尺寸 (宽 高)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    plot_boxplot(args.csv_path, args.output_path, tuple(args.figsize))


if __name__ == "__main__":
    main()

