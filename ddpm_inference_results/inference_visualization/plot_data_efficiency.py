"""
Figure 4: 数据效率分析曲线 (Data Efficiency Plot) - 完整版
展示不同训练数据比例 (1%, 5%, 10%, 50%, 100%) 下，三种实验设置的 Accuracy 变化趋势。
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.ticker as ticker

# ================= 数据录入 =================
# x轴: 数据比例
ratios = [0.01, 0.05, 0.10, 0.50, 1.00]

# 估算的实际训练样本数 (基于 Total Train ~ 592)
# 0.01:~6, 0.05:~30, 0.1:~59, 0.5:~296, 1.0:592
counts = [6, 30, 59, 296, 592]

# y轴: Accuracy (%) - 均来自你的实验数据
# Exp 1: Raw Noisy Baseline (原始图片 + 原始标签)
# Data: 0.01(67.59), 0.05(75.70), 0.1(90.38), 0.5(96.20), 1.0(94.68)
acc_noisy = [67.59, 75.70, 90.38, 96.20, 94.68]

# Exp 2: Strong Clean Baseline (原始图片 + 纠正标签)
# Data: 0.01(62.78), 0.05(78.48), 0.1(92.41), 0.5(95.70), 1.0(95.19)
acc_clean = [62.78, 78.48, 92.41, 95.70, 95.19]

# Exp 3: Ours (增强图片 + 纠正标签)
# Data: 0.01(76.20), 0.05(90.89), 0.1(89.37), 0.5(95.44), 1.0(96.46)
acc_ours  = [76.20, 90.89, 89.37, 95.44, 96.46]

# ================= 绘图配置 =================
OUTPUT_DIR = '/data/xuwen/ddpm_inference_results/inference_visualization/efficiency_curve'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置学术风格
sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=1.1)
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'

def plot_efficiency_curve():
    plt.figure(figsize=(10, 7))
    
    # 1. 绘制三条曲线
    # Line 1: Raw Noisy (灰色虚线，表示最基础的底线)
    plt.plot(ratios, acc_noisy, marker='s', markersize=7, linestyle='--', color='#7f8c8d', 
             linewidth=2, label='Exp 1: Raw Noisy Baseline', alpha=0.7)
    
    # Line 2: Strong Clean (蓝色实线，表示仅清洗标签的效果)
    plt.plot(ratios, acc_clean, marker='^', markersize=8, linestyle='-', color='#2980b9', 
             linewidth=2.5, label='Exp 2: Strong Clean Baseline')
    
    # Line 3: Ours (红色实线，表示最终效果)
    plt.plot(ratios, acc_ours, marker='o', markersize=9, linestyle='-', color='#c0392b', 
             linewidth=3, label='Exp 3: Ours (Enhanced)')

    # 2. 关键点高亮标注
    
    # --- 亮点 A: 1% 数据时的巨大提升 (Data Efficiency) ---
    # 计算提升幅度: Ours (76.20) - Clean (62.78)
    lift_1pct = acc_ours[0] - acc_clean[0]
    
    plt.annotate(f"Low Data Boost\n(1% Data)\n+{lift_1pct:.1f}%", 
                 xy=(ratios[0], acc_ours[0]), 
                 xytext=(ratios[0] * 1.3, acc_ours[0] - 2), # 文字位置微调
                 color='#c0392b', fontweight='bold', fontsize=11,
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2, connectionstyle="arc3,rad=0.2"),
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c0392b", alpha=0.9))

    # --- 亮点 B: 5% 数据时的显著优势 (补充标注) ---
    # 这里的提升也非常大 (90.89 vs 78.48, +12.4%)
    # 可以选择标注，也可以略过以免图太乱。这里简单画个圈或文字。
    plt.text(ratios[1], acc_ours[1] + 1.5, f"{acc_ours[1]:.1f}%", 
             ha='center', va='bottom', color='#c0392b', fontweight='bold', fontsize=10)

    # --- 亮点 C: 100% 数据时的最终提升 ---
    # 计算提升幅度: Ours (96.46) - Clean (95.19)
    lift_full = acc_ours[-1] - acc_clean[-1]
    
    plt.annotate(f"Final Gain\n+{lift_full:.2f}%", 
                 xy=(ratios[-1], acc_ours[-1]), 
                 xytext=(ratios[-1] * 0.6, acc_ours[-1] - 3), 
                 color='#c0392b', fontweight='bold', fontsize=12,
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c0392b", alpha=0.9))

    # 3. 坐标轴设置
    plt.xscale('log') # 使用对数坐标轴
    
    # 自定义 X 轴刻度显示
    plt.xticks(ratios, [f"{int(r*100)}%" for r in ratios])
    plt.xlabel("Training Data Ratio (Log Scale)", fontsize=13, labelpad=10)
    plt.ylabel("Test Accuracy (%)", fontsize=13, labelpad=10)
    
    # Y轴范围微调 (从 50 开始，展示更多差异)
    plt.ylim(55, 100)
    
    # 网格线优化
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 4. 图例
    plt.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='gray', fontsize=11)
    
    plt.title("Data Efficiency Analysis: Impact of Diffusion Enhancement", fontsize=15, pad=20, fontweight='bold')
    
    # 保存
    save_path = os.path.join(OUTPUT_DIR, 'data_efficiency_v3.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表已保存至: {save_path}")

if __name__ == "__main__":
    plot_efficiency_curve()