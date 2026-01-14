import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# ================= 数据录入 =================
# x轴: 数据比例 (0.01 = 1%, 1.0 = 100%)
ratios = [0.01, 0.05, 0.10, 0.50, 1.00]
# 对应的实际图片数量 (TUM+NORM)
counts = [15, 78, 157, 789, 1974]

# Accuracy 数据 (根据你提供的数据填入)
# Baseline: 50.13 -> 76.96 -> 92.91 -> 97.72 -> 99.75
baseline_acc = [50.13, 76.96, 92.91, 97.72, 99.75]

# Ours: 75.19 -> 89.11 -> 91.90 -> 96.71 -> 99.24
ours_acc = [75.19, 89.11, 91.90, 96.71, 99.24]

# ================= 绘图配置 =================
OUTPUT_DIR = '/data/xuwen/ddpm_inference_results/inference_visualization/efficiency_curve'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置学术风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
# 使用更通用的字体设置，避免字体缺失问题
plt.rcParams['font.family'] = 'sans-serif'
# 尝试多个字体，如果都找不到就使用默认字体
try:
    import matplotlib.font_manager as fm
    # 获取系统可用的sans-serif字体
    available_fonts = [f.name for f in fm.fontManager.ttflist if 'sans' in f.name.lower() or 'arial' in f.name.lower()]
    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts[:3] + ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']

plt.figure(figsize=(10, 6))

# 1. 绘制曲线
plt.plot(ratios, baseline_acc, 'o--', color='#95a5a6', linewidth=2, markersize=8, label='Baseline (Scratch)')
plt.plot(ratios, ours_acc, '^-', color='#e74c3c', linewidth=3, markersize=10, label='Ours (Diff-Enhancement)')

# 2. 填充“收益区域” (Highlight Zone)
# 只填充 Ours > Baseline 的部分 (前两个点)
x_fill = ratios[:3] # 0.01, 0.05, 0.10
y1_fill = baseline_acc[:3]
y2_fill = ours_acc[:3]
plt.fill_between(x_fill, y1_fill, y2_fill, where=[o > b for o, b in zip(y2_fill, y1_fill)],
                 color='#e74c3c', alpha=0.15, interpolate=True)

# 3. 添加标注 (Text Annotations)
# 计算提升值
improve_1 = ours_acc[0] - baseline_acc[0]  # 75.19 - 50.13 = 25.06
improve_5 = ours_acc[1] - baseline_acc[1]  # 89.11 - 76.96 = 12.15

# 计算y轴范围，用于动态调整标注偏移
y_range = max(max(ours_acc), max(baseline_acc)) - min(min(ours_acc), min(baseline_acc))
offset_1 = y_range * 0.08  # 1%处的偏移量（y轴范围的8%）
offset_5 = y_range * 0.06  # 5%处的偏移量（y轴范围的6%）

# 在 1% 处标注提升 - 使用 annotate 可以更好地控制位置
# 标注在数据点上方，避免被线遮挡
plt.annotate(f"+{improve_1:.1f}%", 
             xy=(ratios[0], ours_acc[0]),  # 标注点位置
             xytext=(ratios[0], ours_acc[0] + offset_1),  # 文本位置（点上方动态偏移）
             color='#e74c3c', fontweight='bold', fontsize=11,
             ha='center', va='bottom',  # 水平居中，垂直底部对齐
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.8, linewidth=1),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, alpha=0.7, connectionstyle='arc3,rad=0'))

# 在 5% 处标注提升
plt.annotate(f"+{improve_5:.1f}%", 
             xy=(ratios[1], ours_acc[1]),  # 标注点位置
             xytext=(ratios[1], ours_acc[1] + offset_5),  # 文本位置（点上方动态偏移）
             color='#e74c3c', fontweight='bold', fontsize=11,
             ha='center', va='bottom',  # 水平居中，垂直底部对齐
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.8, linewidth=1),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, alpha=0.7, connectionstyle='arc3,rad=0'))

# 4. 坐标轴设置
plt.xscale('log') # 关键！使用对数坐标，否则 1% 和 5% 挤在一起看不清
plt.xlabel("Training Data Ratio (Log Scale)", fontweight='bold')
plt.ylabel("Test Accuracy (%)", fontweight='bold')
# 调整y轴范围，为标注留出空间
plt.ylim(40, 110)

# 自定义 X 轴刻度显示
plt.xticks(ratios, [f"{int(r*100)}%\n(N={c})" for r, c in zip(ratios, counts)])

# 5. 装饰
plt.title("Data Efficiency Analysis: Accuracy vs. Data Scale", fontweight='bold', pad=20)
plt.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
plt.grid(True, which="both", ls="-", alpha=0.2)

# 6. 保存
save_path = os.path.join(OUTPUT_DIR, 'data_efficiency_curve.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')

print(f"✅ 图表已生成: {save_path}")