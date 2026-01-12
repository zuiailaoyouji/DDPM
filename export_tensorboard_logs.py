"""
TensorBoard 日志导出脚本
将 TensorBoard events 文件导出为 CSV 表格和 PNG 图像
适合用于毕设论文的图表生成
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 设置字体（使用默认字体，避免字体丢失问题）
# 使用英文标签，确保在所有系统上都能正常显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def export_scalar_to_csv(log_dir, scalar_name, output_path, smooth_alpha=0.1):
    """
    导出标量数据到 CSV 文件
    
    Args:
        log_dir: TensorBoard 日志目录
        scalar_name: 标量标签名称
        output_path: 输出 CSV 文件路径
        smooth_alpha: 平滑系数 (0-1)
    """
    # 加载 TensorBoard 日志
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # 检查标量是否存在
    available_scalars = event_acc.Tags().get('scalars', [])
    if scalar_name not in available_scalars:
        print(f"⚠️ 警告: 未找到标量 '{scalar_name}'")
        return False
    
    # 提取数据
    events = event_acc.Scalars(scalar_name)
    df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
    
    # 平滑处理
    df['value_smooth'] = df['value'].ewm(alpha=smooth_alpha).mean()
    
    # 保存 CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig 支持 Excel 打开
    print(f"✓ CSV 已保存: {output_path} (数据点数: {len(df)})")
    return True


def plot_scalar(log_dir, scalar_name, output_path, smooth_alpha=0.1, show_raw=False, figsize=(10, 6), dpi=300):
    """
    绘制标量数据图像
    
    Args:
        log_dir: TensorBoard 日志目录
        scalar_name: 标量标签名称
        output_path: 输出图像文件路径
        smooth_alpha: 平滑系数 (0-1)
        show_raw: 是否同时显示原始数据
        figsize: 图像尺寸 (宽, 高)
        dpi: 图像分辨率
    """
    # 加载 TensorBoard 日志
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # 检查标量是否存在
    available_scalars = event_acc.Tags().get('scalars', [])
    if scalar_name not in available_scalars:
        print(f"⚠️ 警告: 未找到标量 '{scalar_name}'")
        return False
    
    # 提取数据
    events = event_acc.Scalars(scalar_name)
    df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
    
    # 平滑处理
    df['value_smooth'] = df['value'].ewm(alpha=smooth_alpha).mean()
    
    # 创建图像
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制原始数据（可选）
    if show_raw:
        ax.plot(df['step'], df['value'], alpha=0.3, label='Raw Data', color='gray', linewidth=0.5)
    
    # 绘制平滑曲线
    ax.plot(df['step'], df['value_smooth'], label='Smoothed' if show_raw else None, linewidth=2)
    
    # 设置标签和标题
    ylabel = scalar_name.split('/')[-1]  # 使用标签的最后一部分作为 Y 轴标签
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(scalar_name, fontsize=14, fontweight='bold')
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle='--')
    if show_raw:
        ax.legend(fontsize=10)
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✓ 图像已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='从 TensorBoard 日志导出 CSV 和 PNG 图像')
    parser.add_argument('--log_dir', type=str, required=True, help='TensorBoard 日志目录路径')
    parser.add_argument('--output_dir', type=str, default='./exported_logs', help='输出目录')
    parser.add_argument('--scalar', type=str, nargs='+', help='要导出的标量名称（可指定多个，如: Train/Total_Loss Train/MSE_Loss）')
    parser.add_argument('--all', action='store_true', help='导出所有标量')
    parser.add_argument('--no_csv', action='store_true', help='不导出 CSV')
    parser.add_argument('--no_images', action='store_true', help='不导出图像')
    parser.add_argument('--smooth_alpha', type=float, default=0.1, help='平滑系数 (0-1)，默认 0.1')
    parser.add_argument('--show_raw', action='store_true', help='在图像中同时显示原始数据')
    
    args = parser.parse_args()
    
    # 检查日志目录
    if not os.path.exists(args.log_dir):
        print(f"❌ 错误: 日志目录不存在: {args.log_dir}")
        sys.exit(1)
    
    # 加载日志以获取可用标量
    print(f"正在加载 TensorBoard 日志: {args.log_dir}")
    event_acc = EventAccumulator(args.log_dir)
    event_acc.Reload()
    available_scalars = event_acc.Tags().get('scalars', [])
    print(f"✓ 找到 {len(available_scalars)} 个标量")
    
    # 确定要导出的标量列表
    if args.all:
        scalar_names = available_scalars
        print(f"将导出所有 {len(scalar_names)} 个标量")
    elif args.scalar:
        scalar_names = [s for s in args.scalar if s in available_scalars]
        not_found = [s for s in args.scalar if s not in available_scalars]
        if not_found:
            print(f"⚠️ 警告: 以下标量未找到: {not_found}")
        if not scalar_names:
            print("❌ 错误: 没有有效的标量可导出")
            sys.exit(1)
    else:
        # 默认导出常用标量
        common_scalars = ['Train/Total_Loss', 'Train/MSE_Loss', 'Train/L1_Diff']
        scalar_names = [s for s in common_scalars if s in available_scalars]
        if not scalar_names:
            scalar_names = available_scalars[:5]  # 至少导出前 5 个
        print(f"未指定标量，将导出默认标量: {scalar_names}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 导出每个标量
    print(f"\n开始导出到: {args.output_dir}")
    print("-" * 60)
    
    success_count = 0
    for scalar_name in scalar_names:
        # 生成文件名（将 '/' 替换为 '_'）
        safe_name = scalar_name.replace('/', '_').replace('\\', '_')
        
        success = True
        
        # 导出 CSV
        if not args.no_csv:
            csv_path = os.path.join(args.output_dir, f"{safe_name}.csv")
            if not export_scalar_to_csv(args.log_dir, scalar_name, csv_path, args.smooth_alpha):
                success = False
        
        # 导出图像
        if not args.no_images:
            img_path = os.path.join(args.output_dir, f"{safe_name}.png")
            if not plot_scalar(args.log_dir, scalar_name, img_path, args.smooth_alpha, args.show_raw):
                success = False
        
        if success:
            success_count += 1
    
    # 打印总结
    print("-" * 60)
    print(f"\n导出完成: {success_count}/{len(scalar_names)} 个标量成功导出")
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()

