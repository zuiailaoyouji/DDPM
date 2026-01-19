"""
批量生成所有可视化图表
支持单文件模式或双文件模式（TUM 和 NORM 分开）
"""
import os
import argparse
from plot_boxplot import plot_boxplot
from plot_scatter import plot_scatter
from plot_trajectory import plot_trajectory


def plot_all(csv_path=None, output_dir='./inference_visualization', 
             csv_path_tum=None, csv_path_norm=None):
    """
    生成所有可视化图表
    
    Args:
        csv_path: CSV 文件路径（单文件模式，可选）
        output_dir: 输出目录
        csv_path_tum: TUM 数据 CSV 文件路径（双文件模式，可选）
        csv_path_norm: NORM 数据 CSV 文件路径（双文件模式，可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始生成所有可视化图表...")
    print("=" * 60)
    
    # 确定模式
    if csv_path_tum is not None and csv_path_norm is not None:
        print("\n模式: 双文件模式 (TUM 和 NORM 分开)")
        print(f"  TUM 文件: {csv_path_tum}")
        print(f"  NORM 文件: {csv_path_norm}")
    else:
        print("\n模式: 单文件模式")
        print(f"  CSV 文件: {csv_path}")
    
    # 图1: 箱线图
    print("\n[1/3] 正在生成箱线图...")
    if csv_path_tum is not None and csv_path_norm is not None:
        plot_boxplot(
            None,
            os.path.join(output_dir, 'Figure_1_Boxplot.png'),
            csv_path_tum=csv_path_tum,
            csv_path_norm=csv_path_norm
        )
    else:
        plot_boxplot(
            csv_path, 
            os.path.join(output_dir, 'Figure_1_Boxplot.png')
        )
    
    # 图2: 散点图
    print("\n[2/3] 正在生成散点图...")
    if csv_path_tum is not None and csv_path_norm is not None:
        plot_scatter(
            None,
            os.path.join(output_dir, 'Figure_2_Scatter.png'),
            csv_path_tum=csv_path_tum,
            csv_path_norm=csv_path_norm
        )
    else:
        plot_scatter(
            csv_path, 
            os.path.join(output_dir, 'Figure_2_Scatter.png')
        )
    
    # 图3: 轨迹图（合并显示 TUM 和 NORM）
    print("\n[3/3] 正在生成轨迹图 (TUM vs NORM)...")
    if csv_path_tum is not None and csv_path_norm is not None:
        plot_trajectory(
            None,
            os.path.join(output_dir, 'Figure_3_Trajectory.png'),
            csv_path_tum=csv_path_tum,
            csv_path_norm=csv_path_norm
        )
    else:
        # 单文件模式：分别生成两个图
        plot_trajectory(
            csv_path, 
            os.path.join(output_dir, 'Figure_3_Trajectory_TUM.png'),
            label='TUM'
        )
        plot_trajectory(
            csv_path, 
            os.path.join(output_dir, 'Figure_3_Trajectory_NORM.png'),
            label='NORM'
        )
    
    print("\n" + "=" * 60)
    print(f"✓ 所有图表已保存至: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='批量生成所有可视化图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  单文件模式:
    python plot_all.py --csv_path ./results/guided_results.csv
  
  双文件模式:
    python plot_all.py --csv_path_tum ./results/tum_results.csv --csv_path_norm ./results/norm_results.csv
        """
    )
    parser.add_argument('--csv_path', type=str, default=None,
                       help='CSV 文件路径（单文件模式）')
    parser.add_argument('--csv_path_tum', type=str, default=None,
                       help='TUM 数据 CSV 文件路径（双文件模式）')
    parser.add_argument('--csv_path_norm', type=str, default=None,
                       help='NORM 数据 CSV 文件路径（双文件模式）')
    parser.add_argument('--output_dir', type=str,
                       default='/data/xuwen/ddpm_inference_results/inference_visualization',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.csv_path_tum is not None and args.csv_path_norm is not None:
        # 双文件模式
        plot_all(
            output_dir=args.output_dir,
            csv_path_tum=args.csv_path_tum,
            csv_path_norm=args.csv_path_norm
        )
    elif args.csv_path is not None:
        # 单文件模式
        plot_all(
            csv_path=args.csv_path,
            output_dir=args.output_dir
        )
    else:
        print("❌ 错误: 请提供 --csv_path (单文件模式) 或 --csv_path_tum 和 --csv_path_norm (双文件模式)")
        parser.print_help()


if __name__ == "__main__":
    main()

