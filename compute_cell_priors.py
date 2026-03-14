"""
细胞先验分布统计脚本
用于计算重采样后数据集中，真实肿瘤细胞与正常细胞的全局比例，
从而生成静态的全局 Alpha 权重供 Focal Loss 使用。
"""
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from ddpm_dataset import NCTDataset
from ddpm_utils import load_hovernet, get_device, print_gpu_info


def compute_priors(tum_dir, norm_dir, hovernet, batch_size=8, num_batches=100, device='cuda'):
    """
    计算数据集中的细胞分布先验

    Args:
        tum_dir: 肿瘤图像目录
        norm_dir: 正常图像目录
        hovernet: HoVer-Net 模型
        batch_size: 批次大小
        num_batches: 采样的批次数量，默认 100 个 Batch 足以估计全局分布
        device: 设备 ('cuda' 或 'cpu')
    """
    # 1. 加载启用了过采样 (oversample=True) 的数据集
    print("正在加载数据集 (oversample=True)...")
    dataset = NCTDataset(tum_dir, norm_dir, oversample=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    hovernet.eval()

    total_tumor_cells = 0.0
    total_normal_cells = 0.0

    # 确定实际要跑的 batch 数量（防止数据集太小不够跑 100 个）
    actual_batches = min(num_batches, len(dataloader))
    print(f"\n开始统计细胞数量，将采样 {actual_batches} 个 Batch...")

    progress_bar = tqdm(dataloader, total=actual_batches, desc="Computing Priors")

    # 2. 遍历 DataLoader 进行统计
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= actual_batches:
                break

            # 准备数据，沿用 feedback_loss.py 的预处理逻辑
            clean_images, _ = batch
            clean_input = clean_images.to(device)
            clean_input = clean_input * 255.0  # HoVer-Net 需要 0-255 范围

            # HoVer-Net 推理
            clean_output = hovernet(clean_input)
            clean_probs = torch.softmax(clean_output['tp'], dim=1)
            mask = torch.softmax(clean_output['np'], dim=1)[:, 1, :, :]  # 细胞核掩膜
            clean_p_neo = clean_probs[:, 1, :, :]

            # 使用 0.5 作为硬阈值划分
            pseudo_target = (clean_p_neo >= 0.5).float()

            # 统计当前 batch 的细胞数量
            tumor_cells_batch = (pseudo_target * mask).sum().item()
            normal_cells_batch = ((1.0 - pseudo_target) * mask).sum().item()

            total_tumor_cells += tumor_cells_batch
            total_normal_cells += normal_cells_batch

            # 实时更新进度条显示当前的比例
            current_total = total_tumor_cells + total_normal_cells
            if current_total > 0:
                t_ratio = total_tumor_cells / current_total
                n_ratio = total_normal_cells / current_total
                progress_bar.set_postfix({'Tumor%': f"{t_ratio:.1%}", 'Norm%': f"{n_ratio:.1%}"})

    # 3. 计算并输出最终的静态 Alpha 建议值
    total_cells = total_tumor_cells + total_normal_cells

    print("\n" + "="*50)
    print("📊 统计完成！")
    print("="*50)

    if total_cells == 0:
        print("❌ 错误：未检测到任何细胞，请检查 HoVer-Net 路径或数据格式。")
        return

    # 反比例计算：数量越少，权重越大
    alpha_tumor = total_normal_cells / total_cells
    alpha_normal = total_tumor_cells / total_cells

    print(f"检测到的总细胞数: {int(total_cells)}")
    print(f"  - 癌细胞 (Tumor): {int(total_tumor_cells)} 个 ({total_tumor_cells/total_cells:.1%})")
    print(f"  - 正常细胞 (Normal): {int(total_normal_cells)} 个 ({total_normal_cells/total_cells:.1%})")
    print("-" * 50)
    print("💡 建议在 train.py / ddpm_config.py 中设置以下全局静态权重：")
    print(f"  --alpha_tumor = {alpha_tumor:.4f}")
    print(f"  --alpha_normal = {alpha_normal:.4f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='计算数据集细胞分布先验以设定 Focal Loss 权重')
    parser.add_argument('--tum_dir', type=str, required=True, help='肿瘤图像目录')
    parser.add_argument('--norm_dir', type=str, required=True, help='正常图像目录')
    parser.add_argument('--hovernet_path', type=str, required=True, help='HoVer-Net 模型路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_batches', type=int, default=100, help='要采样的 Batch 数量（默认 100 足矣）')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=None, help='指定使用的 GPU ID')

    args = parser.parse_args()

    print_gpu_info()

    # 确定使用的设备
    if args.gpu_id is not None:
        device = get_device(gpu_id=args.gpu_id)
    elif args.device is not None:
        device = args.device
    else:
        device = get_device()

    print(f"使用设备: {device}")

    # 加载 HoVer-Net
    print(f"加载 HoVer-Net 模型: {args.hovernet_path}")
    hovernet = load_hovernet(args.hovernet_path, device=device)

    if hovernet is None:
        raise RuntimeError("HoVer-Net 模型加载失败！")

    compute_priors(
        tum_dir=args.tum_dir,
        norm_dir=args.norm_dir,
        hovernet=hovernet,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=device
    )


if __name__ == '__main__':
    main()
