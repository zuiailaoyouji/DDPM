"""
下游任务验证主程序
对比 Baseline（原始数据）和 Ours（增强数据）的分类性能
"""
import os
import argparse
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CRCDataset
from model import get_resnet_model
from trainer import train_model
from utils import collect_original_images, build_enh_map


def get_data_transforms():
    """获取数据预处理变换"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def main():
    parser = argparse.ArgumentParser(description='下游任务验证：对比原始数据和增强数据的分类性能')
    parser.add_argument('--orig_tum_dir', type=str,
                       default='/data/xuwen/NCT-CRC-HE-100K/TUM',
                       help='原始 TUM 图片目录')
    parser.add_argument('--orig_norm_dir', type=str,
                       default='/data/xuwen/NCT-CRC-HE-100K/NORM',
                       help='原始 NORM 图片目录')
    parser.add_argument('--enh_tum_dir', type=str,
                       default='/data/xuwen/ddpm_inference_results/TUM',
                       help='增强 TUM 图片目录（包含子文件夹）')
    parser.add_argument('--enh_norm_dir', type=str,
                       default='/data/xuwen/ddpm_inference_results/NORM',
                       help='增强 NORM 图片目录（包含子文件夹）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=15,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output_dir', type=str,
                       default='./inference_visualization/downstream_results',
                       help='结果输出目录')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)，默认自动选择')
    parser.add_argument('--data_ratio', type=float, default=1.0,
                       help='训练数据保留比例（用于少样本实验，默认1.0即使用全部数据）')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 开始下游任务验证实验...")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ================= 1. 准备数据列表 =================
    print("\n" + "="*50)
    print("收集原始图像...")
    
    all_files = []
    all_labels = []
    
    # 读取 TUM (Label 1)
    tum_files = collect_original_images(args.orig_tum_dir, label=1)
    all_files.extend(tum_files)
    all_labels.extend([1] * len(tum_files))
    print(f"  TUM: {len(tum_files)} 张")
    
    # 读取 NORM (Label 0)
    norm_files = collect_original_images(args.orig_norm_dir, label=0)
    all_files.extend(norm_files)
    all_labels.extend([0] * len(norm_files))
    print(f"  NORM: {len(norm_files)} 张")
    
    print(f"总样本数: {len(all_files)}")
    
    if len(all_files) == 0:
        print("❌ 未找到任何原始图像，请检查路径")
        return
    
    # ================= 2. 划分训练/测试集 =================
    print("\n" + "="*50)
    print("划分训练/测试集...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        all_files, all_labels, 
        test_size=args.test_size, 
        random_state=args.random_seed, 
        stratify=all_labels
    )
    
    print(f"初始训练集: {len(X_train)} (TUM: {sum(y_train)}, NORM: {len(y_train)-sum(y_train)})")
    print(f"测试集: {len(X_test)} (TUM: {sum(y_test)}, NORM: {len(y_test)-sum(y_test)})")
    
    # =========== [新增] 少样本采样逻辑 ===========
    if args.data_ratio < 1.0:
        print(f"\n✂️ [少样本模式] 仅保留 {args.data_ratio*100:.1f}% 的训练数据...")
        
        # 再次进行 split，这次是为了丢弃数据
        # subset_X_train 是我们实际要用的，_rest 是丢弃的
        subset_X_train, _rest_X, subset_y_train, _rest_y = train_test_split(
            X_train, y_train, 
            train_size=args.data_ratio, 
            random_state=args.random_seed,
            stratify=y_train
        )
        
        # 覆盖原变量
        X_train, y_train = subset_X_train, subset_y_train
        original_train_size = int(len(X_train) / args.data_ratio)
        print(f"最终训练集大小: {len(X_train)} (原始大小: {original_train_size})")
        print(f"  TUM: {sum(y_train)}, NORM: {len(y_train)-sum(y_train)}")
    # ===========================================
    
    # ================= 3. 建立增强图映射 =================
    print("\n" + "="*50)
    print("建立增强图映射...")
    enh_map = build_enh_map(X_train, args.enh_tum_dir, args.enh_norm_dir)
    
    if len(enh_map) == 0:
        print("⚠️ 警告: 未找到任何增强图像，Ours 实验将使用原始图像")
    
    # ================= 4. 数据预处理 =================
    train_transform, test_transform = get_data_transforms()
    
    # ================= 5. 实验 A: Baseline =================
    print("\n" + "="*50)
    print("🔥 Running Experiment 1: Baseline (Original Data)")
    print("="*50)
    
    train_ds_base = CRCDataset(X_train, y_train, transform=train_transform, use_enhanced=False)
    test_ds = CRCDataset(X_test, y_test, transform=test_transform, use_enhanced=False)
    
    train_loader_base = DataLoader(train_ds_base, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, 
                             shuffle=False, num_workers=4)
    
    model_base = get_resnet_model(device)
    best_acc_base, best_f1_base, best_auc_base = train_model(
        model_base, train_loader_base, test_loader, 
        args.epochs, args.lr, device
    )
    
    print(f"\n✅ Baseline Best Results:")
    print(f"   Accuracy: {best_acc_base:.4f}")
    print(f"   F1 Score: {best_f1_base:.4f}")
    print(f"   AUC:      {best_auc_base:.4f}")
    
    # ================= 6. 实验 B: Ours =================
    print("\n" + "="*50)
    print("✨ Running Experiment 2: Ours (Enhanced Data)")
    print("="*50)
    
    # 关键：这里开启 use_enhanced=True
    train_ds_ours = CRCDataset(X_train, y_train, transform=train_transform, 
                               use_enhanced=True, enh_map=enh_map)
    # 测试集保持不变（使用原始图像）
    
    train_loader_ours = DataLoader(train_ds_ours, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=4)
    # test_loader 复用上面的
    
    model_ours = get_resnet_model(device)
    best_acc_ours, best_f1_ours, best_auc_ours = train_model(
        model_ours, train_loader_ours, test_loader, 
        args.epochs, args.lr, device
    )
    
    print(f"\n✅ Ours Best Results:")
    print(f"   Accuracy: {best_acc_ours:.4f}")
    print(f"   F1 Score: {best_f1_ours:.4f}")
    print(f"   AUC:      {best_auc_ours:.4f}")
    
    # ================= 7. 结果汇总 =================
    print("\n" + "#"*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("#"*60)
    print(f"{'Metric':<15} {'Baseline':<12} {'Ours':<12} {'Improvement':<15}")
    print("-"*60)
    
    acc_improve = best_acc_ours - best_acc_base
    f1_improve = best_f1_ours - best_f1_base
    auc_improve = best_auc_ours - best_auc_base
    
    print(f"{'Accuracy':<15} {best_acc_base:<12.4f} {best_acc_ours:<12.4f} {acc_improve:+.4f} ({acc_improve*100:+.2f}%)")
    print(f"{'F1 Score':<15} {best_f1_base:<12.4f} {best_f1_ours:<12.4f} {f1_improve:+.4f} ({f1_improve*100:+.2f}%)")
    print(f"{'AUC':<15} {best_auc_base:<12.4f} {best_auc_ours:<12.4f} {auc_improve:+.4f} ({auc_improve*100:+.2f}%)")
    print("#"*60)
    
    # 保存结果
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Downstream Task Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Experiment Configuration:\n")
        f.write(f"  Training Data Ratio: {args.data_ratio*100:.1f}%\n")
        f.write(f"  Training Set Size: {len(X_train)}\n")
        f.write(f"  Test Set Size: {len(X_test)}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Batch Size: {args.batch_size}\n\n")
        f.write(f"Baseline (Original Data):\n")
        f.write(f"  Accuracy: {best_acc_base:.4f}\n")
        f.write(f"  F1 Score: {best_f1_base:.4f}\n")
        f.write(f"  AUC:      {best_auc_base:.4f}\n\n")
        f.write(f"Ours (Enhanced Data):\n")
        f.write(f"  Accuracy: {best_acc_ours:.4f}\n")
        f.write(f"  F1 Score: {best_f1_ours:.4f}\n")
        f.write(f"  AUC:      {best_auc_ours:.4f}\n\n")
        f.write(f"Improvement:\n")
        f.write(f"  Accuracy: {acc_improve:+.4f} ({acc_improve*100:+.2f}%)\n")
        f.write(f"  F1 Score: {f1_improve:+.4f} ({f1_improve*100:+.2f}%)\n")
        f.write(f"  AUC:      {auc_improve:+.4f} ({auc_improve*100:+.2f}%)\n")
        f.write("="*60 + "\n")
    
    print(f"\n结果已保存至: {results_file}")
    
    # 结论
    if acc_improve > 0:
        print("\n🎉 结论: 增强后的数据显著提升了下游分类器的性能！实验成功！")
    elif acc_improve == 0:
        print("\n🤔 结论: 性能持平，增强数据未带来明显提升。")
    else:
        print("\n⚠️ 结论: 性能下降，可能需要检查增强图的分布偏移 (Domain Shift)。")


if __name__ == "__main__":
    main()

