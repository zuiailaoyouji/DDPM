"""
下游任务验证主程序 (双CSV输入版)
适配变更：
1. 接收两个推理结果 CSV (TUM 和 NORM)，合并后作为标签纠正源。
2. 对比三组实验：
   - Exp 1: Raw Noisy (原图+原标签)
   - Exp 2: Strong Clean (原图+纠正标签)
   - Exp 3: Ours (增强图+纠正标签)
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
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


def load_corrected_labels_from_two_csvs(csv_tum, csv_norm):
    """
    读取并合并两个 CSV 文件，构建纠正标签映射
    """
    label_map = {}
    
    # 辅助读取函数
    def read_and_merge(path, source_name):
        if not path or not os.path.exists(path):
            print(f"⚠️ 警告: 未找到 {source_name} CSV 文件: {path}")
            return
        
        try:
            df = pd.read_csv(path)
            # 清理列名空格
            df.columns = [c.strip() for c in df.columns]
            
            count = 0
            for _, row in df.iterrows():
                fname = row['Filename']
                mode = str(row['Mode']).lower()
                
                # 解析纠正标签
                if 'maximize' in mode:
                    label = 1 # TUM
                elif 'minimize' in mode:
                    label = 0 # NORM
                else:
                    continue 
                
                label_map[fname] = label
                count += 1
            print(f"  已从 {source_name} 加载 {count} 条记录")
            
        except Exception as e:
            print(f"❌ 读取 {source_name} CSV 出错: {e}")

    print("正在合并推理结果 CSV...")
    read_and_merge(csv_tum, "TUM")
    read_and_merge(csv_norm, "NORM")
    
    return label_map


def main():
    parser = argparse.ArgumentParser(description='下游任务验证：三组完整对比实验 (双CSV版)')
    # 变更点：接收两个 CSV 路径
    parser.add_argument('--csv_tum_path', type=str, required=True, help='TUM 推理结果 CSV')
    parser.add_argument('--csv_norm_path', type=str, required=True, help='NORM 推理结果 CSV')
    
    parser.add_argument('--orig_tum_dir', type=str, default='/data/xuwen/NCT-CRC-HE-100K/TUM')
    parser.add_argument('--orig_norm_dir', type=str, default='/data/xuwen/NCT-CRC-HE-100K/NORM')
    parser.add_argument('--enh_tum_dir', type=str, default='/data/xuwen/ddpm_inference_results/TUM')
    parser.add_argument('--enh_norm_dir', type=str, default='/data/xuwen/ddpm_inference_results/NORM')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./inference_visualization/downstream_results_full')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    
    args = parser.parse_args()
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 开始全量对比实验 (双 CSV 输入)...")
    
    # ================= 1. 数据准备与对齐 =================
    print("\n" + "="*50)
    # 1.1 加载纠正后的标签 (合并两个CSV)
    corrected_map = load_corrected_labels_from_two_csvs(args.csv_tum_path, args.csv_norm_path)
    
    if len(corrected_map) == 0:
        print("❌ 错误：未能从 CSV 中加载到任何标签，请检查路径。")
        return

    # 1.2 匹配图片文件
    # data_items 存储: {path, orig_label, corr_label}
    data_items = []
    
    def collect_items(folder, original_label_guess):
        files = collect_original_images(folder, label=original_label_guess)
        count = 0
        for fpath in files:
            fname = os.path.basename(fpath)
            # 只有在推理结果(CSV)中存在的图片才参与实验
            if fname in corrected_map:
                data_items.append({
                    'path': fpath,
                    'orig_label': original_label_guess, # 原始标签 (基于文件夹)
                    'corr_label': corrected_map[fname]  # 纠正标签 (基于扩散模型)
                })
                count += 1
        return count

    print("正在匹配图片文件...")
    n_tum = collect_items(args.orig_tum_dir, 1) # 原始文件夹认为是 1
    n_norm = collect_items(args.orig_norm_dir, 0) # 原始文件夹认为是 0
    
    print(f"匹配完成: 总样本数 {len(data_items)} (原始TUM: {n_tum}, 原始NORM: {n_norm})")
    
    if len(data_items) == 0:
        print("❌ 未匹配到任何数据")
        return

    # ================= 2. 数据划分 =================
    # 根据【纠正后的标签】进行分层划分
    all_corr_labels = [d['corr_label'] for d in data_items]
    
    train_items, test_items = train_test_split(
        data_items, 
        test_size=args.test_size, 
        random_state=args.random_seed, 
        stratify=all_corr_labels
    )
    
    # 少样本采样
    if args.data_ratio < 1.0:
        train_corr_labels = [d['corr_label'] for d in train_items]
        train_items, _ = train_test_split(
            train_items,
            train_size=args.data_ratio,
            random_state=args.random_seed,
            stratify=train_corr_labels
        )
        print(f"✂️ 少样本模式: 训练集缩减为 {len(train_items)}")

    # ================= 3. 构建数据集列表 =================
    X_train = [d['path'] for d in train_items]
    y_train_orig = [d['orig_label'] for d in train_items] # Exp 1 用
    y_train_corr = [d['corr_label'] for d in train_items] # Exp 2 & 3 用
    
    X_test = [d['path'] for d in test_items]
    y_test_corr = [d['corr_label'] for d in test_items]   # 所有测试集都用纠正标签
    
    print(f"\n数据集规模:")
    print(f"  训练集: {len(X_train)}")
    print(f"  测试集: {len(X_test)}")
    
    # 构建增强图映射 (从增强文件夹找)
    enh_map = build_enh_map(X_train, args.enh_tum_dir, args.enh_norm_dir)
    train_transform, test_transform = get_data_transforms()
    
    # 统一测试集 Loader
    test_ds = CRCDataset(X_test, y_test_corr, transform=test_transform, use_enhanced=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    results = {}

    # ================= Exp 1: Raw Noisy Baseline =================
    print("\n" + "="*50)
    print("🔥 Exp 1: Raw Noisy Baseline")
    print("   Input: Original Images")
    print("   Label: Original Labels (Human/Noisy)")
    print("   Test : Corrected Labels")
    print("="*50)
    
    train_ds_1 = CRCDataset(X_train, y_train_orig, transform=train_transform, use_enhanced=False)
    train_loader_1 = DataLoader(train_ds_1, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model_1 = get_resnet_model(device)
    res_1 = train_model(model_1, train_loader_1, test_loader, args.epochs, args.lr, device)
    results['Exp1'] = res_1
    
    # ================= Exp 2: Strong Clean Baseline =================
    print("\n" + "="*50)
    print("🛡️ Exp 2: Strong Clean Baseline")
    print("   Input: Original Images")
    print("   Label: Corrected Labels (Clean)")
    print("   Test : Corrected Labels")
    print("="*50)
    
    train_ds_2 = CRCDataset(X_train, y_train_corr, transform=train_transform, use_enhanced=False)
    train_loader_2 = DataLoader(train_ds_2, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model_2 = get_resnet_model(device, pretrained=False)
    res_2 = train_model(model_2, train_loader_2, test_loader, args.epochs, args.lr, device)
    results['Exp2'] = res_2

    # ================= Exp 3: Ours =================
    print("\n" + "="*50)
    print("✨ Exp 3: Ours (Label Cleaning + Feature Enhancement)")
    print("   Input: Original + Enhanced Images")
    print("   Label: Corrected Labels (Clean)")
    print("   Test : Corrected Labels")
    print("="*50)
    
    # 混合原图与增强图
    ds_orig = CRCDataset(X_train, y_train_corr, transform=train_transform, use_enhanced=False)
    ds_enh = CRCDataset(X_train, y_train_corr, transform=train_transform, use_enhanced=True, enh_map=enh_map)
    train_ds_3 = ConcatDataset([ds_orig, ds_enh])
    
    train_loader_3 = DataLoader(train_ds_3, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model_3 = get_resnet_model(device, pretrained=False)
    res_3 = train_model(model_3, train_loader_3, test_loader, args.epochs, args.lr, device)
    results['Exp3'] = res_3

    # ================= 结果汇总 =================
    print("\n" + "#"*80)
    print("🏆 FULL EXPERIMENT REPORT")
    print("#"*80)
    print(f"{'Experiment':<25} {'Accuracy':<12} {'F1 Score':<12} {'AUC':<12}")
    print("-" * 80)
    
    row_fmt = "{:<25} {:<12.4f} {:<12.4f} {:<12.4f}"
    print(row_fmt.format("1. Raw Noisy Baseline", *results['Exp1']))
    print(row_fmt.format("2. Strong Clean Baseline", *results['Exp2']))
    print(row_fmt.format("3. Ours (Enhanced)", *results['Exp3']))
    
    print("-" * 80)
    
    diff_clean = results['Exp2'][0] - results['Exp1'][0]
    diff_enh = results['Exp3'][0] - results['Exp2'][0]
    
    print(f"Impact of Label Cleaning:    {diff_clean:+.4f}")
    print(f"Impact of Image Enhancement: {diff_enh:+.4f}")
    
    res_file = os.path.join(args.output_dir, 'final_comparison_results.txt')
    with open(res_file, 'w') as f:
        f.write("Full Comparison Results\n")
        f.write("="*60 + "\n")
        f.write(f"Exp 1 (Noisy): Acc={results['Exp1'][0]:.4f}\n")
        f.write(f"Exp 2 (Clean): Acc={results['Exp2'][0]:.4f}\n")
        f.write(f"Exp 3 (Ours) : Acc={results['Exp3'][0]:.4f}\n")
        
    print(f"\n报告已保存至: {res_file}")

if __name__ == "__main__":
    main()