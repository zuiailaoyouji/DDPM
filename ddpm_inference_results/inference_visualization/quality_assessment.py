"""
质量评估脚本 (CSV索引版)
功能：
1. 读取 batch_results.csv 获取样本列表。
2. 根据 Filename 自动寻找对应的 original.png 和 BEST.png。
3. 计算 SSIM 和 PSNR 指标。
4. 自动过滤掉未修改的样本 (Stop_Reason=Degradation 或 Best_Iter=0)，只评估有效增强的样本。
5. 生成差异热力图。
"""
import cv2
import numpy as np
import os
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def generate_heatmap(img_a, img_b):
    """生成高对比度的差值热力图"""
    # 1. 计算绝对差值
    diff = cv2.absdiff(img_a, img_b)
    
    # 2. 转为灰度
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 3. 归一化增强对比度 (让微小差异也显形)
    if diff_gray.max() > 0:
        diff_norm = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    else:
        diff_norm = diff_gray
        
    # 4. 伪彩色映射
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return heatmap

def find_image_pair(base_dir, filename_stem):
    """
    在 base_dir 下寻找对应的 original 和 BEST 图片
    假设结构: base_dir/{filename_stem}/original.png
    """
    # 尝试多种可能的命名模式
    # 1. 子文件夹模式 (标准)
    sub_dir = os.path.join(base_dir, filename_stem)
    
    orig_path = None
    best_path = None
    
    # 找 Original
    patterns_orig = [
        os.path.join(sub_dir, "original.png"),
        os.path.join(sub_dir, "original_*.png"),
        os.path.join(base_dir, f"{filename_stem}_original.png") # 扁平结构备选
    ]
    for p in patterns_orig:
        matches = glob(p)
        if matches:
            orig_path = matches[0]
            break
            
    # 找 Best
    patterns_best = [
        os.path.join(sub_dir, "BEST.png"),
        os.path.join(sub_dir, "BEST_*.png"),
        os.path.join(base_dir, f"{filename_stem}_BEST.png") # 扁平结构备选
    ]
    for p in patterns_best:
        matches = glob(p)
        if matches:
            best_path = matches[0]
            break
            
    return orig_path, best_path

def main():
    parser = argparse.ArgumentParser(description='图像质量评估 (SSIM/PSNR)')
    parser.add_argument('--csv_path', type=str, required=True, help='batch_results.csv 的路径')
    parser.add_argument('--image_root', type=str, required=True, help='包含图片子文件夹的根目录')
    parser.add_argument('--output_dir', type=str, default='./quality_report', help='结果输出目录')
    parser.add_argument('--save_diff', action='store_true', help='是否保存差异对比图')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 读取 CSV
    try:
        df = pd.read_csv(args.csv_path)
        # 清理列名空格
        df.columns = [c.strip() for c in df.columns]
        print(f"✅ 成功读取 CSV: {len(df)} 条记录")
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 统计变量
    total_count = 0
    skipped_count = 0  # 未修改的
    missing_count = 0  # 找不到文件的
    valid_scores = {'ssim': [], 'psnr': []}
    
    # 2. 遍历 CSV
    pbar = tqdm(total=len(df), desc="评估中")
    
    for idx, row in df.iterrows():
        filename = row['Filename']
        # 去掉扩展名获取 stem (例如 TUM-TCGA-XXX)
        stem = os.path.splitext(filename)[0]
        
        # 检查是否发生实质修改
        # 如果 Best_Iter 是 0，或者 Stop_Reason 是 Degradation，说明图片可能回滚了
        best_iter = int(row.get('Best_Iter', -1))
        stop_reason = str(row.get('Stop_Reason', ''))
        
        if best_iter == 0 or 'Degradation' in stop_reason:
            skipped_count += 1
            pbar.update(1)
            continue
            
        # 3. 寻找图片对
        orig_path, best_path = find_image_pair(args.image_root, stem)
        
        if not orig_path or not best_path:
            # print(f"⚠️ 找不到文件: {stem}")
            missing_count += 1
            pbar.update(1)
            continue
            
        # 4. 读取图片
        img_orig = cv2.imread(orig_path)
        img_best = cv2.imread(best_path)
        
        if img_orig is None or img_best is None:
            missing_count += 1
            pbar.update(1)
            continue
            
        # 确保尺寸一致
        if img_orig.shape != img_best.shape:
            img_best = cv2.resize(img_best, (img_orig.shape[1], img_orig.shape[0]))
            
        # 5. 计算指标
        # SSIM 需要灰度图
        gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        gray_best = cv2.cvtColor(img_best, cv2.COLOR_BGR2GRAY)
        
        try:
            score_ssim = ssim(gray_orig, gray_best)
            score_psnr = psnr(img_orig, img_best)
            
            # 过滤掉完全一样的图片 (SSIM=1.0)
            # 虽然前面过滤了 Iter=0，但可能存在 Iter>0 但图没变的情况
            if score_ssim > 0.9999:
                skipped_count += 1
                pbar.update(1)
                continue
                
            valid_scores['ssim'].append(score_ssim)
            valid_scores['psnr'].append(score_psnr)
            total_count += 1
            
            # 6. 保存差异图 (可选)
            if args.save_diff and total_count <= 50: # 只保存前50张避免刷屏
                heatmap = generate_heatmap(img_orig, img_best)
                
                # 添加文字标签
                cv2.putText(img_orig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img_best, "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 在热力图上添加 SSIM 值（右边那副图）
                ssim_text = f"SSIM={score_ssim:.4f}"
                # 使用白色文字，带黑色描边以提高可读性
                cv2.putText(heatmap, ssim_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # 黑色描边
                cv2.putText(heatmap, ssim_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 白色文字
                
                concat = np.hstack([img_orig, img_best, heatmap])
                save_name = os.path.join(args.output_dir, f"Diff_{stem}.jpg")
                cv2.imwrite(save_name, concat)
                
        except Exception as e:
            print(f"计算出错 {stem}: {e}")
        
        pbar.update(1)
        
    pbar.close()
    
    # 7. 输出报告
    print("\n" + "="*60)
    print("📊 图像质量评估报告 (Quality Assessment Report)")
    print("="*60)
    print(f"CSV 总记录数     : {len(df)}")
    print(f"已跳过 (未修改)  : {skipped_count} (Iter=0 或 Degradation)")
    print(f"文件缺失         : {missing_count}")
    print(f"有效评估样本数   : {total_count}")
    print("-" * 60)
    
    if total_count > 0:
        avg_ssim = np.mean(valid_scores['ssim'])
        avg_psnr = np.mean(valid_scores['psnr'])
        
        print(f"✅ 平均 SSIM : {avg_ssim:.4f} (结构相似度，越低代表改变越大)")
        print(f"✅ 平均 PSNR : {avg_psnr:.4f} (峰值信噪比)")
        print("\n注：SSIM 越低，说明增强后的图像在结构上与原图差异越大（即修改了更多纹理）。")
        print("    如果 SSIM 接近 1.0，说明模型几乎没干活。")
    else:
        print("⚠️ 没有有效的评估样本。")
    print("="*60)

if __name__ == "__main__":
    main()