"""
质量评估脚本 - 计算原始图像与增强图像的 SSIM 和 PSNR 指标
适配文件结构：{input_dir}/{subfolder}/original_*.png 和 {input_dir}/{subfolder}/BEST_*.png
"""
import cv2
import numpy as np
import os
import argparse
from glob import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def generate_heatmap(img_a, img_b):
    """
    生成高对比度的差值热力图
    """
    # 1. 计算绝对差值
    diff = cv2.absdiff(img_a, img_b)
    
    # 2. 转为灰度图 (以此为强度)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 3. [关键步骤] 归一化增强对比度
    # 将差值拉伸到 0-255 区间，让微小的差异变亮
    # 只要有差异，就让它显形。
    if diff_gray.max() > 0:
        diff_norm = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    else:
        diff_norm = diff_gray
        
    # 4. 应用伪彩色映射 (JET 是经典的彩虹色，适合热力图)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    
    return heatmap


def find_matching_pairs(input_dir):
    """
    在输入目录的所有子文件夹中查找 original 和 BEST 图片对
    
    返回: [(original_path, best_path, subfolder_name), ...]
    """
    pairs = []
    
    # 获取所有子文件夹
    subfolders = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
    
    if not subfolders:
        print(f"⚠️ 在 {input_dir} 下未找到子文件夹")
        return pairs
    
    print(f"🔍 找到 {len(subfolders)} 个子文件夹，开始扫描...")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_dir, subfolder)
        
        # 查找 original_*.png
        original_files = glob(os.path.join(subfolder_path, "original_*.png"))
        
        if not original_files:
            continue
        
        # 对于每个 original 文件，查找对应的 BEST 文件
        for original_path in original_files:
            # 获取文件名前缀（用于匹配）
            # 例如：original_conf0.23.png -> 可能需要匹配到对应的样本
            # 或者直接在同一文件夹下找 BEST_*.png（如果只有一个 original）
            
            # 如果只有一个 original，直接找 BEST_*.png
            if len(original_files) == 1:
                best_files = glob(os.path.join(subfolder_path, "BEST_*.png"))
                if best_files:
                    pairs.append((original_path, best_files[0], subfolder))
            else:
                # 多个 original 的情况，尝试根据文件名匹配
                # 这里简化处理：取第一个 BEST 文件（如果用户有更复杂的匹配规则，可以后续修改）
                best_files = glob(os.path.join(subfolder_path, "BEST_*.png"))
                if best_files:
                    # 可以根据需要添加更精确的匹配逻辑
                    pairs.append((original_path, best_files[0], subfolder))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description='质量评估：计算原始图像与增强图像的 SSIM 和 PSNR')
    parser.add_argument('--input_dir', type=str, 
                       default='./results/guided_inference',
                       help='推理结果根目录（包含多个子文件夹）')
    parser.add_argument('--output_dir', type=str, 
                       default='./inference_visualization/quality_analysis',
                       help='保存分析结果的文件夹')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有匹配的图片对
    pairs = find_matching_pairs(args.input_dir)
    
    if len(pairs) == 0:
        print(f"❌ 在 {args.input_dir} 下未找到 original 和 BEST 图片对。")
        print("   请检查文件命名格式：")
        print("   - 原始图: {subfolder}/original_*.png")
        print("   - 增强图: {subfolder}/BEST_*.png")
        return

    print(f"✓ 找到 {len(pairs)} 对图片，开始评估...")

    # 两个列表：只存【发生改变】的样本分数
    valid_ssim_scores = []
    valid_psnr_scores = []
    
    # 计数器
    total_count = 0
    unchanged_count = 0
    
    # 准备 CSV 记录
    csv_path = os.path.join(args.output_dir, 'metrics_filtered.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Subfolder,Original_File,BEST_File,Status,SSIM,PSNR\n")

    for original_path, best_path, subfolder in tqdm(pairs, desc="处理图片"):
        total_count += 1
        
        # 读取图片
        img_orig = cv2.imread(original_path)
        img_best = cv2.imread(best_path)
        
        if img_orig is None:
            print(f"⚠️ 无法读取原始图: {original_path}")
            continue
        if img_best is None:
            print(f"⚠️ 无法读取增强图: {best_path}")
            continue
            
        # 确保尺寸一致
        if img_orig.shape != img_best.shape:
            img_best = cv2.resize(img_best, (img_orig.shape[1], img_orig.shape[0]))
            
        # ---------------------------------------------------
        # [核心逻辑] 检测图像是否完全一致
        # ---------------------------------------------------
        # 计算 L1 距离，如果为 0 说明像素完全一样
        diff_sum = cv2.norm(img_orig, img_best, cv2.NORM_L1)
        
        orig_name = os.path.basename(original_path)
        best_name = os.path.basename(best_path)
        
        if diff_sum == 0:
            # === 情况 A: 图像没变 ===
            unchanged_count += 1
            # 记录到 CSV 但不计入均值
            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{subfolder},{orig_name},{best_name},Unchanged,1.0,INF\n")
            # 不生成热力图（因为是全黑的）
            continue
        
        else:
            # === 情况 B: 图像变了 (计算指标) ===
            # SSIM 需要指定 channel_axis=2 (对彩色图像)
            try:
                val_ssim = ssim(img_orig, img_best, channel_axis=2)
            except TypeError: 
                # 旧版 skimage 可能用 multichannel=True
                val_ssim = ssim(img_orig, img_best, multichannel=True)
                
            val_psnr = psnr(img_orig, img_best)
            
            valid_ssim_scores.append(val_ssim)
            valid_psnr_scores.append(val_psnr)
            
            # 写入 CSV
            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{subfolder},{orig_name},{best_name},Modified,{val_ssim:.4f},{val_psnr:.4f}\n")
            
            # ---------------------------
            # 生成可视化图 (原图 | 增强图 | 热力图)
            # ---------------------------
            heatmap = generate_heatmap(img_orig, img_best)
            
            # 加文字标签
            h, w, _ = img_orig.shape
            cv2.putText(img_orig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_best, "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(heatmap, f"Diff (SSIM={val_ssim:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 拼接
            concat_img = np.hstack([img_orig, img_best, heatmap])
            
            # 保存图片（使用子文件夹名作为前缀）
            save_name = f"Diff_{subfolder}.png"
            cv2.imwrite(os.path.join(args.output_dir, save_name), concat_img)

    # ---------------------------------------------------
    # 输出统计结果
    # ---------------------------------------------------
    modified_count = len(valid_ssim_scores)
    avg_ssim = np.mean(valid_ssim_scores) if modified_count > 0 else 0
    avg_psnr = np.mean(valid_psnr_scores) if modified_count > 0 else 0
    
    print("\n" + "="*50)
    print("📊 过滤版质量评估报告 (Filtered Quality Report)")
    print("="*50)
    print(f"总样本数       : {total_count}")
    if total_count > 0:
        print(f"未修改 (Original): {unchanged_count} (占比 {unchanged_count/total_count*100:.1f}%) -> 已剔除")
        print(f"已修改 (Active)  : {modified_count} (占比 {modified_count/total_count*100:.1f}%) -> 参与计算")
    else:
        print("未修改 (Original): 0")
        print("已修改 (Active)  : 0")
    print("-" * 50)
    if modified_count > 0:
        print(f"平均 SSIM (Active Only): {avg_ssim:.4f}")
        print(f"平均 PSNR (Active Only): {avg_psnr:.4f} dB")
    else:
        print("⚠️ 没有发生修改的样本，无法计算平均指标")
    print("="*50)
    print(f"CSV 已保存至: {csv_path}")
    print(f"对比图片已保存至: {args.output_dir}/Diff_*.png")
    print("="*50)


if __name__ == "__main__":
    main()

