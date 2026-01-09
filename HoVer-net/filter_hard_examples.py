"""
难样本筛选脚本 (Hard Example Filtering Script)
根据 HoVer-Net 的 Neoplastic (类别1) 预测概率筛选难样本

任务配置:
- TUM: 筛选区间 [0.5, 0.75] - 寻找中等置信度的肿瘤难样本
- NORM: 筛选区间 [0.25, 0.5] - 寻找"具有欺骗性、容易被误判为癌的正常细胞"

输入格式:
- 支持图像格式: .tif, .tiff, .png, .jpg, .jpeg, .bmp
- 注意: 传入的图像都是 tif 格式
"""

import sys
import os
import shutil
import csv
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import cv2
from models.hovernet.net_desc import HoVerNet
from run_utils.utils import convert_pytorch_checkpoint

print("=" * 70)
print("HoVer-Net 难样本筛选工具 (多任务版本)")
print("=" * 70)

# ============================================================================
# 任务配置映射表
# ============================================================================

TASK_CONFIG = {
    'TUM': {
        'input_dir': '/data/xuwen/NCT-CRC-HE-100K/TUM/',
        'output_subdir': 'TUM',
        'lower_bound': 0.5,
        'upper_bound': 0.75,
        'description': '肿瘤样本 - 寻找中等置信度的难样本'
    },
    'NORM': {
        'input_dir': '/data/xuwen/NCT-CRC-HE-100K/NORM/',
        'output_subdir': 'NORM',
        'lower_bound': 0.25,
        'upper_bound': 0.5,
        'description': '正常样本 - 寻找容易被误判为癌的高风险样本'
    }
}

# ============================================================================
# 阶段一：环境与模型初始化
# ============================================================================

# 检查 CUDA 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[阶段一] 环境初始化")
print(f"  使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 1. 初始化模型 (Fast 模式, 6个类别对应 PanNuke)
print("\n[1/3] 初始化模型...")
model = HoVerNet(nr_types=6, mode='fast')
print("  ✓ 模型初始化完成 (Fast 模式, 6 个类别)")

# 2. 加载权重
print("\n[2/3] 加载模型权重...")
checkpoint_path = "hovernet_fast_pannuke_type_tf2pytorch.tar"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"权重文件未找到: {checkpoint_path}\n"
        "请确保权重文件在项目根目录下"
    )

checkpoint = torch.load(checkpoint_path, map_location=device)
print(f"  检查点文件键: {list(checkpoint.keys())}")

# 使用 convert_pytorch_checkpoint 处理可能的 DataParallel 格式
saved_state_dict = checkpoint.get('desc', checkpoint.get('model', checkpoint))
if isinstance(saved_state_dict, dict) and 'desc' in saved_state_dict:
    saved_state_dict = saved_state_dict['desc']
    
saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
load_result = model.load_state_dict(saved_state_dict, strict=False)

if load_result.missing_keys:
    print(f"  ⚠ 警告: 缺失的键数量: {len(load_result.missing_keys)}")
    if len(load_result.missing_keys) <= 5:
        print(f"    缺失的键: {load_result.missing_keys}")
if load_result.unexpected_keys:
    print(f"  ⚠ 警告: 意外的键数量: {len(load_result.unexpected_keys)}")
    if len(load_result.unexpected_keys) <= 5:
        print(f"    意外的键: {load_result.unexpected_keys}")

print("  ✓ 权重加载完成")

# 3. 部署模型到设备
model = model.to(device)
model.eval()
print("  ✓ 模型部署完成 (eval 模式)")

# ============================================================================
# IO 准备
# ============================================================================

print("\n[3/3] IO 路径准备...")

# 创建总输出目录
base_output_dir = Path("hard_examples")
base_output_dir.mkdir(exist_ok=True)
print(f"  ✓ 总输出目录: {base_output_dir}")

# 初始化全局日志记录
all_log_data = []

# ============================================================================
# 外层循环：遍历任务映射表
# ============================================================================

print("\n" + "=" * 70)
print("开始筛选难样本 (多任务模式)")
print("=" * 70)

for task_name, task_config in TASK_CONFIG.items():
    print("\n" + "=" * 70)
    print(f"处理任务: {task_name}")
    print(f"描述: {task_config['description']}")
    print(f"筛选区间: [{task_config['lower_bound']}, {task_config['upper_bound']}]")
    print("=" * 70)
    
    # 获取当前任务的配置
    input_dir = Path(task_config['input_dir'])
    output_subdir = task_config['output_subdir']
    lower_bound = task_config['lower_bound']
    upper_bound = task_config['upper_bound']
    
    # 检查输入文件夹是否存在
    if not input_dir.exists():
        print(f"  ⚠ 警告: 输入文件夹未找到: {input_dir}，跳过此任务")
        continue
    
    # 创建当前任务的输出子文件夹
    output_dir = base_output_dir / output_subdir
    output_dir.mkdir(exist_ok=True)
    print(f"  ✓ 输出文件夹: {output_dir}")
    
    # 获取所有图像文件（主要支持 tif 格式）
    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    image_files = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    # 优先显示 tif 文件数量
    tif_files = [f for f in image_files if f.suffix.lower() in {'.tif', '.tiff'}]
    if tif_files:
        print(f"  ✓ 找到 {len(image_files)} 张图像 (其中 {len(tif_files)} 张为 tif 格式)")
    else:
        print(f"  ✓ 找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print(f"  ⚠ 警告: 在 {input_dir} 中未找到图像文件，跳过此任务")
        continue
    
    print(f"  ✓ 找到 {len(image_files)} 张图像")
    
    # 初始化当前任务的日志记录
    task_log_data = []
    
    # 统计信息（当前任务）
    task_total_processed = 0
    task_total_selected = 0
    task_score_list = []
    
    # ========================================================================
    # 内层循环：图像处理与筛选（保持原有逻辑）
    # ========================================================================
    
    # 遍历当前任务的所有图像
    for img_path in tqdm(image_files, desc=f"处理 {task_name}", unit="张"):
        try:
            # ================================================================
            # 阶段二：图像预处理流程
            # ================================================================
            
            # 1. 加载图像（支持 tif 格式）
            # OpenCV 默认支持 tif/tiff 格式，但某些情况下可能需要指定标志
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                # 如果读取失败，尝试使用 IMREAD_ANYDEPTH | IMREAD_ANYCOLOR 标志
                img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                if img is None:
                    print(f"  ⚠ 警告: 无法读取图像 {img_path.name}，跳过")
                    continue
                # 如果是 16-bit 图像，转换为 8-bit
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)
            
            # 2. 色彩空间转换：BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. 缩放至 256x256
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # 4. 像素缩放：模型内部会除以 255.0，所以这里保持 0-255 范围
            # 转换为 NCHW 格式: (H, W, C) -> (C, H, W) -> (1, C, H, W)
            input_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # ================================================================
            # 阶段三：特征打分与判定
            # ================================================================
            
            # 5. 前向推理
            with torch.no_grad():
                output = model(input_tensor)
            
            # 6. 获取 tp 分支输出，形状为 [1, 6, 164, 164]
            tp_output = output['tp']  # [1, 6, 164, 164]
            
            # 7. 概率映射：对维度 1 (类别维度) 执行 Softmax
            tp_softmax = torch.nn.functional.softmax(tp_output, dim=1)  # [1, 6, 164, 164]
            
            # 8. 提取类别 1 (Neoplastic) 的概率图
            neoplastic_probs = tp_softmax[0, 1, :, :].cpu().numpy()  # [164, 164]
            
            # 9. 计算全局得分 S (所有像素的平均概率)
            score = float(np.mean(neoplastic_probs))
            task_score_list.append(score)
            
            # 10. 动态阈值判定：根据当前任务使用对应的 Lower_Bound 和 Upper_Bound
            if lower_bound <= score <= upper_bound:
                # 11. 物理操作：复制到目标文件夹
                output_path = output_dir / img_path.name
                shutil.copy2(img_path, output_path)
                
                # 12. 记录日志（包含 source_class）
                log_entry = {
                    'filename': img_path.name,
                    'source_class': task_name,
                    'score': score
                }
                task_log_data.append(log_entry)
                all_log_data.append(log_entry)  # 同时添加到全局日志
                task_total_selected += 1
            
            task_total_processed += 1
            
        except Exception as e:
            print(f"  ⚠ 错误: 处理 {img_path.name} 时出错: {str(e)}")
            continue
    
    # ========================================================================
    # 当前任务的统计信息输出
    # ========================================================================
    
    print(f"\n[{task_name}] 任务统计:")
    print(f"  总处理图像数: {task_total_processed}")
    print(f"  筛选出难样本数: {task_total_selected}")
    if task_total_processed > 0:
        print(f"  筛选比例: {task_total_selected/task_total_processed*100:.2f}%")
    
    if task_score_list:
        scores_array = np.array(task_score_list)
        print(f"\n  得分统计:")
        print(f"    最小值: {scores_array.min():.4f}")
        print(f"    最大值: {scores_array.max():.4f}")
        print(f"    平均值: {scores_array.mean():.4f}")
        print(f"    中位数: {np.median(scores_array):.4f}")
        print(f"    标准差: {scores_array.std():.4f}")
        
        # 得分分布（基于当前任务的阈值）
        in_range = np.sum((scores_array >= lower_bound) & (scores_array <= upper_bound))
        below_range = np.sum(scores_array < lower_bound)
        above_range = np.sum(scores_array > upper_bound)
        
        print(f"\n  得分分布:")
        print(f"    < {lower_bound} (低置信度): {below_range} ({below_range/len(task_score_list)*100:.2f}%)")
        print(f"    [{lower_bound}, {upper_bound}] (难样本): {in_range} ({in_range/len(task_score_list)*100:.2f}%)")
        print(f"    > {upper_bound} (高置信度): {above_range} ({above_range/len(task_score_list)*100:.2f}%)")

# ============================================================================
# 保存全局日志
# ============================================================================

print("\n" + "=" * 70)
print("保存结果...")
print("=" * 70)

# 保存全局 CSV 日志（包含所有任务）
log_file = base_output_dir / "filter_log.csv"
if all_log_data:
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'source_class', 'score'])
        writer.writeheader()
        writer.writerows(all_log_data)
    print(f"  ✓ 全局日志已保存: {log_file} ({len(all_log_data)} 条记录)")
    
    # 按类别统计
    tum_scores = [entry['score'] for entry in all_log_data if entry['source_class'] == 'TUM']
    norm_scores = [entry['score'] for entry in all_log_data if entry['source_class'] == 'NORM']
    
    if tum_scores:
        tum_array = np.array(tum_scores)
        print(f"\n  [TUM] 难样本得分统计:")
        print(f"    数量: {len(tum_scores)}")
        print(f"    平均值: {tum_array.mean():.4f}")
        print(f"    中位数: {np.median(tum_array):.4f}")
        print(f"    标准差: {tum_array.std():.4f}")
    
    if norm_scores:
        norm_array = np.array(norm_scores)
        print(f"\n  [NORM] 高风险误报样本得分统计:")
        print(f"    数量: {len(norm_scores)}")
        print(f"    平均值: {norm_array.mean():.4f}")
        print(f"    中位数: {np.median(norm_array):.4f}")
        print(f"    标准差: {norm_array.std():.4f}")
else:
    print(f"  ⚠ 警告: 没有筛选到任何样本，未生成日志文件")

# ============================================================================
# 全局统计信息输出
# ============================================================================

print("\n" + "=" * 70)
print("全局筛选统计")
print("=" * 70)

# 统计所有输入图像数量（主要支持 tif 格式）
image_extensions_all = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
total_all = sum(len([f for f in Path(task['input_dir']).iterdir() 
                     if f.suffix.lower() in image_extensions_all])
                for task in TASK_CONFIG.values() 
                if Path(task['input_dir']).exists())

print(f"  总筛选样本数: {len(all_log_data)}")
print(f"  其中 TUM: {sum(1 for e in all_log_data if e['source_class'] == 'TUM')}")
print(f"  其中 NORM: {sum(1 for e in all_log_data if e['source_class'] == 'NORM')}")

if all_log_data:
    all_scores = np.array([entry['score'] for entry in all_log_data])
    print(f"\n  全局得分统计:")
    print(f"    最小值: {all_scores.min():.4f}")
    print(f"    最大值: {all_scores.max():.4f}")
    print(f"    平均值: {all_scores.mean():.4f}")
    print(f"    中位数: {np.median(all_scores):.4f}")
    print(f"    标准差: {all_scores.std():.4f}")

print("\n" + "=" * 70)
print("筛选完成！")
print(f"难样本已保存至: {base_output_dir.absolute()}")
print(f"  - TUM 样本: {base_output_dir / 'TUM'}")
print(f"  - NORM 样本: {base_output_dir / 'NORM'}")
print(f"  - 日志文件: {log_file}")
print("=" * 70)

