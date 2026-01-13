# 下游任务验证模块

用于对比原始数据和增强数据在分类任务上的性能差异。

## 文件结构

```
downstream_task/
├── __init__.py      # 包初始化文件
├── main.py          # 主程序入口
├── dataset.py       # 数据集类
├── model.py         # 模型定义
├── trainer.py       # 训练和评估逻辑
├── utils.py         # 工具函数（文件匹配等）
└── README.md        # 说明文档
```

## 功能说明

### 实验设计

1. **Baseline 实验**：使用原始图像训练，原始图像测试
2. **Ours 实验**：使用增强图像训练，原始图像测试

通过对比两个实验的结果，评估增强数据对下游分类任务的提升效果。

### 文件匹配逻辑

脚本会自动匹配原始图像和增强图像：

- **原始图像路径**：`{orig_dir}/*.png` 或 `{orig_dir}/**/*.png`
- **增强图像路径**：`{enh_dir}/{subfolder}/BEST_*.png`

匹配策略：
1. 直接匹配文件名（去除扩展名）
2. 子文件夹名包含原始文件名的一部分
3. 提取 TCGA 部分进行匹配

例如：
- 原始文件：`TUM-TCGA-ADFGWHNR.tif`
- 增强文件：`/path/to/TUM/TUM-TCGA-ADFGWHNR/BEST_iter5_conf0.98.png`

## 使用方法

### 基本使用

```bash
cd downstream_task
python main.py \
    --orig_tum_dir /data/xuwen/NCT-CRC-HE-100K/TUM \
    --orig_norm_dir /data/xuwen/NCT-CRC-HE-100K/NORM \
    --enh_tum_dir /data/xuwen/ddpm_inference_results/TUM \
    --enh_norm_dir /data/xuwen/ddpm_inference_results/NORM
```

### 参数说明

- `--orig_tum_dir`: 原始 TUM 图片目录
- `--orig_norm_dir`: 原始 NORM 图片目录
- `--enh_tum_dir`: 增强 TUM 图片目录（包含子文件夹）
- `--enh_norm_dir`: 增强 NORM 图片目录（包含子文件夹）
- `--batch_size`: 批次大小（默认：32）
- `--epochs`: 训练轮数（默认：15）
- `--lr`: 学习率（默认：1e-4）
- `--test_size`: 测试集比例（默认：0.2）
- `--random_seed`: 随机种子（默认：42）
- `--output_dir`: 结果输出目录（默认：`./inference_visualization/downstream_results`）
- `--device`: 计算设备（默认：自动选择）

### 输出结果

结果会保存在 `--output_dir` 目录下：

- `results.txt`: 详细的实验结果（准确率、F1、AUC 等）

## 注意事项

1. **文件匹配**：确保增强图像的子文件夹名与原始文件名有对应关系
2. **数据划分**：两个实验使用相同的训练/测试集划分（固定随机种子）
3. **测试集**：两个实验都使用原始图像作为测试集，确保公平对比
4. **匹配率**：如果匹配率较低，可能需要检查文件命名规则

## 示例输出

```
🏆 FINAL RESULTS SUMMARY
============================================================
Metric          Baseline     Ours         Improvement    
------------------------------------------------------------
Accuracy        0.8500       0.8800       +0.0300 (+3.00%)
F1 Score        0.8450       0.8750       +0.0300 (+3.00%)
AUC             0.9200       0.9400       +0.0200 (+2.00%)
============================================================

🎉 结论: 增强后的数据显著提升了下游分类器的性能！实验成功！
```

