# DDPM 去噪分类项目

基于 DDPM (Denoising Diffusion Probabilistic Model) 的图像去噪与分类项目，结合 HoVer-Net 进行反馈式训练。

## 项目结构

```
DDPM/
├── dataset.py          # 数据集加载模块
├── unet_wrapper.py     # U-Net 模型架构
├── feedback_loss.py    # 反馈损失计算
├── train.py            # 训练主循环
├── utils.py            # 工具函数（模型加载、设备管理等）
├── config.py           # 配置管理
├── example_usage.py    # 使用示例
├── loss.py             # 损失函数（旧版本）
├── requirements.txt    # 依赖包
└── README.md          # 项目说明
```

## 核心特性

### 1. 数据集加载 (dataset.py)
- **归一化方式**: 仅使用 `/255.0` 归一化到 [0, 1]，**不移除** -1 到 1 的归一化
- **过采样**: 自动平衡正常样本和肿瘤样本数量
- **图像尺寸**: 统一调整为 256x256

### 2. 模型架构 (unet_wrapper.py)
- **输入通道**: 6 通道（3 通道原始图 + 3 通道加噪图）
- **输出通道**: 3 通道（预测噪声）
- **架构**: 基于 diffusers 的 UNet2DModel

### 3. 反馈损失 (feedback_loss.py)
- **核心公式**: 从噪声预测还原原始图像 x0
  ```
  x0 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)
  ```
- **HoVer-Net 引导**: 利用分类结果引导去噪过程
- **双重损失**: 概率损失（BCE）+ 熵损失（分类置信度）

### 4. 训练流程 (train.py)
- **加噪过程**: 使用 DDPMScheduler 进行前向扩散
- **输入拼接**: `[clean_images, noisy_images]` 拼接成 6 通道
- **损失组合**: MSE 损失 + 反馈损失（可配置权重）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本训练

```bash
python train.py \
    --tum_dir /path/to/tumor/images \
    --norm_dir /path/to/normal/images \
    --hovernet_path /path/to/hovernet/model.pth \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --device cuda
```

### 不使用反馈损失训练（仅 MSE 损失）

如果暂时没有 HoVer-Net 模型，可以只使用 MSE 损失进行训练：

```bash
python train.py \
    --tum_dir /path/to/tumor/images \
    --norm_dir /path/to/normal/images \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

### 运行示例代码

```bash
python example_usage.py
```

### 参数说明

- `--tum_dir`: 肿瘤图像目录路径
- `--norm_dir`: 正常图像目录路径
- `--hovernet_path`: HoVer-Net 模型文件路径
- `--epochs`: 训练轮数（默认 100）
- `--batch_size`: 批次大小（默认 8）
- `--lr`: 学习率（默认 1e-4）
- `--device`: 设备类型，'cuda' 或 'cpu'（默认 'cuda'）
- `--save_dir`: 模型保存目录（默认 './checkpoints'）
- `--feedback_weight_prob`: 概率损失权重（默认 0.05）
- `--feedback_weight_entropy`: 熵损失权重（默认 0.01）
- `--use_feedback_from_epoch`: 从第几个 epoch 开始使用反馈损失（默认 5）

## 训练策略

1. **前 N 个 Epoch**: 仅使用 MSE 损失训练基础去噪能力
2. **后续 Epoch**: 加入反馈损失，利用 HoVer-Net 引导特征学习
3. **损失权重**: 建议从较小的权重开始（0.01-0.05），根据训练效果调整

## 注意事项

1. **HoVer-Net 加载**: 需要在 `utils.py` 中实现 `load_hovernet` 函数，根据实际的 HoVer-Net 模型结构进行调整
2. **数据格式**: 支持 PNG、JPG、JPEG、TIF、TIFF 格式
3. **内存管理**: 如果显存不足，可以减小 `batch_size` 或图像尺寸
4. **梯度裁剪**: 已内置梯度裁剪（max_norm=1.0）防止梯度爆炸
5. **反馈损失**: 如果未提供 HoVer-Net 模型，训练将仅使用 MSE 损失，仍然可以正常进行

## 模型保存

模型每 10 个 epoch 自动保存一次，保存在 `--save_dir` 指定的目录下，文件名格式为 `unet_epoch_{epoch}.pth`。

## 开发说明

- 所有模块采用模块化设计，便于维护和扩展
- 代码遵循 PEP 8 规范
- 关键函数都有详细的文档字符串

