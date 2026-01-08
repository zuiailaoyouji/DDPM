"""
使用示例
展示如何使用各个模块进行训练
"""
import torch
from ddpm_dataset import NCTDataset
from unet_wrapper import create_model
from feedback_loss import FeedbackLoss
from ddpm_utils import load_hovernet, get_device, count_parameters
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader


def example_dataset():
    """数据集使用示例"""
    print("=" * 50)
    print("数据集使用示例")
    print("=" * 50)
    
    # 创建数据集
    dataset = NCTDataset(
        tum_dir='./data/tumor',
        norm_dir='./data/normal',
        oversample=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    img, label = dataset[0]
    print(f"图像形状: {img.shape}, 标签: {label.item()}")
    print(f"图像值范围: [{img.min():.3f}, {img.max():.3f}]")
    print()


def example_model():
    """模型使用示例"""
    print("=" * 50)
    print("模型使用示例")
    print("=" * 50)
    
    # 创建模型
    model = create_model()
    total_params, trainable_params = count_parameters(model)
    
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # 模拟输入：6 通道 (3 原始 + 3 加噪)
    batch_size = 2
    clean_img = torch.randn(batch_size, 3, 256, 256).to(device)
    noisy_img = torch.randn(batch_size, 3, 256, 256).to(device)
    model_input = torch.cat([clean_img, noisy_img], dim=1)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()
    
    with torch.no_grad():
        output = model(model_input, timesteps).sample
    
    print(f"输入形状: {model_input.shape}")
    print(f"输出形状: {output.shape}")
    print()


def example_training_step():
    """训练步骤示例"""
    print("=" * 50)
    print("训练步骤示例")
    print("=" * 50)
    
    device = get_device()
    
    # 1. 创建模型和调度器
    unet = create_model().to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 2. 创建数据集和数据加载器
    dataset = NCTDataset('./data/tumor', './data/normal', oversample=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. 模拟一个训练步骤
    clean_images, labels = next(iter(dataloader))
    clean_images = clean_images.to(device)
    labels = labels.to(device)
    bs = clean_images.shape[0]
    
    # 采样噪声和时间步
    noise = torch.randn_like(clean_images).to(device)
    timesteps = torch.randint(0, 1000, (bs,), device=device).long()
    
    # 加噪
    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
    
    # 拼接输入
    model_input = torch.cat([clean_images, noisy_images], dim=1)
    
    # 预测噪声
    noise_pred = unet(model_input, timesteps).sample
    
    # 计算 MSE 损失
    loss_mse = torch.nn.functional.mse_loss(noise_pred, noise)
    
    print(f"批次大小: {bs}")
    print(f"干净图像形状: {clean_images.shape}")
    print(f"加噪图像形状: {noisy_images.shape}")
    print(f"模型输入形状: {model_input.shape}")
    print(f"预测噪声形状: {noise_pred.shape}")
    print(f"MSE 损失: {loss_mse.item():.4f}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("DDPM 项目模块使用示例")
    print("=" * 50 + "\n")
    
    try:
        example_dataset()
    except Exception as e:
        print(f"数据集示例失败: {e}\n")
    
    try:
        example_model()
    except Exception as e:
        print(f"模型示例失败: {e}\n")
    
    try:
        example_training_step()
    except Exception as e:
        print(f"训练步骤示例失败: {e}\n")
    
    print("=" * 50)
    print("示例完成")
    print("=" * 50)


if __name__ == '__main__':
    main()

