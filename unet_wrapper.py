"""
U-Net 模型架构模块
核心改变：输入通道改为 6（3通道原始图 + 3通道加噪图）
"""
from diffusers import UNet2DModel


def create_model(sample_size=256, in_channels=6, out_channels=3):
    """
    创建 U-Net 模型
    
    Args:
        sample_size: 输入图像尺寸，默认 256
        in_channels: 输入通道数，默认 6 (3通道原始图 + 3通道加噪图)
        out_channels: 输出通道数，默认 3 (预测噪声的通道数)
    
    Returns:
        UNet2DModel: 配置好的 U-Net 模型
    """
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,      # 关键：3 (原始) + 3 (加噪)
        out_channels=out_channels,     # 预测噪声，还是 3 通道
        layers_per_block=2, 
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )
    return model


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

