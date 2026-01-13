"""
模型定义：ResNet-18 分类器
"""
import torch
import torch.nn as nn
from torchvision import models


def get_resnet_model(device='cuda', pretrained=False):
    """
    加载 ResNet-18 并修改全连接层用于二分类
    
    Args:
        device: 计算设备
        pretrained: 是否加载 ImageNet 预训练权重，默认 False（随机初始化）
    
    Returns:
        model: ResNet-18 模型
    """
    # 兼容不同版本的 PyTorch
    if pretrained:
        try:
            model = models.resnet18(weights='IMAGENET1K_V1')
        except TypeError:
            # 旧版本使用 pretrained=True
            model = models.resnet18(pretrained=True)
    else:
        try:
            model = models.resnet18(weights=None)
        except TypeError:
            model = models.resnet18(pretrained=False)
    
    # 修改全连接层用于二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 二分类：NORM (0) 和 TUM (1)
    
    return model.to(device)

