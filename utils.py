"""
工具函数模块
包含模型加载、设备管理等辅助函数
"""
import torch
import os


def load_hovernet(model_path, device='cuda'):
    """
    加载 HoVer-Net 模型
    
    Args:
        model_path: 模型文件路径
        device: 设备类型
    
    Returns:
        hovernet: 加载的 HoVer-Net 模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"HoVer-Net 模型文件不存在: {model_path}")
    
    # 注意：这里需要根据实际的 HoVer-Net 模型结构进行调整
    # 以下是示例代码，实际使用时需要替换为正确的模型加载方式
    
    # 示例 1: 如果 HoVer-Net 是标准的 PyTorch 模型
    # from hovernet import HoVerNet  # 需要根据实际导入路径调整
    # hovernet = HoVerNet()
    # hovernet.load_state_dict(torch.load(model_path, map_location=device))
    # hovernet = hovernet.to(device)
    # hovernet.eval()
    # return hovernet
    
    # 示例 2: 如果直接保存了整个模型
    # hovernet = torch.load(model_path, map_location=device)
    # hovernet.eval()
    # return hovernet
    
    # 暂时返回 None，需要用户根据实际情况实现
    print(f"警告: load_hovernet 函数需要根据实际的 HoVer-Net 模型结构进行实现")
    print(f"模型路径: {model_path}")
    return None


def get_device():
    """
    获取可用设备
    
    Returns:
        device: 'cuda' 或 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    else:
        print("警告: CUDA 不可用，将使用 CPU")
        return 'cpu'


def count_parameters(model):
    """
    计算模型参数量
    
    Args:
        model: PyTorch 模型
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

