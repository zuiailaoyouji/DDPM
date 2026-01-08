"""
工具函数模块 (DDPM/utils.py)
包含模型加载、设备管理等辅助函数
"""
import torch
import os
import sys

# 关键：将 HoVer-net 目录加入 python 路径，以便能导入 models 和 run_utils
# 目录结构：
#   /home/xuwen/DDPM
#     /utils.py      <- 当前文件所在位置
#     /HoVer-net     <- HoVer-Net 代码所在位置（与 utils.py 同级）
#       /models
#       /run_utils
current_dir = os.path.dirname(os.path.abspath(__file__))  # DDPM 目录
hovernet_dir = os.path.join(current_dir, 'HoVer-net')  # /home/xuwen/DDPM/HoVer-net

# 将 HoVer-net 目录添加到路径
if hovernet_dir not in sys.path:
    sys.path.append(hovernet_dir)

# 尝试导入 HoVer-Net 和工具函数
HoVerNet = None
convert_pytorch_checkpoint = None

try:
    from models.hovernet.net_desc import HoVerNet
    print(f"✓ 成功导入 HoVerNet")
except ImportError as e:
    print(f"警告: 无法导入 HoVer-Net 模块: {e}")
    print(f"请确保 'HoVer-net' 目录在 DDPM 目录下（与 utils.py 同级）")
    print(f"当前 DDPM 目录: {current_dir}")
    print(f"预期 HoVer-net 路径: {hovernet_dir}")

# 尝试导入 convert_pytorch_checkpoint
try:
    from run_utils.utils import convert_pytorch_checkpoint
    print(f"✓ 成功导入 convert_pytorch_checkpoint")
except ImportError as e:
    print(f"警告: 无法导入 convert_pytorch_checkpoint: {e}")
    # 创建占位函数
    def convert_pytorch_checkpoint(state_dict):
        """占位函数：如果导入失败，直接返回原始字典"""
        return state_dict


def load_hovernet(model_path, device='cuda'):
    """
    加载 HoVer-Net 模型
    逻辑移植自 filter_hard_examples.py
    
    Args:
        model_path: 权重文件路径 (.tar)
        device: 设备类型
    
    Returns:
        hovernet: 加载好并冻结参数的 HoVer-Net 模型
    """
    if HoVerNet is None:
        raise ImportError("未找到 HoVer-Net 定义，无法加载模型。请检查 models 文件夹位置。")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"HoVer-Net 权重文件未找到: {model_path}")
    
    print(f"正在加载 HoVer-Net: {model_path} ...")
    
    # 1. 初始化模型结构 (Fast 模式, 6个类别对应 PanNuke)
    # 注意：这里的配置必须与你训练/筛选时一致
    net = HoVerNet(nr_types=6, mode='fast')
    
    # 2. 加载权重字典
    checkpoint = torch.load(model_path, map_location=device)
    
    # 3. 提取权重状态字典 (处理可能的嵌套结构)
    # 优先找 'desc' (HoVer-Net 标准格式)，其次找 'model'，最后直接用 checkpoint
    state_dict = checkpoint.get('desc', checkpoint.get('model', checkpoint))
    
    # 4. 转换权重名称 (处理 DataParallel 的 module. 前缀)
    # 需要依赖 run_utils.utils 中的 convert_pytorch_checkpoint
    if convert_pytorch_checkpoint is not None:
        try:
            state_dict = convert_pytorch_checkpoint(state_dict)
        except Exception as e:
            print(f"  注意: 权重名称转换失败或不需要转换: {e}")
    
    # 5. 加载权重到模型
    # strict=False 是为了容忍一些不匹配的键 (如不需要的 loss 参数)
    keys = net.load_state_dict(state_dict, strict=False)
    
    # 打印加载情况（可选）
    if len(keys.missing_keys) > 0:
        print(f"  缺失键 (部分是正常的): {len(keys.missing_keys)} 个")
    
    # 6. 部署到设备并冻结
    net = net.to(device)
    net.eval()
    
    # 彻底冻结所有参数，防止在 DDPM 训练中意外更新 HoVer-Net
    for param in net.parameters():
        param.requires_grad = False
        
    print("✓ HoVer-Net 加载完成并已冻结 (Eval Mode)")
    return net


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        print("警告: CUDA 不可用，将使用 CPU")
        return 'cpu'


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

