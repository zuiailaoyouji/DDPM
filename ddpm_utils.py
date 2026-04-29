"""
工具函数模块 (DDPM/ddpm_utils.py)
包含模型加载、设备管理等辅助函数
"""
"""
cellvit_utils.py
CellViT 模型加载工具，对应 ddpm_utils.py 里的 load_hovernet。
"""
import os
import sys
import torch


def load_cellvit(
    model_path: str,
    cellvit_repo_path: str = '/home/xuwen/DDPM/CellViT',
    device: str = 'cuda',
    variant: str = 'sam_h',          # 'cellvit_256' 或 'sam_h'
):
    """
    加载 CellViT 模型并冻结参数。支持 CellViT-256 与 CellViT-SAM-H。

    Args:
        model_path        : PanNuke fine-tuned 权重 (.pth)
        cellvit_repo_path : CellViT 代码仓库根目录
        device            : 'cuda' / 'cpu'
        variant           : 'cellvit_256' 或 'sam_h'
    """
    import inspect

    if cellvit_repo_path not in sys.path:
        sys.path.insert(0, cellvit_repo_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CellViT 权重文件未找到: {model_path}")

    print(f"正在加载 CellViT[{variant}]: {model_path} ...")
    checkpoint = torch.load(model_path, map_location='cpu')

    run_conf           = checkpoint.get('config', {})
    num_nuclei_classes = int(run_conf.get('data.num_nuclei_classes', 6))
    num_tissue_classes = int(run_conf.get('data.num_tissue_classes', 19))

    if variant == 'cellvit_256':
        from models.segmentation.cell_segmentation.cellvit import CellViT256
        model = CellViT256(
            model256_path      = model_path,
            num_nuclei_classes = num_nuclei_classes,
            num_tissue_classes = num_tissue_classes,
        )

    elif variant == 'sam_h':
        # CellViTSAM 与 CellViT256 在同一个文件
        from models.segmentation.cell_segmentation.cellvit import CellViTSAM

        # 自动签名匹配:不同版本 CellViT 仓库对参数名命名不同
        # 这套逻辑沿用 test_cellvit_sam_h_standalone.py 中已验证可跑的方案
        sig = inspect.signature(CellViTSAM.__init__)
        params = sig.parameters
        common_values = {
            'model_path':         model_path,
            'model256_path':      model_path,
            'model_sam_path':     model_path,
            'sam_model_path':     model_path,
            'pretrained_encoder': model_path,
            'num_nuclei_classes': num_nuclei_classes,
            'num_tissue_classes': num_tissue_classes,
            'nuclei_classes':     num_nuclei_classes,
            'tissue_classes':     num_tissue_classes,
            'vit_structure':      'SAM-H',
        }
        kwargs = {n: common_values[n] for n in params
                  if n != 'self' and n in common_values}
        print(f"  CellViTSAM kwargs: {list(kwargs.keys())}")
        model = CellViTSAM(**kwargs)

    else:
        raise ValueError(f"未知 variant: {variant}, 应为 'cellvit_256' 或 'sam_h'")

    # SAM-H checkpoint 可能有少量键不完全匹配,用 strict=False
    state = checkpoint.get('model_state_dict', checkpoint)
    msg   = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        print(f"  缺失键: {len(msg.missing_keys)} 个 "
              f"(前5: {msg.missing_keys[:5]})")
    if msg.unexpected_keys:
        print(f"  多余键: {len(msg.unexpected_keys)} 个 "
              f"(前5: {msg.unexpected_keys[:5]})")
    if len(msg.missing_keys) > 50 or len(msg.unexpected_keys) > 50:
        print("  ⚠️ missing/unexpected 数量异常,请检查权重文件是否匹配")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"✓ CellViT[{variant}] 加载完成并已冻结 "
          f"(num_nuclei_classes={num_nuclei_classes})")
    return model


def get_device(gpu_id=None):
    """
    获取可用设备
    
    Args:
        gpu_id: GPU ID（整数），如果为 None，则自动选择或使用 CPU
               例如：0 表示使用 GPU 0，1 表示使用 GPU 1
    
    Returns:
        device: 设备字符串，例如 'cuda:0' 或 'cpu'
    """
    if gpu_id is not None:
        if torch.cuda.is_available():
            if gpu_id < torch.cuda.device_count():
                device_str = f'cuda:{gpu_id}'
                print(f"使用指定的 GPU: {device_str}")
                return device_str
            else:
                print(f"警告: GPU {gpu_id} 不存在，可用 GPU 数量: {torch.cuda.device_count()}")
                print("将使用默认 GPU (cuda:0)")
                return 'cuda:0'
        else:
            print("警告: CUDA 不可用，将使用 CPU")
            return 'cpu'
    else:
        # 默认行为：自动检测
        if torch.cuda.is_available():
            return 'cuda:0'
        else:
            print("警告: CUDA 不可用，将使用 CPU")
            return 'cpu'


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def predict_x0_from_noise_shared(x_t, noise_pred, t, scheduler):
    """
    通用的 x0 还原函数
    从加噪图像 x_t 和预测的噪声 noise_pred 还原原始图像 x0
    
    Args:
        x_t: 加噪图像 [B, C, H, W]
        noise_pred: 预测的噪声 [B, C, H, W]
        t: 时间步 [B]
        scheduler: DDPMScheduler 实例
    
    Returns:
        pred_x0: 预测的原始图像 [B, C, H, W]，范围 [0, 1]
    """
    device = x_t.device
    dtype = x_t.dtype
    # 先把 alphas_cumprod 移动到 device，然后再用 t 去取值
    # 这样可以避免设备不匹配的问题（t 在 GPU，alphas_cumprod 在 CPU）
    alpha_prod_t = scheduler.alphas_cumprod.to(device)[t].to(dtype).view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (x_t - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)
    return torch.clamp(pred_x0, 0.0, 1.0)


def print_gpu_info():
    """打印可用的 GPU 信息"""
    if torch.cuda.is_available():
        print(f"\n可用的 GPU 设备:")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        print()
    else:
        print("\n警告: 未检测到可用的 CUDA 设备，将使用 CPU\n")

