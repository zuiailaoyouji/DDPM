import torch
import sys
import os

# 确保能找到 utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_hovernet, get_device


def verify():
    print("=" * 60)
    print("HoVer-Net 加载测试")
    print("=" * 60)
    
    # 指向你的权重文件路径 (请修改为你实际的路径)
    # 假设权重文件在项目根目录
    weight_path = "../hovernet_fast_pannuke_type_tf2pytorch.tar"
    
    device = get_device()
    
    try:
        # 尝试加载
        model = load_hovernet(weight_path, device=device)
        
        # 测试前向传播
        print("\n执行前向传播测试 (Dry Run)...")
        # HoVer-Net 输入要求: [B, 3, 256, 256], 0-1 范围
        # Fast 模式输出为 164x164
        dummy_input = torch.rand(1, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        print("✓ 前向传播成功")
        print(f"  输出键: {output.keys()}")
        if 'tp' in output:
            print(f"  TP分支形状: {output['tp'].shape} (预期: [1, 6, 164, 164])")
        if 'np' in output:
            print(f"  NP分支形状: {output['np'].shape} (预期: [1, 2, 164, 164])")
            
        print("\n验证通过！阶段二所需的组件已就绪。")
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify()

