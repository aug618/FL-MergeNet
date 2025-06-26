import torch
import yaml
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention

def test_param_attention():
    # 加载配置
    config = {
        'd_attention': 64,
        'h': 8,
        'num_layers': 2,
        'a_size_conv': [160, 960],
        'a_size_linear': [100, 1280],
        'b_size_linear': [100, 2048],
        'mode': 5
    }
    
    # 创建模型
    mbv2 = mobilenetv2()
    res = resnet50()
    param_attention_l = ParamAttention(config, mode='a')
    
    print("=== 测试参数注意力模块 ===")
    
    # 提取实际参数
    param_a = {
        'conv': mbv2.stage6[2].residual[6].weight.data.clone().detach()
    }
    param_b = {
        'linear_weight': res.fc.weight.data.clone().detach()
    }
    
    print(f"输入参数尺寸:")
    print(f"  param_a['conv']: {param_a['conv'].shape}")
    print(f"  param_b['linear_weight']: {param_b['linear_weight'].shape}")
    
    # 测试参数注意力
    try:
        out_a = param_attention_l(param_a, param_b)
        print(f"输出参数尺寸: {out_a.shape}")
        print(f"输出形状是否匹配: {out_a.shape == param_a['conv'].shape}")
        print("✅ 参数注意力模块工作正常!")
        return True
    except Exception as e:
        print(f"❌ 参数注意力模块错误: {e}")
        return False

if __name__ == '__main__':
    test_param_attention()