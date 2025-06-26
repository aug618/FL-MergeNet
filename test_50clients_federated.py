"""
测试50个客户端联邦学习+MergeNet的基本功能
"""
import torch
import yaml
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from dataset.cls_dataloader import train_dataloader
from dataset.federated_data_partition import create_federated_dataloaders, select_random_clients

def test_federated_data_partition():
    """测试联邦数据划分功能"""
    print("=== 测试联邦数据划分 ===")
    
    # 创建联邦数据划分
    partitioner, client_dataloaders = create_federated_dataloaders(
        dataset=train_dataloader.dataset,
        num_clients=50,
        alpha=0.5,
        batch_size=32,
        num_workers=2,
        min_samples_per_client=50
    )
    
    # 检查数据划分
    stats = partitioner.get_data_distribution()
    print(f"✅ 创建了{len(client_dataloaders)}个客户端数据加载器")
    print(f"✅ 客户端样本数范围: {min(stats['client_sizes'])} - {max(stats['client_sizes'])}")
    
    # 测试随机选择客户端
    selected = select_random_clients(50, 15, seed=42)
    print(f"✅ 随机选择15个客户端: {selected[:5]}...等")
    
    # 测试数据加载
    first_client_loader = client_dataloaders[0]
    for batch_idx, (data, target) in enumerate(first_client_loader):
        if batch_idx == 0:
            print(f"✅ 第一个客户端数据形状: {data.shape}, 标签形状: {target.shape}")
            break
    
    return True

def test_model_creation():
    """测试50个客户端模型创建"""
    print("\n=== 测试模型创建 ===")
    
    # 创建50个客户端模型
    num_clients = 50
    client_models = [mobilenetv2() for _ in range(num_clients)]
    res_model = resnet50()
    
    print(f"✅ 创建了{len(client_models)}个MobileNetV2客户端模型")
    print(f"✅ ResNet50模型参数量: {sum(p.numel() for p in res_model.parameters()):,}")
    
    # 测试模型前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 移动第一个客户端模型到设备
    client_models[0].to(device)
    res_model.to(device)
    
    # 创建测试数据
    test_input = torch.randn(2, 3, 32, 32).to(device)
    
    with torch.no_grad():
        client_output = client_models[0](test_input)
        res_output = res_model(test_input)
    
    print(f"✅ 客户端模型输出形状: {client_output.shape}")
    print(f"✅ ResNet模型输出形状: {res_output.shape}")
    
    return True

def test_param_attention():
    """测试参数注意力模块"""
    print("\n=== 测试参数注意力模块 ===")
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    mbv2 = mobilenetv2().to(device)
    res = resnet50().to(device)
    param_attention = ParamAttention(config, mode='a').to(device)
    
    # 提取参数
    param_a = {
        'conv': mbv2.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device),
    }
    param_b = {
        'linear_weight': res.fc.weight.data.clone().detach().requires_grad_(True).to(device),
    }
    
    print(f"✅ 输入参数A形状: {param_a['conv'].shape}")
    print(f"✅ 输入参数B形状: {param_b['linear_weight'].shape}")
    
    # 测试参数注意力
    try:
        out_a = param_attention(param_a, param_b)
        print(f"✅ 参数注意力输出形状: {out_a.shape}")
        print(f"✅ 输出与输入A形状匹配: {out_a.shape == param_a['conv'].shape}")
        return True
    except Exception as e:
        print(f"❌ 参数注意力测试失败: {e}")
        return False

def test_federated_averaging():
    """测试联邦平均功能"""
    print("\n=== 测试联邦平均 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建3个测试客户端模型
    client_models = [mobilenetv2().to(device) for _ in range(3)]
    
    # 给模型添加一些差异
    for i, model in enumerate(client_models):
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01 * (i + 1))
    
    # 测试联邦平均
    from federated_batch_mergenet import federated_average_models
    
    avg_state_dict = federated_average_models(client_models, device)
    
    if avg_state_dict is not None:
        print(f"✅ 联邦平均成功，包含{len(avg_state_dict)}个参数层")
        
        # 创建平均模型并加载参数
        avg_model = mobilenetv2().to(device)
        avg_model.load_state_dict(avg_state_dict)
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            output = avg_model(test_input)
        print(f"✅ 平均模型前向传播成功，输出形状: {output.shape}")
        return True
    else:
        print("❌ 联邦平均失败")
        return False

def test_configuration():
    """测试配置文件加载"""
    print("\n=== 测试配置文件 ===")
    
    try:
        config = yaml.load(open('config/param_attention_config.yaml', 'r'), Loader=yaml.Loader)
        print(f"✅ 配置文件加载成功")
        print(f"✅ 配置参数: {list(config.keys())}")
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 50个客户端联邦学习+MergeNet 功能测试")
    print("=" * 60)
    
    tests = [
        ("配置文件", test_configuration),
        ("联邦数据划分", test_federated_data_partition),
        ("模型创建", test_model_creation),
        ("参数注意力", test_param_attention),
        ("联邦平均", test_federated_averaging),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("🔍 测试总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统已准备好运行50个客户端的联邦学习实验")
        print(f"\n🚀 运行完整实验:")
        print(f"   python federated_batch_mergenet.py")
    else:
        print("⚠️  部分测试失败，请检查系统配置")
        
    return passed == total

if __name__ == "__main__":
    main()
