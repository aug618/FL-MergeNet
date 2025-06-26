"""
测试联邦学习 + MergeNet 环境配置
验证所有组件是否正确安装和配置
"""
import sys
import torch
import importlib.util
import yaml
import os

def test_imports():
    """测试所有必要的包导入"""
    print("=== 测试包导入 ===")
    
    required_packages = [
        'torch',
        'torchvision', 
        'flwr',
        'numpy',
        'tqdm',
        'yaml',
        'matplotlib'
    ]
    
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✅ {package} - 导入成功")
        except ImportError as e:
            print(f"❌ {package} - 导入失败: {e}")
            return False
    
    return True

def test_models():
    """测试模型导入和初始化"""
    print("\n=== 测试模型 ===")
    
    try:
        from model.MobileNet_v2 import mobilenetv2
        from model.ResNet import resnet50
        from model.param_attention import ParamAttention
        
        print("✅ 模型导入成功")
        
        # 测试模型初始化
        mbv2 = mobilenetv2()
        res = resnet50()
        
        print(f"✅ MobileNetV2 参数量: {sum(p.numel() for p in mbv2.parameters()):,}")
        print(f"✅ ResNet50 参数量: {sum(p.numel() for p in res.parameters()):,}")
        
        # 测试参数注意力模块
        config = {
            'd_attention': 64,
            'h': 8,
            'num_layers': 2,
            'a_size_conv': [160, 960],
            'a_size_linear': [100, 1280],
            'b_size_linear': [100, 2048],
            'mode': 5
        }
        
        param_attention = ParamAttention(config, mode='a')
        print("✅ ParamAttention 初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n=== 测试数据加载器 ===")
    
    try:
        from dataset.cls_dataloader import train_dataloader, test_dataloader
        
        print(f"✅ 训练数据集大小: {len(train_dataloader.dataset)}")
        print(f"✅ 测试数据集大小: {len(test_dataloader.dataset)}")
        print(f"✅ 训练批次数: {len(train_dataloader)}")
        print(f"✅ 测试批次数: {len(test_dataloader)}")
        
        # 测试数据加载
        for data, target in train_dataloader:
            print(f"✅ 数据形状: {data.shape}, 标签形状: {target.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False

def test_config_files():
    """测试配置文件"""
    print("\n=== 测试配置文件 ===")
    
    config_files = [
        'config/param_attention_config.yaml',
        'config/federated_mergenet_config.py'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} 存在")
            
            if config_file.endswith('.yaml'):
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.load(f, Loader=yaml.Loader)
                    print(f"✅ {config_file} 格式正确")
                except Exception as e:
                    print(f"❌ {config_file} 格式错误: {e}")
                    return False
        else:
            print(f"❌ {config_file} 不存在")
            return False
    
    return True

def test_cuda():
    """测试CUDA可用性"""
    print("\n=== 测试CUDA ===")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"✅ CUDA 设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"✅ 设备 {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA 不可用，将使用CPU训练")
    
    return True

def test_directories():
    """测试目录结构"""
    print("\n=== 测试目录结构 ===")
    
    required_dirs = [
        'logs',
        'checkpoints', 
        'model',
        'dataset',
        'config'
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ 目录存在")
        else:
            print(f"❌ {dir_name}/ 目录不存在")
            return False
    
    return True

def test_federated_components():
    """测试联邦学习组件"""
    print("\n=== 测试联邦学习组件 ===")
    
    # 测试Flower
    try:
        import flwr as fl
        print(f"✅ Flower 版本: {fl.__version__}")
    except Exception as e:
        print(f"❌ Flower 导入失败: {e}")
        return False
    
    # 测试我们的联邦学习文件是否存在
    federated_files = [
        'federated_mergenet_server.py',
        'federated_mergenet_client.py',
        'run_federated_mergenet.py'
    ]
    
    for file_name in federated_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 不存在")
            return False
    
    return True

def main():
    """运行所有测试"""
    print("联邦学习 + MergeNet 环境测试")
    print("=" * 50)
    
    tests = [
        ("包导入", test_imports),
        ("模型", test_models), 
        ("数据加载器", test_data_loader),
        ("配置文件", test_config_files),
        ("CUDA", test_cuda),
        ("目录结构", test_directories),
        ("联邦学习组件", test_federated_components)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置正确，可以开始实验。")
        print("\n下一步:")
        print("1. 运行 ./start_federated_experiment.sh 开始实验")
        print("2. 或者手动启动: python run_federated_mergenet.py --num_clients 3")
    else:
        print("⚠️  部分测试失败，请检查配置。")
        
    return passed == total

if __name__ == "__main__":
    main()
