"""
实验体系完整性验证脚本
检查所有必要的文件、依赖和配置是否正确
"""
import os
import sys
import importlib
import torch
from typing import List, Dict

def check_file_exists(file_path: str) -> bool:
    """检查文件是否存在"""
    return os.path.exists(file_path)

def check_import(module_name: str) -> bool:
    """检查模块是否能正常导入"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_experiment_files() -> Dict[str, bool]:
    """检查实验脚本文件"""
    files = {
        'run_baseline_mobilenetv2.py': '基线实验脚本',
        'run_res50_mbv2.py': 'MergeNet实验脚本',
        'federated_batch_mergenet.py': '联邦MergeNet实验脚本',
        'pure_federated_learning.py': '纯联邦学习实验脚本',
        'compare_results.py': '结果对比分析脚本',
        'auto_experiment_runner.py': '自动化实验运行器',
        'demo_comparison.py': '实验演示脚本',
        'run_experiments.sh': '快速启动脚本'
    }
    
    results = {}
    print("📁 检查实验脚本文件...")
    for file_path, description in files.items():
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path} - {description}")
        results[file_path] = exists
    
    return results

def check_model_files() -> Dict[str, bool]:
    """检查模型文件"""
    files = {
        'model/MobileNet_v2.py': 'MobileNetV2模型',
        'model/ResNet.py': 'ResNet模型',
        'model/param_attention.py': '参数注意力模块',
        'config/param_attention_config.yaml': 'MergeNet配置',
        'config/federated_mergenet_config.py': '联邦学习配置'
    }
    
    results = {}
    print("\n🏗️ 检查模型和配置文件...")
    for file_path, description in files.items():
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path} - {description}")
        results[file_path] = exists
    
    return results

def check_dataset_files() -> Dict[str, bool]:
    """检查数据集文件"""
    files = {
        'dataset/cls_dataloader.py': '数据加载器',
        'dataset/federated_data_partition.py': '联邦数据划分',
        'data/cifar-100-python.tar.gz': 'CIFAR-100数据集'
    }
    
    results = {}
    print("\n📊 检查数据集文件...")
    for file_path, description in files.items():
        exists = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path} - {description}")
        results[file_path] = exists
    
    return results

def check_dependencies() -> Dict[str, bool]:
    """检查Python依赖"""
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM',
        'yaml': 'PyYAML',
        'swanlab': 'SwanLab'
    }
    
    results = {}
    print("\n📦 检查Python依赖...")
    for module, description in dependencies.items():
        can_import = check_import(module)
        status = "✅" if can_import else "❌"
        print(f"  {status} {module} - {description}")
        results[module] = can_import
    
    return results

def check_hardware() -> Dict[str, bool]:
    """检查硬件环境"""
    results = {}
    print("\n🖥️ 检查硬件环境...")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    status = "✅" if cuda_available else "⚠️"
    print(f"  {status} CUDA可用: {cuda_available}")
    results['cuda'] = cuda_available
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"  📱 GPU数量: {device_count}")
        results['gpu_count'] = device_count
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"    GPU {i}: {gpu_name}")
    
    # 检查内存
    if hasattr(torch.cuda, 'get_device_properties'):
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1024**3  # GB
            print(f"  💾 GPU内存: {total_memory:.1f} GB")
            results['gpu_memory'] = total_memory
    
    return results

def check_directories():
    """检查和创建必要的目录"""
    directories = ['logs', 'checkpoints', 'results']
    
    print("\n📁 检查目录结构...")
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  ✅ 创建目录: {dir_name}")
        else:
            print(f"  ✅ 目录已存在: {dir_name}")

def run_simple_test():
    """运行简单的功能测试"""
    print("\n🧪 运行简单功能测试...")
    
    try:
        # 测试模型加载
        from model.MobileNet_v2 import mobilenetv2
        model = mobilenetv2()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ MobileNetV2加载成功，参数量: {param_count:,}")
        
        # 测试数据加载
        from dataset.cls_dataloader import train_dataloader, test_dataloader
        print(f"  ✅ 数据加载器创建成功")
        print(f"    训练集批次数: {len(train_dataloader)}")
        print(f"    测试集批次数: {len(test_dataloader)}")
        
        # 测试联邦数据划分
        from dataset.federated_data_partition import create_federated_dataloaders
        partitioner, client_loaders = create_federated_dataloaders(
            train_dataloader.dataset, 5, 0.5, 32, 1, 50  # 小规模测试
        )
        print(f"  ✅ 联邦数据划分测试成功，创建{len(client_loaders)}个客户端")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 功能测试失败: {e}")
        return False

def generate_report(results: Dict[str, Dict[str, bool]]):
    """生成检查报告"""
    print("\n📋 生成完整性检查报告...")
    
    all_passed = True
    report_lines = []
    
    for category, checks in results.items():
        category_passed = all(checks.values())
        all_passed &= category_passed
        
        status = "✅ 通过" if category_passed else "❌ 失败"
        report_lines.append(f"{category}: {status}")
        
        for item, passed in checks.items():
            item_status = "✅" if passed else "❌"
            report_lines.append(f"  {item_status} {item}")
    
    # 保存报告
    report_content = [
        "# 实验体系完整性检查报告",
        f"检查时间: {__import__('datetime').datetime.now()}",
        "",
        "## 检查结果",
        ""
    ] + report_lines
    
    with open('results/system_check_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"\n📊 检查报告已保存到: results/system_check_report.txt")
    
    return all_passed

def main():
    """主函数"""
    print("🔍 联邦学习 + MergeNet 实验体系完整性检查")
    print("=" * 50)
    
    # 检查各个组件
    results = {
        '实验脚本': check_experiment_files(),
        '模型配置': check_model_files(),
        '数据集': check_dataset_files(),
        'Python依赖': check_dependencies(),
        '硬件环境': check_hardware()
    }
    
    # 创建目录
    check_directories()
    
    # 功能测试
    test_passed = run_simple_test()
    results['功能测试'] = {'简单测试': test_passed}
    
    # 生成报告
    all_passed = generate_report(results)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有检查都通过！实验体系已就绪。")
        print("\n🚀 可以开始运行实验:")
        print("  ./run_experiments.sh")
        print("  python auto_experiment_runner.py --run-all")
    else:
        print("⚠️ 部分检查未通过，请根据上述信息修复问题。")
        print("\n🔧 常见解决方案:")
        print("  pip install -r requirements_federated.txt")
        print("  python test_environment.py")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
