"""
对比演示：原始MergeNet vs 联邦batch-level MergeNet
展示每f个batch时的不同处理方式
"""
import torch
import torch.nn as nn
import yaml
import numpy as np
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from dataset.cls_dataloader import train_dataloader

def original_mergenet_demo():
    """原始MergeNet方法演示"""
    print("🔸 原始MergeNet方法:")
    print("   每f个batch → 直接从ResNet50融合知识到MobileNetV2")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型初始化
    mbv2 = mobilenetv2().to(device)
    res = resnet50().to(device)
    
    config = {
        'd_attention': 64, 'h': 8, 'num_layers': 2,
        'a_size_conv': [160, 960], 'a_size_linear': [100, 1280],
        'b_size_linear': [100, 2048], 'mode': 5
    }
    param_attention = ParamAttention(config, mode='a').to(device)
    
    f = 2  # 每2个batch融合一次
    cnt = 0
    
    print(f"   MobileNetV2初始参数示例: {mbv2.stage6[2].residual[6].weight[0, 0, :3].detach().cpu().numpy()}")
    
    train_iter = iter(train_dataloader)
    for batch_idx in range(5):  # 模拟5个batch
        try:
            data, target = next(train_iter)
            data, target = data.to(device), target.to(device)
            
            if cnt % f == 0:
                print(f"   📝 Batch {cnt}: 执行知识融合")
                
                # 提取参数
                param_a = {
                    'conv': mbv2.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device),
                }
                param_b = {
                    'linear_weight': res.fc.weight.data.clone().detach().requires_grad_(True).to(device),
                }
                
                # 知识融合
                with torch.no_grad():
                    out_a = param_attention(param_a, param_b)
                    
                    # 更新MobileNetV2参数
                    new_param_mbv = {'stage6.2.residual.6.weight': out_a}
                    mbv2.load_state_dict(new_param_mbv, strict=False)
                
                print(f"   ✅ 知识融合完成，参数已更新")
                print(f"   MobileNetV2更新后参数: {mbv2.stage6[2].residual[6].weight[0, 0, :3].detach().cpu().numpy()}")
            else:
                print(f"   📝 Batch {cnt}: 常规训练")
            
            cnt += 1
            
        except StopIteration:
            break

def federated_batch_mergenet_demo():
    """联邦batch-level MergeNet方法演示"""
    print("\n🔹 联邦batch-level MergeNet方法:")
    print("   每f个batch → 联邦平均所有客户端 → 从ResNet50融合知识 → 分发给客户端")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型初始化
    num_clients = 3
    client_models = [mobilenetv2().to(device) for _ in range(num_clients)]
    res = resnet50().to(device)
    
    # 给客户端添加一些差异性（模拟不同客户端的训练状态）
    for i, model in enumerate(client_models):
        with torch.no_grad():
            model.stage6[2].residual[6].weight += torch.randn_like(model.stage6[2].residual[6].weight) * 0.01 * (i + 1)
    
    config = {
        'd_attention': 64, 'h': 8, 'num_layers': 2,
        'a_size_conv': [160, 960], 'a_size_linear': [100, 1280],
        'b_size_linear': [100, 2048], 'mode': 5
    }
    param_attention = ParamAttention(config, mode='a').to(device)
    
    f = 2  # 每2个batch进行联邦平均+融合
    cnt = 0
    
    print(f"   客户端初始参数示例:")
    for i, model in enumerate(client_models):
        print(f"     Client {i}: {model.stage6[2].residual[6].weight[0, 0, :3].detach().cpu().numpy()}")
    
    train_iter = iter(train_dataloader)
    for batch_idx in range(5):  # 模拟5个batch
        try:
            data, target = next(train_iter)
            data, target = data.to(device), target.to(device)
            
            if cnt % f == 0:
                print(f"   📝 Batch {cnt}: 执行联邦平均 + 知识融合")
                
                # 1. 联邦平均
                print("     🔄 步骤1: 联邦平均客户端模型")
                avg_state_dict = {}
                first_model = client_models[0]
                
                for key in first_model.state_dict().keys():
                    param_type = first_model.state_dict()[key].dtype
                    avg_state_dict[key] = torch.zeros_like(first_model.state_dict()[key], dtype=torch.float32, device=device)
                    
                    for client_model in client_models:
                        avg_state_dict[key] += client_model.state_dict()[key].float()
                    
                    avg_state_dict[key] /= len(client_models)
                    
                    if param_type != torch.float32:
                        avg_state_dict[key] = avg_state_dict[key].to(param_type)
                
                # 创建平均模型
                averaged_model = mobilenetv2().to(device)
                averaged_model.load_state_dict(avg_state_dict)
                
                print(f"     平均后参数: {averaged_model.stage6[2].residual[6].weight[0, 0, :3].detach().cpu().numpy()}")
                
                # 2. 知识融合
                print("     🧠 步骤2: MergeNet知识融合")
                param_a = {
                    'conv': averaged_model.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device),
                }
                param_b = {
                    'linear_weight': res.fc.weight.data.clone().detach().requires_grad_(True).to(device),
                }
                
                with torch.no_grad():
                    out_a = param_attention(param_a, param_b)
                    new_param_dict = {'stage6.2.residual.6.weight': out_a}
                    averaged_model.load_state_dict(new_param_dict, strict=False)
                
                print(f"     融合后参数: {averaged_model.stage6[2].residual[6].weight[0, 0, :3].detach().cpu().numpy()}")
                
                # 3. 分发给客户端
                print("     📤 步骤3: 分发融合后参数给所有客户端")
                fused_state_dict = averaged_model.state_dict()
                for i, client_model in enumerate(client_models):
                    client_model.load_state_dict(fused_state_dict)
                    print(f"       Client {i} 更新后: {client_model.stage6[2].residual[6].weight[0, 0, :3].detach().cpu().numpy()}")
                
                print("   ✅ 联邦平均 + 知识融合完成")
            else:
                print(f"   📝 Batch {cnt}: 客户端本地训练")
                # 模拟本地训练导致的参数差异
                for model in client_models:
                    with torch.no_grad():
                        model.stage6[2].residual[6].weight += torch.randn_like(model.stage6[2].residual[6].weight) * 0.005
            
            cnt += 1
            
        except StopIteration:
            break

def main():
    """主演示函数"""
    print("=" * 60)
    print("📊 原始MergeNet vs 联邦batch-level MergeNet 对比演示")
    print("=" * 60)
    
    # 原始方法演示
    original_mergenet_demo()
    
    # 联邦方法演示
    federated_batch_mergenet_demo()
    
    print("\n" + "=" * 60)
    print("📈 关键差异总结:")
    print("=" * 60)
    print("🔸 原始方法:")
    print("   - 单一MobileNetV2模型")
    print("   - 每f个batch直接融合知识")
    print("   - 简单直接，但缺乏分布式优势")
    
    print("\n🔹 联邦batch-level方法:")
    print("   - 多个客户端MobileNetV2模型")
    print("   - 每f个batch先联邦平均，再融合知识，最后分发")
    print("   - 结合了联邦学习的分布式优势和MergeNet的知识融合")
    print("   - 可能获得更好的泛化能力和鲁棒性")
    
    print(f"\n🚀 要运行完整对比实验:")
    print(f"   原始方法: python run_res50_mbv2.py")
    print(f"   联邦方法: python federated_batch_mergenet.py")

if __name__ == "__main__":
    main()
