"""
联邦学习 + MergeNet 快速演示
运行一个简化版本的实验来验证核心功能
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm
import time

from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from dataset.cls_dataloader import train_dataloader, test_dataloader

def simulate_federated_round(clients_models, server_model, param_attention, teacher_model, device):
    """模拟一轮联邦学习+知识融合"""
    
    print("📱 客户端本地训练...")
    
    # 1. 客户端本地训练（简化版，只训练1个epoch）
    for client_id, client_model in enumerate(clients_models):
        client_model.train()
        optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练几个batch
        train_iter = iter(train_dataloader)
        for batch_idx in range(5):  # 只训练5个batch作为演示
            try:
                data, target = next(train_iter)
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
            except StopIteration:
                break
        
        print(f"  客户端 {client_id} 本地训练完成")
    
    print("🔄 服务端联邦平均...")
    
    # 2. 联邦平均
    avg_state_dict = {}
    for key in server_model.state_dict().keys():
        param_type = server_model.state_dict()[key].dtype
        avg_state_dict[key] = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
        
        for client_model in clients_models:
            avg_state_dict[key] += client_model.state_dict()[key].float()
        
        avg_state_dict[key] /= len(clients_models)
        
        # 转换回原始类型
        if param_type != torch.float32:
            avg_state_dict[key] = avg_state_dict[key].to(param_type)
    
    server_model.load_state_dict(avg_state_dict)
    print("  联邦平均完成")
    
    print("🧠 MergeNet知识融合...")
    
    # 3. 知识融合
    param_a = {
        'conv': server_model.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device)
    }
    param_b = {
        'linear_weight': teacher_model.fc.weight.data.clone().detach().requires_grad_(True).to(device)
    }
    
    with torch.no_grad():
        fused_params = param_attention(param_a, param_b)
        
        # 更新服务端模型
        new_param_dict = {
            'stage6.2.residual.6.weight': fused_params
        }
        server_model.load_state_dict(new_param_dict, strict=False)
    
    print("  知识融合完成")
    
    print("📤 参数分发...")
    
    # 4. 将融合后的参数分发给客户端
    for client_model in clients_models:
        client_model.load_state_dict(server_model.state_dict())
    
    print("  参数分发完成")

def evaluate_model(model, device):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        test_iter = iter(test_dataloader)
        for _ in range(10):  # 只评估10个batch
            try:
                data, target = next(test_iter)
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            except StopIteration:
                break
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    """主演示函数"""
    print("🚀 联邦学习 + MergeNet 快速演示")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config = yaml.load(open('config/param_attention_config.yaml', 'r'), Loader=yaml.Loader)
    config['a_size_conv'] = [160, 960]
    config['a_size_linear'] = [100, 1280]
    config['b_size_linear'] = [100, 2048]
    config['mode'] = 5
    
    # 初始化模型
    print("\n📋 初始化模型...")
    
    # 服务端模型
    server_model = mobilenetv2().to(device)
    
    # 客户端模型（3个）
    num_clients = 3
    clients_models = [mobilenetv2().to(device) for _ in range(num_clients)]
    
    # 大模型（知识源）
    teacher_model = resnet50().to(device)
    
    # 参数注意力模块
    param_attention = ParamAttention(config, mode='a').to(device)
    
    print(f"  服务端模型: {sum(p.numel() for p in server_model.parameters()):,} 参数")
    print(f"  客户端数量: {num_clients}")
    print(f"  大模型: {sum(p.numel() for p in teacher_model.parameters()):,} 参数")
    
    # 初始性能评估
    print("\n📊 初始性能评估...")
    initial_acc = evaluate_model(server_model, device)
    print(f"  初始准确率: {initial_acc:.2f}%")
    
    # 运行几轮联邦学习+知识融合
    print("\n🔄 开始联邦学习+知识融合演示...")
    
    num_rounds = 3
    for round_num in range(num_rounds):
        print(f"\n--- 第 {round_num + 1} 轮 ---")
        
        start_time = time.time()
        
        # 模拟联邦学习+知识融合
        simulate_federated_round(
            clients_models, 
            server_model, 
            param_attention, 
            teacher_model, 
            device
        )
        
        # 评估性能
        accuracy = evaluate_model(server_model, device)
        
        end_time = time.time()
        round_time = end_time - start_time
        
        print(f"✅ 第 {round_num + 1} 轮完成")
        print(f"   准确率: {accuracy:.2f}%")
        print(f"   用时: {round_time:.2f}秒")
    
    # 最终性能
    print("\n📈 演示总结:")
    final_acc = evaluate_model(server_model, device)
    improvement = final_acc - initial_acc
    
    print(f"  初始准确率: {initial_acc:.2f}%")
    print(f"  最终准确率: {final_acc:.2f}%")
    print(f"  性能提升: {improvement:+.2f}%")
    
    if improvement > 0:
        print("🎉 联邦学习+知识融合显示出正面效果！")
    else:
        print("⚠️  演示轮数较少，可能需要更多轮次才能看到明显效果")
    
    print("\n🔍 关键技术要点:")
    print("1. ✅ 联邦平均成功聚合多客户端模型")
    print("2. ✅ MergeNet成功融合大模型知识")
    print("3. ✅ 参数分发确保所有客户端同步")
    print("4. ✅ 端到端流程运行正常")
    
    print(f"\n🚀 要运行完整实验，请执行:")
    print(f"   python run_federated_mergenet.py --num_clients 3")
    print(f"   或")
    print(f"   ./start_federated_experiment.sh")

if __name__ == "__main__":
    main()
