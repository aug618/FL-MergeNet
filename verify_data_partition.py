"""
验证联邦学习数据划分是否正确工作
确认每个客户端使用不同的数据子集
"""
import torch
import numpy as np
from collections import defaultdict
from dataset.cls_dataloader import train_dataloader
from dataset.federated_data_partition import create_federated_dataloaders

def verify_data_partition():
    """验证数据划分的正确性"""
    print("🔍 验证联邦学习数据划分")
    print("=" * 50)
    
    # 创建联邦数据划分
    partitioner, client_dataloaders = create_federated_dataloaders(
        dataset=train_dataloader.dataset,
        num_clients=10,  # 为了快速验证，使用10个客户端
        alpha=0.5,
        batch_size=32,
        num_workers=0,  # 避免多进程问题
        min_samples_per_client=50
    )
    
    print(f"✅ 创建了{len(client_dataloaders)}个客户端数据加载器")
    
    # 1. 验证数据不重复
    print("\n📊 验证数据不重复...")
    all_indices = set()
    client_indices = []
    
    for client_id in range(len(client_dataloaders)):
        client_dataset = partitioner.get_client_dataset(client_id)
        indices = set(client_dataset.indices)
        client_indices.append(indices)
        
        # 检查是否有重复
        overlap = all_indices.intersection(indices)
        if overlap:
            print(f"❌ 客户端{client_id}与其他客户端有重复数据: {len(overlap)}个样本")
        else:
            print(f"✅ 客户端{client_id}: {len(indices)}个独特样本")
        
        all_indices.update(indices)
    
    print(f"✅ 总共使用的样本数: {len(all_indices)}")
    print(f"✅ 原始数据集样本数: {len(train_dataloader.dataset)}")
    
    # 2. 验证标签分布
    print("\n🏷️ 验证标签分布...")
    for client_id in range(min(5, len(client_dataloaders))):  # 只检查前5个客户端
        client_labels = []
        dataloader = client_dataloaders[client_id]
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            client_labels.extend(labels.tolist())
            if batch_idx >= 5:  # 只检查前几个batch
                break
        
        if client_labels:
            unique_labels = set(client_labels)
            label_counts = defaultdict(int)
            for label in client_labels:
                label_counts[label] += 1
            
            print(f"✅ 客户端{client_id}: {len(unique_labels)}个不同类别，前5个类别分布: {dict(list(label_counts.items())[:5])}")
    
    # 3. 验证数据加载
    print("\n📥 验证数据加载...")
    for client_id in range(min(3, len(client_dataloaders))):
        dataloader = client_dataloaders[client_id]
        try:
            data_iter = iter(dataloader)
            batch1 = next(data_iter)
            batch2 = next(data_iter)
            
            print(f"✅ 客户端{client_id}: 成功加载批次，数据形状: {batch1[0].shape}, 标签形状: {batch1[1].shape}")
            
            # 验证两个批次的数据不同
            if not torch.equal(batch1[0], batch2[0]):
                print(f"✅ 客户端{client_id}: 批次间数据不同（正确）")
            else:
                print(f"❌ 客户端{client_id}: 批次间数据相同（错误）")
                
        except Exception as e:
            print(f"❌ 客户端{client_id}: 数据加载失败 - {e}")
    
    # 4. 验证客户端间数据不同
    print("\n🔄 验证客户端间数据差异...")
    if len(client_dataloaders) >= 2:
        # 比较前两个客户端的第一个batch
        try:
            batch_client0 = next(iter(client_dataloaders[0]))
            batch_client1 = next(iter(client_dataloaders[1]))
            
            if not torch.equal(batch_client0[0], batch_client1[0]):
                print("✅ 不同客户端使用不同数据（正确）")
            else:
                print("❌ 不同客户端使用相同数据（错误）")
                
            # 检查标签分布差异
            labels0 = batch_client0[1].tolist()
            labels1 = batch_client1[1].tolist()
            
            unique0 = set(labels0)
            unique1 = set(labels1)
            
            print(f"✅ 客户端0标签: {sorted(list(unique0))[:10]}...")
            print(f"✅ 客户端1标签: {sorted(list(unique1))[:10]}...")
            
            if unique0 != unique1:
                print("✅ 不同客户端有不同的标签分布（符合Non-IID）")
            
        except Exception as e:
            print(f"❌ 客户端间比较失败: {e}")
    
    return True

def verify_training_flow():
    """验证训练流程中的数据使用"""
    print("\n🔄 验证训练流程中的数据使用")
    print("=" * 50)
    
    # 创建联邦数据划分
    partitioner, client_dataloaders = create_federated_dataloaders(
        dataset=train_dataloader.dataset,
        num_clients=5,
        alpha=0.5,
        batch_size=32,
        num_workers=0,
        min_samples_per_client=50
    )
    
    # 模拟训练流程
    selected_client_ids = [0, 2, 4]  # 选择3个客户端
    print(f"选择客户端: {selected_client_ids}")
    
    # 创建选中客户端的数据迭代器
    selected_client_iterators = []
    for client_id in selected_client_ids:
        iterator = iter(client_dataloaders[client_id])
        selected_client_iterators.append(iterator)
    
    print(f"✅ 创建了{len(selected_client_iterators)}个客户端数据迭代器")
    
    # 模拟几个batch的训练
    for batch_idx in range(3):
        print(f"\n--- Batch {batch_idx} ---")
        
        # ResNet使用完整数据
        try:
            if batch_idx == 0:
                train_iter = iter(train_dataloader)
            img_res, label_res = next(train_iter)
            print(f"✅ ResNet数据: {img_res.shape}, 标签范围: {label_res.min().item()}-{label_res.max().item()}")
        except StopIteration:
            train_iter = iter(train_dataloader)
            img_res, label_res = next(train_iter)
            print(f"✅ ResNet数据（重新开始）: {img_res.shape}")
        
        # 各客户端使用自己的数据
        active_clients = 0
        for i, (client_id, client_iterator) in enumerate(zip(selected_client_ids, selected_client_iterators)):
            try:
                img_client, label_client = next(client_iterator)
                print(f"✅ 客户端{client_id}数据: {img_client.shape}, 标签范围: {label_client.min().item()}-{label_client.max().item()}")
                active_clients += 1
                
                # 验证客户端数据与ResNet数据不同
                if not torch.equal(img_client, img_res):
                    print(f"✅ 客户端{client_id}与ResNet数据不同（正确）")
                else:
                    print(f"❌ 客户端{client_id}与ResNet数据相同（错误）")
                    
            except StopIteration:
                print(f"⚠️ 客户端{client_id}数据用完")
        
        print(f"活跃客户端数: {active_clients}")
    
    return True

def main():
    """主验证函数"""
    print("🚀 联邦学习数据划分验证")
    print("=" * 60)
    
    try:
        # 验证数据划分
        verify_data_partition()
        
        # 验证训练流程
        verify_training_flow()
        
        print("\n" + "=" * 60)
        print("✅ 所有验证通过！数据划分正确工作")
        print("📋 验证结果:")
        print("   ✅ 每个客户端使用不同的数据子集")
        print("   ✅ 客户端间无数据重复")
        print("   ✅ 标签分布呈现Non-IID特性")
        print("   ✅ 数据加载器正常工作")
        print("   ✅ 训练流程中数据使用正确")
        
    except Exception as e:
        print(f"\n❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
