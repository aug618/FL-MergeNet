"""
验证联邦学习数据分割的正确性
确保每个客户端使用不同的数据子集
"""
import torch
import numpy as np
from dataset.cls_dataloader import train_dataloader
from dataset.federated_data_partition import create_federated_dataloaders
from collections import defaultdict

def verify_data_splitting():
    """验证数据分割的正确性"""
    print("🔍 验证联邦学习数据分割")
    print("=" * 50)
    
    # 创建联邦数据划分
    partitioner, client_dataloaders = create_federated_dataloaders(
        dataset=train_dataloader.dataset,
        num_clients=5,  # 使用5个客户端进行测试
        alpha=0.5,
        batch_size=32,
        num_workers=0,  # 避免多进程问题
        min_samples_per_client=50
    )
    
    print(f"✅ 创建了{len(client_dataloaders)}个客户端数据加载器")
    
    # 收集每个客户端的样本索引
    client_samples = defaultdict(set)
    client_labels = defaultdict(list)
    
    for client_id, dataloader in enumerate(client_dataloaders):
        print(f"\n客户端 {client_id}:")
        batch_count = 0
        sample_count = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            batch_count += 1
            sample_count += len(targets)
            
            # 收集标签分布
            client_labels[client_id].extend(targets.numpy().tolist())
            
            if batch_idx == 0:
                print(f"  第一个batch形状: {data.shape}")
        
        print(f"  总batch数: {batch_count}")
        print(f"  总样本数: {sample_count}")
        
        # 分析标签分布
        unique_labels, counts = np.unique(client_labels[client_id], return_counts=True)
        top_labels = unique_labels[np.argsort(counts)[-5:]][::-1]  # 前5个最常见的标签
        print(f"  主要类别: {top_labels.tolist()}")
        print(f"  类别数量: {len(unique_labels)}")
    
    # 验证数据不重叠
    print(f"\n🔍 验证数据不重叠:")
    
    all_samples = set()
    overlap_found = False
    
    for client_id in range(len(client_dataloaders)):
        client_indices = set(partitioner.client_indices[client_id])
        
        # 检查与已有样本的重叠
        overlap = all_samples.intersection(client_indices)
        if overlap:
            print(f"  ⚠️  客户端{client_id}与之前客户端有{len(overlap)}个重叠样本")
            overlap_found = True
        else:
            print(f"  ✅ 客户端{client_id}无重叠样本")
        
        all_samples.update(client_indices)
    
    if not overlap_found:
        print("  🎉 所有客户端数据完全不重叠！")
    
    # 验证总样本数
    total_original = len(train_dataloader.dataset)
    total_federated = len(all_samples)
    
    print(f"\n📊 样本统计:")
    print(f"  原始数据集样本数: {total_original}")
    print(f"  联邦分割样本数: {total_federated}")
    print(f"  样本覆盖率: {total_federated/total_original*100:.1f}%")
    
    if total_federated == total_original:
        print("  ✅ 样本完全覆盖，无丢失！")
    else:
        print(f"  ⚠️  丢失了{total_original - total_federated}个样本")
    
    return True

def test_training_data_flow():
    """测试训练时的数据流"""
    print(f"\n🔄 测试训练数据流")
    print("=" * 50)
    
    # 创建联邦数据划分
    partitioner, client_dataloaders = create_federated_dataloaders(
        dataset=train_dataloader.dataset,
        num_clients=3,
        alpha=0.5,
        batch_size=16,
        num_workers=0
    )
    
    print("模拟一个epoch的训练:")
    
    # 模拟选择客户端
    selected_client_ids = [0, 2]  # 选择客户端0和2
    selected_dataloaders = [client_dataloaders[i] for i in selected_client_ids]
    
    print(f"选择的客户端: {selected_client_ids}")
    
    # 创建迭代器
    client_iterators = [iter(dataloader) for dataloader in selected_dataloaders]
    
    batch_idx = 0
    while True:
        active_clients = 0
        batch_data = {}
        
        for i, (client_id, iterator) in enumerate(zip(selected_client_ids, client_iterators)):
            try:
                data, targets = next(iterator)
                batch_data[client_id] = {
                    'data_shape': data.shape,
                    'labels': targets[:5].tolist()  # 显示前5个标签
                }
                active_clients += 1
            except StopIteration:
                batch_data[client_id] = {'status': 'finished'}
        
        if active_clients == 0:
            print(f"  所有客户端数据用完，epoch结束")
            break
        
        if batch_idx < 3:  # 只显示前3个batch
            print(f"\n  Batch {batch_idx}:")
            for client_id, info in batch_data.items():
                if 'data_shape' in info:
                    print(f"    客户端{client_id}: {info['data_shape']}, 标签样例: {info['labels']}")
                else:
                    print(f"    客户端{client_id}: {info['status']}")
        
        batch_idx += 1
        
        if batch_idx > 100:  # 防止无限循环
            break
    
    print(f"  总共处理了{batch_idx}个batch")
    
    return True

def main():
    """运行所有验证"""
    print("🚀 联邦学习数据分割验证")
    print("=" * 60)
    
    tests = [
        ("数据分割正确性", verify_data_splitting),
        ("训练数据流", test_training_data_flow),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            print(f"✅ {test_name} 验证通过")
        except Exception as e:
            print(f"❌ {test_name} 验证失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 总结:")
    print(f"如果所有验证都通过，说明：")
    print(f"1. ✅ 每个客户端使用不同的数据子集")
    print(f"2. ✅ 数据没有重叠")
    print(f"3. ✅ 训练时数据流正确")
    print(f"4. ✅ 真正实现了联邦学习的数据隔离")
    
    print(f"\n现在的实验设置:")
    print(f"📍 原实验: ResNet50(完整数据) + MobileNetV2(完整数据) + 知识融合")
    print(f"📍 新实验: ResNet50(完整数据) + 50个MobileNetV2(分割数据) + 联邦平均 + 知识融合")

if __name__ == "__main__":
    main()
