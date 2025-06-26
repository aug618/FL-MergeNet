"""
联邦学习数据划分模块
为50个客户端创建异构数据分布（Non-IID）
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class FederatedDataPartitioner:
    """联邦学习数据划分器"""
    
    def __init__(self, dataset, num_clients=50, alpha=0.5, min_samples_per_client=50):
        """
        初始化数据划分器
        
        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            alpha: Dirichlet分布参数，越小数据越异构
            min_samples_per_client: 每个客户端最少样本数
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        
        # 获取标签信息
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.labels = np.array(dataset.labels)
        else:
            # 如果没有直接的标签属性，遍历数据集获取标签
            labels = []
            for _, label in dataset:
                labels.append(label)
            self.labels = np.array(labels)
        
        self.num_classes = len(np.unique(self.labels))
        self.client_indices = self._partition_data()
        
        logger.info(f"数据划分完成: {num_clients}个客户端，{self.num_classes}个类别")
        logger.info(f"Alpha: {alpha}，最少样本数: {min_samples_per_client}")
        
    def _partition_data(self):
        """使用Dirichlet分布划分数据"""
        
        # 按类别组织数据索引
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        
        client_indices = [[] for _ in range(self.num_clients)]
        
        # 为每个类别分配数据到客户端
        for class_id in range(self.num_classes):
            class_samples = class_indices[class_id]
            np.random.shuffle(class_samples)
            
            # 使用Dirichlet分布生成每个客户端的数据比例
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # 确保每个客户端至少有一些数据
            proportions = proportions * len(class_samples)
            proportions = proportions.astype(int)
            
            # 调整确保所有样本都被分配
            diff = len(class_samples) - proportions.sum()
            for i in range(abs(diff)):
                if diff > 0:
                    proportions[i % self.num_clients] += 1
                else:
                    if proportions[i % self.num_clients] > 0:
                        proportions[i % self.num_clients] -= 1
            
            # 分配样本到客户端
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + proportions[client_id]
                client_indices[client_id].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx
        
        # 确保每个客户端有最少样本数
        self._ensure_min_samples(client_indices)
        
        # 打乱每个客户端的数据
        for client_id in range(self.num_clients):
            random.shuffle(client_indices[client_id])
        
        return client_indices
    
    def _ensure_min_samples(self, client_indices):
        """确保每个客户端有最少样本数"""
        for client_id in range(self.num_clients):
            if len(client_indices[client_id]) < self.min_samples_per_client:
                # 从其他客户端借用数据
                deficit = self.min_samples_per_client - len(client_indices[client_id])
                borrowed = 0
                
                for other_client in range(self.num_clients):
                    if other_client != client_id and len(client_indices[other_client]) > self.min_samples_per_client + 10:
                        # 可以借用的数量
                        can_borrow = min(
                            deficit - borrowed,
                            len(client_indices[other_client]) - self.min_samples_per_client
                        )
                        
                        if can_borrow > 0:
                            # 借用数据
                            borrowed_samples = client_indices[other_client][-can_borrow:]
                            client_indices[other_client] = client_indices[other_client][:-can_borrow]
                            client_indices[client_id].extend(borrowed_samples)
                            borrowed += can_borrow
                            
                            if borrowed >= deficit:
                                break
    
    def get_client_dataset(self, client_id):
        """获取指定客户端的数据集"""
        if client_id >= self.num_clients:
            raise ValueError(f"Client ID {client_id} 超出范围 [0, {self.num_clients-1}]")
        
        return Subset(self.dataset, self.client_indices[client_id])
    
    def get_client_dataloader(self, client_id, batch_size=32, shuffle=True, num_workers=2):
        """获取指定客户端的数据加载器"""
        client_dataset = self.get_client_dataset(client_id)
        return DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_data_distribution(self):
        """获取数据分布统计信息"""
        stats = {
            'client_sizes': [],
            'client_class_distributions': [],
            'total_samples': len(self.dataset),
            'num_classes': self.num_classes
        }
        
        for client_id in range(self.num_clients):
            client_labels = self.labels[self.client_indices[client_id]]
            
            # 客户端样本数
            client_size = len(client_labels)
            stats['client_sizes'].append(client_size)
            
            # 客户端类别分布
            class_counts = np.bincount(client_labels, minlength=self.num_classes)
            class_dist = class_counts / client_size if client_size > 0 else np.zeros(self.num_classes)
            stats['client_class_distributions'].append(class_dist.tolist())
        
        return stats
    
    def print_statistics(self):
        """打印数据分布统计信息"""
        stats = self.get_data_distribution()
        
        print(f"\n=== 联邦数据分布统计 ===")
        print(f"总客户端数: {self.num_clients}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"类别数: {stats['num_classes']}")
        print(f"Alpha参数: {self.alpha}")
        
        client_sizes = stats['client_sizes']
        print(f"\n客户端样本数统计:")
        print(f"  平均: {np.mean(client_sizes):.1f}")
        print(f"  最小: {np.min(client_sizes)}")
        print(f"  最大: {np.max(client_sizes)}")
        print(f"  标准差: {np.std(client_sizes):.1f}")
        
        # 计算数据异构程度（类别分布的平均KL散度）
        uniform_dist = np.ones(stats['num_classes']) / stats['num_classes']
        kl_divergences = []
        
        for class_dist in stats['client_class_distributions']:
            if sum(class_dist) > 0:  # 避免除零
                kl_div = self._kl_divergence(np.array(class_dist), uniform_dist)
                kl_divergences.append(kl_div)
        
        avg_kl_div = np.mean(kl_divergences)
        print(f"\n数据异构程度 (平均KL散度): {avg_kl_div:.4f}")
        
        # 显示前5个客户端的类别分布示例
        print(f"\n前5个客户端类别分布示例:")
        for i in range(min(5, self.num_clients)):
            class_dist = stats['client_class_distributions'][i]
            dominant_classes = np.argsort(class_dist)[-3:][::-1]  # 前3个主要类别
            print(f"  客户端{i}: 样本数={client_sizes[i]}, 主要类别={dominant_classes.tolist()}")
    
    def _kl_divergence(self, p, q):
        """计算KL散度"""
        # 添加小的epsilon避免log(0)
        epsilon = 1e-8
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log(p / q))

def select_random_clients(num_total_clients, num_selected_clients, seed=None):
    """随机选择参与训练的客户端"""
    if seed is not None:
        random.seed(seed)
    
    selected_clients = random.sample(range(num_total_clients), num_selected_clients)
    return sorted(selected_clients)

def create_federated_dataloaders(dataset, num_clients=50, alpha=0.5, batch_size=32, 
                                num_workers=2, min_samples_per_client=50):
    """
    创建联邦学习数据加载器
    
    Returns:
        partitioner: 数据划分器
        client_dataloaders: 客户端数据加载器列表
    """
    partitioner = FederatedDataPartitioner(
        dataset=dataset,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=min_samples_per_client
    )
    
    client_dataloaders = []
    for client_id in range(num_clients):
        dataloader = partitioner.get_client_dataloader(
            client_id=client_id,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        client_dataloaders.append(dataloader)
    
    return partitioner, client_dataloaders
