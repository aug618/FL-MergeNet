# 50个客户端联邦学习 + MergeNet 系统

本项目实现了一个大规模联邦学习系统，支持50个客户端，每轮随机选择10-20个客户端进行训练，并集成了MergeNet知识融合机制。

## 🎯 核心特性

### 1. 大规模联邦学习
- **50个客户端**: 支持大规模分布式训练
- **智能选择**: 每轮随机选择15个客户端参与训练
- **异构数据**: 使用Dirichlet分布模拟真实的Non-IID数据分布
- **高效通信**: 批次级联邦平均减少通信频率

### 2. MergeNet知识融合
- **参数注意力**: 使用ParamAttention模块进行参数级知识融合
- **批次融合**: 每f个batch进行一次联邦平均+知识融合
- **大模型指导**: ResNet50向MobileNetV2传递知识
- **全局同步**: 融合后的参数分发给所有客户端

### 3. 系统优势
- **隐私保护**: 客户端数据不离开本地
- **可扩展性**: 轻松扩展到更多客户端
- **效率优化**: 只有选中的客户端参与训练
- **知识增强**: 小模型获得大模型的知识

## 📁 文件结构

```
milkDragon/
├── federated_batch_mergenet.py          # 主训练脚本（50客户端版本）
├── dataset/
│   ├── federated_data_partition.py      # 联邦数据划分模块
│   └── cls_dataloader.py               # 原始数据加载器
├── model/
│   ├── MobileNet_v2.py                 # MobileNetV2模型
│   ├── ResNet.py                       # ResNet50模型
│   └── param_attention.py              # 参数注意力模块
├── test_50clients_federated.py         # 系统功能测试
├── demo_50clients_federated.py         # 系统演示脚本
├── config/
│   └── param_attention_config.yaml     # 配置文件
├── logs/                               # 训练日志
├── checkpoints/                        # 模型检查点
└── README_50CLIENTS.md                 # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision tqdm pyyaml numpy matplotlib swanlab

# 创建必要目录
mkdir -p logs checkpoints results
```

### 2. 系统测试

```bash
# 运行功能测试
python test_50clients_federated.py

# 运行系统演示
python demo_50clients_federated.py
```

### 3. 运行训练

```bash
# 运行完整的50个客户端联邦学习实验
python federated_batch_mergenet.py
```

## ⚙️ 核心配置

### 联邦学习配置

```python
NUM_TOTAL_CLIENTS = 50      # 总客户端数
NUM_SELECTED_CLIENTS = 15   # 每轮选择的客户端数
ALPHA = 0.5                 # Dirichlet分布参数，控制数据异构程度
MIN_SAMPLES_PER_CLIENT = 50 # 每个客户端最少样本数
```

### MergeNet配置

```yaml
d_attention: 64      # 注意力维度
h: 8                # 注意力头数
num_layers: 2       # 层数
f: 2                # 知识融合频率（每f个batch）
lr: 0.01            # 学习率
mode: 5             # 模式
```

## 📊 数据分布

### 异构数据特性
- **Dirichlet分布**: Alpha=0.5实现中等程度的数据异构
- **类别不平衡**: 每个客户端主要包含2-5个类别的数据
- **样本数差异**: 客户端样本数在600-1300之间变化
- **真实模拟**: 模拟真实联邦学习场景中的数据分布

### 数据统计示例
```
总客户端数: 50
总样本数: 50,000 (CIFAR-100训练集)
客户端样本数均值: 1,000
客户端样本数范围: 674 - 1,213
数据异构程度: 中等 (Alpha=0.5)
```

## 🔄 训练流程

### 1. 每个Epoch
1. **客户端选择**: 随机选择15个客户端参与训练
2. **本地训练**: 选中的客户端在本地数据上训练
3. **定期融合**: 每f个batch执行联邦平均+知识融合
4. **参数分发**: 融合后的参数分发给所有客户端

### 2. 联邦平均+知识融合
```python
# 1. 联邦平均（选中的客户端）
averaged_state = federated_average(selected_clients)

# 2. MergeNet知识融合
fused_params = mergenet_fusion(averaged_model, teacher_model)

# 3. 参数分发（所有客户端）
distribute_params(fused_params, all_clients)
```

### 3. 评估策略
- **定期评估**: 每10个epoch评估所有客户端
- **性能指标**: Top1和Top5准确率
- **模型保存**: 保存最佳性能的模型检查点

## 📈 实验结果

### 关键指标
- **客户端平均准确率**: 跟踪所有客户端的平均性能
- **最佳客户端准确率**: 记录最好客户端的性能
- **ResNet50准确率**: 大模型（知识源）的性能
- **收敛速度**: 达到目标准确率的轮数

### SwanLab监控
```python
log_dict = {
    'epoch': epoch,
    'selected_clients_count': num_selected,
    'test_acc_all_clients_avg_top1': avg_accuracy,
    'test_acc_all_clients_max_top1': max_accuracy,
    'test_acc_all_clients_min_top1': min_accuracy,
    'train_loss_selected_clients_avg': avg_loss,
    'knowledge_fusion_rounds': fusion_count
}
```

## 🔍 系统优势对比

### vs 原始3客户端版本
| 特性 | 原始版本 | 50客户端版本 |
|------|----------|-------------|
| 客户端数量 | 3 | 50 |
| 客户端选择 | 全部参与 | 随机选择15个 |
| 数据分布 | 均匀分布 | Dirichlet异构分布 |
| 通信效率 | 低（全部通信） | 高（部分通信） |
| 扩展性 | 差 | 优秀 |
| 真实性 | 低 | 高 |

### vs 标准联邦学习
| 特性 | 标准联邦学习 | 联邦学习+MergeNet |
|------|-------------|------------------|
| 知识融合 | 无 | 有（大模型→小模型） |
| 模型性能 | 基础 | 增强 |
| 参数效率 | 标准 | 优化 |
| 收敛速度 | 标准 | 可能更快 |

## 🛠️ 扩展配置

### 调整客户端数量
```python
NUM_TOTAL_CLIENTS = 100     # 扩展到100个客户端
NUM_SELECTED_CLIENTS = 20   # 每轮选择20个
```

### 调整数据异构程度
```python
alpha = 0.1    # 高度异构
alpha = 0.5    # 中等异构  
alpha = 1.0    # 低度异构
```

### 调整知识融合频率
```python
f = 1    # 每个batch融合（高频率）
f = 5    # 每5个batch融合（低频率）
```

## 📝 日志分析

### 训练日志
```
logs/federated_50clients_mergenet_optimized.log
```

### 关键信息
- 每轮选择的客户端ID
- 联邦平均执行时机
- 知识融合应用状态
- 客户端和ResNet性能

## 🔧 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 减少batch size或客户端数量
   batch_size = 32  # 降低到32
   NUM_SELECTED_CLIENTS = 10  # 减少选择数量
   ```

2. **训练速度慢**
   ```python
   # 减少评估频率
   if epoch % 20 == 0:  # 每20个epoch评估一次
   ```

3. **数据加载慢**
   ```python
   num_workers = 0  # 在某些系统上设置为0
   ```

## 🎯 实验建议

### 推荐配置
- **生产环境**: 50客户端，选择15个，Alpha=0.5，f=2
- **快速测试**: 10客户端，选择5个，Alpha=1.0，f=1
- **研究实验**: 100客户端，选择20个，Alpha=0.1，f=3

### 性能调优
1. 根据硬件资源调整客户端数量
2. 根据数据分布调整Alpha参数
3. 根据收敛情况调整知识融合频率
4. 监控SwanLab指标进行参数调优

## 📚 相关论文

- **MergeNet**: Knowledge Migration across Heterogeneous Models, Tasks, and Modalities
- **联邦学习**: Communication-Efficient Learning of Deep Networks from Decentralized Data
- **Non-IID数据**: Federated Learning with Non-IID Data

## 📄 许可证

本项目基于原有MergeNet项目，遵循相同的开源许可证。
