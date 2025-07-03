# 联邦学习 + MergeNet 核心实验指南

## 🎯 实验目标

本实验体系专注于验证联邦学习与MergeNet知识融合技术的结合效果，通过四个核心对比实验验证：

1. **MergeNet知识融合的有效性**
2. **联邦学习在知识融合场景下的适应性**  
3. **分布式训练与知识融合结合的性能增益**

## 📊 实验设计

### 核心对比实验

| 实验类型 | 脚本文件 | 模型配置 | 数据分布 | 特点 |
|---------|---------|---------|---------|------|
| **基线实验** | `run_alone.py` | 单MobileNetV2 | 完整数据 | 标准单模型训练 |
| **MergeNet实验** | `run_res50_mbv2.py` | ResNet50 + MobileNetV2 | 完整数据 | 知识融合 |
| **联邦MergeNet** | `federated_batch_mergenet.py` | 50客户端MobileNetV2 + ResNet50 | 分割数据 | 联邦平均 + 知识融合 |
| **纯联邦学习** | `pure_federated_learning.py` | 50客户端MobileNetV2 | 分割数据 | 仅联邦平均 |

### 对比逻辑

```
基线实验 ←→ MergeNet实验        → 验证知识融合效果
MergeNet ←→ 联邦MergeNet       → 验证联邦学习适应性  
纯联邦 ←→ 联邦MergeNet         → 验证联邦环境下MergeNet增益
```

## 🔧 核心技术细节

### 1. 数据划分策略

- **联邦数据划分**: 使用Dirichlet分布 (α=0.5) 创建Non-IID数据分布
- **50个客户端**: 每个客户端至少50个样本
- **标签异构**: 不同客户端具有不同的类别分布
- **无数据重叠**: 确保客户端间数据完全分离

### 2. 联邦学习配置

- **总客户端数**: 50个
- **每轮选择**: 15个客户端参与训练
- **联邦平均频率**: 每2个batch进行一次聚合
- **本地训练**: 每个客户端使用独立的数据子集

### 3. MergeNet知识融合

- **教师模型**: ResNet50 (大模型，知识源)
- **学生模型**: MobileNetV2 (小模型，知识接收者)
- **融合机制**: 参数注意力模块
- **融合频率**: 联邦平均后立即进行知识融合

### 4. 训练参数统一

```python
# 所有实验统一参数
EPOCH_NUM = 200
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
lr_milestones = [60, 120, 160]
lr_gamma = 0.2
batch_size = 128
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib tqdm pyyaml swanlab

# 创建目录
mkdir -p logs checkpoints results

# 环境检查
python check_experiment_system.py
```

### 2. 运行实验

#### 核心对比实验（推荐）
```bash
# 基线实验 - 单MobileNetV2
python run_alone.py

# MergeNet实验 - 知识融合
python run_res50_mbv2.py

# 纯联邦学习实验 - 仅联邦平均
python pure_federated_learning.py

# 联邦MergeNet实验 - 联邦平均+知识融合
python federated_batch_mergenet.py
```

### 3. 验证和演示

```bash
# 验证联邦学习数据划分一致性
python verify_federated_consistency.py

# 验证数据划分
python verify_data_partition.py

# 系统完整性检查
python check_experiment_system.py
```

## 📈 实验监控

### SwanLab监控

所有实验都集成了SwanLab实时监控：

- **训练指标**: loss, accuracy, learning_rate
- **测试指标**: test_loss, test_accuracy
- **联邦指标**: selected_clients, federated_rounds
- **知识融合**: fusion_applied, parameter_changes

### 日志文件

```
logs/
├── baseline_mobilenetv2.log                    # 基线实验日志（来自run_alone.py）
├── train_res50_mbv2.log                         # MergeNet实验日志
├── federated_50clients_mergenet_optimized.log  # 联邦MergeNet日志
└── pure_federated_50clients.log                # 纯联邦学习日志
```

### 模型保存

```
checkpoints/
├── best_mobilenetv2_alone.pth                  # 基线最佳模型
├── best_resnet50_lr_0.01_freq_2.pth            # MergeNet ResNet50
├── federated_best_client_*.pth                 # 联邦客户端模型
└── pure_federated_best_client_*.pth            # 纯联邦客户端模型
```

## 📊 结果分析

### 手动对比分析

由于当前实验专注于核心对比，建议手动分析日志文件中的关键指标：

- **基线准确率**: 查看 `logs/baseline_mobilenetv2.log`
- **MergeNet准确率**: 查看 `logs/train_res50_mbv2.log`  
- **纯联邦准确率**: 查看 `logs/pure_federated_50clients.log`
- **联邦MergeNet准确率**: 查看 `logs/federated_50clients_mergenet_optimized.log`

## 🔍 重要发现

### MergeNet性能问题分析

经过深入分析，发现原始MergeNet版本性能下降的主要原因：

1. **资源竞争问题**：同时训练ResNet50和50个客户端模型，导致GPU资源竞争
2. **知识融合逻辑问题**：跨架构融合（MobileNetV2 ↔ ResNet50）可能不合理
3. **训练流程复杂性**：复杂的更新过程引入不稳定性
4. **评估设置不一致**：不同的评估频率影响性能比较

### 解决方案

创建了修复版本：
- **脚本**: `federated_mergenet_fixed.py`
- **改进**: 移除ResNet50实时训练，使用固定teacher参数
- **优化**: 简化融合流程，减少资源竞争
- **统一**: 确保除MergeNet外，其他配置完全一致

### 完整实验对比

| 实验版本 | 脚本文件 | 状态 | 特点 |
|---------|---------|------|------|
| 纯联邦学习 | `pure_federated_learning.py` | ✅ 稳定 | 仅联邦平均 |
| 原始MergeNet | `federated_batch_mergenet.py` | ⚠️ 有问题 | 资源竞争严重 |
| 修复MergeNet | `federated_mergenet_fixed.py` | 🔧 待测试 | 优化后版本 |

### 关键对比指标

- **最高准确率 (Best Accuracy)**
- **最终稳定准确率 (Final Accuracy)**
- **训练时间 (Training Time)**
- **收敛速度 (Convergence Speed)**

## 🔍 实验验证

### 数据完整性验证

```bash
# 验证数据划分
python verify_data_partition.py

# 检查数据分布
python -c "
from dataset.federated_data_partition import create_federated_dataloaders
from dataset.cls_dataloader import train_dataloader
partitioner, loaders = create_federated_dataloaders(
    train_dataloader.dataset, 50, 0.5, 128, 2, 50
)
partitioner.print_statistics()
"
```

### 功能测试

```bash
# 验证联邦学习一致性
python verify_federated_consistency.py

# 系统完整性检查
python check_experiment_system.py
```

## 📝 实验记录模板

### 实验日志格式

```
实验名称: [实验类型]
开始时间: [时间戳]
模型配置: [模型参数]
数据配置: [数据分布信息]
训练参数: [优化器、学习率等]
实验结果: [最终准确率、最佳准确率]
结束时间: [时间戳]
总用时: [训练时长]
```

### 对比分析记录

```
对比实验: [实验A] vs [实验B]
准确率提升: [数值]%
训练时间对比: [A时长] vs [B时长]
收敛速度: [达到目标准确率的epoch数]
结论: [分析结果]
```

## 🎯 实验目标与假设

### 预期结果

1. **MergeNet > 基线**: 知识融合应该带来性能提升
2. **联邦MergeNet ≈ MergeNet**: 联邦学习不应显著损害融合效果
3. **联邦MergeNet > 纯联邦**: 知识融合在联邦环境中应有额外增益

### 可能的挑战

1. **通信开销**: 联邦学习的额外通信成本
2. **数据异构性**: Non-IID数据对模型性能的影响
3. **同步开销**: 客户端同步等待时间
4. **知识融合适应性**: 联邦环境下融合效果可能降低

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少batch_size或客户端数量
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **SwanLab连接问题**
   ```bash
   # 检查网络连接或使用离线模式
   export SWANLAB_MODE=offline
   ```

3. **数据加载错误**
   ```bash
   # 重新下载CIFAR-100数据
   rm -rf data/cifar-100-python*
   ```

3. **依赖包问题**
   ```bash
   # 重新安装基础依赖
   pip install torch torchvision numpy matplotlib tqdm pyyaml swanlab
   ```

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=/home/vit/milkDragon:$PYTHONPATH
python -v [实验脚本]

# 单步调试
python -m pdb [实验脚本]
```

## 📄 引用和参考

### 相关论文

1. MergeNet知识融合技术
2. 联邦学习算法 (FedAvg)
3. 参数注意力机制
4. Non-IID数据分布处理

### 代码库结构

```
milkDragon/
├── model/                              # 模型定义
│   ├── MobileNet_v2.py                # MobileNetV2模型
│   ├── ResNet.py                      # ResNet50模型
│   └── param_attention.py             # 参数注意力模块
├── dataset/                           # 数据处理
│   ├── cls_dataloader.py              # 数据加载器
│   └── federated_data_partition.py    # 联邦数据划分
├── config/                            # 配置文件
├── logs/                              # 训练日志
├── checkpoints/                       # 模型权重
├── utils/                             # 工具函数
├── run_alone.py                       # 基线实验
├── run_res50_mbv2.py                  # MergeNet实验
├── federated_batch_mergenet.py        # 联邦MergeNet实验
├── pure_federated_learning.py        # 纯联邦学习实验
├── verify_federated_consistency.py   # 联邦学习一致性验证
├── verify_data_partition.py          # 数据划分验证
├── check_experiment_system.py        # 系统完整性检查
└── README.md                          # 本文件
```

---

**实验体系版本**: v2.0  
**最后更新**: 2025年7月  
**维护者**: MilkDragon团队  

🚀 现在可以开始运行核心对比实验了！
