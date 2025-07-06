# 联邦学习 + MergeNet Alpha对比实验指南

## 🎯 实验目标

本实验体系专注于验证**不同数据异构程度**下联邦学习与MergeNet知识融合技术的结合效果，通过Alpha对比实验验证：

1. **MergeNet知识融合在不同数据异构性下的有效性**
2. **联邦学习在不同数据分布下的性能表现**  
3. **固定ResNet vs 可训练ResNet在异构环境下的差异**
4. **数据异构程度对知识融合效果的影响**

## 📊 实验设计

### Alpha对比实验概览

| 实验类型 | 脚本文件 | Alpha值 | 数据异构性 | 知识融合 |
|---------|---------|---------|-----------|---------|
| **纯联邦学习** | `pure_federated_learning.py` | 0.5, 10, 100 | 高→低 | 无 |
| **固定ResNet MergeNet** | `federated_mergenet_fixed.py` | 0.5, 10, 100 | 高→低 | 固定teacher |
| **可训练ResNet MergeNet** | `federated_mergenet_trainable.py` | 0.5, 10, 100 | 高→低 | 可训练teacher |

### Alpha值含义

- **Alpha = 0.5**: 高数据异构性（Non-IID严重）
- **Alpha = 10**: 中等数据异构性（适中Non-IID）
- **Alpha = 100**: 低数据异构性（接近IID）

### 对比逻辑

```
纯联邦 ←→ 固定MergeNet ←→ 可训练MergeNet  → 三种方法在不同异构程度下的对比
Alpha=0.5 ←→ Alpha=10 ←→ Alpha=100      → 数据异构程度对性能的影响
```

## 🔧 核心技术细节

### 1. 数据划分策略

- **联邦数据划分**: 使用Dirichlet分布创建Non-IID数据分布
- **Alpha参数控制**: 
  - α=0.5: 高异构性，客户端数据分布差异很大
  - α=10: 中异构性，客户端数据分布有一定差异
  - α=100: 低异构性，客户端数据分布接近均匀
- **50个客户端**: 每个客户端至少50个样本
- **标签异构**: 不同客户端具有不同的类别分布偏好
- **无数据重叠**: 确保客户端间数据完全分离

### 2. 联邦学习配置

- **总客户端数**: 50个
- **每轮选择**: 15个客户端参与训练
- **联邦平均频率**: 每2个batch进行一次聚合
- **本地训练**: 每个客户端使用独立的数据子集

### 3. MergeNet知识融合

- **固定ResNet版本**: 使用预训练ResNet50参数作为固定teacher
- **可训练ResNet版本**: ResNet50与客户端同时训练
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
python validate_alpha_experiments.py
```

### 2. 运行Alpha对比实验

#### 完整对比实验（推荐按顺序运行）
```bash
# 纯联邦学习实验 - Alpha: 0.5, 10, 100
python pure_federated_learning.py

# 固定ResNet MergeNet实验 - Alpha: 0.5, 10, 100  
python federated_mergenet_fixed.py

# 可训练ResNet MergeNet实验 - Alpha: 0.5, 10, 100
python federated_mergenet_trainable.py
```

#### 单独运行基线实验
```bash
# 基线实验 - 单MobileNetV2
python run_alone.py

# 标准MergeNet实验 - 知识融合
python run_res50_mbv2.py
```

### 3. 验证和检查

```bash
# 验证Alpha实验配置
python validate_alpha_experiments.py

# 验证联邦学习数据划分一致性
python verify_federated_consistency.py

# 系统完整性检查
python check_experiment_system.py
```

## 📈 实验监控

### SwanLab监控

所有Alpha对比实验都集成了SwanLab实时监控：

- **Pure-Federated-Alpha-Comparison**: 纯联邦学习不同Alpha值对比
- **Fixed-MergeNet-Alpha-Comparison**: 固定ResNet MergeNet不同Alpha值对比  
- **Trainable-MergeNet-Alpha-Comparison**: 可训练ResNet MergeNet不同Alpha值对比

#### 监控指标
- **训练指标**: loss, accuracy, learning_rate, selected_clients
- **测试指标**: test_loss, test_accuracy (top1/top5)
- **联邦指标**: federated_rounds, client_selection, data_heterogeneity
- **知识融合**: fusion_applied, parameter_changes (MergeNet实验)

### 日志文件

```
logs/
├── pure_federated_50clients.log                    # 纯联邦学习日志
├── federated_mergenet_fixed.log                    # 固定ResNet MergeNet日志
├── federated_mergenet_trainable.log                # 可训练ResNet MergeNet日志
├── baseline_mobilenetv2.log                        # 基线实验日志
└── train_res50_mbv2.log                            # 标准MergeNet日志
```

### 模型保存

```
checkpoints/
├── pure_federated_best_client_*.pth                # 纯联邦客户端模型
├── fixed_mergenet_best_client_*.pth                # 固定MergeNet客户端模型  
├── trainable_mergenet_best_client_*.pth            # 可训练MergeNet客户端模型
├── run_alone_best_mobilenet.pth                    # 基线最佳模型
└── best_resnet50_lr_*.pth                          # MergeNet ResNet50模型
```

## 📊 结果分析

### Alpha对比分析方法

#### 1. 性能对比矩阵

| 方法 \ Alpha | 0.5 (高异构) | 10 (中异构) | 100 (低异构) |
|-------------|-------------|-----------|-------------|
| 纯联邦学习 | acc_0.5 | acc_10 | acc_100 |
| 固定MergeNet | acc_0.5 | acc_10 | acc_100 |
| 可训练MergeNet | acc_0.5 | acc_10 | acc_100 |

#### 2. 关键分析指标

- **客户端平均准确率**: 所有客户端的平均性能
- **客户端最佳准确率**: 最优客户端的性能
- **客户端性能方差**: 客户端间性能差异程度
- **收敛速度**: 达到目标准确率的轮次
- **训练稳定性**: 性能波动程度

#### 3. 日志分析方法

```bash
# 查看特定Alpha值的结果
grep "Alpha = 0.5" logs/pure_federated_50clients.log
grep "Alpha = 10" logs/federated_mergenet_fixed.log  
grep "Alpha = 100" logs/federated_mergenet_trainable.log

# 提取最终准确率
grep "平均最佳准确率" logs/*.log

# 查看训练时间对比
grep "总训练时间" logs/*.log
```

### 预期发现

#### 1. Alpha值影响
- **α=0.5**: 高异构性，性能较低，MergeNet提升更明显
- **α=10**: 中异构性，平衡的性能表现
- **α=100**: 低异构性，接近IID，所有方法性能最好

#### 2. 方法对比
- **纯联邦 < 固定MergeNet < 可训练MergeNet** (在高异构环境)
- **固定MergeNet ≈ 可训练MergeNet** (在低异构环境)

#### 3. 异构性敏感度
- **纯联邦学习**: 对异构性最敏感
- **固定MergeNet**: 中等敏感度
- **可训练MergeNet**: 对异构性最鲁棒

## 🔍 重要发现

### Alpha对比实验的关键洞察

经过深入分析和优化，我们构建了完整的Alpha对比实验体系：

1. **数据异构性的系统性影响**：
   - 通过α=0.5, 10, 100三个值系统性研究数据异构程度
   - 每个方法在相同的数据分布下进行公平对比

2. **MergeNet在异构环境下的适应性**：
   - 固定ResNet版本：使用预训练参数，减少资源竞争
   - 可训练ResNet版本：服务器端持续学习，适应数据分布

3. **联邦学习的异构性挑战**：
   - 高异构性(α=0.5)下客户端性能差异显著
   - 低异构性(α=100)下接近集中式训练效果

### 实验优化改进

#### 原始版本问题
- **资源竞争**: 同时训练ResNet50和50个客户端模型
- **评估不一致**: 不同实验使用不同的评估频率
- **配置差异**: 各实验间参数设置不统一

#### 当前版本优化
- **统一配置**: 所有实验使用相同的基础参数
- **分离关注点**: 固定vs可训练ResNet清晰对比
- **系统性对比**: 三个Alpha值 × 三种方法 = 9个实验点
- **资源管理**: 合理分配GPU资源，避免竞争

### 完整实验矩阵

| 实验版本 | 脚本文件 | Alpha值 | 状态 | 特点 |
|---------|---------|---------|------|------|
| 纯联邦学习 | `pure_federated_learning.py` | 0.5,10,100 | ✅ 就绪 | 仅联邦平均 |
| 固定MergeNet | `federated_mergenet_fixed.py` | 0.5,10,100 | ✅ 就绪 | 预训练teacher |
| 可训练MergeNet | `federated_mergenet_trainable.py` | 0.5,10,100 | ✅ 就绪 | 动态teacher |
| 基线对比 | `run_alone.py` | - | ✅ 可用 | 单模型训练 |
| 标准MergeNet | `run_res50_mbv2.py` | - | ✅ 可用 | 集中式融合 |

## 🔍 实验验证

### Alpha配置验证

```bash
# 验证所有文件的Alpha实验配置
python validate_alpha_experiments.py

# 检查特定文件配置
grep -n "alpha_values" *.py
grep -n "Alpha.*Comparison" *.py
```

### 数据完整性验证

```bash
# 验证不同Alpha值的数据划分
python -c "
from dataset.federated_data_partition import create_federated_dataloaders
from dataset.cls_dataloader import train_dataloader

for alpha in [0.5, 10, 100]:
    print(f'\\n=== Alpha = {alpha} ===')
    partitioner, loaders = create_federated_dataloaders(
        train_dataloader.dataset, 50, alpha, 128, 2, 50
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

# 快速smoke test
python -c "
import torch
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
print('✅ 模型导入正常')

from dataset.cls_dataloader import train_dataloader, test_dataloader
print('✅ 数据加载正常')

print('✅ 所有依赖就绪')
"
```

## 📝 实验记录模板

### Alpha对比实验日志格式

```
=== Alpha = [0.5/10/100] [方法名称] 实验 ===
开始时间: [时间戳]
数据异构程度: Alpha = [值]
客户端配置: 50总数, 15选择
联邦配置: 每2batch平均, 200epochs
模型配置: [MobileNetV2/ResNet50配置]
知识融合: [无/固定ResNet/可训练ResNet]

训练过程:
- Epoch 0-20: [初期表现]
- Epoch 50: [中期表现]  
- Epoch 100: [后期表现]
- 最终结果: [最终准确率]

最终统计:
- 客户端平均准确率: [数值]%
- 客户端最佳准确率: [数值]%
- 客户端最差准确率: [数值]%
- 性能标准差: [数值]
- 总训练时间: [时长]

结束时间: [时间戳]
```

### 跨Alpha对比分析记录

```
=== [方法名称] 跨Alpha对比分析 ===

性能变化趋势:
- Alpha=0.5  → Alpha=10:  提升 [数值]%
- Alpha=10   → Alpha=100: 提升 [数值]%
- Alpha=0.5  → Alpha=100: 总提升 [数值]%

异构性敏感度:
- 高异构(α=0.5): [性能表现]
- 中异构(α=10):  [性能表现]  
- 低异构(α=100): [性能表现]

收敛分析:
- 收敛速度: [epochs]
- 稳定性: [波动程度]

结论: [分析总结]
```

## 🎯 实验目标与假设

### 核心研究问题

1. **数据异构性如何影响联邦学习性能？**
   - 假设：Alpha值越大(越接近IID)，性能越好

2. **MergeNet知识融合在异构环境下是否依然有效？**
   - 假设：MergeNet在各种异构程度下都能带来提升

3. **固定ResNet vs 可训练ResNet哪个更适合异构联邦学习？**
   - 假设：可训练ResNet在高异构环境下表现更好

4. **联邦学习的异构性挑战有多严重？**
   - 假设：高异构性会显著降低联邦学习效果

### 预期结果假设

#### 性能排序预期
```
低异构(α=100): 可训练MergeNet ≥ 固定MergeNet > 纯联邦
中异构(α=10):  可训练MergeNet > 固定MergeNet > 纯联邦  
高异构(α=0.5): 可训练MergeNet >> 固定MergeNet > 纯联邦
```

#### 异构性敏感度预期
- **纯联邦学习**: 对异构性最敏感，性能下降最明显
- **固定MergeNet**: 中等敏感度，有一定鲁棒性
- **可训练MergeNet**: 最鲁棒，能适应不同数据分布

### 可能的挑战与限制

1. **计算资源挑战**
   - 9个实验 × 200 epochs = 大量计算需求
   - GPU内存和时间管理

2. **数据异构性建模**
   - Dirichlet分布的局限性
   - 真实异构性可能更复杂

3. **知识融合的架构差异**
   - MobileNetV2 ↔ ResNet50跨架构融合的合理性
   - 参数注意力机制的有效性

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
├── model/                                      # 模型定义
│   ├── MobileNet_v2.py                        # MobileNetV2模型
│   ├── ResNet.py                              # ResNet50模型
│   └── param_attention.py                     # 参数注意力模块(MergeNet核心)
├── dataset/                                   # 数据处理
│   ├── cls_dataloader.py                      # CIFAR-100数据加载器
│   └── federated_data_partition.py            # 联邦数据划分(Dirichlet)
├── config/                                    # 配置文件
│   └── param_attention_config.yaml            # MergeNet参数配置
├── logs/                                      # 训练日志
├── checkpoints/                               # 模型权重
├── utils/                                     # 工具函数
│
├── === Alpha对比实验核心文件 ===
├── pure_federated_learning.py                 # 纯联邦学习(Alpha: 0.5,10,100)
├── federated_mergenet_fixed.py                # 固定ResNet MergeNet(Alpha: 0.5,10,100)
├── federated_mergenet_trainable.py            # 可训练ResNet MergeNet(Alpha: 0.5,10,100)
│
├── === 基线对比实验 ===
├── run_alone.py                               # 单MobileNetV2基线
├── run_res50_mbv2.py                          # 标准MergeNet实验
│
├── === 验证和工具 ===
├── validate_alpha_experiments.py              # Alpha实验配置验证
├── verify_federated_consistency.py            # 联邦学习一致性验证
├── check_experiment_system.py                 # 系统完整性检查
│
└── README.md                                  # 本文件
```

### 关键文件说明

#### 核心实验文件
- **pure_federated_learning.py**: 纯联邦学习基线，无知识融合
- **federated_mergenet_fixed.py**: 使用固定ResNet参数的MergeNet
- **federated_mergenet_trainable.py**: ResNet与客户端同步训练的MergeNet

#### 验证工具
- **validate_alpha_experiments.py**: 自动检查三个文件的Alpha配置
- **verify_federated_consistency.py**: 确保联邦实验的一致性
- **check_experiment_system.py**: 系统依赖和环境检查

---

**实验体系版本**: v3.0 - Alpha对比实验  
**最后更新**: 2025年7月6日  
**维护者**: MilkDragon团队  

🚀 **现在可以开始运行Alpha对比实验了！**

### 推荐运行顺序

1. **环境验证**: `python validate_alpha_experiments.py`
2. **纯联邦基线**: `python pure_federated_learning.py` 
3. **固定MergeNet**: `python federated_mergenet_fixed.py`
4. **可训练MergeNet**: `python federated_mergenet_trainable.py`

### 实验监控

- 监控SwanLab面板查看实时训练进度
- 检查日志文件获取详细训练信息  
- 每个方法运行3个Alpha值，总计9组实验

### 预期收获

通过本实验体系，您将深入理解：
- 数据异构性对联邦学习的影响程度
- MergeNet知识融合在不同异构环境下的适应性
- 固定vs可训练teacher模型的权衡关系
- 联邦学习在现实异构场景下的挑战与解决方案
