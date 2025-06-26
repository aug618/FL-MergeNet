# 联邦学习 + MergeNet 实现

本项目实现了联邦学习与MergeNet知识融合的结合，在原有的MobileNetV2从ResNet50融合知识的基础上，加入了联邦平均的训练机制。

## 🎯 核心思想

在原始的`run_res50_mbv2.py`中，MobileNetV2每隔`f`个batch从ResNet50融合一次知识。现在我们在这个过程中加入联邦学习：

1. **联邦平均**: 多个客户端训练MobileNetV2模型
2. **服务端聚合**: 服务端对客户端模型进行联邦平均
3. **知识融合**: 平均后的模型从ResNet50融合知识
4. **参数分发**: 融合知识后的模型参数分发给客户端

## 📁 文件结构

```
federated_mergenet/
├── federated_mergenet_server.py    # 联邦学习服务端（含知识融合）
├── federated_mergenet_client.py    # 联邦学习客户端
├── run_federated_mergenet.py       # 实验启动脚本
├── compare_results.py              # 结果对比分析
├── start_federated_experiment.sh   # 快速启动脚本
├── config/
│   ├── federated_mergenet_config.py # 联邦学习配置
│   └── param_attention_config.yaml  # 原有MergeNet配置
├── requirements_federated.txt       # 依赖包
└── results/                        # 实验结果
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements_federated.txt

# 创建必要目录
mkdir -p logs/federated checkpoints/federated results
```

### 2. 使用快速启动脚本

```bash
# 给脚本执行权限（如果需要）
chmod +x start_federated_experiment.sh

# 启动实验
./start_federated_experiment.sh
```

### 3. 手动启动

#### 启动服务端
```bash
python federated_mergenet_server.py
```

#### 启动客户端（需要多个终端）
```bash
# 客户端1
python federated_mergenet_client.py --client_id 0 --device cuda

# 客户端2  
python federated_mergenet_client.py --client_id 1 --device cpu

# 客户端3
python federated_mergenet_client.py --client_id 2 --device cuda
```

#### 完整实验启动
```bash
# 启动3个客户端的完整实验
python run_federated_mergenet.py --num_clients 3
```

## 📊 实验对比

### 与原始方法对比
```bash
# 生成对比分析
python compare_results.py

# 或使用启动脚本
python run_federated_mergenet.py --compare
```

### 预期改进

1. **分布式训练**: 多客户端并行训练，提高训练效率
2. **知识聚合**: 联邦平均 + 知识融合双重聚合机制
3. **隐私保护**: 客户端数据不出本地，只交换模型参数
4. **泛化能力**: 多客户端数据分布可能提升模型泛化能力

## ⚙️ 核心技术细节

### 联邦学习策略

我们扩展了Flower的`FedAvg`策略，在`aggregate_fit`方法中加入了MergeNet知识融合：

```python
def aggregate_fit(self, server_round, results, failures):
    # 1. 标准联邦平均
    aggregated_parameters, metrics = super().aggregate_fit(...)
    
    # 2. 应用MergeNet知识融合
    fused_parameters = self._apply_mergenet_knowledge_fusion(averaged_weights)
    
    # 3. 返回融合后的参数
    return fused_parameters, metrics
```

### 知识融合流程

1. **参数提取**: 从平均后的MobileNetV2和ResNet50提取关键参数
2. **注意力计算**: 使用ParamAttention模块计算参数间的注意力
3. **参数融合**: 生成融合了大模型知识的新参数
4. **模型更新**: 将融合参数加载回MobileNetV2

### 客户端训练

每个客户端进行本地训练：
- 本地epochs: 5
- 优化器: SGD (lr=0.1, momentum=0.9)
- 学习率调度: MultiStepLR

## 📈 实验结果

### 评估指标

- **准确率**: Top1和Top5准确率
- **损失**: 训练和测试损失
- **收敛速度**: 达到特定准确率阈值的轮数
- **最终性能**: 最高准确率和最终稳定准确率

### 对比维度

1. **基线方法**: 原始`run_res50_mbv2.py`的单机训练+知识融合
2. **联邦方法**: 联邦学习+知识融合的分布式训练
3. **效率对比**: 训练时间、通信开销、收敛速度
4. **性能对比**: 最终准确率、模型泛化能力

## 🔧 配置参数

### 联邦学习配置

```python
federated_config = {
    "server": {
        "num_rounds": 50,      # 联邦学习轮数
        "min_fit_clients": 2,  # 最少参与客户端数
    },
    "client": {
        "local_epochs": 5,     # 本地训练轮数
        "learning_rate": 0.1,  # 学习率
    },
    "knowledge_fusion": {
        "start_round": 2,      # 开始融合的轮数
        "fusion_frequency": 1, # 融合频率
    }
}
```

### MergeNet配置

保持与原始实验一致的参数注意力配置：

```yaml
d_attention: 64
h: 8
num_layers: 2
lr: 0.001
f: 1  # 每轮都融合
```

## 📝 日志和监控

### 日志文件

- `logs/federated_mergenet_server.log`: 服务端日志
- `logs/federated_mergenet_client.log`: 客户端日志

### 监控指标

- 每轮聚合后的准确率和损失
- 知识融合应用状态
- 客户端参与情况
- 通信轮次和时间

## 🎯 实验目标

通过这个实现，我们期望验证：

1. **联邦学习是否能与MergeNet有效结合**
2. **分布式训练+知识融合是否优于单机训练+知识融合**
3. **额外的联邦平均步骤是否有助于提升最终性能**
4. **在保护数据隐私的前提下是否能达到相当的性能**

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**: 调整batch_size或使用CPU训练部分客户端
2. **网络连接问题**: 检查服务端地址和端口配置
3. **依赖包版本**: 确保PyTorch和Flower版本兼容

### 调试建议

1. 先运行单客户端测试
2. 检查日志文件中的错误信息
3. 验证模型参数维度匹配
4. 确认数据加载正常

## 📄 许可证

本项目基于原有MergeNet项目，遵循相同的开源许可证。
