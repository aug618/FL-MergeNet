"""
联邦学习 + MergeNet 服务端实现
在联邦平均后使用MergeNet从大模型融合知识
"""
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, FitRes, Parameters, Scalar
from flwr.server.strategy import FedAvg
import yaml
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
import logging

logging.basicConfig(
    filename='logs/federated_mergenet_server.log',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FederatedMergeNetStrategy(FedAvg):
    """自定义联邦学习策略，集成MergeNet知识融合"""
    
    def __init__(self, config_path: str, **kwargs):
        super().__init__(**kwargs)
        
        # 加载配置
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        self.config['a_size_conv'] = [160, 960]
        self.config['a_size_linear'] = [100, 1280]
        self.config['b_size_linear'] = [100, 2048]
        self.config['mode'] = 5
        
        # 初始化大模型（知识源）
        self.teacher_model = resnet50()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_model.to(self.device)
        
        # 初始化参数注意力模块
        self.param_attention = ParamAttention(self.config, mode='a')
        self.param_attention.to(self.device)
        
        # 预训练大模型或加载预训练权重
        self._init_teacher_model()
        
        logger.info("FederatedMergeNetStrategy initialized")
        logger.info(f"Teacher model parameters: {sum(p.numel() for p in self.teacher_model.parameters())}")
        
    def _init_teacher_model(self):
        """初始化或加载预训练的大模型"""
        try:
            # 尝试加载预训练权重
            checkpoint = torch.load('checkpoints/best_resnet50.pth', map_location=self.device)
            self.teacher_model.load_state_dict(checkpoint)
            logger.info("Loaded pre-trained teacher model")
        except FileNotFoundError:
            logger.info("No pre-trained teacher model found, using random initialization")
            # 可以在这里添加预训练逻辑
            pass
    
    def _get_teacher_parameters(self) -> Dict[str, torch.Tensor]:
        """提取大模型的关键参数"""
        return {
            'linear_weight': self.teacher_model.fc.weight.data.clone().detach()
        }
    
    def _apply_mergenet_knowledge_fusion(self, averaged_weights: List[np.ndarray]) -> List[np.ndarray]:
        """应用MergeNet知识融合到平均后的模型权重"""
        
        # 创建临时学生模型来加载平均权重
        temp_student = mobilenetv2()
        temp_student.to(self.device)
        
        # 将平均权重加载到临时学生模型
        params_dict = zip(temp_student.state_dict().keys(), averaged_weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        temp_student.load_state_dict(state_dict, strict=False)
        
        # 提取学生模型参数
        param_a = {
            'conv': temp_student.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(self.device)
        }
        
        # 提取大模型参数
        param_b = self._get_teacher_parameters()
        param_b = {k: v.requires_grad_(True).to(self.device) for k, v in param_b.items()}
        
        try:
            # 使用参数注意力进行知识融合
            with torch.no_grad():
                fused_params = self.param_attention(param_a, param_b)
                
                # 更新对应的参数
                new_param_dict = {
                    'stage6.2.residual.6.weight': fused_params
                }
                temp_student.load_state_dict(new_param_dict, strict=False)
                
                logger.info("Successfully applied MergeNet knowledge fusion")
                
        except Exception as e:
            logger.error(f"Error in knowledge fusion: {e}")
            # 如果融合失败，返回原始平均权重
            return averaged_weights
        
        # 将融合后的权重转换回numpy格式
        fused_weights = []
        for param in temp_student.parameters():
            fused_weights.append(param.data.cpu().numpy())
            
        return fused_weights
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """自定义聚合函数，在联邦平均后应用知识融合"""
        
        logger.info(f"Aggregating round {server_round} with {len(results)} clients")
        
        # 1. 首先进行标准的联邦平均
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is None:
            return None, {}
        
        # 2. 将参数转换为numpy数组
        averaged_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        # 3. 应用MergeNet知识融合
        if server_round > 1:  # 从第2轮开始应用知识融合
            logger.info("Applying MergeNet knowledge fusion to aggregated model")
            fused_weights = self._apply_mergenet_knowledge_fusion(averaged_weights)
            
            # 4. 转换回Parameters格式
            fused_parameters = fl.common.ndarrays_to_parameters(fused_weights)
            
            # 记录融合信息
            aggregated_metrics["knowledge_fusion_applied"] = 1
            logger.info("Knowledge fusion completed and applied to aggregated model")
            
            return fused_parameters, aggregated_metrics
        else:
            aggregated_metrics["knowledge_fusion_applied"] = 0
            return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """聚合评估结果"""
        
        if not results:
            return None, {}
        
        # 计算加权平均准确率
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        aggregated_accuracy = sum(accuracies) / sum(examples)
        
        # 计算加权平均损失
        losses = [r.loss * r.num_examples for _, r in results]
        aggregated_loss = sum(losses) / sum(examples)
        
        logger.info(f"Round {server_round} - Aggregated accuracy: {aggregated_accuracy:.4f}, loss: {aggregated_loss:.4f}")
        
        return aggregated_loss, {
            "accuracy": aggregated_accuracy,
            "round": server_round
        }

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """计算指标的加权平均"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    """启动联邦学习服务器"""
    
    # 配置路径
    config_path = 'config/param_attention_config.yaml'
    
    # 创建自定义策略
    strategy = FederatedMergeNetStrategy(
        config_path=config_path,
        fraction_fit=1.0,  # 每轮使用所有客户端
        fraction_evaluate=1.0,
        min_fit_clients=2,  # 最少2个客户端
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # 启动服务器
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),  # 50轮联邦学习
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
