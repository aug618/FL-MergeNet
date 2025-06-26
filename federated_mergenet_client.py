"""
联邦学习 + MergeNet 客户端实现
每个客户端训练MobileNetV2模型
"""
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from typing import Dict, List, Tuple
import logging
import time
import math
from tqdm import tqdm

from dataset.cls_dataloader import train_dataloader, test_dataloader
from model.MobileNet_v2 import mobilenetv2

logging.basicConfig(
    filename='logs/federated_mergenet_client.log',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MobileNetV2Client(fl.client.NumPyClient):
    """MobileNetV2联邦学习客户端"""
    
    def __init__(self, client_id: int, device: str = "cuda"):
        self.client_id = client_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = mobilenetv2()
        self.model.to(self.device)
        
        # 训练配置
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = 5  # 每轮联邦学习的本地训练轮数
        self.best_acc = 0.0
        
        logger.info(f"Client {client_id} initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """获取模型参数"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """本地训练"""
        
        # 设置从服务器接收的参数（已经经过知识融合）
        self.set_parameters(parameters)
        
        # 配置优化器和学习率调度器
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = MultiStepLR(optimizer, milestones=[3, 8], gamma=0.2)  # 适应本地训练轮数
        
        # 本地训练
        train_loss = self._train_local_epochs(optimizer, lr_scheduler)
        
        # 获取训练后的参数
        updated_parameters = self.get_parameters({})
        
        # 计算训练样本数
        num_examples = len(train_dataloader.dataset)
        
        logger.info(f"Client {self.client_id} completed local training - Loss: {train_loss:.4f}")
        
        return updated_parameters, num_examples, {"train_loss": train_loss}
    
    def _train_local_epochs(self, optimizer, lr_scheduler) -> float:
        """执行本地训练epochs"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f'Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}',
                leave=False,
                disable=False
            )
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            lr_scheduler.step()
            total_loss += epoch_loss
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Client {self.client_id} - Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.4f}")
        
        return total_loss / num_batches
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """模型评估"""
        
        # 设置参数
        self.set_parameters(parameters)
        
        # 评估模型
        loss, accuracy, top5_accuracy = self._evaluate_model()
        
        # 计算测试样本数
        num_examples = len(test_dataloader.dataset)
        
        # 更新最佳准确率
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            # 保存最佳模型
            torch.save(
                self.model.state_dict(),
                f'checkpoints/federated_best_client_{self.client_id}.pth'
            )
        
        logger.info(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        return loss, num_examples, {
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "best_accuracy": self.best_acc
        }
    
    def _evaluate_model(self) -> Tuple[float, float, float]:
        """执行模型评估"""
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 计算损失
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 计算top1准确率
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                # 计算top5准确率
                _, pred5 = output.topk(5, 1, True, True)
                target_resize = target.view(-1, 1)
                top5_correct += torch.eq(pred5, target_resize).sum().float().item()
                
                total += target.size(0)
        
        avg_loss = total_loss / len(test_dataloader)
        accuracy = correct / total
        top5_accuracy = top5_correct / total
        
        return avg_loss, accuracy, top5_accuracy

def main():
    """启动客户端"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated MergeNet Client")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = MobileNetV2Client(client_id=args.client_id, device=args.device)
    
    # 连接到服务器
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()
