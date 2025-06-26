"""
联邦学习 + MergeNet 实验启动脚本
用于启动完整的联邦学习实验
"""
import subprocess
import time
import os
import signal
import sys
import argparse
from typing import List
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FederatedExperiment:
    """联邦学习实验管理器"""
    
    def __init__(self, num_clients: int = 3, server_address: str = "127.0.0.1:8080"):
        self.num_clients = num_clients
        self.server_address = server_address
        self.processes: List[subprocess.Popen] = []
        
    def start_server(self):
        """启动服务器"""
        logger.info("Starting federated server...")
        
        server_cmd = [
            sys.executable, "federated_mergenet_server.py"
        ]
        
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(server_process)
        logger.info(f"Server started with PID: {server_process.pid}")
        
        # 等待服务器启动
        time.sleep(5)
        
    def start_clients(self):
        """启动所有客户端"""
        logger.info(f"Starting {self.num_clients} clients...")
        
        for client_id in range(self.num_clients):
            client_cmd = [
                sys.executable, "federated_mergenet_client.py",
                "--client_id", str(client_id),
                "--server_address", self.server_address,
                "--device", "cuda" if client_id % 2 == 0 else "cpu"  # 交替使用GPU和CPU
            ]
            
            client_process = subprocess.Popen(
                client_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(client_process)
            logger.info(f"Client {client_id} started with PID: {client_process.pid}")
            
            # 客户端启动间隔
            time.sleep(2)
    
    def wait_for_completion(self):
        """等待实验完成"""
        logger.info("Waiting for federated learning to complete...")
        
        try:
            # 等待服务器进程完成
            server_process = self.processes[0]
            server_process.wait()
            logger.info("Server process completed")
            
        except KeyboardInterrupt:
            logger.info("Experiment interrupted by user")
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during experiment: {e}")
            self.cleanup()
    
    def cleanup(self):
        """清理所有进程"""
        logger.info("Cleaning up processes...")
        
        for process in self.processes:
            if process.poll() is None:  # 进程仍在运行
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        logger.info("All processes cleaned up")
    
    def run_experiment(self):
        """运行完整实验"""
        try:
            # 创建必要的目录
            os.makedirs("logs", exist_ok=True)
            os.makedirs("checkpoints", exist_ok=True)
            
            # 启动服务器
            self.start_server()
            
            # 启动客户端
            self.start_clients()
            
            # 等待完成
            self.wait_for_completion()
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.cleanup()
            raise
        finally:
            self.cleanup()

def compare_with_baseline():
    """与原始run_res50_mbv2.py结果进行对比"""
    logger.info("=== 实验对比分析 ===")
    
    # 这里可以添加结果对比逻辑
    baseline_log = "logs/train_res50_mbv2.log"
    federated_log = "logs/federated_mergenet_server.log"
    
    if os.path.exists(baseline_log) and os.path.exists(federated_log):
        logger.info("找到基线实验和联邦实验日志，可以进行对比分析")
        # 可以在这里添加自动化的结果分析代码
    else:
        logger.info("未找到完整的对比日志文件")

def main():
    parser = argparse.ArgumentParser(description="Federated MergeNet Experiment")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline results")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_baseline()
        return
    
    # 运行联邦学习实验
    experiment = FederatedExperiment(
        num_clients=args.num_clients,
        server_address=args.server_address
    )
    
    logger.info("=== 联邦学习 + MergeNet 实验开始 ===")
    logger.info(f"客户端数量: {args.num_clients}")
    logger.info(f"服务器地址: {args.server_address}")
    
    experiment.run_experiment()
    
    logger.info("=== 实验完成 ===")

if __name__ == "__main__":
    main()
