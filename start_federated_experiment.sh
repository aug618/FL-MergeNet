#!/bin/bash

# 联邦学习 + MergeNet 快速启动脚本

echo "=== 联邦学习 + MergeNet 实验启动 ==="

# 检查Python环境
echo "检查Python环境..."
python --version

# 安装依赖
echo "安装依赖包..."
pip install -r requirements_federated.txt

# 创建必要目录
echo "创建目录结构..."
mkdir -p logs/federated
mkdir -p checkpoints/federated
mkdir -p results

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

echo "环境配置完成!"

# 提供多种启动选项
echo ""
echo "请选择启动方式:"
echo "1. 启动完整联邦学习实验 (3个客户端)"
echo "2. 启动服务器 (手动启动客户端)"
echo "3. 启动单个客户端"
echo "4. 与基线结果对比"

read -p "请输入选项 (1-4): " option

case $option in
    1)
        echo "启动完整联邦学习实验..."
        python run_federated_mergenet.py --num_clients 3
        ;;
    2)
        echo "启动服务器..."
        python federated_mergenet_server.py
        ;;
    3)
        read -p "请输入客户端ID (0-2): " client_id
        echo "启动客户端 $client_id..."
        python federated_mergenet_client.py --client_id $client_id
        ;;
    4)
        echo "进行结果对比..."
        python run_federated_mergenet.py --compare
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo "操作完成!"
