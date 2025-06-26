"""
联邦学习 + MergeNet 配置文件
包含联邦学习和知识融合的相关参数
"""

# 联邦学习配置
federated_config = {
    # 服务器配置
    "server": {
        "address": "0.0.0.0:8080",
        "num_rounds": 50,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0
    },
    
    # 客户端配置
    "client": {
        "local_epochs": 5,
        "batch_size": 128,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_milestones": [3, 8],
        "lr_gamma": 0.2
    },
    
    # MergeNet知识融合配置
    "knowledge_fusion": {
        "enabled": True,
        "start_round": 2,  # 从第2轮开始应用知识融合
        "teacher_model_path": "checkpoints/best_resnet50.pth",
        "fusion_frequency": 1,  # 每轮都进行融合
        "param_attention_lr": 0.001,
        "attention_layers": 2,
        "attention_heads": 8,
        "attention_dim": 64
    },
    
    # 实验配置
    "experiment": {
        "num_clients": 3,
        "seed": 42,
        "log_level": "INFO",
        "save_checkpoints": True,
        "checkpoint_dir": "checkpoints/federated",
        "log_dir": "logs/federated"
    }
}

# 参数注意力配置（与原始配置保持一致）
param_attention_config = {
    "d_attention": 64,
    "h": 8,
    "num_layers": 2,
    "num_layers_a": 2,
    "num_layers_b": 2,
    "lr": 0.001,
    "f": 1,  # 在联邦学习中，每轮都进行知识融合
    "a_size_conv": [160, 960],
    "a_size_linear": [100, 1280],
    "b_size_conv": [4, 1024],
    "b_size_linear": [100, 2048],
    "mode": 5
}

# 对比实验配置
comparison_config = {
    "baseline_log_path": "logs/train_res50_mbv2.log",
    "federated_log_path": "logs/federated_mergenet_server.log",
    "metrics_to_compare": [
        "accuracy",
        "loss",
        "convergence_speed",
        "final_performance"
    ],
    "output_comparison_report": "results/federated_vs_baseline_comparison.txt"
}
