"""
纯联邦学习实验（无MergeNet）
50个客户端，每轮随机选择10-20个，只进行联邦平均，不进行知识融合
用于与federated_batch_mergenet.py对比，验证MergeNet在联邦学习中的增益
"""
import torch
import time
import yaml
from tqdm import tqdm
import random
import os
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import math
from dataset.cls_dataloader import train_dataloader, test_dataloader
from dataset.federated_data_partition import create_federated_dataloaders, select_random_clients
import logging
import torch.nn as nn
import torch.nn.functional as F
from model.MobileNet_v2 import mobilenetv2
from torch.optim.lr_scheduler import MultiStepLR

import swanlab

EPOCH_NUM = 200
NUM_TOTAL_CLIENTS = 50  # 总客户端数
NUM_SELECTED_CLIENTS = 15  # 每轮选择的客户端数 (与federated_batch_mergenet.py保持一致)
best_acc_clients = [0.0] * NUM_TOTAL_CLIENTS  # 50个客户端的最佳准确率
current_config = None

logging.basicConfig(filename='logs/pure_federated_50clients.log',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def stats_params(model, weight_decay=5e-4):
    """设置不同的权重衰减策略"""
    params_without_wd = []
    params_with_wd = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bias", "norm"]]):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
         
    param = [
        {"params": params_without_wd, "weight_decay": 0},
        {
            "params": params_with_wd,
            "weight_decay": weight_decay,
        },
    ]
    return param

def federated_average_models(selected_client_models, device):
    """联邦平均选中的客户端模型"""
    if not selected_client_models:
        return None
    
    # 获取第一个模型的状态字典作为模板
    avg_state_dict = {}
    first_model = selected_client_models[0]
    
    for key in first_model.state_dict().keys():
        param_type = first_model.state_dict()[key].dtype
        # 初始化为0，使用float32进行计算
        avg_state_dict[key] = torch.zeros_like(first_model.state_dict()[key], dtype=torch.float32, device=device)
        
        # 累加所有选中客户端的参数
        for client_model in selected_client_models:
            avg_state_dict[key] += client_model.state_dict()[key].float()
        
        # 求平均
        avg_state_dict[key] /= len(selected_client_models)
        
        # 转换回原始类型
        if param_type != torch.float32:
            avg_state_dict[key] = avg_state_dict[key].to(param_type)
    
    return avg_state_dict

def train_pure_federated(all_client_models, all_client_dataloaders, config):
    """纯联邦学习训练主函数 - 50个客户端版本（无MergeNet）"""
    global current_config, best_acc_clients
    current_config = config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running at {device}")
    print(torch.cuda.get_device_name())
    
    num_total_clients = len(all_client_models)
    print(f"总客户端数: {num_total_clients}")
    print(f"每轮选择客户端数: {NUM_SELECTED_CLIENTS}")
    
    # 将所有模型移动到设备
    for i, model in enumerate(all_client_models):
        all_client_models[i] = model.to(device)
    
    f = config['f']  # 每f个batch进行一次联邦平均（与federated_batch_mergenet.py保持一致）
    
    # 所有客户端优化器
    all_client_optimizers = []
    all_client_lr_schedulers = []
    for model in all_client_models:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        all_client_optimizers.append(optimizer)
        all_client_lr_schedulers.append(scheduler)
    
    criterion = nn.CrossEntropyLoss()
    cnt = 0
    
    for epoch in range(EPOCH_NUM):
        for model in all_client_models:
            model.train()
        
        # 每个epoch随机选择客户端参与训练
        selected_client_ids = select_random_clients(
            num_total_clients=num_total_clients, 
            num_selected_clients=NUM_SELECTED_CLIENTS,
            seed=epoch  # 使用epoch作为seed确保可重现性
        )
        
        selected_client_models = [all_client_models[i] for i in selected_client_ids]
        selected_client_optimizers = [all_client_optimizers[i] for i in selected_client_ids]
        selected_client_dataloaders = [all_client_dataloaders[i] for i in selected_client_ids]
        
        print(f'Epoch {epoch}: 选择客户端 {selected_client_ids[:5]}...等{len(selected_client_ids)}个')
        logger.info(f'Epoch {epoch}: 选择客户端 {selected_client_ids}')
        
        loss_total_selected_clients = [0.0] * len(selected_client_ids)
        
        print(f'epoch: {epoch} | lr: {all_client_optimizers[0].param_groups[0]["lr"]:.6f}')
        logger.info(f'epoch: {epoch} | lr: {all_client_optimizers[0].param_groups[0]["lr"]:.6f}')
        
        # 客户端并行训练（每个客户端使用自己的数据子集）
        # 获取选中客户端中最大的batch数，用于控制训练轮次
        max_batches_selected = max(len(all_client_dataloaders[client_id]) for client_id in selected_client_ids)
        
        # 创建选中客户端的数据迭代器
        selected_client_iterators = []
        for client_id in selected_client_ids:
            iterator = iter(all_client_dataloaders[client_id])
            selected_client_iterators.append(iterator)
        
        progress_bar = tqdm(range(max_batches_selected),
                            desc=f'Epoch {epoch}',
                            leave=False,
                            disable=False)
        
        for batch_idx in progress_bar:
            # 每f个batch进行联邦平均（不进行知识融合）
            if cnt % f == 0:
                logger.info(f'Batch {cnt}: 开始联邦平均 (选中{len(selected_client_ids)}个客户端)')
                
                # 1. 联邦平均（只对选中的客户端）
                averaged_state_dict = federated_average_models(selected_client_models, device)
                
                # 2. 将平均后的参数分发给所有客户端（包括未选中的）
                if averaged_state_dict is not None:
                    for client_model in all_client_models:  # 分发给所有客户端
                        client_model.load_state_dict(averaged_state_dict)
                
                logger.info(f'Batch {cnt}: 联邦平均完成，参数已分发给所有{num_total_clients}个客户端')
            
            # 选中的客户端使用各自的数据子集训练
            client_losses = []
            active_clients = 0
            
            for i, (client_idx, client_model, client_optimizer, client_iterator) in enumerate(
                zip(selected_client_ids, selected_client_models, selected_client_optimizers, selected_client_iterators)
            ):
                try:
                    img_client, label_client = next(client_iterator)
                    img_client, label_client = img_client.to(device), label_client.to(device)
                    
                    client_optimizer.zero_grad()
                    out_client = client_model(img_client)
                    loss_client = criterion(out_client, label_client)
                    loss_client.backward()
                    loss_total_selected_clients[i] += loss_client.item()
                    client_optimizer.step()
                    client_losses.append(loss_client.item())
                    active_clients += 1
                    
                except StopIteration:
                    # 客户端数据用完，跳过该客户端
                    continue
            
            cnt += 1
            
            # 更新进度条
            if client_losses:
                avg_client_loss = np.mean(client_losses)
                progress_bar.set_postfix({
                    'Active_clients': active_clients,
                    'Loss_clients_avg': f'{avg_client_loss:.6f}',
                    'lr': f'{all_client_optimizers[0].param_groups[0]["lr"]:.6f}'
                })
            
            # 如果所有选中的客户端数据都用完了，结束这个epoch
            if active_clients == 0:
                break
        
        # 更新学习率（所有客户端）
        for scheduler in all_client_lr_schedulers:
            scheduler.step()
        
        # 计算选中客户端的平均损失
        loss_train_ave_selected_clients = []
        for i, total_loss in enumerate(loss_total_selected_clients):
            if total_loss > 0:  # 只计算有训练的客户端
                # 获取该客户端的实际batch数
                client_batches = len(all_client_dataloaders[selected_client_ids[i]])
                loss_train_ave_selected_clients.append(total_loss / max(client_batches, 1))
        
        tqdm.write(f'Epoch {epoch}')
        logger.info(f'\nEpoch {epoch}')
        
        if loss_train_ave_selected_clients:
            avg_selected_client_loss = np.mean(loss_train_ave_selected_clients)
            tqdm.write(f'Selected Clients Avg Training Loss: {avg_selected_client_loss:.6f}')
            logger.info(f'Selected Clients Avg Training Loss: {avg_selected_client_loss:.6f}')
        
        # 评估所有客户端（每10个epoch评估一次以节省时间）
        if epoch % 10 == 0 or epoch == EPOCH_NUM - 1:
            # 评估所有客户端
            client_accs_top1 = []
            client_accs_top5 = []
            client_test_losses = []
            
            print(f"评估所有{num_total_clients}个客户端...")
            for i, client_model in enumerate(all_client_models):
                top1_acc, top5_acc, test_loss = test_model(client_model, device, f'Client_{i}')
                client_accs_top1.append(top1_acc)
                client_accs_top5.append(top5_acc)
                client_test_losses.append(test_loss)
                
                if i < 5:  # 只显示前5个客户端的详细信息
                    tqdm.write(f'Client {i} - Top1 Acc: {top1_acc:.2f}%, Top5 Acc: {top5_acc:.2f}%')
                    logger.info(f'Client {i} - Top1 Acc: {top1_acc:.2f}%, Top5 Acc: {top5_acc:.2f}%')
            
            # 显示统计信息
            avg_acc_top1 = np.mean(client_accs_top1)
            max_acc_top1 = np.max(client_accs_top1)
            min_acc_top1 = np.min(client_accs_top1)
            
            tqdm.write(f'所有客户端 Top1 Acc - 平均: {avg_acc_top1:.2f}%, 最高: {max_acc_top1:.2f}%, 最低: {min_acc_top1:.2f}%')
            logger.info(f'所有客户端 Top1 Acc - 平均: {avg_acc_top1:.2f}%, 最高: {max_acc_top1:.2f}%, 最低: {min_acc_top1:.2f}%')
            
            # 记录到SwanLab（只记录关键指标）
            log_dict = {
                'epoch': epoch,
                'selected_clients_count': len(selected_client_ids),
                'train/loss_selected_clients_avg': avg_selected_client_loss if loss_train_ave_selected_clients else 0.0,
                'test/loss_all_clients_avg': np.mean(client_test_losses),
                'test/acc_all_clients_avg_top1': avg_acc_top1,
                'test/acc_all_clients_max_top1': max_acc_top1,
                'test/acc_all_clients_min_top1': min_acc_top1,
                'test/acc_all_clients_avg_top5': np.mean(client_accs_top5),
            }
            
            swanlab.log(log_dict)
        else:
            # 非评估轮次，只记录训练损失
            log_dict = {
                'epoch': epoch,
                'selected_clients_count': len(selected_client_ids),
            }
            if loss_train_ave_selected_clients:
                log_dict['train/loss_selected_clients_avg'] = avg_selected_client_loss
            
            swanlab.log(log_dict)
    
    logger.info(f'训练完成！前10个客户端最佳准确率: {best_acc_clients[:10]}')
    
    return all_client_models

def test_model(model, device, model_name):
    """测试模型性能"""
    global best_acc_clients
    
    model.eval()
    top1_acc = 0.
    top5_acc = 0.
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            img, label = data[0], data[1]
            img, label = img.to(device), label.to(device)
            out = model(img)
            
            loss = criterion(out, label)
            running_loss += loss.item()
            pred1 = out.argmax(dim=1)
            label_resize = label.view(-1,1)
            _, pred5 = out.topk(5, 1, True, True)
            total += label.size(0)
            top1_acc += pred1.eq(label).sum().item()
            top5_acc += torch.eq(pred5, label_resize).sum().float().item()
    
    final_top1_acc = 100. * top1_acc / total
    final_top5_acc = 100. * top5_acc / total
    final_test_loss = running_loss / len(test_dataloader)
    
    # 保存最佳模型
    if 'Client' in model_name:
        client_id = int(model_name.split('_')[1])
        if client_id < len(best_acc_clients) and final_top1_acc / 100 > best_acc_clients[client_id]:
            best_acc_clients[client_id] = final_top1_acc / 100
            # 只保存前几个客户端的模型文件以节省空间
            if client_id < 10:
                torch.save(model.state_dict(), 
                          f'checkpoints/pure_federated_best_client_{client_id}_freq_{current_config["f"]}.pth')
    
    return final_top1_acc, final_top5_acc, final_test_loss

def main():
    """主函数"""
    warnings.filterwarnings("ignore")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    cpu_num = 4 
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    # 加载配置
    config = yaml.load(open('config/param_attention_config.yaml', 'r'), Loader=yaml.Loader)
    
    logger.info(f'Pure Federated Learning with {NUM_TOTAL_CLIENTS} clients, selecting {NUM_SELECTED_CLIENTS} per round')
    
    start = time.time()
    logger.info('Start training!!!\n')

    global best_acc_clients
    
    # 创建联邦数据划分（与federated_batch_mergenet.py完全一致）
    print("创建联邦数据划分...")
    partitioner, client_dataloaders = create_federated_dataloaders(
        dataset=train_dataloader.dataset,
        num_clients=NUM_TOTAL_CLIENTS,
        alpha=0.5,  # Dirichlet参数，控制数据异构程度
        batch_size=train_dataloader.batch_size,
        num_workers=2,
        min_samples_per_client=50
    )
    
    # 打印数据分布统计
    partitioner.print_statistics()
    
    # 测试不同的参数组合
    for j in [2]:  # f值（与federated_batch_mergenet.py保持一致）
        config['f'] = j
        best_acc_clients = [0.] * NUM_TOTAL_CLIENTS

        # 初始化SwanLab实验
        swanlab.init(
            project="Pure-Federated-7-2",
            experiment_name=f"pure_fed_50clients_freq_{j}_select_{NUM_SELECTED_CLIENTS}",
            description=f"纯联邦学习：50个客户端，每轮选择{NUM_SELECTED_CLIENTS}个，每{j}个batch进行联邦平均（无MergeNet）",
            config={
                **config,
                'num_total_clients': NUM_TOTAL_CLIENTS,
                'num_selected_clients': NUM_SELECTED_CLIENTS,
                'data_alpha': 0.5,
                'min_samples_per_client': 50,
                'merge_net': False  # 标记这是无MergeNet的实验
            },
        )
        
        # 创建50个客户端模型
        print(f"创建{NUM_TOTAL_CLIENTS}个客户端模型...")
        all_client_models = [mobilenetv2() for _ in range(NUM_TOTAL_CLIENTS)]
        
        print(f"总客户端数量: {NUM_TOTAL_CLIENTS}")
        print(f"每轮选择客户端数: {NUM_SELECTED_CLIENTS}")
        print(f"MobileNet v2 参数量: {sum(p.numel() for p in all_client_models[0].parameters()):,}")
        print(f"联邦平均频率: 每{j}个batch")
        print("注意: 本实验不使用MergeNet知识融合")
        
        logger.info(f'f:{j}, clients:{NUM_TOTAL_CLIENTS}, selected:{NUM_SELECTED_CLIENTS}, mergenet: False')
        
        # 开始训练
        train_pure_federated(all_client_models, client_dataloaders, config)

        # 实验结束
        swanlab.finish()
        
        # 保存实验总结
        end_time = time.time()
        total_time = end_time - start
        
        print(f"\n=== 纯联邦学习实验完成 ===")
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"前10个客户端最佳准确率: {[f'{acc:.4f}' for acc in best_acc_clients[:10]]}")
        print(f"所有客户端平均最佳准确率: {np.mean(best_acc_clients):.4f}")
        
        logger.info(f"实验完成，总时间: {total_time/3600:.2f} 小时")
        logger.info(f"所有客户端平均最佳准确率: {np.mean(best_acc_clients):.4f}")

if __name__ == '__main__':
    main()
