"""
改进的联邦学习 + MergeNet 实现
在每f个batch时先进行联邦平均，再进行知识融合
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
import logging
import torch.nn as nn
import torch.nn.functional as F
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import swanlab

EPOCH_NUM = 200
best_acc_clients = [0.0, 0.0, 0.0]  # 3个客户端的最佳准确率
best_acc_res = 0.0 
current_config = None

logging.basicConfig(filename='logs/federated_batch_mergenet.log',
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

def hypernetwork_update(model, param, final_param, optimizer, lr_scheduler, epoch):
    """更新超网络参数"""
    optimizer.zero_grad()
    delta_theta = param - final_param
    hn_grads = torch.autograd.grad(
        [param], model.parameters(), grad_outputs=delta_theta, allow_unused=True
    )

    # 更新超网络权重
    for p, g in zip(model.parameters(), hn_grads):
        p.grad = g

    torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
    optimizer.step()
    if epoch >= 4:
        lr_scheduler.step()

def federated_average_models(client_models, device):
    """联邦平均多个客户端模型"""
    if not client_models:
        return None
    
    # 获取第一个模型的状态字典作为模板
    avg_state_dict = {}
    first_model = client_models[0]
    
    for key in first_model.state_dict().keys():
        param_type = first_model.state_dict()[key].dtype
        # 初始化为0，使用float32进行计算
        avg_state_dict[key] = torch.zeros_like(first_model.state_dict()[key], dtype=torch.float32, device=device)
        
        # 累加所有客户端的参数
        for client_model in client_models:
            avg_state_dict[key] += client_model.state_dict()[key].float()
        
        # 求平均
        avg_state_dict[key] /= len(client_models)
        
        # 转换回原始类型
        if param_type != torch.float32:
            avg_state_dict[key] = avg_state_dict[key].to(param_type)
    
    return avg_state_dict

def apply_mergenet_fusion(averaged_model, teacher_model, param_attention, device):
    """对平均后的模型应用MergeNet知识融合"""
    
    # 提取平均模型的参数
    param_a = {
        'conv': averaged_model.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device),
    }
    
    # 提取大模型参数
    param_b = {
        'linear_weight': teacher_model.fc.weight.data.clone().detach().requires_grad_(True).to(device),
    }
    
    # 使用参数注意力模块生成新参数
    out_a = param_attention(param_a, param_b)
    
    # 更新平均模型的参数
    new_param_dict = {
        'stage6.2.residual.6.weight': out_a
    }
    averaged_model.load_state_dict(new_param_dict, strict=False)
    
    return out_a, averaged_model.stage6[2].residual[6].weight

def train_federated_mergenet(client_models, res_model, config):
    """联邦学习 + MergeNet训练主函数"""
    global current_config, best_acc_clients
    current_config = config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running at {device}")
    print(torch.cuda.get_device_name())
    
    num_clients = len(client_models)
    
    # 创建参数注意力模块
    param_attention = ParamAttention(config, mode='a')
    param_attention.to(device)
    
    # 将所有模型移动到设备
    for i, model in enumerate(client_models):
        client_models[i] = model.to(device)
    res_model = res_model.to(device)
    
    f = config['f']  # 每f个batch进行一次联邦平均+知识融合
    
    # 设置优化器
    param_atten_params = stats_params(param_attention)
    
    # 客户端优化器
    client_optimizers = []
    client_lr_schedulers = []
    for model in client_models:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        client_optimizers.append(optimizer)
        client_lr_schedulers.append(scheduler)
    
    # ResNet优化器
    optimizer_res = optim.SGD(res_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler_res = MultiStepLR(optimizer_res, milestones=[60, 120, 160], gamma=0.2)
    
    # 参数注意力优化器
    optimizer_atten = optim.AdamW(param_atten_params, lr=config['lr'])
    lr_scheduler_atten = CosineAnnealingLR(optimizer_atten, T_max=(EPOCH_NUM - 4) * len(train_dataloader) // f)
    
    criterion = nn.CrossEntropyLoss()
    cnt = 0
    
    for epoch in range(EPOCH_NUM):
        param_attention.train()
        for model in client_models:
            model.train()
        res_model.train()
        
        loss_total_clients = [0.0] * num_clients
        loss_total_res = 0.0
        
        print(f'epoch: {epoch} | lr: {client_optimizers[0].param_groups[0]["lr"]:.6f}')
        logger.info(f'epoch: {epoch} | lr: {client_optimizers[0].param_groups[0]["lr"]:.6f}')
        
        progress_bar = tqdm(train_dataloader,
                            desc=f'Epoch {epoch}',
                            leave=False,
                            disable=False)
        
        for data in progress_bar:
            img, label = data[0], data[1]
            img, label = img.to(device), label.to(device)
            
            # 每f个batch进行联邦平均和知识融合
            if cnt % f == 0:
                logger.info(f'Batch {cnt}: 开始联邦平均 + 知识融合')
                
                # 1. 联邦平均
                averaged_state_dict = federated_average_models(client_models, device)
                
                # 创建临时的平均模型
                averaged_model = mobilenetv2().to(device)
                averaged_model.load_state_dict(averaged_state_dict)
                
                # 2. MergeNet知识融合
                out_a, final_state = apply_mergenet_fusion(averaged_model, res_model, param_attention, device)
                
                # 3. 将融合后的参数分发给所有客户端
                fused_state_dict = averaged_model.state_dict()
                for client_model in client_models:
                    client_model.load_state_dict(fused_state_dict)
                
                # 4. 更新参数注意力模块
                hypernetwork_update(param_attention, out_a, final_state, optimizer_atten, lr_scheduler_atten, epoch)
                
                logger.info(f'Batch {cnt}: 联邦平均 + 知识融合完成')
            
            # ResNet50训练
            optimizer_res.zero_grad()
            out_res = res_model(img)
            loss_res = criterion(out_res, label)
            loss_res.backward()
            loss_total_res += loss_res.item()
            optimizer_res.step()
            
            # 各客户端并行训练
            client_losses = []
            for i, (client_model, client_optimizer) in enumerate(zip(client_models, client_optimizers)):
                client_optimizer.zero_grad()
                out_client = client_model(img)
                loss_client = criterion(out_client, label)
                loss_client.backward()
                loss_total_clients[i] += loss_client.item()
                client_optimizer.step()
                client_losses.append(loss_client.item())
            
            cnt += 1
            
            # 更新进度条
            avg_client_loss = np.mean(client_losses)
            progress_bar.set_postfix({
                'Loss_clients_avg': f'{avg_client_loss:.6f}',
                'Loss_res': f'{loss_res.item():.6f}',
                'lr': f'{client_optimizers[0].param_groups[0]["lr"]:.6f}'
            })
        
        # 更新学习率
        for scheduler in client_lr_schedulers:
            scheduler.step()
        lr_scheduler_res.step()
        
        # 计算平均损失
        loss_train_ave_clients = [loss / len(train_dataloader) for loss in loss_total_clients]
        loss_train_ave_res = loss_total_res / len(train_dataloader)
        
        tqdm.write(f'Epoch {epoch}')
        logger.info(f'\nEpoch {epoch}')
        
        avg_client_loss = np.mean(loss_train_ave_clients)
        tqdm.write(f'Clients Avg Training Loss: {avg_client_loss:.6f}, Res Training Loss: {loss_train_ave_res:.6f}')
        logger.info(f'Clients Avg Training Loss: {avg_client_loss:.6f}, Res Training Loss: {loss_train_ave_res:.6f}')
        
        # 评估所有客户端和ResNet
        client_accs_top1 = []
        client_accs_top5 = []
        client_test_losses = []
        
        for i, client_model in enumerate(client_models):
            top1_acc, top5_acc, test_loss = test_model(client_model, device, f'Client_{i}')
            client_accs_top1.append(top1_acc)
            client_accs_top5.append(top5_acc)
            client_test_losses.append(test_loss)
            
            tqdm.write(f'Client {i} - Top1 Acc: {top1_acc:.2f}%, Top5 Acc: {top5_acc:.2f}%')
            logger.info(f'Client {i} - Top1 Acc: {top1_acc:.2f}%, Top5 Acc: {top5_acc:.2f}%')
        
        # 评估ResNet
        top1_acc_res, top5_acc_res, test_loss_res = test_model(res_model, device, 'Res')
        tqdm.write(f'Res Top1 Acc: {top1_acc_res:.2f}%, Top5 Acc: {top5_acc_res:.2f}%')
        logger.info(f'Res Top1 Acc: {top1_acc_res:.2f}%, Top5 Acc: {top5_acc_res:.2f}%')
        
        # 记录到SwanLab
        log_dict = {
            'train_loss_res_avg': loss_train_ave_res,
            'test_loss_res': test_loss_res,
            'test_acc_res_top1': top1_acc_res,
            'test_acc_res_top5': top5_acc_res,
            'train_best_acc_res': best_acc_res,
        }
        
        # 添加客户端指标
        for i in range(num_clients):
            log_dict.update({
                f'train_loss_client_{i}': loss_train_ave_clients[i],
                f'test_loss_client_{i}': client_test_losses[i],
                f'test_acc_client_{i}_top1': client_accs_top1[i],
                f'test_acc_client_{i}_top5': client_accs_top5[i],
                f'train_best_acc_client_{i}': best_acc_clients[i],
            })
        
        # 平均客户端指标
        log_dict.update({
            'train_loss_clients_avg': avg_client_loss,
            'test_loss_clients_avg': np.mean(client_test_losses),
            'test_acc_clients_avg_top1': np.mean(client_accs_top1),
            'test_acc_clients_avg_top5': np.mean(client_accs_top5),
        })
        
        swanlab.log(log_dict)
    
    logger.info(f'Best accuracies - Clients: {best_acc_clients}, Res: {best_acc_res}')

def test_model(model, device, model_name):
    """测试模型性能"""
    global best_acc_clients, best_acc_res
    
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
        if final_top1_acc / 100 > best_acc_clients[client_id]:
            best_acc_clients[client_id] = final_top1_acc / 100
            torch.save(model.state_dict(), 
                      f'checkpoints/federated_best_client_{client_id}_lr_{current_config["lr"]}_freq_{current_config["f"]}.pth')
    elif model_name == 'Res' and final_top1_acc / 100 > best_acc_res:
        best_acc_res = final_top1_acc / 100
        torch.save(model.state_dict(), 
                  f'checkpoints/federated_best_resnet50_lr_{current_config["lr"]}_freq_{current_config["f"]}.pth')
    
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
    
    logger.info('Federated MergeNet with batch-level fusion')
    
    start = time.time()
    logger.info('Start training!!!\n')

    config['a_size_conv'] = [160, 960]
    config['a_size_linear'] = [100, 1280] 
    config['b_size_conv'] = [4, 1024]
    config['b_size_linear'] = [100, 2048]
    config['mode'] = 5

    global best_acc_clients, best_acc_res
    
    # 测试不同的参数组合
    for i in [0.01]:  # 简化测试
        for j in [2]:  # f值
            config['lr'] = i
            config['f'] = j
            best_acc_clients = [0., 0., 0.]
            best_acc_res = 0.

            # 初始化SwanLab实验
            swanlab.init(
                project="Federated-MergeNet-Batch",
                experiment_name=f"fed_batch_mergenet_lr_{i}_freq_{j}",
                description=f"每{j}个batch进行联邦平均+知识融合, lr: {i}",
                config=config,
            )
            
            # 创建模型
            num_clients = 3
            client_models = [mobilenetv2() for _ in range(num_clients)]
            res_model = resnet50()
            
            print(f"客户端数量: {num_clients}")
            print(f"MobileNet v2 参数量: {sum(p.numel() for p in client_models[0].parameters()):,}")
            print(f"ResNet 50 参数量: {sum(p.numel() for p in res_model.parameters()):,}")
            
            logger.info(f'lr:{i}, f:{j}')
            
            # 开始训练
            train_federated_mergenet(client_models, res_model, config)

            # 实验结束
            swanlab.finish()

if __name__ == '__main__':
    main()
