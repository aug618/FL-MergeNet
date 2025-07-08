"""
联邦学习 + MergeNet 实现（ResNet可训练版本）
这是第五个实验：联邦学习 + MergeNet，其中ResNet使用完整数据训练

实验设计：
- 50个客户端，每个使用独立的数据子集训练MobileNetV2（联邦学习）
- 服务器端ResNet50使用完整数据训练（与实验2相同）
- 通过MergeNet将训练的ResNet知识融合到联邦学习的MobileNetV2中
- 每f个batch进行一次：ResNet训练 -> 联邦平均 -> MergeNet知识融合

对比目的：
- 与federated_mergenet_fixed.py对比，验证训练ResNet vs 固定ResNet的知识融合效果
- 验证服务器端能访问完整数据时，对联邦学习客户端的提升程度
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import yaml
import logging
import warnings
import random
import os
import numpy as np
from tqdm import tqdm
from dataset.cls_dataloader import train_dataloader, test_dataloader
from dataset.federated_data_partition import create_federated_dataloaders, select_random_clients
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import swanlab

EPOCH_NUM = 200
NUM_TOTAL_CLIENTS = 50  # 总客户端数
NUM_SELECTED_CLIENTS = 15  # 每轮选择的客户端数（与其他实验保持一致）
best_acc_clients = [0.0] * NUM_TOTAL_CLIENTS  # 50个客户端的最佳准确率
best_acc_resnet = 0.0  # ResNet的最佳准确率
current_config = None

# 日志设置
logging.basicConfig(filename='logs/federated_mergenet_trainable.log',
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
            if len(param.shape) == 1 or name.endswith(".bias"):
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
    
    # 确保参数有梯度信息
    if not param.requires_grad:
        param = param.requires_grad_(True)
    if not final_param.requires_grad:
        final_param = final_param.requires_grad_(True)
    
    delta_theta = param - final_param
    hn_grads = torch.autograd.grad(
        [param], model.parameters(), grad_outputs=delta_theta, allow_unused=True
    )

    # 更新超网络权重
    for p, g in zip(model.parameters(), hn_grads):
        if g is not None:
            p.grad = g

    torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
    optimizer.step()
    if epoch >= 4:
        lr_scheduler.step()

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

def train_federated_mergenet_trainable(all_client_models, all_client_dataloaders, resnet_model, config):
    """联邦学习 + MergeNet训练主函数（ResNet可训练版本）"""
    global current_config, best_acc_clients, best_acc_resnet
    current_config = config
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Running at {device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(1))
    else:
        print("CPU mode")
    
    num_total_clients = len(all_client_models)
    print(f"总客户端数: {num_total_clients}")
    print(f"每轮选择客户端数: {NUM_SELECTED_CLIENTS}")
    
    # 创建参数注意力模块
    param_attention = ParamAttention(config, mode='a')
    param_attention.to(device)
    
    # 将所有模型移动到设备
    for i, model in enumerate(all_client_models):
        all_client_models[i] = model.to(device)
    resnet_model = resnet_model.to(device)
    
    f = config['f']  # 每f个batch进行一次联邦平均+知识融合
    
    # 初始化融合参数变量
    fusion_generated_param = None
    
    # 设置优化器
    param_atten_params = stats_params(param_attention)
    
    # 所有客户端优化器（与pure_federated保持一致）
    all_client_optimizers = []
    all_client_lr_schedulers = []
    for model in all_client_models:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        all_client_optimizers.append(optimizer)
        all_client_lr_schedulers.append(scheduler)
    
    # ResNet优化器（与run_res50_mbv2.py保持一致）
    optimizer_resnet = optim.SGD(resnet_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler_resnet = MultiStepLR(optimizer_resnet, milestones=[60, 120, 160], gamma=0.2)
    
    # 参数注意力优化器
    optimizer_atten = optim.AdamW(param_atten_params, lr=config['lr'])
    lr_scheduler_atten = CosineAnnealingLR(optimizer_atten, T_max=(EPOCH_NUM - 4) * len(train_dataloader) // f)
    
    criterion = nn.CrossEntropyLoss()
    cnt = 0
    
    
    for epoch in range(EPOCH_NUM):
        # 设置模型为训练模式
        for model in all_client_models:
            model.train()
        resnet_model.train()
        param_attention.train()
        
        # 每个epoch随机选择客户端参与训练
        selected_client_ids = select_random_clients(
            num_total_clients=num_total_clients, 
            num_selected_clients=NUM_SELECTED_CLIENTS,
            seed=epoch
        )
        
        selected_client_models = [all_client_models[i] for i in selected_client_ids]
        selected_client_optimizers = [all_client_optimizers[i] for i in selected_client_ids]

        print(f'Epoch {epoch}: 选择客户端 {selected_client_ids[:5]}...等{len(selected_client_ids)}个')
        logger.info(f'Epoch {epoch}: 选择客户端 {selected_client_ids}')
        print(f'epoch: {epoch} | lr: {all_client_optimizers[0].param_groups[0]["lr"]:.6f}')
        logger.info(f'epoch: {epoch} | lr: {all_client_optimizers[0].param_groups[0]["lr"]:.6f}')
        
        # 1. 先用完整数据训练ResNet一轮（遍历整个训练集）
        print(f"Epoch {epoch}: 训练ResNet...")
        loss_total_resnet = 0.0
        resnet_progress = tqdm(train_dataloader, desc=f'Epoch {epoch} - ResNet', leave=False)
        for batch_idx, (img, label) in enumerate(resnet_progress):
            img, label = img.to(device), label.to(device)
            
            optimizer_resnet.zero_grad()
            out = resnet_model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer_resnet.step()
            loss_total_resnet += loss.item()
            
            resnet_progress.set_postfix({'ResNet_Loss': f'{loss.item():.6f}'})
        
        avg_resnet_loss = loss_total_resnet / len(train_dataloader)
        print(f"Epoch {epoch}: ResNet训练完成，平均损失: {avg_resnet_loss:.6f}")
        
        # 2. 联邦客户端训练（与其他实验保持完全一致）
        # 获取选中客户端中最大的batch数，用于控制训练轮次
        max_batches_selected = max(len(all_client_dataloaders[client_id]) for client_id in selected_client_ids)
        
        # 创建选中客户端的数据迭代器
        selected_client_iterators = []
        for client_id in selected_client_ids:
            iterator = iter(all_client_dataloaders[client_id])
            selected_client_iterators.append(iterator)
        
        loss_total_clients = 0.0
        
        progress_bar = tqdm(range(max_batches_selected),
                           desc=f'Epoch {epoch} - Clients',
                           leave=False,
                           disable=False)
        
        for batch_idx in progress_bar:
            # 每f个batch进行联邦平均和知识融合
            if cnt % f == 0:
                logger.info(f'Batch {cnt}: 开始联邦平均和MergeNet知识融合 (选中{len(selected_client_ids)}个客户端)')
                
                # 1. 联邦平均（仅用于计算融合参数，不分发完整模型）
                avg_state_dict = federated_average_models(selected_client_models, device)
                if avg_state_dict is not None:
                    # 创建临时模型存储平均参数
                    avg_model = mobilenetv2()
                    avg_model.to(device)
                    avg_model.load_state_dict(avg_state_dict)
                    
                    # 2. MergeNet知识融合 - 生成融合参数
                    param_a = {
                        'conv': avg_model.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device),
                    }
                    param_b = {
                        'linear_weight': resnet_model.fc.weight.data.clone().detach().requires_grad_(True).to(device),
                    }
                    
                    # 用平均参数+teacher参数生成融合参数
                    out_a = param_attention(param_a, param_b)
                    # 保存生成的融合参数供后续使用，保持梯度信息
                    fusion_generated_param = out_a.clone().requires_grad_(True)
                    
                    # 3. 只将融合参数更新到选中客户端的对应层（避免额外的联邦同步）
                    fused_param_dict = {'stage6.2.residual.6.weight': out_a.detach()}
                    for client_model in selected_client_models:
                        client_model.load_state_dict(fused_param_dict, strict=False)
                
                logger.info(f'Batch {cnt}: 联邦平均和MergeNet知识融合完成，融合参数已分发给选中的{len(selected_client_ids)}个客户端')
            
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
                    client_optimizer.step()
                    
                    client_losses.append(loss_client.item())
                    active_clients += 1
                    
                except StopIteration:
                    # 客户端数据用完，跳过该客户端
                    continue
            
            cnt += 1
            
            # 每f个batch进行第二次联邦平均和超网络更新（在SGD训练后）
            if cnt % f == 0 and fusion_generated_param is not None:
                logger.info(f'Batch {cnt}: 开始第二次联邦平均和超网络参数更新')
                
                # 5. 再次对选中客户端进行联邦平均（SGD训练后）
                sgd_avg_state_dict = federated_average_models(selected_client_models, device)
                if sgd_avg_state_dict is not None:
                    # 6. 创建SGD训练后的平均模型
                    sgd_avg_model = mobilenetv2()
                    sgd_avg_model.to(device)
                    sgd_avg_model.load_state_dict(sgd_avg_state_dict)
                    
                    # 7. 获取SGD训练后的实际参数
                    final_param_sgd = sgd_avg_model.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True)
                    
                    # 8. 更新参数注意力模块（使用之前生成的融合参数和SGD后参数）
                    hypernetwork_update(param_attention, 
                                      fusion_generated_param,  # 之前生成的融合参数
                                      final_param_sgd,         # SGD训练后的实际参数
                                      optimizer_atten, lr_scheduler_atten, epoch)
                    
                    # 9. 将SGD后的参数分发给所有客户端（包括未选中的）
                    for client_model in all_client_models:
                        client_model.load_state_dict(sgd_avg_state_dict)
                
                logger.info(f'Batch {cnt}: 参数注意力模块已更新，SGD后参数已分发给所有{num_total_clients}个客户端')
            
            if client_losses:
                avg_client_loss = sum(client_losses) / len(client_losses)
                loss_total_clients += avg_client_loss
            
            # 更新进度条
            progress_bar.set_postfix({
                'Client_Loss': f'{avg_client_loss:.6f}' if client_losses else 'N/A',
                'ResNet_Loss': f'{avg_resnet_loss:.6f}',
                'Active_Clients': active_clients,
                'Batch': f'{batch_idx+1}/{max_batches_selected}',
                'lr': f'{selected_client_optimizers[0].param_groups[0]["lr"]:.6f}' if selected_client_optimizers else 'N/A'
            })
        
        # 更新学习率
        for scheduler in all_client_lr_schedulers:
            scheduler.step()
        lr_scheduler_resnet.step()
        
        # 每轮评估
        if (epoch + 1) % 2 == 0:  # 每2个epoch评估一次
            tqdm.write(f'\nEpoch {epoch} 评估:')
            logger.info(f'\nEpoch {epoch} 评估:')
            
            # 测试ResNet
            resnet_acc, resnet_acc5 = test_model(resnet_model, device, "ResNet50")
            
            # 测试部分客户端
            client_accs = []
            client_accs5 = []
            for i in range(min(5, num_total_clients)):
                acc1, acc5 = test_model(all_client_models[i], device, f"Client_{i}")
                client_accs.append(acc1)
                client_accs5.append(acc5)
            
            # 记录训练过程指标
            avg_client_acc = sum(client_accs) / len(client_accs) if client_accs else 0
            avg_client_acc5 = sum(client_accs5) / len(client_accs5) if client_accs5 else 0
            
            swanlab.log({
                # 训练损失
                "train/loss_selected_clients_avg": loss_total_clients / max(1, batch_idx + 1),
                "train/loss_resnet_avg": avg_resnet_loss,
                # ResNet准确率
                "test/acc_resnet_top1": resnet_acc,
                "test/acc_resnet_top5": resnet_acc5,
                # 客户端平均准确率
                "test/acc_all_clients_avg_top1": avg_client_acc,
                "test/acc_all_clients_avg_top5": avg_client_acc5,
                # 最佳准确率
                "test/best_acc_resnet": best_acc_resnet,
                "test/best_acc_clients_avg": sum(best_acc_clients) / len(best_acc_clients),
                # 学习率
                "lr_clients": selected_client_optimizers[0].param_groups[0]["lr"],
                "lr_resnet": optimizer_resnet.param_groups[0]["lr"],
                # 当前epoch
                "epoch": epoch
            })
            
            tqdm.write(f'Epoch {epoch} 完成')
            logger.info(f'Epoch {epoch} 完成')
    
    logger.info(f'训练完成！ResNet最佳准确率: {best_acc_resnet:.4f}')
    logger.info(f'前10个客户端最佳准确率: {best_acc_clients[:10]}')
    
    return all_client_models, resnet_model

def test_model(model, device, model_name):
    """测试模型性能"""
    global best_acc_clients, best_acc_resnet
    
    model.eval()
    top1_acc = 0.
    top5_acc = 0.
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(test_dataloader):
            img, label = img.to(device), label.to(device)
            out = model(img)
            
            loss = criterion(out, label)
            running_loss += loss.item()
            pred1 = out.argmax(dim=1)
            label_resize = label.view(-1, 1)
            _, pred5 = out.topk(5, 1, True, True)
            total += label.size(0)
            top1_acc += pred1.eq(label).sum().item()
            top5_acc += torch.eq(pred5, label_resize).sum().float().item()
    
    final_top1_acc = 100. * top1_acc / total
    final_top5_acc = 100. * top5_acc / total
    final_test_loss = running_loss / len(test_dataloader)
    
    # 保存最佳模型
    if 'Client' in model_name:
        # 处理 "Client_{i}" 和 "Final_Client_{i}" 两种格式
        parts = model_name.split('_')
        if len(parts) >= 3 and parts[0] == "Final":
            # "Final_Client_{i}" 格式
            client_id = int(parts[2])
        else:
            # "Client_{i}" 格式
            client_id = int(parts[1])
        
        if final_top1_acc > best_acc_clients[client_id] * 100:
            best_acc_clients[client_id] = final_top1_acc / 100
            torch.save(model.state_dict(), f'checkpoints/federated_mergenet_trainable_best_client_{client_id}.pth')
    elif 'ResNet' in model_name:
        if final_top1_acc > best_acc_resnet * 100:
            best_acc_resnet = final_top1_acc / 100
            torch.save(model.state_dict(), f'checkpoints/federated_mergenet_trainable_best_resnet50.pth')
    
    tqdm.write(f'{model_name} - Top1: {final_top1_acc:.2f}%, Top5: {final_top5_acc:.2f}%, Loss: {final_test_loss:.4f}')
    logger.info(f'{model_name} - Top1: {final_top1_acc:.2f}%, Top5: {final_top5_acc:.2f}%, Loss: {final_test_loss:.4f}')
    
    # 记录详细的测试统计信息
    if 'Client' in model_name:
        # 处理 "Client_{i}" 和 "Final_Client_{i}" 两种格式
        parts = model_name.split('_')
        if len(parts) >= 3 and parts[0] == "Final":
            # "Final_Client_{i}" 格式
            client_id = int(parts[2])
        else:
            # "Client_{i}" 格式
            client_id = int(parts[1])
        best_acc_value = best_acc_clients[client_id] * 100
    else:
        best_acc_value = best_acc_resnet * 100
    
    swanlab.log({
        f'test_acc_{model_name}_top1': final_top1_acc,
        f'test_acc_{model_name}_top5': final_top5_acc,
        f'test_loss_{model_name}': final_test_loss,
        f'best_acc_{model_name}': best_acc_value
    })
    
    return final_top1_acc, final_top5_acc

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
    
    logger.info(f'Federated MergeNet (Trainable ResNet) with {NUM_TOTAL_CLIENTS} clients, selecting {NUM_SELECTED_CLIENTS} per round')
    
    start = time.time()
    logger.info('Start training!!!\n')

    global best_acc_clients, best_acc_resnet
    
    # 测试不同alpha值的对比实验
    alpha_values = [0.5, 10, 100]  # 不同的数据异构程度
    
    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"开始 Alpha = {alpha} 的可训练ResNet MergeNet实验")
        print(f"{'='*60}")
        logger.info(f"开始 Alpha = {alpha} 的可训练ResNet MergeNet实验")
        
        # 创建联邦数据划分（与其他实验完全一致）
        print(f"创建联邦数据划分 (alpha={alpha})...")
        partitioner, client_dataloaders = create_federated_dataloaders(
            dataset=train_dataloader.dataset,
            num_clients=NUM_TOTAL_CLIENTS,
            alpha=alpha,
            batch_size=train_dataloader.batch_size,
            num_workers=2,
            min_samples_per_client=50
        )
        
        # 打印数据分布统计
        partitioner.print_statistics()
        
        # 测试不同的参数组合
        for j in [2]:  # f=2，与其他实验保持一致
            logger.info(f'\n========== Alpha = {alpha}, f = {j} ==========')
            print(f'\n========== Alpha = {alpha}, f = {j} ==========')
            
            # 重置最佳准确率
            best_acc_clients = [0.0] * NUM_TOTAL_CLIENTS
            best_acc_resnet = 0.0
            
            # 为当前参数组合创建SwanLab实验
            swanlab.init(
                project="FL2Merget",
                experiment_name=f"Trainable_resnet_alpha_{alpha}_freq_{j}",
                description=f"可训练ResNet MergeNet对比实验：Alpha={alpha}, 联邦学习 + MergeNet (可训练ResNet), 联邦平均频率: {j}",
                config={
                    "experiment_type": "federated_mergenet_trainable",
                    "num_total_clients": NUM_TOTAL_CLIENTS,
                    "num_selected_clients": NUM_SELECTED_CLIENTS,
                    "epoch_num": EPOCH_NUM,
                    "federated_frequency": j,
                    "alpha": alpha,  # 记录当前alpha值
                    "lr_clients": 0.1,
                    "lr_resnet": 0.1,
                    "lr_attention": 0.01,
                    "batch_size": train_dataloader.batch_size,
                    "min_samples_per_client": 50
                }
                )
            
            # 创建所有客户端模型
            print("创建客户端模型...")
            all_client_models = []
            for i in range(NUM_TOTAL_CLIENTS):
                model = mobilenetv2()
                all_client_models.append(model)
            
            # 创建ResNet模型（服务器端，可训练）
            print("创建ResNet模型...")
            resnet_model = resnet50()
            print(f"数据异构程度 (Alpha): {alpha}")
            print(f"ResNet50参数量: {sum(p.numel() for p in resnet_model.parameters()):,}")
            print(f"MobileNetV2参数量: {sum(p.numel() for p in all_client_models[0].parameters()):,}")
            
            # 设置配置参数
            config['f'] = j
            config['a_size_conv'] = [160, 960]
            config['a_size_linear'] = [100, 1280]
            config['b_size_conv'] = [4, 1024]
            config['b_size_linear'] = [100, 2048]
            config['mode'] = 5
            config['lr'] = 0.01  # 参数注意力学习率
            
            # 开始训练
            trained_client_models, trained_resnet = train_federated_mergenet_trainable(
                all_client_models, client_dataloaders, resnet_model, config
            )
            
            # 最终测试
            print("\n" + "="*50)
            print(f"Alpha = {alpha} 最终测试结果:")
            logger.info(f"Alpha = {alpha} 最终测试结果:")
            
            # 测试ResNet
            final_resnet_acc, _ = test_model(trained_resnet, torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'), "Final_ResNet50")
            
            # 测试所有客户端
            final_client_accs = []
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            for i in range(NUM_TOTAL_CLIENTS):
                acc, _ = test_model(trained_client_models[i], device, f"Final_Client_{i}")
                final_client_accs.append(acc)
            
            avg_client_acc = sum(final_client_accs) / len(final_client_accs)
            
            print(f"\nAlpha = {alpha} 最终结果总结:")
            print(f"ResNet50最终准确率: {final_resnet_acc:.2f}%")
            print(f"客户端平均准确率: {avg_client_acc:.2f}%")
            print(f"客户端最高准确率: {max(final_client_accs):.2f}%")
            print(f"客户端最低准确率: {min(final_client_accs):.2f}%")
            
            logger.info(f"Alpha = {alpha} ResNet50最终准确率: {final_resnet_acc:.2f}%")
            logger.info(f"Alpha = {alpha} 客户端平均准确率: {avg_client_acc:.2f}%")
            logger.info(f"Alpha = {alpha} 客户端最高准确率: {max(final_client_accs):.2f}%")
            logger.info(f"Alpha = {alpha} 客户端最低准确率: {min(final_client_accs):.2f}%")
            
            # 计算当前alpha实验的训练时间
            alpha_end_time = time.time()
            alpha_total_time = alpha_end_time - start
            
            # 结束当前实验
            swanlab.finish()
            
            print(f"\nAlpha = {alpha} 训练时间: {alpha_total_time//3600:.0f}小时 {(alpha_total_time%3600)//60:.0f}分钟")
            logger.info(f"Alpha = {alpha} 训练时间: {alpha_total_time//3600:.0f}小时 {(alpha_total_time%3600)//60:.0f}分钟")
    
    # 所有alpha实验完成
    final_end_time = time.time()
    total_experiment_time = final_end_time - start
    print(f"\n{'='*60}")
    print(f"所有Alpha对比实验完成！")
    print(f"总实验时间: {total_experiment_time//3600:.0f}小时 {(total_experiment_time%3600)//60:.0f}分钟")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
