"""
修复后的联邦学习 + MergeNet 实现
解决了原版本中的多个关键问题：

1. 移除ResNet50额外训练，减少资源竞争
2. 使用预训练的teacher模型代替实时训练的ResNet
3. 简化MergeNet融合流程
4. 统一评估频率和参数设置
5. 确保除了MergeNet外，其他所有配置与pure_federated完全一致

主要修改点：
- 去除ResNet50的训练循环
- 使用固定的teacher参数
- 简化apply_mergenet_fusion函数
- 统一评估和日志设置
"""
import torch
import time
import yaml
from tqdm import tqdm
import random
import os
import warnings
import torch.optim as optim
import numpy as np
from dataset.cls_dataloader import train_dataloader, test_dataloader
from dataset.federated_data_partition import create_federated_dataloaders, select_random_clients
import logging
import torch.nn as nn
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import swanlab

EPOCH_NUM = 200
NUM_TOTAL_CLIENTS = 50  # 总客户端数
NUM_SELECTED_CLIENTS = 15  # 每轮选择的客户端数 (与pure_federated保持一致)
best_acc_clients = [0.0] * NUM_TOTAL_CLIENTS  # 50个客户端的最佳准确率
current_config = None

# 修改日志文件名，便于区分
logging.basicConfig(filename='logs/federated_mergenet_fixed.log',
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

def train_federated_mergenet_fixed(all_client_models, all_client_dataloaders, teacher_params, config):
    """修复后的联邦学习 + MergeNet训练主函数"""
    global current_config, best_acc_clients
    current_config = config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running at {device}")
    print(torch.cuda.get_device_name())
    
    num_total_clients = len(all_client_models)
    print(f"总客户端数: {num_total_clients}")
    print(f"每轮选择客户端数: {NUM_SELECTED_CLIENTS}")
    
    # 创建参数注意力模块
    param_attention = ParamAttention(config, mode='a')
    param_attention.to(device)
    
    # 将所有模型移动到设备
    for i, model in enumerate(all_client_models):
        all_client_models[i] = model.to(device)
    
    f = config['f']  # 每f个batch进行一次联邦平均+知识融合
    
    # 设置优化器
    param_atten_params = stats_params(param_attention)
    
    # 所有客户端优化器 (与pure_federated保持一致)
    all_client_optimizers = []
    all_client_lr_schedulers = []
    for model in all_client_models:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        all_client_optimizers.append(optimizer)
        all_client_lr_schedulers.append(scheduler)
    
    # 参数注意力优化器
    optimizer_atten = optim.AdamW(param_atten_params, lr=config['lr'])
    lr_scheduler_atten = CosineAnnealingLR(optimizer_atten, T_max=(EPOCH_NUM - 4) * len(train_dataloader) // f)
    
    criterion = nn.CrossEntropyLoss()
    cnt = 0
    
    for epoch in range(EPOCH_NUM):
        param_attention.train()
        for model in all_client_models:
            model.train()
        
        # 每个epoch随机选择客户端参与训练 (与pure_federated保持一致)
        selected_client_ids = select_random_clients(
            num_total_clients=num_total_clients, 
            num_selected_clients=NUM_SELECTED_CLIENTS,
            seed=epoch  # 使用epoch作为seed确保可重现性
        )
        
        selected_client_models = [all_client_models[i] for i in selected_client_ids]
        selected_client_optimizers = [all_client_optimizers[i] for i in selected_client_ids]
        
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
            # 选中的客户端使用各自的数据子集训练
            client_losses = []
            active_clients = 0
            
            # 每f个batch进行联邦平均和知识融合（在客户端训练前）
            if cnt % f == 0:
                # 1. 先进行联邦平均
                averaged_state_dict = federated_average_models(selected_client_models, device)
                
                if averaged_state_dict is not None and len(selected_client_models) > 0:
                    # 2. 创建平均模型
                    averaged_model = mobilenetv2().to(device)
                    averaged_model.load_state_dict(averaged_state_dict)
                    
                    # 3. 使用参数注意力生成融合参数
                    param_a = {
                        'conv': averaged_model.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device),
                    }
                    param_b = {
                        'linear_weight': teacher_params.clone().detach().requires_grad_(True).to(device),
                    }
                    out_a = param_attention(param_a, param_b)  # 参数注意力生成的融合参数
                    
                    # 4. 将融合参数只应用到选中的客户端（让选中的客户端用融合参数训练）
                    fused_param_dict = {'stage6.2.residual.6.weight': out_a}
                    for client_model in selected_client_models:
                        client_model.load_state_dict(fused_param_dict, strict=False)
                    
                    # 5. 保存融合参数，用于后续的参数注意力更新
                    fusion_generated_param = out_a.clone().detach().requires_grad_(True)
                    
                    logger.info(f'Batch {cnt}: 联邦平均 + 知识融合完成，融合参数已应用到{len(selected_client_ids)}个选中客户端')
            
            # 客户端使用融合后的参数进行训练
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
                    client_optimizer.step()  # ← 客户端用融合参数进行SGD训练
                    client_losses.append(loss_client.item())
                    active_clients += 1
                    
                except StopIteration:
                    # 客户端数据用完，跳过该客户端
                    continue
            
            # 训练完成后，更新参数注意力模块并分发SGD后的参数
            if cnt % f == 0 and 'fusion_generated_param' in locals():
                # 7. 重新计算联邦平均（基于SGD训练后的选中客户端参数）
                averaged_state_dict_post_sgd = federated_average_models(selected_client_models, device)
                
                if averaged_state_dict_post_sgd is not None:
                    # 8. 创建SGD训练后的平均模型
                    averaged_model_post_sgd = mobilenetv2().to(device)
                    averaged_model_post_sgd.load_state_dict(averaged_state_dict_post_sgd)
                    
                    # 9. 获取SGD训练后的实际参数
                    final_param_post_sgd = averaged_model_post_sgd.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True)
                    
                    # 10. 更新参数注意力模块（用SGD后的参数与之前生成的融合参数）
                    hypernetwork_update(param_attention, 
                                      fusion_generated_param,  # 之前生成的融合参数
                                      final_param_post_sgd,    # SGD训练后的实际参数
                                      optimizer_atten, lr_scheduler_atten, epoch)
                    
                    # 11. 将SGD训练后的参数分发给所有客户端（包括未选中的）
                    for client_model in all_client_models:
                        client_model.load_state_dict(averaged_state_dict_post_sgd)
                    
                    logger.info(f'Batch {cnt}: 参数注意力模块已更新，SGD后参数已分发给所有{num_total_clients}个客户端')
                
                # 清理临时变量
                del fusion_generated_param
            
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
        logger.info(f'\\nEpoch {epoch}')
        
        if loss_train_ave_selected_clients:
            avg_selected_client_loss = np.mean(loss_train_ave_selected_clients)
            tqdm.write(f'Selected Clients Avg Training Loss: {avg_selected_client_loss:.6f}')
            logger.info(f'Selected Clients Avg Training Loss: {avg_selected_client_loss:.6f}')
        
        # 评估所有客户端（与pure_federated保持一致：每2个epoch评估一次）
        if epoch % 2 == 0 or epoch == EPOCH_NUM - 1:
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
                          f'checkpoints/federated_mergenet_fixed_best_client_{client_id}_freq_{current_config["f"]}.pth')
    
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
    
    logger.info(f'Fixed Federated MergeNet with {NUM_TOTAL_CLIENTS} clients, selecting {NUM_SELECTED_CLIENTS} per round')
    
    start = time.time()
    logger.info('Start training!!!\\n')

    global best_acc_clients
    
    # 测试不同alpha值的对比实验
    alpha_values = [0.5, 10, 100]  # 不同的数据异构程度
    
    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"开始 Alpha = {alpha} 的固定ResNet MergeNet实验")
        print(f"{'='*60}")
        logger.info(f"开始 Alpha = {alpha} 的固定ResNet MergeNet实验")
        
        # 创建联邦数据划分（与pure_federated完全一致）
        print(f"创建联邦数据划分 (alpha={alpha})...")
        partitioner, client_dataloaders = create_federated_dataloaders(
            dataset=train_dataloader.dataset,
            num_clients=NUM_TOTAL_CLIENTS,
            alpha=alpha,  # Dirichlet参数，控制数据异构程度
            batch_size=train_dataloader.batch_size,
            num_workers=2,
            min_samples_per_client=50
        )
        
        # 打印数据分布统计
        partitioner.print_statistics()
        
        # 创建固定的teacher参数（使用预训练ResNet50的参数）
        print("创建固定的teacher参数...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        teacher_model = resnet50().to(device)
        teacher_params = teacher_model.fc.weight.data.clone().detach()
        print(f"Teacher参数形状: {teacher_params.shape}")
        
        # 测试不同的参数组合
        for j in [2]:  # f值（与pure_federated保持一致）
            config['f'] = j
            best_acc_clients = [0.] * NUM_TOTAL_CLIENTS

            # 初始化SwanLab实验
            swanlab.init(
                project="FL2Merget",
                experiment_name=f"fixed_mergenet_alpha_{alpha}_freq_{j}_select_{NUM_SELECTED_CLIENTS}",
                description=f"固定ResNet MergeNet对比实验：Alpha={alpha}, 50个客户端，每轮选择{NUM_SELECTED_CLIENTS}个，每{j}个batch进行联邦平均+知识融合",
                config={
                    **config,
                    'num_total_clients': NUM_TOTAL_CLIENTS,
                    'num_selected_clients': NUM_SELECTED_CLIENTS,
                    'data_alpha': alpha,  # 记录当前alpha值
                    'min_samples_per_client': 50,
                    'merge_net': True,  # 标记这是带MergeNet的实验
                    'fixed_teacher': True,  # 使用固定teacher参数
                    'no_resnet_training': True,  # 不训练ResNet
                },
                )
            
            # 创建50个客户端模型
            print(f"创建{NUM_TOTAL_CLIENTS}个客户端模型...")
            all_client_models = [mobilenetv2() for _ in range(NUM_TOTAL_CLIENTS)]
            
            print(f"总客户端数量: {NUM_TOTAL_CLIENTS}")
            print(f"每轮选择客户端数: {NUM_SELECTED_CLIENTS}")
            print(f"数据异构程度 (Alpha): {alpha}")
            print(f"MobileNet v2 参数量: {sum(p.numel() for p in all_client_models[0].parameters()):,}")
            print(f"联邦平均频率: 每{j}个batch")
            print("注意: 本实验使用修复后的MergeNet知识融合")
            
            logger.info(f'Alpha:{alpha}, f:{j}, clients:{NUM_TOTAL_CLIENTS}, selected:{NUM_SELECTED_CLIENTS}, mergenet: fixed')
            
            # 开始训练
            train_federated_mergenet_fixed(all_client_models, client_dataloaders, teacher_params, config)

            # 实验结束
            swanlab.finish()
            
            print(f"\nAlpha = {alpha} 固定ResNet MergeNet实验完成")
            print(f"前10个客户端最佳准确率: {[f'{acc:.4f}' for acc in best_acc_clients[:10]]}")
            print(f"所有客户端平均最佳准确率: {np.mean(best_acc_clients):.4f}")
            
            logger.info(f"Alpha = {alpha} 实验完成，平均最佳准确率: {np.mean(best_acc_clients):.4f}")
        
        # 保存实验总结
        end_time = time.time()
        total_time = end_time - start
        
        print(f"\n=== Alpha = {alpha} 固定ResNet MergeNet实验完成 ===")
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"前10个客户端最佳准确率: {[f'{acc:.4f}' for acc in best_acc_clients[:10]]}")
        print(f"所有客户端平均最佳准确率: {np.mean(best_acc_clients):.4f}")
        
        logger.info(f"Alpha = {alpha} 实验完成，总时间: {total_time/3600:.2f} 小时")
        logger.info(f"所有客户端平均最佳准确率: {np.mean(best_acc_clients):.4f}")
    
    # 所有alpha实验完成
    final_end_time = time.time()
    total_experiment_time = final_end_time - start
    print(f"\n{'='*60}")
    print(f"所有Alpha对比实验完成！")
    print(f"总实验时间: {total_experiment_time/3600:.2f} 小时")
    print(f"{'='*60}")
    logger.info(f"所有Alpha对比实验完成，总时间: {total_experiment_time/3600:.2f} 小时")

if __name__ == '__main__':
    main()
