"""
联邦学习 + MergeNet 与基线方法结果对比分析
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

class ExperimentComparer:
    """实验结果对比分析器"""
    
    def __init__(self, baseline_log: str, federated_log: str):
        self.baseline_log = baseline_log
        self.federated_log = federated_log
        self.baseline_results = {}
        self.federated_results = {}
        
    def parse_baseline_log(self) -> Dict:
        """解析基线实验日志"""
        if not os.path.exists(self.baseline_log):
            print(f"基线日志文件不存在: {self.baseline_log}")
            return {}
            
        results = {
            'mbv_accuracies': [],
            'res_accuracies': [],
            'mbv_losses': [],
            'res_losses': [],
            'epochs': []
        }
        
        with open(self.baseline_log, 'r') as f:
            content = f.read()
            
            # 提取准确率
            mbv_acc_pattern = r'Mbv Top1 Acc: ([\d.]+)'
            res_acc_pattern = r'Res Top1 Acc: ([\d.]+)'
            
            mbv_accs = re.findall(mbv_acc_pattern, content)
            res_accs = re.findall(res_acc_pattern, content)
            
            results['mbv_accuracies'] = [float(acc) for acc in mbv_accs]
            results['res_accuracies'] = [float(acc) for acc in res_accs]
            
            # 提取损失
            mbv_loss_pattern = r'Mbv Training Loss: ([\d.]+)'
            res_loss_pattern = r'Res Training Loss: ([\d.]+)'
            
            mbv_losses = re.findall(mbv_loss_pattern, content)
            res_losses = re.findall(res_loss_pattern, content)
            
            results['mbv_losses'] = [float(loss) for loss in mbv_losses]
            results['res_losses'] = [float(loss) for loss in res_losses]
            
            results['epochs'] = list(range(len(results['mbv_accuracies'])))
            
        return results
    
    def parse_federated_log(self) -> Dict:
        """解析联邦学习日志"""
        if not os.path.exists(self.federated_log):
            print(f"联邦学习日志文件不存在: {self.federated_log}")
            return {}
            
        results = {
            'round_accuracies': [],
            'round_losses': [],
            'rounds': [],
            'knowledge_fusion_rounds': []
        }
        
        with open(self.federated_log, 'r') as f:
            content = f.read()
            
            # 提取聚合准确率
            acc_pattern = r'Aggregated accuracy: ([\d.]+)'
            loss_pattern = r'loss: ([\d.]+)'
            round_pattern = r'Round (\d+)'
            fusion_pattern = r'Knowledge fusion completed'
            
            accuracies = re.findall(acc_pattern, content)
            losses = re.findall(loss_pattern, content)
            rounds = re.findall(round_pattern, content)
            
            results['round_accuracies'] = [float(acc) for acc in accuracies]
            results['round_losses'] = [float(loss) for loss in losses]
            results['rounds'] = [int(r) for r in rounds if r.isdigit()]
            
            # 找到应用知识融合的轮数
            fusion_matches = re.finditer(fusion_pattern, content)
            for match in fusion_matches:
                # 在融合信息前查找最近的轮数
                text_before = content[:match.start()]
                round_matches = re.findall(round_pattern, text_before)
                if round_matches:
                    results['knowledge_fusion_rounds'].append(int(round_matches[-1]))
            
        return results
    
    def generate_comparison_plots(self):
        """生成对比图表"""
        
        # 解析数据
        baseline = self.parse_baseline_log()
        federated = self.parse_federated_log()
        
        if not baseline or not federated:
            print("无法生成对比图表：缺少必要的日志数据")
            return
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 准确率对比
        if baseline.get('mbv_accuracies') and federated.get('round_accuracies'):
            ax1.plot(baseline['epochs'][:len(baseline['mbv_accuracies'])], 
                    baseline['mbv_accuracies'], 
                    label='Baseline MobileNetV2', marker='o', alpha=0.7)
            
            ax1.plot(federated['rounds'][:len(federated['round_accuracies'])], 
                    federated['round_accuracies'], 
                    label='Federated + MergeNet', marker='s', alpha=0.7)
            
            # 标记知识融合点
            for round_num in federated.get('knowledge_fusion_rounds', []):
                if round_num < len(federated['round_accuracies']):
                    ax1.axvline(x=round_num, color='red', linestyle='--', alpha=0.5)
            
            ax1.set_xlabel('Epoch/Round')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('Accuracy Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 损失对比
        if baseline.get('mbv_losses') and federated.get('round_losses'):
            ax2.plot(baseline['epochs'][:len(baseline['mbv_losses'])], 
                    baseline['mbv_losses'], 
                    label='Baseline MobileNetV2', marker='o', alpha=0.7)
            
            ax2.plot(federated['rounds'][:len(federated['round_losses'])], 
                    federated['round_losses'], 
                    label='Federated + MergeNet', marker='s', alpha=0.7)
            
            ax2.set_xlabel('Epoch/Round')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 收敛速度分析
        self._plot_convergence_analysis(ax3, baseline, federated)
        
        # 4. 最终性能对比
        self._plot_final_performance(ax4, baseline, federated)
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/federated_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 生成文字报告
        self._generate_text_report(baseline, federated)
    
    def _plot_convergence_analysis(self, ax, baseline, federated):
        """收敛速度分析"""
        
        # 计算达到特定准确率阈值的时间
        thresholds = [60, 65, 70, 75]
        
        baseline_convergence = []
        federated_convergence = []
        
        for threshold in thresholds:
            # 基线方法
            baseline_epoch = None
            for i, acc in enumerate(baseline.get('mbv_accuracies', [])):
                if acc >= threshold:
                    baseline_epoch = i
                    break
            baseline_convergence.append(baseline_epoch if baseline_epoch is not None else len(baseline.get('mbv_accuracies', [])))
            
            # 联邦方法
            federated_round = None
            for i, acc in enumerate(federated.get('round_accuracies', [])):
                if acc >= threshold:
                    federated_round = i
                    break
            federated_convergence.append(federated_round if federated_round is not None else len(federated.get('round_accuracies', [])))
        
        x = np.arange(len(thresholds))
        width = 0.35
        
        ax.bar(x - width/2, baseline_convergence, width, label='Baseline', alpha=0.7)
        ax.bar(x + width/2, federated_convergence, width, label='Federated + MergeNet', alpha=0.7)
        
        ax.set_xlabel('Accuracy Threshold (%)')
        ax.set_ylabel('Epochs/Rounds to Converge')
        ax.set_title('Convergence Speed Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}%' for t in thresholds])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_performance(self, ax, baseline, federated):
        """最终性能对比"""
        
        # 计算最终性能指标
        metrics = ['Max Accuracy', 'Final Accuracy', 'Avg Loss']
        
        baseline_metrics = [
            max(baseline.get('mbv_accuracies', [0])),
            baseline.get('mbv_accuracies', [0])[-1] if baseline.get('mbv_accuracies') else 0,
            np.mean(baseline.get('mbv_losses', [0])[-10:]) if baseline.get('mbv_losses') else 0
        ]
        
        federated_metrics = [
            max(federated.get('round_accuracies', [0])),
            federated.get('round_accuracies', [0])[-1] if federated.get('round_accuracies') else 0,
            np.mean(federated.get('round_losses', [0])[-10:]) if federated.get('round_losses') else 0
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, baseline_metrics, width, label='Baseline', alpha=0.7)
        ax.bar(x + width/2, federated_metrics, width, label='Federated + MergeNet', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Final Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (b_val, f_val) in enumerate(zip(baseline_metrics, federated_metrics)):
            ax.text(i - width/2, b_val + 0.01, f'{b_val:.2f}', ha='center', va='bottom')
            ax.text(i + width/2, f_val + 0.01, f'{f_val:.2f}', ha='center', va='bottom')
    
    def _generate_text_report(self, baseline, federated):
        """生成文字对比报告"""
        
        report = []
        report.append("=" * 50)
        report.append("联邦学习 + MergeNet vs 基线方法对比报告")
        report.append("=" * 50)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 基线结果
        if baseline.get('mbv_accuracies'):
            max_baseline_acc = max(baseline['mbv_accuracies'])
            final_baseline_acc = baseline['mbv_accuracies'][-1]
            report.append(f"基线方法 (MobileNetV2):")
            report.append(f"  最高准确率: {max_baseline_acc:.2f}%")
            report.append(f"  最终准确率: {final_baseline_acc:.2f}%")
            report.append(f"  训练轮数: {len(baseline['mbv_accuracies'])}")
        
        # 联邦结果
        if federated.get('round_accuracies'):
            max_federated_acc = max(federated['round_accuracies'])
            final_federated_acc = federated['round_accuracies'][-1]
            report.append(f"")
            report.append(f"联邦学习 + MergeNet:")
            report.append(f"  最高准确率: {max_federated_acc:.2f}%")
            report.append(f"  最终准确率: {final_federated_acc:.2f}%")
            report.append(f"  联邦轮数: {len(federated['round_accuracies'])}")
            report.append(f"  知识融合次数: {len(federated.get('knowledge_fusion_rounds', []))}")
        
        # 对比分析
        if baseline.get('mbv_accuracies') and federated.get('round_accuracies'):
            acc_improvement = max_federated_acc - max_baseline_acc
            report.append(f"")
            report.append(f"对比分析:")
            report.append(f"  准确率提升: {acc_improvement:+.2f}%")
            report.append(f"  提升幅度: {(acc_improvement/max_baseline_acc)*100:+.1f}%")
            
            if acc_improvement > 0:
                report.append(f"  结论: 联邦学习 + MergeNet 方法在准确率上优于基线方法")
            else:
                report.append(f"  结论: 基线方法在准确率上优于联邦学习 + MergeNet 方法")
        
        report.append("")
        report.append("详细分析:")
        report.append("1. 联邦学习引入了分布式训练，可能增加了通信开销")
        report.append("2. MergeNet知识融合提供了额外的知识迁移能力")
        report.append("3. 两种方法的结合需要在效率和性能间权衡")
        
        # 保存报告
        os.makedirs('results', exist_ok=True)
        with open('results/federated_vs_baseline_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))

def main():
    """主函数"""
    
    # 日志文件路径
    baseline_log = "logs/train_res50_mbv2.log"
    federated_log = "logs/federated_mergenet_server.log"
    
    # 创建对比分析器
    comparer = ExperimentComparer(baseline_log, federated_log)
    
    # 生成对比分析
    comparer.generate_comparison_plots()
    
    print("对比分析完成！")
    print("- 图表已保存至: results/federated_vs_baseline_comparison.png")
    print("- 报告已保存至: results/federated_vs_baseline_report.txt")

if __name__ == "__main__":
    main()
