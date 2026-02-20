"""
Trust-Weighted Federated Learning - Comprehensive Experimental Framework

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
import pandas as pd
from datetime import datetime
import os

# Import the main FL classes
from trust_fl_simulation import Client, TrustWeightedFL, generate_data

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

class ExperimentRunner:
    
    def __init__(self, output_dir: str = './experiment_results'):
        self.output_dir = output_dir
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single_experiment(self, num_clients: int = 20, num_good: int = 12, 
                              num_noisy: int = 5, num_malicious: int = 3, 
                              num_rounds: int = 50, clients_per_round: int = 10,
                              trust_decay: float = 0.9, seed: int = 42) -> Dict:
        """Run a single FL simulation with given parameters."""
        np.random.seed(seed)
        
        # Generate data
        client_data, X_val, y_val = generate_data(num_clients, samples_per_client=100)
        
        # Create clients
        clients_standard = []
        clients_trust = []
        
        # Assign client types
        client_types = []
        for i in range(num_clients):
            if i < num_malicious:
                client_types.append('good')
            elif i < num_malicious + num_good:
                client_types.append('good')
            elif i < num_malicious + num_good + num_noisy:
                client_types.append('noisy')
            else:
                client_types.append('unstable')
        
        for i in range(num_clients):
            is_malicious = i < num_malicious
            quality = client_types[i]
            
            client_std = Client(i, quality, is_malicious)
            client_std.set_data(client_data[i][0], client_data[i][1])
            clients_standard.append(client_std)
            
            client_trust = Client(i, quality, is_malicious)
            client_trust.set_data(client_data[i][0], client_data[i][1])
            clients_trust.append(client_trust)
        
        # Initialize FL systems
        fl_standard = TrustWeightedFL(num_features=10)
        fl_standard.set_validation_data(X_val, y_val)
        
        fl_trust = TrustWeightedFL(num_features=10, trust_decay=trust_decay, selection_threshold=0.3)
        fl_trust.set_validation_data(X_val, y_val)
        
        # Training loop
        for round_num in range(num_rounds):
            fl_standard.train_round(clients_standard, clients_per_round, use_trust=False)
            fl_trust.train_round(clients_trust, clients_per_round, use_trust=True)
        
        # Compile results
        results = {
            'standard_fl_accuracy': fl_standard.history['global_accuracy'],
            'trust_fl_accuracy': fl_trust.history['global_accuracy'],
            'trust_history': fl_trust.history['average_trust'],
            'final_std_acc': fl_standard.history['global_accuracy'][-1],
            'final_trust_acc': fl_trust.history['global_accuracy'][-1],
            'improvement': ((fl_trust.history['global_accuracy'][-1] - 
                           fl_standard.history['global_accuracy'][-1]) / 
                          fl_standard.history['global_accuracy'][-1] * 100),
            'client_trust_scores': {c.client_id: c.trust_score for c in clients_trust},
            'client_selection_counts': {c.client_id: c.history['selection_count'] 
                                       for c in clients_trust},
            'malicious_detection_round': self._detect_isolation_round(clients_trust[:num_malicious]),
            'convergence_round_std': self._get_convergence_round(fl_standard.history['global_accuracy'], 0.80),
            'convergence_round_trust': self._get_convergence_round(fl_trust.history['global_accuracy'], 0.80),
        }
        
        return results
    
    @staticmethod
    def _detect_isolation_round(malicious_clients: List[Client]) -> int:
        """Find the round when malicious clients are isolated (trust < 0.3)."""
        for client in malicious_clients:
            for round_num, trust in enumerate(client.history['trust_scores']):
                if trust < 0.3:
                    return round_num
        return -1
    
    @staticmethod
    def _get_convergence_round(accuracy_history: List[float], threshold: float) -> int:
        """Find the round when accuracy reaches threshold."""
        for round_num, acc in enumerate(accuracy_history):
            if acc >= threshold:
                return round_num + 1
        return -1
    
    def experiment_1_malicious_percentage(self, num_runs: int = 3):
        """
        Experiment 1: Varying malicious client percentages
        Tests robustness to different attack intensities
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: MALICIOUS CLIENT PERCENTAGE IMPACT")
        print("="*70)
        
        malicious_percentages = [0, 5, 10, 15, 20]
        results_by_percentage = {}
        
        for mal_pct in malicious_percentages:
            print(f"\nTesting with {mal_pct}% malicious clients ({num_runs} runs)...")
            run_results = []
            
            num_malicious = max(1, int(20 * mal_pct / 100))
            
            for run_id in range(num_runs):
                result = self.run_single_experiment(
                    num_clients=20,
                    num_malicious=num_malicious,
                    num_rounds=50,
                    seed=42 + run_id
                )
                run_results.append(result)
                print(f"  Run {run_id+1}: Standard={result['final_std_acc']:.4f}, "
                      f"Trust={result['final_trust_acc']:.4f}, "
                      f"Improvement={result['improvement']:.2f}%")
            
            # Aggregate statistics
            results_by_percentage[mal_pct] = {
                'std_acc_mean': np.mean([r['final_std_acc'] for r in run_results]),
                'std_acc_std': np.std([r['final_std_acc'] for r in run_results]),
                'trust_acc_mean': np.mean([r['final_trust_acc'] for r in run_results]),
                'trust_acc_std': np.std([r['final_trust_acc'] for r in run_results]),
                'improvement_mean': np.mean([r['improvement'] for r in run_results]),
                'improvement_std': np.std([r['improvement'] for r in run_results]),
                'conv_std_mean': np.mean([r['convergence_round_std'] for r in run_results if r['convergence_round_std'] > 0]),
                'conv_trust_mean': np.mean([r['convergence_round_trust'] for r in run_results if r['convergence_round_trust'] > 0]),
            }
        
        self.results['exp1_malicious_pct'] = results_by_percentage
        self._plot_malicious_percentage(results_by_percentage)
        self._table_malicious_percentage(results_by_percentage)
        
        return results_by_percentage
    
    def experiment_2_trust_decay_sensitivity(self, num_runs: int = 2):
        """
        Experiment 2: Trust decay factor sensitivity
        Tests different penalty mechanisms for bad contributions
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: TRUST DECAY FACTOR SENSITIVITY")
        print("="*70)
        
        decay_factors = [0.7, 0.8, 0.9, 0.95]
        results_by_decay = {}
        
        for decay in decay_factors:
            print(f"\nTesting with decay factor {decay} ({num_runs} runs)...")
            run_results = []
            
            for run_id in range(num_runs):
                result = self.run_single_experiment(
                    num_clients=20,
                    num_malicious=3,
                    num_rounds=50,
                    trust_decay=decay,
                    seed=42 + run_id
                )
                run_results.append(result)
                print(f"  Run {run_id+1}: Trust FL Accuracy={result['final_trust_acc']:.4f}, "
                      f"Improvement={result['improvement']:.2f}%")
            
            results_by_decay[decay] = {
                'std_acc_mean': np.mean([r['final_std_acc'] for r in run_results]),
                'trust_acc_mean': np.mean([r['final_trust_acc'] for r in run_results]),
                'trust_acc_std': np.std([r['final_trust_acc'] for r in run_results]),
                'improvement_mean': np.mean([r['improvement'] for r in run_results]),
                'malicious_detection': np.mean([r['malicious_detection_round'] for r in run_results 
                                               if r['malicious_detection_round'] > 0]),
            }
        
        self.results['exp2_decay_sensitivity'] = results_by_decay
        self._plot_decay_sensitivity(results_by_decay)
        self._table_decay_sensitivity(results_by_decay)
        
        return results_by_decay
    
    def experiment_3_scalability(self, num_runs: int = 2):
        """
        Experiment 3: System scalability
        Tests performance with different number of clients
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: SYSTEM SCALABILITY")
        print("="*70)
        
        client_counts = [10, 20, 40, 80]
        results_by_scale = {}
        
        for num_clients in client_counts:
            print(f"\nTesting with {num_clients} clients ({num_runs} runs)...")
            run_results = []
            
            num_malicious = max(1, int(num_clients * 0.15))  # 15% malicious
            
            for run_id in range(num_runs):
                result = self.run_single_experiment(
                    num_clients=num_clients,
                    num_malicious=num_malicious,
                    num_rounds=50,
                    clients_per_round=max(5, int(num_clients * 0.5)),
                    seed=42 + run_id
                )
                run_results.append(result)
                print(f"  Run {run_id+1}: Trust FL Accuracy={result['final_trust_acc']:.4f}")
            
            results_by_scale[num_clients] = {
                'std_acc_mean': np.mean([r['final_std_acc'] for r in run_results]),
                'trust_acc_mean': np.mean([r['final_trust_acc'] for r in run_results]),
                'trust_acc_std': np.std([r['final_trust_acc'] for r in run_results]),
                'improvement_mean': np.mean([r['improvement'] for r in run_results]),
            }
        
        self.results['exp3_scalability'] = results_by_scale
        self._plot_scalability(results_by_scale)
        self._table_scalability(results_by_scale)
        
        return results_by_scale
    
    def _plot_malicious_percentage(self, results_by_percentage: Dict):
        """Plot malicious client percentage experiment."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        percentages = list(results_by_percentage.keys())
        std_accs = [results_by_percentage[p]['std_acc_mean'] for p in percentages]
        std_stds = [results_by_percentage[p]['std_acc_std'] for p in percentages]
        trust_accs = [results_by_percentage[p]['trust_acc_mean'] for p in percentages]
        trust_stds = [results_by_percentage[p]['trust_acc_std'] for p in percentages]
        improvements = [results_by_percentage[p]['improvement_mean'] for p in percentages]
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0]
        ax1.errorbar(percentages, std_accs, yerr=std_stds, label='Standard FL', 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax1.errorbar(percentages, trust_accs, yerr=trust_stds, label='Trust-Weighted FL', 
                    marker='s', linewidth=2, markersize=8, capsize=5)
        ax1.set_xlabel('Malicious Clients (%)', fontsize=12)
        ax1.set_ylabel('Final Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs. Malicious Client Percentage', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.3, 1.0])
        
        # Plot 2: Improvement
        ax2 = axes[1]
        ax2.bar(percentages, improvements, color='green', alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_xlabel('Malicious Clients (%)', fontsize=12)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_title('Performance Improvement with Trust Mechanism', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (p, imp) in enumerate(zip(percentages, improvements)):
            ax2.text(p, imp + 1, f'{imp:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp1_malicious_percentage.png', dpi=300, bbox_inches='tight')
        print(f"Saved: exp1_malicious_percentage.png")
    
    def _plot_decay_sensitivity(self, results_by_decay: Dict):
        """Plot trust decay sensitivity experiment."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        decays = sorted(list(results_by_decay.keys()))
        improvements = [results_by_decay[d]['improvement_mean'] for d in decays]
        trust_accs = [results_by_decay[d]['trust_acc_mean'] for d in decays]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(decays, improvements, 'o-', label='Improvement (%)', 
                       linewidth=2.5, markersize=10, color='green')
        line2 = ax2.plot(decays, trust_accs, 's-', label='Final Accuracy', 
                        linewidth=2.5, markersize=10, color='blue')
        
        ax.set_xlabel('Trust Decay Factor', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12, color='green')
        ax2.set_ylabel('Final Accuracy', fontsize=12, color='blue')
        ax.set_title('Trust Decay Factor Sensitivity Analysis', fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp2_decay_sensitivity.png', dpi=300, bbox_inches='tight')
        print(f"Saved: exp2_decay_sensitivity.png")
    
    def _plot_scalability(self, results_by_scale: Dict):
        """Plot system scalability experiment."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        clients = sorted(list(results_by_scale.keys()))
        std_accs = [results_by_scale[c]['std_acc_mean'] for c in clients]
        trust_accs = [results_by_scale[c]['trust_acc_mean'] for c in clients]
        trust_stds = [results_by_scale[c]['trust_acc_std'] for c in clients]
        
        ax.errorbar(clients, std_accs, label='Standard FL', marker='o', linewidth=2.5, markersize=10)
        ax.errorbar(clients, trust_accs, yerr=trust_stds, label='Trust-Weighted FL', 
                   marker='s', linewidth=2.5, markersize=10, capsize=5)
        
        ax.set_xlabel('Number of Clients', fontsize=12)
        ax.set_ylabel('Final Accuracy', fontsize=12)
        ax.set_title('System Scalability: Performance vs. Number of Clients', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3, 1.0])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exp3_scalability.png', dpi=300, bbox_inches='tight')
        print(f"Saved: exp3_scalability.png")
    
    def _table_malicious_percentage(self, results_by_percentage: Dict):
        """Generate table for malicious percentage experiment."""
        data = []
        for pct in sorted(results_by_percentage.keys()):
            r = results_by_percentage[pct]
            data.append({
                'Malicious %': f"{pct}%",
                'Std FL Acc': f"{r['std_acc_mean']:.4f}±{r['std_acc_std']:.4f}",
                'Trust FL Acc': f"{r['trust_acc_mean']:.4f}±{r['trust_acc_std']:.4f}",
                'Improvement': f"{r['improvement_mean']:.2f}±{r['improvement_std']:.2f}%",
                'Conv Rounds (Std)': f"{r['conv_std_mean']:.0f}" if r['conv_std_mean'] > 0 else "N/A",
                'Conv Rounds (Trust)': f"{r['conv_trust_mean']:.0f}" if r['conv_trust_mean'] > 0 else "N/A",
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{self.output_dir}/table_exp1_malicious_percentage.csv', index=False)
        print(f"Saved: table_exp1_malicious_percentage.csv")
        print(df.to_string(index=False))
    
    def _table_decay_sensitivity(self, results_by_decay: Dict):
        """Generate table for decay sensitivity experiment."""
        data = []
        for decay in sorted(results_by_decay.keys()):
            r = results_by_decay[decay]
            data.append({
                'Decay Factor': f"{decay:.2f}",
                'Std FL Acc': f"{r['std_acc_mean']:.4f}",
                'Trust FL Acc': f"{r['trust_acc_mean']:.4f}±{r['trust_acc_std']:.4f}",
                'Improvement': f"{r['improvement_mean']:.2f}%",
                'Malicious Detection (rounds)': f"{r['malicious_detection']:.0f}" if r['malicious_detection'] > 0 else "N/A",
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{self.output_dir}/table_exp2_decay_sensitivity.csv', index=False)
        print(f"Saved: table_exp2_decay_sensitivity.csv")
        print(df.to_string(index=False))
    
    def _table_scalability(self, results_by_scale: Dict):
        """Generate table for scalability experiment."""
        data = []
        for clients in sorted(results_by_scale.keys()):
            r = results_by_scale[clients]
            data.append({
                'Number of Clients': clients,
                'Std FL Acc': f"{r['std_acc_mean']:.4f}",
                'Trust FL Acc': f"{r['trust_acc_mean']:.4f}±{r['trust_acc_std']:.4f}",
                'Improvement': f"{r['improvement_mean']:.2f}%",
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{self.output_dir}/table_exp3_scalability.csv', index=False)
        print(f"Saved: table_exp3_scalability.csv")
        print(df.to_string(index=False))
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        report = []
        report.append("="*80)
        report.append("TRUST-WEIGHTED FEDERATED LEARNING - EXPERIMENTAL RESULTS SUMMARY")
        report.append(f"Generated: {self.timestamp}")
        report.append("="*80)
        
        for exp_name, exp_results in self.results.items():
            report.append(f"\n{exp_name.upper().replace('_', ' ')}")
            report.append("-"*80)
            report.append(json.dumps(exp_results, indent=2, default=str))
        
        report_text = "\n".join(report)
        
        with open(f'{self.output_dir}/experiment_summary.txt', 'w') as f:
            f.write(report_text)
        
        print(f"\nSaved: experiment_summary.txt")
        print(report_text)


if __name__ == "__main__":
    runner = ExperimentRunner(output_dir='./experiment_results')
    
    print("\n" + "="*70)
    print("TRUST-WEIGHTED FL - COMPREHENSIVE EXPERIMENTAL FRAMEWORK")
    print("="*70)
    print(f"Output directory: {runner.output_dir}")
    
    # Run experiments
    print("\n[Running Experiment 1: Malicious Client Percentage Impact...]")
    runner.experiment_1_malicious_percentage(num_runs=3)
    
    print("\n[Running Experiment 2: Trust Decay Sensitivity...]")
    runner.experiment_2_trust_decay_sensitivity(num_runs=2)
    
    print("\n[Running Experiment 3: System Scalability...]")
    runner.experiment_3_scalability(num_runs=2)
    
    # Generate final report
    runner.generate_summary_report()
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {runner.output_dir}")
    print("="*70)
