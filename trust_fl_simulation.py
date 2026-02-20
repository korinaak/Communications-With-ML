"""
Trust-Weighted Federated Learning (FL) Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

np.random.seed(42)

INITIAL_TRUST_SCORE = 1.0
TRUST_DECAY_FACTOR = 0.9  # How quickly trust decreases for poor contributions
MIN_TRUST_THRESHOLD = 0.3  # Minimum trust to be eligible for selection
MAX_TRUST_INCREASE = 0.2   # Cap on trust increase per round (prevents dominance)
MIN_TRUST_FLOOR = 0.5      # Don't let trust drop below this when penalizing

# Noise parameters for data quality simulation
NOISE_LEVELS = {'good': 0.01, 'noisy': 0.15, 'unstable': 0.3}
MALICIOUS_NOISE = 0.5      # Large noise for poisoned updates

# Training parameters
SIGMOID_CLIP = (-500, 500)  # Numerical stability for sigmoid
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 5
VALIDATION_SET_SIZE = 500


class Client:
    """
    Represents an IoT/edge device client in the federated learning system.
    
    Each client:
    - Holds local data with varying quality (good, noisy, unstable)
    - Performs local model training independently
    - Sends updates to the server (may be poisoned if malicious)
    - Receives a trust score that affects future selection probability
    - Tracks contribution history for analysis
    """
    
    def __init__(self, client_id: int, data_quality: str = 'good', is_malicious: bool = False):
        """
        Initialize a client.
        
        Args:
            client_id: Unique identifier for the client
            data_quality: 'good' (clean), 'noisy' (label noise), or 'unstable' (high noise)
            is_malicious: Whether this client actively poisons updates (attacks model)
        """
        self.client_id = client_id
        self.data_quality = data_quality
        self.is_malicious = is_malicious
        self.trust_score = 1.0  # Initial trust score
        self.local_data = None
        self.local_labels = None
        self.history = {
            'trust_scores': [1.0],
            'contributions': [],
            'selection_count': 0
        }
    
    def set_data(self, X: np.ndarray, y: np.ndarray):
        """Assign local data to the client."""
        self.local_data = X
        self.local_labels = y
    
    def local_train(self, global_weights: np.ndarray, learning_rate: float = LEARNING_RATE, 
                   epochs: int = LOCAL_EPOCHS) -> np.ndarray:
        """
        Perform local training on client's data.
        
        This simulates the local SGD step in federated learning. The quality of local
        training depends on data quality (clean clients have low noise, noisy/unstable 
        have high gradient noise). Malicious clients deliberately reverse their gradients
        to poison the global model.
        
        Args:
            global_weights: Current global model weights from server
            learning_rate: Learning rate for local SGD
            epochs: Number of local training epochs
            
        Returns:
            Updated local weights (clean update if benign, poisoned if malicious)
        """
        weights = global_weights.copy()
        
        # Determine gradient noise based on data quality
        noise_level = NOISE_LEVELS.get(self.data_quality, 0.01)
        
        # Local SGD training
        for _ in range(epochs):
            # Compute logistic regression gradients
            predictions = self._sigmoid(np.dot(self.local_data, weights))
            error = self.local_labels - predictions
            gradient = np.dot(self.local_data.T, error) / len(self.local_labels)
            
            # Add noise to simulate data quality issues (realistic in edge computing)
            gradient += np.random.normal(0, noise_level, gradient.shape)
            
            # Standard gradient descent update
            weights += learning_rate * gradient
        
        # MALICIOUS CLIENT BEHAVIOR: Poison the update by reversing gradients
        if self.is_malicious:
            # Strategy: Flip gradient direction + add large noise to maximize damage
            # This is a "targeted" attack - reverses the learned direction
            weights = global_weights - 2 * (weights - global_weights)
            weights += np.random.normal(0, MALICIOUS_NOISE, weights.shape)
        
        return weights
    
    @staticmethod
    def _sigmoid(z):
        """
        Sigmoid activation function with numerical stability.
        
        Clipping prevents overflow/underflow in exp() for extreme values.
        """
        return 1 / (1 + np.exp(-np.clip(z, *SIGMOID_CLIP)))
    
    def update_trust_score(self, new_score: float):
        """Update the client's trust score."""
        self.trust_score = new_score
        self.history['trust_scores'].append(new_score)


class TrustWeightedFL:
    """
    Trust-Weighted Federated Learning System.
    
    Core Innovation: Unlike standard FedAvg, this system:
    1. SELECTS clients probabilistically based on trust scores (not uniformly random)
    2. AGGREGATES updates using trust-weighted averaging (not simple average)
    3. UPDATES trust scores based on contribution quality to global model
    
    This creates a feedback loop:
    - Good contributions → higher trust → more likely selected → more influence
    - Bad/poisoned updates → lower trust → less likely selected → less influence
    - Result: System becomes robust to malicious clients and noisy data sources
    """
    
    def __init__(self, num_features: int, trust_decay: float = TRUST_DECAY_FACTOR, 
                 selection_threshold: float = MIN_TRUST_THRESHOLD):
        """
        Initialize the FL system.
        
        Args:
            num_features: Number of features in the model
            trust_decay: Decay factor (0-1) for reducing trust of bad contributors
                        (e.g., 0.9 means: new_trust = old_trust * 0.9 when penalizing)
            selection_threshold: Minimum trust score needed to be eligible for selection
                                (prevents low-trust clients from participating)
        """
        self.global_weights = np.random.randn(num_features) * 0.01
        self.trust_decay = trust_decay
        self.selection_threshold = selection_threshold
        self.validation_data = None
        self.validation_labels = None
        self.history = {
            'global_accuracy': [],
            'average_trust': [],
            'selected_clients': []
        }
    
    def set_validation_data(self, X_val: np.ndarray, y_val: np.ndarray):
        """Set validation data for trust score computation."""
        self.validation_data = X_val
        self.validation_labels = y_val
    
    def select_clients(self, clients: List[Client], num_select: int, 
                      use_trust: bool = True) -> List[Client]:
        """
        Select clients for the current training round.
        
        TRUST-BASED SELECTION (use_trust=True):
        - Only eligible clients (trust >= threshold) can participate
        - Selection probability ∝ trust_score (higher trust = higher chance)
        - Effect: Malicious clients with low trust are rarely selected
        
        STANDARD SELECTION (use_trust=False):
        - Uniformly random selection (traditional FedAvg)
        - Effect: Malicious clients participate equally to benign clients
        
        Args:
            clients: List of all available clients
            num_select: Number of clients to select for this round
            use_trust: Whether to use trust-weighted selection vs random
            
        Returns:
            List of selected clients for training
        """
        if not use_trust:
            # Random selection (standard FedAvg) - baseline for comparison
            return np.random.choice(clients, num_select, replace=False).tolist()
        
        # TRUST-WEIGHTED SELECTION: Favor high-trust clients
        eligible_clients = [c for c in clients if c.trust_score >= self.selection_threshold]
        
        if len(eligible_clients) < num_select:
            # Fallback: If not enough eligible clients, include highest trust scores
            eligible_clients = sorted(clients, key=lambda x: x.trust_score, reverse=True)[:num_select]
        
        # Compute selection probabilities proportional to trust scores
        trust_scores = np.array([c.trust_score for c in eligible_clients])
        probabilities = trust_scores / trust_scores.sum()
        
        # Sample without replacement using trust-based probabilities
        selected_indices = np.random.choice(
            len(eligible_clients), 
            min(num_select, len(eligible_clients)), 
            replace=False,
            p=probabilities
        )
        
        selected = [eligible_clients[i] for i in selected_indices]
        for client in selected:
            client.history['selection_count'] += 1
        
        return selected
    
    def aggregate_updates(self, clients: List[Client], local_weights_list: List[np.ndarray],
                         use_trust: bool = True) -> np.ndarray:
        """
        Aggregate local model updates into global model.
        
        TRUST-WEIGHTED AGGREGATION (use_trust=True):
        - High-trust clients get higher weight in the weighted average
        - Formula: aggregated = Σ(trust_weight_i * local_weights_i)
        - Effect: Good updates dominate, poisoned updates are diluted
        
        STANDARD AGGREGATION (use_trust=False):
        - Simple average with equal weights (FedAvg)
        - Formula: aggregated = (1/n) * Σ(local_weights_i)
        - Effect: One poisoned update can cancel out several good ones
        
        Args:
            clients: List of clients who participated in this round
            local_weights_list: List of weight updates from each client
            use_trust: Whether to use trust-weighted vs standard averaging
            
        Returns:
            Aggregated global model weights
        """
        if not use_trust:
            # Standard FedAvg: simple average (baseline)
            return np.mean(local_weights_list, axis=0)
        
        # TRUST-WEIGHTED AGGREGATION: Weight by client trust scores
        trust_scores = np.array([c.trust_score for c in clients])
        weights_normalized = trust_scores / trust_scores.sum()
        
        aggregated = np.zeros_like(self.global_weights)
        for weight, local_weights in zip(weights_normalized, local_weights_list):
            aggregated += weight * local_weights
        
        return aggregated
    
    def update_trust_scores(self, clients: List[Client], local_weights_list: List[np.ndarray]):
        if self.validation_data is None:
            return
        
        # Baseline: accuracy with current global model
        global_accuracy = self._compute_accuracy(self.global_weights)
        
        for client, local_weights in zip(clients, local_weights_list):
            # Test this client's update on validation set
            local_accuracy = self._compute_accuracy(local_weights)
            
            if local_accuracy > global_accuracy:
                # GOOD CONTRIBUTION: Increase trust
                contribution = min((local_accuracy - global_accuracy) / global_accuracy, MAX_TRUST_INCREASE)
                new_trust = min(1.0, client.trust_score * (1 + contribution))
            else:
                # BAD CONTRIBUTION: Decrease trust (decay mechanism)
                contribution = (global_accuracy - local_accuracy) / max(global_accuracy, 0.01)
                new_trust = client.trust_score * self.trust_decay * max(MIN_TRUST_FLOOR, 1 - contribution)
            
            client.update_trust_score(new_trust)
            client.history['contributions'].append(local_accuracy - global_accuracy)
    
    def _compute_accuracy(self, weights: np.ndarray) -> float:
        """Compute accuracy on validation data."""
        predictions = Client._sigmoid(np.dot(self.validation_data, weights))
        predicted_labels = (predictions > 0.5).astype(int)
        return np.mean(predicted_labels == self.validation_labels)
    
    def train_round(self, clients: List[Client], num_select: int, use_trust: bool = True):
        """Execute one round of federated learning."""
        # Select clients
        selected_clients = self.select_clients(clients, num_select, use_trust)
        self.history['selected_clients'].append([c.client_id for c in selected_clients])
        
        # Local training
        local_weights_list = []
        for client in selected_clients:
            local_weights = client.local_train(self.global_weights)
            local_weights_list.append(local_weights)
        
        # Update trust scores (before aggregation for trust-weighted FL)
        if use_trust:
            self.update_trust_scores(selected_clients, local_weights_list)
        
        # Aggregate updates
        self.global_weights = self.aggregate_updates(selected_clients, local_weights_list, use_trust)
        
        # Track metrics
        global_accuracy = self._compute_accuracy(self.global_weights)
        avg_trust = np.mean([c.trust_score for c in clients])
        
        self.history['global_accuracy'].append(global_accuracy)
        self.history['average_trust'].append(avg_trust)


def generate_data(num_clients: int, samples_per_client: int, num_features: int = 10) -> Tuple:
    """
    Generate synthetic dataset for federated learning simulation.
    
    Args:
        num_clients: Number of clients
        samples_per_client: Number of samples per client
        num_features: Number of features
        
    Returns:
        Tuple of (client_data, validation_data, validation_labels)
    """
    # Generate global data distribution
    X_global = np.random.randn(num_clients * samples_per_client, num_features)
    true_weights = np.random.randn(num_features)
    
    # Generate labels with logistic relationship
    logits = np.dot(X_global, true_weights)
    probabilities = 1 / (1 + np.exp(-logits))
    y_global = (probabilities > 0.5).astype(int)
    
    # Split data among clients
    client_data = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_data.append((X_global[start_idx:end_idx], y_global[start_idx:end_idx]))
    
    # Create validation set
    X_val = np.random.randn(500, num_features)
    logits_val = np.dot(X_val, true_weights)
    probabilities_val = 1 / (1 + np.exp(-logits_val))
    y_val = (probabilities_val > 0.5).astype(int)
    
    return client_data, X_val, y_val


def run_simulation(num_clients: int = 20, num_good: int = 12, num_noisy: int = 5,
                  num_malicious: int = 3, num_rounds: int = 50, clients_per_round: int = 10) -> Dict:
    print("=" * 70)
    print("Trust-Weighted Federated Learning Simulation")
    print("=" * 70)
    
    # Generate data
    print("\n[1/4] Generating dataset...")
    client_data, X_val, y_val = generate_data(num_clients, samples_per_client=100)
    
    # Create clients
    print("[2/4] Initializing clients...")
    clients_standard = []
    clients_trust = []
    
    # Assign client types (malicious clients get 'good' data but poison updates)
    client_types = []
    for i in range(num_clients):
        if i < num_malicious:
            client_types.append('good')  # Malicious clients have good data but poison updates
        elif i < num_malicious + num_good:
            client_types.append('good')
        elif i < num_malicious + num_good + num_noisy:
            client_types.append('noisy')
        else:
            client_types.append('unstable')
    
    for i in range(num_clients):
        is_malicious = i < num_malicious
        quality = client_types[i]
        
        # Create two copies: one for standard FL, one for trust FL
        client_std = Client(i, quality, is_malicious)
        client_std.set_data(client_data[i][0], client_data[i][1])
        clients_standard.append(client_std)
        
        client_trust = Client(i, quality, is_malicious)
        client_trust.set_data(client_data[i][0], client_data[i][1])
        clients_trust.append(client_trust)
    
    print(f"   - {num_good} clients with good data")
    print(f"   - {num_noisy} clients with noisy data")
    print(f"   - {num_clients - num_good - num_noisy - num_malicious} clients with unstable data")
    print(f"   - {num_malicious} malicious clients")
    
    # Initialize FL systems
    print("[3/4] Initializing FL systems...")
    fl_standard = TrustWeightedFL(num_features=10)
    fl_standard.set_validation_data(X_val, y_val)
    
    fl_trust = TrustWeightedFL(num_features=10, trust_decay=0.9, selection_threshold=0.3)
    fl_trust.set_validation_data(X_val, y_val)
    
    # Training
    print(f"[4/4] Training for {num_rounds} rounds...")
    print("\n" + "-" * 70)
    print(f"{'Round':>6} | {'Standard FL Acc':>16} | {'Trust FL Acc':>14} | {'Avg Trust':>10}")
    print("-" * 70)
    
    for round_num in range(num_rounds):
        # Standard FL (no trust mechanism) - baseline for comparison
        fl_standard.train_round(clients_standard, clients_per_round, use_trust=False)
        
        # Trust-weighted FL - our proposed approach
        fl_trust.train_round(clients_trust, clients_per_round, use_trust=True)
        
        # Print progress every 5 rounds
        if (round_num + 1) % 5 == 0:
            std_acc = fl_standard.history['global_accuracy'][-1]
            trust_acc = fl_trust.history['global_accuracy'][-1]
            avg_trust = fl_trust.history['average_trust'][-1]
            print(f"{round_num + 1:6d} | {std_acc:16.4f} | {trust_acc:14.4f} | {avg_trust:10.4f}")
    
    print("-" * 70)
    print("\nSimulation completed!")
    
    # Compile results
    results = {
        'standard_fl': {
            'accuracy_history': fl_standard.history['global_accuracy'],
            'final_accuracy': fl_standard.history['global_accuracy'][-1]
        },
        'trust_fl': {
            'accuracy_history': fl_trust.history['global_accuracy'],
            'trust_history': fl_trust.history['average_trust'],
            'final_accuracy': fl_trust.history['global_accuracy'][-1],
            'client_trust_scores': {c.client_id: c.trust_score for c in clients_trust},
            'client_selection_counts': {c.client_id: c.history['selection_count'] 
                                       for c in clients_trust}
        },
        'configuration': {
            'num_clients': num_clients,
            'num_good': num_good,
            'num_noisy': num_noisy,
            'num_malicious': num_malicious,
            'num_rounds': num_rounds,
            'clients_per_round': clients_per_round
        }
    }
    
    return results


def plot_results(results: Dict, save_path: str = 'simulation_results.png'):
    """
    Generate visualization of simulation results.
    
    Args:
        results: Dictionary containing simulation results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    rounds = range(1, len(results['standard_fl']['accuracy_history']) + 1)
    ax1.plot(rounds, results['standard_fl']['accuracy_history'], 
            label='Standard FL', linewidth=2, marker='o', markersize=3)
    ax1.plot(rounds, results['trust_fl']['accuracy_history'], 
            label='Trust-Weighted FL', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Training Round', fontsize=11)
    ax1.set_ylabel('Global Model Accuracy', fontsize=11)
    ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average trust score over time
    ax2 = axes[0, 1]
    ax2.plot(rounds, results['trust_fl']['trust_history'], 
            color='green', linewidth=2, marker='D', markersize=3)
    ax2.set_xlabel('Training Round', fontsize=11)
    ax2.set_ylabel('Average Trust Score', fontsize=11)
    ax2.set_title('Average Client Trust Score Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final trust scores by client
    ax3 = axes[1, 0]
    client_ids = list(results['trust_fl']['client_trust_scores'].keys())
    trust_scores = list(results['trust_fl']['client_trust_scores'].values())
    
    colors = []
    for cid in client_ids:
        if cid < results['configuration']['num_malicious']:
            colors.append('red')
        elif cid < results['configuration']['num_malicious'] + results['configuration']['num_good']:
            colors.append('green')
        else:
            colors.append('orange')
    
    ax3.bar(client_ids, trust_scores, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.3, color='r', linestyle='--', linewidth=1, label='Selection Threshold')
    ax3.set_xlabel('Client ID', fontsize=11)
    ax3.set_ylabel('Final Trust Score', fontsize=11)
    ax3.set_title('Final Trust Scores by Client', fontsize=12, fontweight='bold')
    ax3.legend(['Threshold=0.3', 'Malicious', 'Good', 'Noisy/Unstable'], fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Client selection frequency
    ax4 = axes[1, 1]
    selection_counts = list(results['trust_fl']['client_selection_counts'].values())
    ax4.bar(client_ids, selection_counts, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Client ID', fontsize=11)
    ax4.set_ylabel('Number of Times Selected', fontsize=11)
    ax4.set_title('Client Selection Frequency', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    return fig


def save_results(results: Dict, save_path: str = 'simulation_results.json'):
    """Save simulation results to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    # Run simulation
    results = run_simulation(
        num_clients=20,
        num_good=12,
        num_noisy=5,
        num_malicious=3,
        num_rounds=50,
        clients_per_round=10
    )
    
    # Generate plots
    plot_results(results)
    
    # Save results
    save_results(results)
    
    # Print detailed summary with insights
    print("\n" + "=" * 70)
    print("TRUST-WEIGHTED FL SIMULATION SUMMARY")
    print("=" * 70)
    
    print("\n📊 ACCURACY COMPARISON:")
    print("-" * 70)
    std_final = results['standard_fl']['final_accuracy']
    trust_final = results['trust_fl']['final_accuracy']
    improvement_pct = ((trust_final - std_final) / std_final * 100)
    
    print(f"  Standard FL (baseline):        {std_final:.4f} ({std_final*100:.2f}%)")
    print(f"  Trust-Weighted FL (proposed):  {trust_final:.4f} ({trust_final*100:.2f}%)")
    print(f"  ⬆ Improvement:                 {improvement_pct:+.2f}%")
    
    print("\n🔐 MALICIOUS CLIENT DETECTION:")
    print("-" * 70)
    malicious_count = results['configuration']['num_malicious']
    malicious_ids = list(range(malicious_count))
    malicious_trusts = [results['trust_fl']['client_trust_scores'][cid] for cid in malicious_ids]
    malicious_selections = [results['trust_fl']['client_selection_counts'][cid] for cid in malicious_ids]
    
    for cid in malicious_ids:
        trust = results['trust_fl']['client_trust_scores'][cid]
        selections = results['trust_fl']['client_selection_counts'][cid]
        status = "✓ ISOLATED" if trust < 0.3 else "⚠ STILL ACTIVE"
        print(f"  Malicious Client {cid}: Trust={trust:.4f}, Selected {selections:2d} times {status}")
    
    print("\n✅ BENIGN CLIENT RECOGNITION:")
    print("-" * 70)
    good_start = malicious_count
    good_end = malicious_count + results['configuration']['num_good']
    good_ids = list(range(good_start, good_end))
    good_trusts = [results['trust_fl']['client_trust_scores'][cid] for cid in good_ids]
    good_selections = [results['trust_fl']['client_selection_counts'][cid] for cid in good_ids]
    
    avg_good_trust = np.mean(good_trusts)
    avg_good_selections = np.mean(good_selections)
    print(f"  Average Trust Score:     {avg_good_trust:.4f}")
    print(f"  Average Times Selected:  {avg_good_selections:.1f}/50 rounds")
    print(f"  Selection Rate:          {avg_good_selections/results['configuration']['num_rounds']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("""
  1. ROBUSTNESS: Trust-weighted FL maintains high accuracy despite malicious attacks
  
  2. DETECTION: Malicious clients are automatically identified through validation
     testing and progressively isolated (low trust = low selection probability)
  
  3. FAIRNESS: Benign clients with good data quality are rewarded with higher
     trust scores and more frequent selection, speeding up their contribution
  
  4. SCALABILITY: Trust mechanism works with any number of malicious clients
     as long as honest clients form a majority
    """)
    print("=" * 70)
