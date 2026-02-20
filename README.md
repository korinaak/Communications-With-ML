# Trust-Weighted Federated Learning for Trustworthy Mobile IoT Systems

A federated learning framework with trust mechanisms for improving robustness against malicious clients and noisy data in IoT environments.

## Overview

This project implements **Trust-Weighted Federated Learning (TWFL)**, addressing the challenge that not all devices in IoT networks are reliable. Some contain noisy data, exhibit instability, and some may even be malicious. The framework introduces a **trust score mechanism** for each client that influences both client selection and weight aggregation, improving system robustness.

## Key Features

- **Trust-Based Client Selection**: Dynamically selects clients based on historical performance
- **Trust-Weighted Aggregation**: Weights client updates according to their trust scores
- **Malicious Attack Resilience**: Tested against Byzantine attacks and gradient poisoning
- **Data Quality Handling**: Manages noisy, unstable, and good data clients
- **Comprehensive Experiments**: Three automated experiments demonstrating robustness, sensitivity, and scalability

## Files

- `trust_fl_simulation.py` - Core FL implementation with Client and TrustWeightedFL classes
- `trust_fl_comprehensive_experiments.py` - Experiment runner framework with automated analysis
- `simulation_results.json` - Experimental results data
- `experiment_results/` - Generated plots and analysis tables

## Requirements

- Python 3.8+
- numpy
- matplotlib
- scikit-learn

## Running Experiments

```bash
python trust_fl_comprehensive_experiments.py
```

This will run three experiments:
1. **Malicious Percentage Impact**: Evaluates robustness with 0-20% malicious clients
2. **Decay Sensitivity Analysis**: Tests trust decay parameter sensitivity
3. **Scalability Evaluation**: Assesses performance with varying network sizes

## Experimental Results

All experiments generate:
- Publication-quality plots (300 DPI PNG)
- Statistical analysis tables (CSV format)
- Summary statistics (TXT format)

Results are saved in `experiment_results/`