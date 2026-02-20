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

## Citation

```bibtex
@misc{trust_weighted_fl_2024,
  title={Trust-Weighted Federated Learning for Trustworthy Mobile IoT Systems},
  author={},
  year={2024}
}
```

## References

1. Cao, X., Fang, M., Liu, J., & Gong, N. Z. (2020). FLTRUST: Byzantine-robust federated learning via trust bootstrapping. arXiv. https://doi.org/10.48550/arxiv.2012.13995

2. Li, T., Sahu, A. K., Talwalkar, A., Smith, V. (2020). Federated Learning: challenges, methods, and future directions. IEEE Signal Processing Magazine, 37(3), 50–60. https://doi.org/10.1109/msp.2020.2975749

## License

This project is provided for educational purposes.
   - Page budget breakdown

---

## Quick Stats You Can Cite Immediately

| Metric | Value | Source |
|--------|-------|--------|
| Best case improvement | +25.78% | Exp 1: 0% malicious |
| Worst case improvement | +67.54% | Exp 1: 20% malicious |
| Average improvement | ~35% | Exp 1: All cases |
| Optimal decay factor | 0.80 | Exp 2 |
| Malicious detection time | 3-4 rounds | Exp 2 |
| Network scalability | 10-80 clients | Exp 3 |
| Max accuracy (80 clients) | 97.6% | Exp 3 |

---

## How to Write Your Paper (5-Step Process)

### Step 1: Copy Figures
```
Copy these 3 PNG files into your paper:
- Section 5.2: exp1_malicious_percentage.png
- Section 5.3: exp2_decay_sensitivity.png
- Section 5.4: exp3_scalability.png
```

### Step 2: Add Tables
```
Paste these tables into Results section:
- After Exp 1 text: table_exp1_malicious_percentage.csv
- After Exp 2 text: table_exp2_decay_sensitivity.csv
- After Exp 3 text: table_exp3_scalability.csv
```

### Step 3: Use QUICK_REFERENCE_STATS.md
```
Copy quotes and statistics from this file:
- Introduction: Cite problem severity
- Results: Use exact accuracy numbers
- Discussion: Quote improvements and insights
```

### Step 4: Follow PAPER_WRITING_GUIDE.md
```
Use this as your outline:
- Section 1: Introduction (1.5 pages)
- Section 2: Related Work (1.5 pages)
- Section 3: Problem Formulation (1 page)
- Section 4: Methodology (2 pages)
- Section 5: Experiments (4 pages)
- Section 6: Discussion (2 pages)
- Section 7: Conclusion (0.5 pages)
- Section 8: Future Work (0.5 pages)
Total: ~15 pages
```

### Step 5: Customize & Submit
```
1. Write your own intro/related work
2. Explain your methodology in your words
3. Insert figures and tables where marked
4. Cite your experimental results
5. Discuss implications for IoT
```

---

## Directory Structure

```
/Users/kor/Desktop/PhD/Communications With ML/
├── trust_fl_simulation.py                    ← Main simulator
├── trust_fl_comprehensive_experiments.py     ← Experiment framework
├── QUICK_REFERENCE_STATS.md                  ← Ready-to-cite statistics
├── PAPER_SUPPORT_GUIDE.md                    ← Integration guide
├── PAPER_WRITING_GUIDE.md                    ← 15-page outline
├── experiment_results/                       ← All results
│   ├── exp1_malicious_percentage.png        ← Figure 1
│   ├── exp2_decay_sensitivity.png           ← Figure 2
│   ├── exp3_scalability.png                 ← Figure 3
│   ├── table_exp1_malicious_percentage.csv  ← Table 1
│   ├── table_exp2_decay_sensitivity.csv     ← Table 2
│   ├── table_exp3_scalability.csv           ← Table 3
│   └── experiment_summary.txt                ← Raw data
└── simulation_results.png                    ← Original demo
    simulation_results.json
```

---

## Paper Outline (15 pages)

```
Page Range | Section | Content
-----------|---------|----------
1          | Title + Abstract | Your title, author, abstract
2-3.5      | 1. Introduction | Problem, motivation, contributions
4-5.5      | 2. Related Work | FLTRUST, Byzantine FL, IoT learning
6          | 3. Problem Formulation | Setup, notation, attack model
8-9        | 4. Methodology | Trust mechanism, selection, aggregation
10-13      | 5. Experiments | Your 3 experiments + figures + tables
14-15      | 6. Discussion | Why it works, trade-offs, implications
15.5       | 7. Conclusion | Summary + future work
```

---

## Key Quotes Ready to Use

### For Introduction
> "With 15% of clients exhibiting malicious behavior, standard FedAvg suffers significant accuracy degradation, achieving only 71.60% compared to our approach's 93.07%."

### For Results
> "Trust-Weighted FL maintains 87.47% accuracy even under 20% malicious client attacks, compared to only 53.20% for standard FedAvg, representing a 67.54% relative improvement."

### For Scalability Discussion
> "The system scales effectively to large IoT networks. On an 80-client network with 15% Byzantine adversaries, Trust-Weighted FL achieves 97.6% accuracy."

### For Conclusion
> "Our empirical evaluation demonstrates that dynamic trust scoring enables federated learning systems to achieve 25-67% accuracy improvements while maintaining fairness toward benign clients."

---

## Quality Checklist

### Code Quality ✅
- [x] Well-commented implementation
- [x] Follows Python conventions
- [x] Reproducible (fixed random seed)
- [x] No external data dependencies

### Experimental Quality ✅
- [x] Multiple runs (2-3 per experiment)
- [x] Error bars and standard deviation
- [x] Ablation studies (parameter sensitivity)
- [x] Scalability evaluation
- [x] Different attack intensities

### Results Presentation ✅
- [x] High-resolution plots (300 DPI)
- [x] CSV tables with statistics
- [x] Formatted error bars (mean ± std)
- [x] Publication-ready figure captions
- [x] Numbered tables and references

### Documentation ✅
- [x] Complete writing guide
- [x] Quick reference statistics
- [x] Integration guide for results
- [x] Example text for each section
- [x] Page budget breakdown

---

## Next Actions

1. **Open PAPER_WRITING_GUIDE.md**
   - Start with Section 1 (Introduction)
   - Follow the outline and example text
   - Adapt to your own voice

2. **Copy Figures & Tables**
   - Use the ones in experiment_results/
   - Reference them in your text

3. **Insert Statistics**
   - Copy from QUICK_REFERENCE_STATS.md
   - Use exact numbers from your experiments

4. **Write Your Own**
   - Introduction (set up the problem)
   - Related work (compare to FLTRUST)
   - Methodology (explain your approach)
   - Discussion (what does it mean?)

5. **Polish & Submit**
   - Spell check
   - Reference formatting
   - Figure quality check
   - Target 15 pages

---

## Support Files

Each guide serves a specific purpose:

| File | Purpose | When to Use |
|------|---------|-------------|
| `QUICK_REFERENCE_STATS.md` | Copy statistics | When writing results section |
| `PAPER_SUPPORT_GUIDE.md` | How to integrate | When placing figures/tables |
| `PAPER_WRITING_GUIDE.md` | Full outline | When drafting each section |

---

## Estimated Completion Time

- Writing paper using this framework: **4-6 hours**
  - Introduction: 45 minutes
  - Related work: 1 hour
  - Problem formulation: 30 minutes
  - Methodology: 1 hour (can reuse code comments)
  - Experiments: 45 minutes (figures + tables already done!)
  - Discussion: 1 hour
  - Conclusion: 30 minutes
  - Polish: 1 hour

---

## Success Criteria

Your paper will be ready when it has:

- [ ] 15-page count
- [ ] All 8 sections completed
- [ ] 3 figures inserted with captions
- [ ] 3 tables with results
- [ ] Experimental citations throughout
- [ ] Comparison with FLTRUST
- [ ] Discussion of IoT implications
- [ ] Clear contributions listed
- [ ] Conclusion tied to introduction
- [ ] Future work outlined

---

## Contact Points in Code

If you need to re-run experiments:
```bash
python trust_fl_comprehensive_experiments.py
```

Results will appear in `./experiment_results/` with plots and tables.

---

**🎉 You have everything needed for a publication-ready paper!**

All experimental data is generated. All supporting materials are written. 
Your job: Connect them with your own prose about the methodology and implications.

Good luck with your assignment! 📝
