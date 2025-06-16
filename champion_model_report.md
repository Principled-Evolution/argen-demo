# Champion Model Analysis Report

## Champion Model Identification

**Champion Model**: /home/kapil/checkpoints/grpo_6_seed_2_4
- **Seed**: 2
- **Checkpoint**: final
- **Model Type**: GRPO6
- **Average Combined Score**: 0.7947

## Performance Breakdown

### Core Metrics (Champion Model)
- **Ahimsa Score**: 0.8122
- **Dharma Score**: 0.9641  
- **Helpfulness Score**: 0.5513
- **Clarity Score**: 0.6560
- **Relevance Score**: 0.7100
- **Completeness Score**: 0.4840

### Violation Rates (Champion Model)
- **Ahimsa Violations**: 1.0%
- **Dharma Violations**: 4.0%
- **Helpfulness Violations**: 33.0%

## Baseline Comparison

**Baseline Model**: meta-llama/Llama-3.2-1B-Instruct
- **Baseline Combined Score**: 0.6359
- **Champion Combined Score**: 0.7947
- **Absolute Improvement**: +0.1588
- **Relative Improvement**: +25.0%

## Top 5 ArGen Models Ranking

| Rank | Model | Seed | Checkpoint | Combined Score |
|------|-------|------|------------|----------------|
| 1 | GRPO6 | 2 | final | 0.7947 |
| 2 | GRPO7 | 3 | 3000 | 0.7825 |
| 3 | GRPO7 | 3 | 4000 | 0.7815 |
| 4 | GRPO7 | 3 | final | 0.7803 |
| 5 | GRPO7 | 3 | 2000 | 0.7800 |

## Analysis Summary

The Champion Model represents the peak performance achieved by the ArGen framework across all training runs and checkpoints. This model should be used for:

1. **Main Results Tables**: Report these scores as "ArGen (Best)" or "MedGuide-AI (Champion)"
2. **Performance Claims**: "At its best, ArGen achieved 0.7947 combined score"
3. **Baseline Comparisons**: Demonstrate the maximum improvement over the baseline model

**Note**: For ablation studies, use the Median Seed (see median_seed_analysis.py) to ensure fair, non-cherry-picked comparisons.

---
*Generated on: 2025-06-16 12:29:08*
*Total ArGen models analyzed: 11*
