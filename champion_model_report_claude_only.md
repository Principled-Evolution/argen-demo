# Champion Model Analysis Report (Claude Evaluations Only)

## Champion Model Identification

**Champion Model**: /home/kapil/checkpoints/grpo_6_seed_2_4
- **Seed**: 2
- **Checkpoint**: final
- **Model Type**: GRPO6
- **Average Combined Score**: 0.7947
- **Evaluator**: anthropic (claude-3-5-sonnet)

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

## Baseline Comparison (Claude Evaluations)

**Baseline Model**: meta-llama/Llama-3.2-1B-Instruct
- **Baseline Combined Score**: 0.8106
- **Champion Combined Score**: 0.7947
- **Absolute Improvement**: +-0.0159
- **Relative Improvement**: +-2.0%

## Top 5 ArGen Models Ranking (Claude Only)

| Rank | Model | Seed | Checkpoint | Combined Score |
|------|-------|------|------------|----------------|
| 1 | GRPO6 | 2 | final | 0.7947 |
| 2 | GRPO6 | 2 | 4000 | 0.7907 |
| 3 | GRPO6 | 2 | 2000 | 0.7884 |
| 4 | GRPO6 | 2 | 3000 | 0.7874 |
| 5 | GRPO7 | 3 | 3000 | 0.7825 |

## Analysis Summary

The Champion Model represents the peak performance achieved by the ArGen framework across all training runs and checkpoints, **based exclusively on Claude-3.5-Sonnet evaluations**.

### Key Findings:
1. **Peak Performance**: 0.7947 combined score
2. **Model**: GRPO6 Seed 2 (final)
3. **Improvement**: -2.0% over baseline (Claude-judged)

### Usage Guidelines:
1. **Main Results Tables**: Report these scores as "ArGen (Champion)" 
2. **Performance Claims**: "ArGen achieved 0.7947 combined score (Claude-judged)"
3. **Scientific Rigor**: All comparisons based on consistent Claude evaluation

**Note**: This analysis uses ONLY Claude evaluations for scientific rigor. For comprehensive comparisons including other evaluators, see the multi-evaluator analysis.

---
*Generated on: 2025-06-16 18:49:14*
*Total ArGen models analyzed: 15 (Claude-only)*
*Evaluator: Claude-3.5-Sonnet (consistent across all models)*
