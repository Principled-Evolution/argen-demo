# Median Seed Analysis Report - Final Evaluations

## Executive Summary

**Median Seed Identified**: Seed 3
- **Peak Performance**: 0.7825
- **Peak Model**: GRPO7 (3000)
- **Purpose**: Representative seed for ablation studies (reward-only, policy-only)
- **Analysis Date**: 2025-06-16 20:32:24

## Methodology

The Median Seed is selected to ensure fair and representative ablation studies:

1. **Calculate Peak Performance**: For each seed, find the highest `average_combined_score` across all checkpoints
2. **Rank Seeds**: Order the three seeds by their peak performance scores
3. **Select Median**: Choose the seed with the middle (median) peak performance
4. **Scientific Rationale**: Avoids cherry-picking the best or worst performing seed

## Seed Peak Performance Analysis

### Peak Scores Summary
| Rank | Seed | Peak Score | Peak Model | Checkpoint | Model Type |
|------|------|------------|------------|------------|------------|
| 1 | 2 | 0.7947 | GRPO6 | final | grpo_6_seed_2_4 |
| 2 | 3 🎯 | 0.7825 | GRPO7 | 3000 | checkpoint-3000 |
| 3 | 1 | 0.7752 | GRPO5 | 3000 | checkpoint-3000 |

### Median Seed Selection
- **Selected Seed**: 3 (middle-ranked performance)
- **Peak Score**: 0.7825
- **Rationale**: Represents typical ArGen performance, avoiding extremes

## Detailed Seed Analysis

### Seed 1 

#### Peak Performance Model
- **Model**: GRPO5 (3000)
- **Combined Score**: 0.7752
- **Ahimsa**: 0.7932
- **Dharma**: 0.9331
- **Helpfulness**: 0.5467

#### Performance Distribution (5 checkpoints)
| Metric | Value |
|--------|-------|
| Mean | 0.7627 |
| Median | 0.7631 |
| Std Dev | 0.0105 |
| Min | 0.7440 |
| Max | 0.7752 |
| Range | 0.0312 |

### Seed 2 

#### Peak Performance Model
- **Model**: GRPO6 (final)
- **Combined Score**: 0.7947
- **Ahimsa**: 0.8122
- **Dharma**: 0.9641
- **Helpfulness**: 0.5513

#### Performance Distribution (5 checkpoints)
| Metric | Value |
|--------|-------|
| Mean | 0.7790 |
| Median | 0.7884 |
| Std Dev | 0.0226 |
| Min | 0.7341 |
| Max | 0.7947 |
| Range | 0.0606 |

### Seed 3 🎯 **MEDIAN SEED**

#### Peak Performance Model
- **Model**: GRPO7 (3000)
- **Combined Score**: 0.7825
- **Ahimsa**: 0.8039
- **Dharma**: 0.9353
- **Helpfulness**: 0.5575

#### Performance Distribution (5 checkpoints)
| Metric | Value |
|--------|-------|
| Mean | 0.7778 |
| Median | 0.7803 |
| Std Dev | 0.0066 |
| Min | 0.7647 |
| Max | 0.7825 |
| Range | 0.0178 |

## Ablation Study Recommendations

### Using Median Seed 3 for Ablations

The Median Seed should be used to train the following ablation models:

1. **Reward-Only Model**: Train using only reward optimization (no policy constraints)
2. **Policy-Only Model**: Train using only policy optimization (no reward shaping)

### Training Configuration
- **Base Seed**: 3
- **Peak Checkpoint Reference**: 3000 (0.7825 score)
- **Training Steps**: Match the peak checkpoint's training duration
- **Evaluation**: Use same 100-scenario benchmark with Claude-3.5-Sonnet

### Scientific Justification
- **Avoids Cherry-Picking**: Not using the best-performing seed prevents bias
- **Representative Performance**: Median seed represents typical ArGen behavior
- **Fair Comparison**: Ablations will be compared against a representative baseline
- **Reproducible**: Clear methodology for seed selection

## Performance Context

### Seed Performance Spread
- **Highest Peak**: 0.7947
- **Median Peak**: 0.7825
- **Lowest Peak**: 0.7752
- **Peak Range**: 0.0195

### Cross-Seed Consistency
- **Mean Peak Score**: 0.7841
- **Peak Score Std Dev**: 0.0080
- **Coefficient of Variation**: 1.0%

## Data Provenance

### Median Seed Model Source
- **Evaluation File**: `../../claude-runs/eval_grpo_7_seed_3_3_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json`
- **Evaluator**: anthropic (claude-3-5-sonnet)
- **Timestamp**: 2025-06-15T18:23:37.241769
- **Scenarios**: 100

### Analysis Methodology
1. **Data Source**: Final corrected evaluations from `training_reports/final-evals/`
2. **Evaluator Filter**: Claude-3.5-Sonnet only (consistent judgment)
3. **Scope**: All ArGen checkpoints across 3 seeds
4. **Selection**: Mathematical median of peak performances

### Next Steps
1. **Train Ablation Models**: Use Seed 3 configuration
2. **Evaluate Ablations**: Same evaluation protocol as main models
3. **Compare Results**: Ablation performance vs. Median Seed peak performance

---
*Report Generated*: 2025-06-16 20:32:24  
*Analysis Script*: `median_seed_analysis_final.py`  
*Data Source*: `../data/consolidated_final_evals.json`  
*Total Models Analyzed*: 15 across 3 seeds
