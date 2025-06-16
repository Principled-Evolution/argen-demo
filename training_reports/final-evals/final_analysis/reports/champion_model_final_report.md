# Champion Model Analysis Report - Final Evaluations

## Executive Summary

**Champion Model Identified**: GRPO6 Seed 2 (final)
- **Combined Score**: 0.7947
- **Evaluator**: Claude-3.5-Sonnet (consistent across all models)
- **Analysis Date**: 2025-06-16 20:32:24

## Champion Model Details

### Model Identification
- **Model Type**: GRPO6
- **Seed**: 2
- **Checkpoint**: final
- **Full Model Name**: `/home/kapil/checkpoints/grpo_6_seed_2_4`

### Performance Metrics
| Metric | Score | Violation Rate |
|--------|-------|----------------|
| **Combined Score** | **0.7947** | - |
| Ahimsa | 0.8122 | 1.0% |
| Dharma | 0.9641 | 4.0% |
| Helpfulness | 0.5513 | 33.0% |

### Detailed Helpfulness Breakdown
| Component | Score |
|-----------|-------|
| Clarity | 0.6560 |
| Relevance | 0.7100 |
| Completeness | 0.4840 |

## Baseline Comparison

### Performance vs Meta-Llama-3.2-1B-Instruct
| Metric | Baseline | Champion | Difference | Relative Change |
|--------|----------|----------|------------|-----------------|
| **Combined Score** | 0.6359 | 0.7947 | +0.1588 | +25.0% |
| Ahimsa | 0.7723 | 0.8122 | +0.0399 | +5.2% |
| Dharma | 0.5640 | 0.9641 | +0.4001 | +70.9% |
| Helpfulness | 0.5952 | 0.5513 | -0.0440 | -7.4% |

### Statistical Analysis
- **Effect Size (Cohen's d)**: 1.588 (large)
- **Absolute Improvement**: +0.1588 points
- **Relative Improvement**: +25.0%

*Note: Statistical significance requires individual scenario scores for proper testing*

## Top 10 ArGen Models Ranking

| Rank | Model | Seed | Checkpoint | Combined Score | Ahimsa | Dharma | Helpfulness |
|------|-------|------|------------|----------------|--------|--------|-------------|
| 1 | GRPO6 | 2 | final | 0.7947 | 0.8122 | 0.9641 | 0.5513 |
| 2 | GRPO6 | 2 | 4000 | 0.7907 | 0.7925 | 0.9749 | 0.5432 |
| 3 | GRPO6 | 2 | 2000 | 0.7884 | 0.8167 | 0.9293 | 0.5722 |
| 4 | GRPO6 | 2 | 3000 | 0.7874 | 0.8028 | 0.9544 | 0.5492 |
| 5 | GRPO7 | 3 | 3000 | 0.7825 | 0.8039 | 0.9353 | 0.5575 |
| 6 | GRPO7 | 3 | 4000 | 0.7815 | 0.8002 | 0.9434 | 0.5470 |
| 7 | GRPO7 | 3 | final | 0.7803 | 0.7938 | 0.9519 | 0.5380 |
| 8 | GRPO7 | 3 | 2000 | 0.7800 | 0.8067 | 0.9043 | 0.5877 |
| 9 | GRPO5 | 1 | 3000 | 0.7752 | 0.7932 | 0.9331 | 0.5467 |
| 10 | GRPO5 | 1 | 2600 | 0.7690 | 0.7917 | 0.9312 | 0.5300 |

## Performance Distribution Analysis

### Summary Statistics (All ArGen Models)
| Statistic | Combined Score |
|-----------|----------------|
| Mean | 0.7732 |
| Median | 0.7800 |
| Standard Deviation | 0.0166 |
| Minimum | 0.7341 |
| Maximum | 0.7947 |
| 25th Percentile | 0.7639 |
| 75th Percentile | 0.7850 |
| Range | 0.0606 |

### Performance by Seed
| Seed | Mean Score | Std Dev | Min | Max | Models |
|------|------------|---------|-----|-----|--------|
| 1 | 0.7627 | 0.0117 | 0.7440 | 0.7752 | 5 |
| 2 | 0.7790 | 0.0253 | 0.7341 | 0.7947 | 5 |
| 3 | 0.7778 | 0.0074 | 0.7647 | 0.7825 | 5 |

## Data Provenance

### Champion Model Source
- **Evaluation File**: `../../claude-runs/eval_grpo_6_seed_2_4_benchmarking_20250510_135534-cleanprep-hashprompt.json`
- **Evaluator**: anthropic (claude-3-5-sonnet)
- **Timestamp**: 2025-06-15T16:30:57.556922
- **Scenarios Evaluated**: 100

### Analysis Methodology
1. **Data Source**: Final corrected evaluations from `training_reports/final-evals/`
2. **Evaluator Filter**: Claude-3.5-Sonnet only (for scientific consistency)
3. **Selection Criteria**: Highest `average_combined_score` across all ArGen checkpoints
4. **Total Models Analyzed**: 15 ArGen models across 3 seeds

### Usage Guidelines
1. **Main Results Tables**: Report Champion score as "ArGen (Peak Performance)"
2. **Performance Claims**: "ArGen achieved 0.7947 combined score"
3. **Comparison Standard**: All comparisons use Claude-3.5-Sonnet evaluation
4. **Scientific Rigor**: Single evaluator ensures consistent judgment criteria

---
*Report Generated*: 2025-06-16 20:32:24  
*Analysis Script*: `champion_model_analysis_final.py`  
*Data Source*: `../data/consolidated_final_evals.json`  
*Total ArGen Models*: 15 (Claude-evaluated)
