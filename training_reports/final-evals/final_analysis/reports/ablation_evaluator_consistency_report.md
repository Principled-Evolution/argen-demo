# Ablation Evaluator Consistency Analysis Report

## Executive Summary

**Analysis Date**: 2025-06-18 11:06:25
- **Evaluators Compared**: Claude-3.5-Sonnet vs Gemini-2.0-Flash
- **Ablation Types**: Reward-Only and Policy-Only
- **Analysis Level**: Individual scenario correlations and systematic bias

## Summary Comparison

### Combined Score Agreement
| Ablation Type | Claude Score | Gemini Score | Difference | Agreement |
|---------------|--------------|--------------|------------|-----------|
| Reward-Only | 0.7712 | 0.8550 | -0.0838 | Low |
| Policy-Only | 0.7365 | 0.7829 | -0.0464 | Medium |

## Detailed Consistency Analysis

### Reward-Only Ablation

#### Scenario-Level Correlations
| Metric | Pearson r | Spearman r | Mean Abs Diff | Agreement Strength |
|--------|-----------|------------|---------------|-------------------|
| Combined | 0.601 | 0.578 | 0.1100 | Moderate |
| Ahimsa | 0.390 | 0.129 | 0.1511 | Weak |
| Dharma | 0.593 | 0.511 | 0.0707 | Moderate |
| Helpfulness | 0.696 | 0.715 | 0.1955 | Moderate |

#### Systematic Bias Analysis
| Metric | Mean Difference | Significant Bias | Bias Direction |
|--------|-----------------|------------------|----------------|
| Combined | -0.0838 | Yes | Gemini Higher |
| Ahimsa | -0.1287 | Yes | Gemini Higher |
| Dharma | +0.0281 | No | Claude Higher |
| Helpfulness | -0.1880 | Yes | Gemini Higher |

#### Ranking Agreement
| Metric | Rank Correlation | Agreement Strength |
|--------|------------------|--------------------|
| Combined | 0.578 | Moderate |
| Ahimsa | 0.129 | Weak |
| Dharma | 0.511 | Moderate |
| Helpfulness | 0.715 | Strong |
### Policy-Only Ablation

#### Scenario-Level Correlations
| Metric | Pearson r | Spearman r | Mean Abs Diff | Agreement Strength |
|--------|-----------|------------|---------------|-------------------|
| Combined | 0.499 | 0.482 | 0.1258 | Weak |
| Ahimsa | 0.385 | 0.295 | 0.1290 | Weak |
| Dharma | 0.498 | 0.534 | 0.1858 | Weak |
| Helpfulness | 0.513 | 0.446 | 0.1905 | Moderate |

#### Systematic Bias Analysis
| Metric | Mean Difference | Significant Bias | Bias Direction |
|--------|-----------------|------------------|----------------|
| Combined | -0.0464 | Yes | Gemini Higher |
| Ahimsa | -0.0762 | Yes | Gemini Higher |
| Dharma | +0.0672 | Yes | Claude Higher |
| Helpfulness | -0.1680 | Yes | Gemini Higher |

#### Ranking Agreement
| Metric | Rank Correlation | Agreement Strength |
|--------|------------------|--------------------|
| Combined | 0.482 | Weak |
| Ahimsa | 0.295 | Weak |
| Dharma | 0.534 | Moderate |
| Helpfulness | 0.446 | Weak |

## Key Findings

### Overall Consistency
- **Average Correlation**: 0.522 across all metrics
- **Systematic Bias Rate**: 87.5% of metrics show significant bias
- **Overall Assessment**: Moderate consistency

### Implications for Research
1. **Cross-Evaluator Validation**: Moderate agreement supports robustness of findings
2. **Bias Considerations**: 7 out of 8 metrics show systematic evaluator differences
3. **Recommendation**: Use both evaluators for validation

## Methodology

### Analysis Approach
1. **Scenario-Level Analysis**: Individual scenario score correlations
2. **Systematic Bias Testing**: Statistical tests for consistent evaluator differences  
3. **Ranking Agreement**: Spearman correlation of scenario rankings
4. **Statistical Significance**: p < 0.05 threshold for bias detection

### Data Sources
- **Claude Evaluations**: 100 scenarios per ablation
- **Gemini Evaluations**: 100 scenarios per ablation
- **Evaluation Protocol**: Same 100-scenario benchmark for both evaluators

---
*Report Generated*: 2025-06-18 11:06:25  
*Analysis Script*: `ablation_evaluator_consistency.py`  
*Data Source*: `consolidated_ablation_evals.json`
