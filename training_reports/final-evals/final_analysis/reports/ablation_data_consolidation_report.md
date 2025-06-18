# Ablation Data Consolidation Report

## Executive Summary

**Consolidation Date**: 2025-06-18 11:46:38
- **Total Ablation Evaluations**: 4
- **Evaluators**: gemini, claude
- **Ablation Types**: reward_only, policy_only
- **Baseline Reference**: ✓ Loaded

## Data Summary

### Ablation Evaluations
| Evaluator | Ablation Type | Combined Score | Scenarios | Status |
|-----------|---------------|----------------|-----------|--------|
| Claude | Policy-Only | 0.7365 | 100 | ✓ Complete |
| Claude | Reward-Only | 0.7712 | 100 | ✓ Complete |
| Gemini | Policy-Only | 0.7829 | 100 | ✓ Complete |
| Gemini | Reward-Only | 0.8550 | 100 | ✓ Complete |

### Baseline Reference (Median Seed)
- **Claude**: GRPO7 Seed 3 (3000) - 0.7825
- **Gemini**: GRPO7 Seed 3 (3000) - 0.8599
- **Purpose**: Representative baseline for ablation comparisons

## Validation Results

### Data Completeness
- **Expected Evaluations**: 4 (2 evaluators × 2 ablation types)
- **Found Evaluations**: 4
- **Completeness**: ✓ Complete

### Data Quality Checks

#### Claude Reward-Only
- Individual Results: ✓
- Valid Combined Score: ✓
- All Metrics Present: ✓

#### Claude Policy-Only
- Individual Results: ✓
- Valid Combined Score: ✓
- All Metrics Present: ✓

#### Gemini Reward-Only
- Individual Results: ✓
- Valid Combined Score: ✓
- All Metrics Present: ✓

#### Gemini Policy-Only
- Individual Results: ✓
- Valid Combined Score: ✓
- All Metrics Present: ✓

## Next Steps

1. **Statistical Analysis**: Run `ablation_statistical_analysis.py`
2. **Comparative Analysis**: Compare ablations against median seed baseline
3. **Cross-Evaluator Validation**: Analyze Claude vs Gemini consistency
4. **Report Generation**: Generate comprehensive ablation reports

## Data Files Generated

- `consolidated_ablation_evals.json`: Unified ablation dataset
- `ablation_data_consolidation_report.md`: This report

---
*Report Generated*: 2025-06-18 11:46:38  
*Script*: `ablation_data_consolidator.py`  
*Data Sources*: Claude and Gemini ablation evaluations
