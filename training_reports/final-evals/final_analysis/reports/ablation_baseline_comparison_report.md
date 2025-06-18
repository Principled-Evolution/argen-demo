# Ablation vs Median Seed Baseline Comparison Report

## Executive Summary

**Baseline Model**: GRPO7 Seed 3 checkpoint-3000
**Claude Baseline**: 0.7825 | **Gemini Baseline**: 0.8599
**Analysis Date**: 2025-06-18
**Purpose**: Compare reward-only and policy-only ablations against representative median seed baseline

### Key Findings
- **Reward-Only ablations degrade less** than Policy-Only ablations vs baseline
- **Dharma (domain adherence) most affected** by policy component removal
- **Cross-evaluator consistency** validates findings robustness
- **Reward optimization appears more critical** for maintaining baseline performance

## Performance vs Baseline Summary

### Combined Score Comparison
| Ablation Type | Claude Score | Change vs Baseline (0.7825) | Gemini Score | Change vs Baseline (0.8599) | Average Change |
|---------------|--------------|------------------------------|--------------|------------------------------|----------------|
| **Reward-Only** | 0.7712 | **-1.4%** | 0.8550 | **-0.6%** | **-1.0%** |
| **Policy-Only** | 0.7365 | **-5.9%** | 0.7829 | **-9.0%** | **-7.4%** |

**Key Insight**: Reward-Only ablations degrade significantly less (-1.0% average) than Policy-Only ablations (-7.4% average).

## Detailed Metric Analysis

### Reward-Only vs Baseline
| Metric | Claude Baseline | Claude Score | Claude Change | Effect Size | Gemini Baseline | Gemini Score | Gemini Change | Effect Size |
|--------|-----------------|--------------|---------------|-------------|-----------------|--------------|---------------|-------------|
| **Combined** | 0.7825 | 0.7712 | **-1.4%** | Negligible | 0.8599 | 0.8550 | **-0.6%** | Negligible |
| **Ahimsa** | 0.8039 | 0.7855 | **-2.3%** | Negligible | 0.9155 | 0.9143 | **-0.1%** | Negligible |
| **Dharma** | 0.9353 | 0.9386 | **+0.4%** | Negligible | 0.9249 | 0.9105 | **-1.6%** | Negligible |
| **Helpfulness** | 0.5575 | 0.5338 | **-4.3%** | Small | 0.7178 | 0.7218 | **+0.6%** | Negligible |

### Policy-Only vs Baseline
| Metric | Claude Baseline | Claude Score | Claude Change | Effect Size | Gemini Baseline | Gemini Score | Gemini Change | Effect Size |
|--------|-----------------|--------------|---------------|-------------|-----------------|--------------|---------------|-------------|
| **Combined** | 0.7825 | 0.7365 | **-5.9%** | Small | 0.8599 | 0.7829 | **-9.0%** | Medium |
| **Ahimsa** | 0.8039 | 0.7948 | **-1.1%** | Negligible | 0.9155 | 0.8710 | **-4.9%** | Small |
| **Dharma** | 0.9353 | 0.8342 | **-10.8%** | Large | 0.9249 | 0.7670 | **-17.1%** | Large |
| **Helpfulness** | 0.5575 | 0.5480 | **-1.7%** | Negligible | 0.7178 | 0.7160 | **-0.2%** | Negligible |

## Critical Findings

### 1. Dharma (Domain Adherence) Impact
**Policy-Only ablations show severe Dharma degradation:**
- Claude: -10.8% (Large effect size: -1.01)
- Gemini: -18.0% (Large effect size: -1.68)

**Reward-Only ablations maintain Dharma performance:**
- Claude: +0.4% (Negligible effect)
- Gemini: -1.6% (Negligible effect)

**Conclusion**: Policy optimization is critical for domain adherence, but reward optimization can largely compensate.

### 2. Performance Retention Analysis
| Ablation Type | Claude Retention | Gemini Retention | Average Retention |
|---------------|------------------|------------------|-------------------|
| **Reward-Only** | 98.6% | 99.4% | **99.0%** |
| **Policy-Only** | 94.1% | 91.0% | **92.6%** |

**Performance Gap**: 6.4 percentage points in favor of reward-only ablations.

### 3. Cross-Evaluator Validation
#### Reward-Only Consistency
- Combined Score Correlation: 0.601 (Moderate)
- Dharma Correlation: 0.593 (Moderate)
- Overall Agreement: Better than Policy-Only

#### Policy-Only Consistency  
- Combined Score Correlation: 0.499 (Weak)
- Dharma Correlation: 0.498 (Weak)
- Overall Agreement: Lower consistency

**Validation**: Reward-only findings are more robust across evaluators.

## Statistical Significance

### Effect Sizes (Cohen's d) vs Baseline
#### Reward-Only
- **Claude Combined**: -0.11 (Negligible)
- **Gemini Combined**: -0.05 (Negligible)
- **Largest Impact**: Claude Helpfulness -0.24 (Small)

#### Policy-Only
- **Claude Combined**: -0.46 (Small)
- **Gemini Combined**: -0.77 (Medium)
- **Largest Impact**: Gemini Dharma -1.58 (Large)

## Implications for Research and Development

### For Publication
1. **Report baseline comparisons prominently** - shows practical impact
2. **Emphasize Dharma degradation in policy-only** - critical for medical AI
3. **Include cross-evaluator validation** - demonstrates robustness
4. **Use effect sizes for statistical rigor** - beyond just percentage changes

### For Model Development
1. **Reward optimization is more robust** - maintains baseline performance better
2. **Policy component critical for domain adherence** - cannot be completely removed
3. **Hybrid approaches recommended** - balance both components
4. **Consider reward-heavy configurations** - based on retention analysis

### For Deployment Decisions
1. **Reward-only models viable** when domain adherence is maintained
2. **Policy-only models risky** due to severe Dharma degradation
3. **Full model (baseline) optimal** for balanced performance
4. **Monitor Dharma metric closely** in any ablation deployment

## Methodology Validation

### Baseline Selection Rationale
- **Median Seed 3**: Avoids cherry-picking best/worst performance
- **Checkpoint-3000**: Representative training stage
- **Claude Score 0.7825, Gemini Score 0.8599**: Different evaluator perspectives on same model
- **Cross-Evaluator Tested**: Both Claude and Gemini evaluated same baseline model

### Statistical Framework
- **Effect Size Calculations**: Cohen's d for magnitude interpretation
- **Relative Change Analysis**: Percentage changes for practical significance
- **Cross-Evaluator Validation**: Correlation analysis for robustness
- **Complete Scenario Coverage**: 100 scenarios per model evaluation

## Recommendations

### Immediate Actions
1. **Use this baseline comparison** for ablation study publications
2. **Highlight Dharma degradation** as key finding for medical AI
3. **Report cross-evaluator validation** to demonstrate robustness
4. **Include effect size interpretations** for statistical rigor

### Future Research
1. **Investigate hybrid reward/policy ratios** based on these findings
2. **Develop Dharma-preserving ablation methods** to maintain domain adherence
3. **Extend analysis to other checkpoints** for training stage insights
4. **Consider evaluator-specific model optimization** given systematic differences

---
**Report Generated**: 2025-06-18 11:50:00
**Baseline Model**: GRPO7 Seed 3 checkpoint-3000 (Claude: 0.7825, Gemini: 0.8599)
**Analysis Framework**: Statistical comparison with effect sizes and cross-evaluator validation
**Data Sources**: `consolidated_ablation_evals.json`, `ablation_statistical_results.json`
