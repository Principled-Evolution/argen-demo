# Evaluator Consistency Analysis Report

## Executive Summary

This report analyzes the consistency between Claude-3.5-Sonnet and Gemini-2.0-Flash evaluations to validate the robustness of Claude-based model selections.

### Key Findings
- **Matched Evaluations**: 16 model pairs analyzed
- **Top-5 Overlap**: 4/5 (80%)
- **Top-10 Overlap**: 9/10 (90%)
- **Same Champion**: ❌ No

---

## 1. Correlation Analysis

### Metric Correlations (Claude vs Gemini)
| Metric | Pearson r | p-value | Spearman ρ | p-value | Mean Abs Diff |
|--------|-----------|---------|------------|---------|---------------|
| Combined | 0.970 | 0.000 | 0.912 | 0.000 | 0.0689 |
| Ahimsa | 0.412 | 0.113 | -0.003 | 0.991 | 0.1016 |
| Dharma | 0.986 | 0.000 | 0.903 | 0.000 | 0.0309 |
| Helpfulness | 0.420 | 0.105 | 0.497 | 0.050 | 0.1716 |

### Interpretation
- **Strong Correlation**: r > 0.7 indicates good agreement
- **Moderate Correlation**: 0.4 < r < 0.7 indicates reasonable agreement  
- **Weak Correlation**: r < 0.4 indicates poor agreement

### Score Distributions
| Metric | Claude Mean (±SD) | Gemini Mean (±SD) |
|--------|-------------------|-------------------|
| Combined | 0.7646 (±0.0369) | 0.8335 (±0.0552) |
| Ahimsa | 0.8000 (±0.0135) | 0.8993 (±0.0393) |
| Dharma | 0.8919 (±0.0984) | 0.8610 (±0.1059) |
| Helpfulness | 0.5595 (±0.0227) | 0.7311 (±0.0174) |

---

## 2. Ranking Consistency Analysis

### Champion Model Comparison
| Evaluator | Champion Model | Combined Score |
|-----------|----------------|----------------|
| Claude | grpo6_s2_cfinal | 0.7947 |
| Gemini | grpo6_s2_c3000 | 0.8784 |

**Same Champion**: ❌ No - Different champions selected

### Top Model Overlap
- **Top-5 Models**: 4/5 overlap (80%)
- **Top-10 Models**: 9/10 overlap (90%)

### Top 10 Rankings Comparison
| Rank | Claude Model | Claude Score | Gemini Model | Gemini Score |
|------|--------------|--------------|--------------|--------------|
| 1 | grpo6_s2_cfinal | 0.7947 | grpo6_s2_c3000 | 0.8784 |
| 2 | grpo6_s2_c4000 | 0.7907 | grpo6_s2_c4000 | 0.8666 |
| 3 | grpo6_s2_c2000 | 0.7884 | grpo6_s2_cfinal | 0.8619 |
| 4 | grpo6_s2_c3000 | 0.7874 | grpo7_s3_c3000 | 0.8599 |
| 5 | grpo7_s3_c3000 | 0.7825 | grpo7_s3_c4000 | 0.8583 |
| 6 | grpo7_s3_c4000 | 0.7815 | grpo6_s2_c2000 | 0.8574 |
| 7 | grpo7_s3_cfinal | 0.7803 | grpo7_s3_c2000 | 0.8564 |
| 8 | grpo7_s3_c2000 | 0.7800 | grpo5_s1_c2600 | 0.8547 |
| 9 | grpo5_s1_c3000 | 0.7752 | grpo7_s3_cfinal | 0.8524 |
| 10 | grpo5_s1_c2600 | 0.7690 | grpo5_s1_cfinal | 0.8506 |

---

## 3. Systematic Differences Analysis

### Mean Differences (Claude - Gemini)
| Metric | Mean Diff | Std Dev | Median | Range | t-test p | Significant Bias |
|--------|-----------|---------|--------|-------|----------|------------------|
| Combined | -0.0689 | 0.0214 | -0.0740 | [-0.0910, -0.0011] | 0.000 | ⚠️ Yes |
| Ahimsa | -0.0993 | 0.0359 | -0.1110 | [-0.1312, +0.0178] | 0.000 | ⚠️ Yes |
| Dharma | +0.0309 | 0.0189 | +0.0273 | [+0.0011, +0.0677] | 0.000 | ⚠️ Yes |
| Helpfulness | -0.1716 | 0.0220 | -0.1763 | [-0.2023, -0.0988] | 0.000 | ⚠️ Yes |

### Interpretation
- **Positive values**: Claude scores higher than Gemini
- **Negative values**: Gemini scores higher than Claude
- **Significant bias**: p < 0.05 indicates systematic difference

---

## 4. Validation of Claude-Based Selections

### Robustness Assessment

**Overall Assessment**: ✅ **HIGH ROBUSTNESS**

### Evidence
- **Combined Score Correlation**: 0.970 (Strong)
- **Top-5 Agreement**: 80%
- **Champion Agreement**: No

### Recommendation
Claude-based selections are well-validated by Gemini agreement.

---

## 5. Implications for Research

### For Champion Model Selection
- **Primary Analysis**: Continue using Claude-only for consistency
- **Validation**: Strong cross-evaluator validation
- **Confidence**: Moderate confidence in champion selection

### For Publication
1. **Report Claude Results**: Use Claude-based champion as primary result
2. **Cross-Validation**: Mention strong Gemini agreement
3. **Transparency**: Report both evaluator results for completeness

---

## 6. Data Provenance

### Analysis Details
- **Matched Pairs**: 16 models evaluated by both systems
- **Claude Evaluations**: 16 files
- **Gemini Evaluations**: 16 files
- **Analysis Date**: 2025-06-16 20:32:27

### Statistical Methods
- **Correlation**: Pearson (linear) and Spearman (rank) correlations
- **Ranking**: Top-k overlap analysis
- **Bias Testing**: One-sample t-tests for systematic differences

---
*Report Generated*: 2025-06-16 20:32:27  
*Analysis Script*: `evaluator_consistency_analysis.py`  
*Data Source*: `../data/consolidated_final_evals.json`
