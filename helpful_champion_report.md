# Helpful-Champion Model Analysis Report

## Helpful-Champion Model Identification

**Helpful-Champion Model**: /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-1000
- **Seed**: 3
- **Checkpoint**: 1000
- **Model Type**: GRPO7
- **Combined Score**: 0.7647
- **Helpfulness Score**: 0.5947

## Helpfulness Analysis

### Baseline Comparison (Helpfulness Focus)
- **Baseline Helpfulness**: 0.5952
- **Helpful-Champion Helpfulness**: 0.5947
- **Helpfulness Change**: -0.0005
- **Helpfulness Change %**: -0.1%

### Overall Performance (Helpful-Champion)
- **Combined Score**: 0.7647
- **Combined Improvement**: +0.1289
- **Combined Improvement %**: +20.3%

## Champion Model Comparison

### Overall Champion vs Helpful-Champion
| Metric | Overall Champion | Helpful-Champion | Difference |
|--------|------------------|------------------|------------|
| **Model** | GRPO6 Seed 2 | GRPO7 Seed 3 | - |
| **Combined Score** | 0.7947 | 0.7647 | -0.0300 |
| **Helpfulness Score** | 0.5513 | 0.5947 | +0.0435 |
| **Ahimsa Score** | 0.8122 | 0.8272 | +0.0151 |
| **Dharma Score** | 0.9641 | 0.8453 | -0.1188 |

## Top 5 Models by Helpfulness Preservation

| Rank | Model | Seed | Checkpoint | Helpfulness Score | Change vs Baseline | Combined Score |
|------|-------|------|------------|-------------------|-------------------|----------------|
| 1 | GRPO7 | 3 | 1000 | 0.5947 | -0.0005 | 0.7647 |
| 2 | GRPO7 | 3 | 2000 | 0.5877 | -0.0075 | 0.7800 |
| 3 | GRPO5 | 1 | 1000 | 0.5705 | -0.0247 | 0.7440 |
| 4 | GRPO5 | 1 | 2000 | 0.5665 | -0.0287 | 0.7620 |
| 5 | GRPO7 | 3 | 3000 | 0.5575 | -0.0377 | 0.7825 |

## Detailed Performance Breakdown (Helpful-Champion)

### Core Metrics
- **Ahimsa Score**: 0.8272
- **Dharma Score**: 0.8453
- **Helpfulness Score**: 0.5947
- **Clarity Score**: 0.6660
- **Relevance Score**: 0.7270
- **Completeness Score**: 0.5120

### Violation Rates
- **Ahimsa Violations**: 1.0%
- **Dharma Violations**: 10.0%
- **Helpfulness Violations**: 21.0%

## Analysis Summary

The Helpful-Champion Model represents the best balance between overall improvement and helpfulness preservation. Key insights:

1. **Helpfulness Preservation**: -0.0005 change vs baseline (-0.1%)
2. **Overall Improvement**: +20.3% combined score improvement
3. **Use Case**: Ideal for scenarios where maintaining helpfulness is critical
4. **Trade-offs**: May sacrifice some safety gains for better user assistance

### Recommended Usage

Use the Helpful-Champion Model when:
- **User Experience Priority**: Helpfulness is the primary concern
- **Balanced Performance**: Need good overall improvement with minimal helpfulness loss
- **Comparative Studies**: Demonstrating that ArGen can maintain helpfulness while improving safety

**Note**: For maximum safety gains, use the Overall Champion Model. For fair ablations, use the Median Seed.

---
*Generated on: 2025-06-16 13:44:53*
*Models analyzed: 11 improved ArGen models*
