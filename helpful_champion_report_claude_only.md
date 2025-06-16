# Helpful-Champion Model Analysis Report (Claude Evaluations Only)

## Helpful-Champion Model Identification

**Helpful-Champion Model**: /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-1000
- **Seed**: 3
- **Checkpoint**: 1000
- **Model Type**: GRPO7
- **Combined Score**: 0.7647
- **Helpfulness Score**: 0.5947
- **Evaluator**: anthropic (claude-3-5-sonnet)

## Helpfulness Analysis (Claude Evaluations)

### Baseline Comparison (Helpfulness Focus)
- **Baseline Helpfulness**: 0.5250
- **Helpful-Champion Helpfulness**: 0.5947
- **Helpfulness Change**: +0.0697
- **Helpfulness Change %**: +13.3%

### Overall Performance (Helpful-Champion)
- **Combined Score**: 0.7647
- **Combined Improvement**: +0.0072
- **Combined Improvement %**: +1.0%

## Champion Model Comparison (Claude Only)

### Overall Champion vs Helpful-Champion
| Metric | Overall Champion | Helpful-Champion | Difference |
|--------|------------------|------------------|------------|
| **Model** | GRPO6 Seed 2 | GRPO7 Seed 3 | - |
| **Combined Score** | 0.7947 | 0.7647 | -0.0300 |
| **Helpfulness Score** | 0.5513 | 0.5947 | +0.0435 |
| **Ahimsa Score** | 0.8122 | 0.8272 | +0.0151 |
| **Dharma Score** | 0.9641 | 0.8453 | -0.1188 |

## Top 5 Models by Helpfulness Preservation (Claude Only)

| Rank | Model | Seed | Checkpoint | Helpfulness Score | Change vs Baseline | Combined Score |
|------|-------|------|------------|-------------------|-------------------|----------------|
| 1 | GRPO7 | 3 | 1000 | 0.5947 | +0.0697 | 0.7647 |
| 2 | GRPO7 | 3 | 2000 | 0.5877 | +0.0627 | 0.7800 |
| 3 | GRPO6 | 2 | 2000 | 0.5722 | +0.0472 | 0.7884 |
| 4 | GRPO5 | 1 | 2000 | 0.5665 | +0.0415 | 0.7620 |
| 5 | GRPO7 | 3 | 3000 | 0.5575 | +0.0325 | 0.7825 |

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

The Helpful-Champion Model represents the best balance between overall improvement and helpfulness preservation, **based exclusively on Claude-3.5-Sonnet evaluations**.

### Key Insights:
1. **Helpfulness Preservation**: +0.0697 change vs baseline (+13.3%)
2. **Overall Improvement**: +1.0% combined score improvement
3. **Scientific Rigor**: All comparisons based on consistent Claude evaluation
4. **Use Case**: Ideal for scenarios where maintaining helpfulness is critical

### Recommended Usage

Use the Helpful-Champion Model when:
- **User Experience Priority**: Helpfulness is the primary concern
- **Balanced Performance**: Need good overall improvement with minimal helpfulness loss
- **Comparative Studies**: Demonstrating that ArGen can maintain helpfulness while improving safety

**Important Notes**:
- This analysis uses ONLY Claude evaluations for scientific rigor
- For maximum safety gains, use the Overall Champion Model
- For fair ablations, use the Median Seed
- All models evaluated by Claude-3.5-Sonnet for consistency

---
*Generated on: 2025-06-16 18:33:58*
*Models analyzed: 13 improved ArGen models (Claude-only)*
*Evaluator: Claude-3.5-Sonnet (consistent across all models)*
