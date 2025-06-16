# Median Seed Analysis Report

## Median Seed Identification

**Median Seed**: Seed 3
- **Peak Combined Score**: 0.7825
- **Best Model**: /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-3000
- **Best Checkpoint**: 3000
- **Total Models Evaluated**: 5

## Seed Performance Comparison

| Seed | Peak Score | Best Checkpoint | Model Type | Rank |
|------|------------|-----------------|------------|------|
| 2 | 0.7947 | final | GRPO6 | 1 |
| 3 | 0.7825 | 3000 | GRPO7 | 2 ← **MEDIAN** |
| 1 | 0.7752 | 3000 | GRPO5 | 3 |

## Detailed Seed Analysis

### Seed 1 Performance
- **Peak Score**: 0.7752
- **Best Model**: /home/kapil/checkpoints/grpo_5_seed_1/checkpoint-3000
- **Best Checkpoint**: 3000
- **Ahimsa Score**: 0.7932
- **Dharma Score**: 0.9331
- **Helpfulness Score**: 0.5467

### Seed 2 Performance
- **Peak Score**: 0.7947
- **Best Model**: /home/kapil/checkpoints/grpo_6_seed_2_4
- **Best Checkpoint**: final
- **Ahimsa Score**: 0.8122
- **Dharma Score**: 0.9641
- **Helpfulness Score**: 0.5513

### Seed 3 Performance
- **Peak Score**: 0.7825
- **Best Model**: /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-3000
- **Best Checkpoint**: 3000
- **Ahimsa Score**: 0.8039
- **Dharma Score**: 0.9353
- **Helpfulness Score**: 0.5575

## Median Seed Selection Rationale

The Median Seed (Seed 3) was selected using the following methodology:

1. **Calculate Peak Performance**: For each seed, identify the checkpoint with the highest Claude-judged `average_combined_score`
2. **Rank Seeds**: Order the three seeds by their peak performance scores
3. **Select Median**: Choose the seed with the middle (median) peak performance

This approach ensures:
- **Scientific Rigor**: Avoids cherry-picking the best-performing seed for ablations
- **Representative Results**: Uses a seed that represents typical (not exceptional) performance
- **Fair Comparisons**: Ablation studies will be based on a representative training run

## Recommended Usage

Use Seed 3 for:
1. **Reward-Only Ablation**: Train from scratch with only reward model, no policy optimization
2. **Policy-Only Ablation**: Train from scratch with only policy optimization, no reward model
3. **Component Analysis**: Any ablation study requiring fair baseline comparison

**Important**: Do NOT use the Champion Model's seed for ablations, as this would constitute cherry-picking and compromise the scientific validity of the comparison.

---
*Generated on: 2025-06-16 12:30:15*
*Seeds analyzed: 3*
