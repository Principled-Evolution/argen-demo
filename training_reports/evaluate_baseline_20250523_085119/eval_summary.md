# Model Evaluation Summary

Evaluation completed on: 2025-05-23 09:07:16

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.2-1B-Instruct | 0.0700 | 7 | 0.8137 | 0.8130 | 0.6156 | 0.5290 | 0.4599 | 0.6250 | 0.9040 | 0.5840 | 0.0000 | 0.3400 | 34 | 0.1800 | 18 | 0 | {'S0': 40, 'S1': 2, 'S2': 47, 'S3': 11} | 11 | 0.1100 |
| /mnt/checkpoints/grpo_L4_1 | 0.0400 | 4 | 0.9003 | 0.9720 | 0.8483 | 0.7690 | 0.8867 | 0.7450 | 0.9450 | 0.9190 | 0.0000 | 0.0700 | 7 | 0.0200 | 2 | 0 | {'S0': 87, 'S1': 1, 'S2': 10, 'S3': 2} | 2 | 0.0200 |
| /mnt/checkpoints/grpo_L4_1/checkpoint-3000 | 0.0400 | 4 | 0.9048 | 0.9750 | 0.8470 | 0.7760 | 0.8772 | 0.7490 | 0.9505 | 0.9170 | 0.0000 | 0.1000 | 10 | 0.0300 | 3 | 0 | {'S0': 86, 'S1': 0, 'S2': 12, 'S3': 2} | 2 | 0.0200 |
| meta-llama/Llama-3.2-3B-Instruct | 0.0400 | 4 | 0.8673 | 0.9020 | 0.7235 | 0.7445 | 0.5687 | 0.7860 | 0.9700 | 0.6770 | 0.0000 | 0.2800 | 28 | 0.0200 | 2 | 0 | {'S0': 52, 'S1': 2, 'S2': 38, 'S3': 8} | 8 | 0.0800 |
| medalpaca/medalpaca-7b | 0.0800 | 8 | 0.8493 | 0.9620 | 0.7229 | 0.6700 | 0.6716 | 0.6650 | 0.9010 | 0.7190 | 0.0000 | 0.2500 | 25 | 0.1300 | 13 | 0 | {'S0': 64, 'S1': 0, 'S2': 25, 'S3': 11} | 11 | 0.1100 |