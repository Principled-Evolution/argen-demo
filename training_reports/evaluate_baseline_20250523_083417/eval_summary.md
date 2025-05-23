# Model Evaluation Summary

Evaluation completed on: 2025-05-23 08:39:41

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.2-1B-Instruct | 0.0900 | 9 | 0.8175 | 0.8190 | 0.6227 | 0.5130 | 0.4883 | 0.6070 | 0.9160 | 0.6190 | 0.0000 | 0.3000 | 30 | 0.1800 | 18 | 0 | {'S0': 47, 'S1': 1, 'S2': 41, 'S3': 11} | 11 | 0.1100 |
| /mnt/checkpoints/grpo_L4_1 | 0.0100 | 1 | 0.9175 | 0.9700 | 0.8381 | 0.7820 | 0.8446 | 0.7500 | 0.9650 | 0.8830 | 0.0000 | 0.1000 | 10 | 0.0200 | 2 | 0 | {'S0': 83, 'S1': 0, 'S2': 13, 'S3': 4} | 4 | 0.0400 |
| /mnt/checkpoints/grpo_L4_1/checkpoint-1000 | 0.0200 | 2 | 0.9058 | 0.9620 | 0.8269 | 0.7510 | 0.8517 | 0.7150 | 0.9550 | 0.8970 | 0.0000 | 0.0900 | 9 | 0.0600 | 6 | 0 | {'S0': 83, 'S1': 1, 'S2': 12, 'S3': 4} | 4 | 0.0400 |
| /mnt/checkpoints/grpo_L4_1/checkpoint-2000 | 0.0600 | 6 | 0.8945 | 0.9750 | 0.8476 | 0.7625 | 0.8924 | 0.7410 | 0.9630 | 0.9180 | 0.0000 | 0.0600 | 6 | 0.0100 | 1 | 0 | {'S0': 87, 'S1': 1, 'S2': 8, 'S3': 4} | 4 | 0.0400 |
| /mnt/checkpoints/grpo_L4_1/checkpoint-3000 | 0.0500 | 5 | 0.9068 | 0.9820 | 0.8478 | 0.7705 | 0.8919 | 0.7300 | 0.9525 | 0.9150 | 0.0000 | 0.0700 | 7 | 0.0000 | 0 | 0 | {'S0': 88, 'S1': 0, 'S2': 7, 'S3': 5} | 5 | 0.0500 |