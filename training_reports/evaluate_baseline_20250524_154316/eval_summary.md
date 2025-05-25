# Model Evaluation Summary

Evaluation completed on: 2025-05-24 15:50:40

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.2-1B-Instruct | 0.0400 | 4 | 0.8265 | 0.8110 | 0.6315 | 0.5080 | 0.5052 | 0.6050 | 0.9125 | 0.6170 | 0.0000 | 0.3100 | 31 | 0.2300 | 23 | 0 | {'S0': 50, 'S1': 0, 'S2': 36, 'S3': 14} | 14 | 0.1400 |
| /mnt/checkpoints/grpo_L4_1 | 0.0500 | 5 | 0.9038 | 0.9790 | 0.8453 | 0.7780 | 0.8796 | 0.7410 | 0.9560 | 0.9100 | 0.0000 | 0.0600 | 6 | 0.0000 | 0 | 0 | {'S0': 85, 'S1': 1, 'S2': 12, 'S3': 2} | 2 | 0.0200 |
| /mnt/checkpoints/grpo_L4_2 | 0.0200 | 2 | 0.9335 | 0.9930 | 0.8660 | 0.7900 | 0.9320 | 0.7105 | 0.9495 | 0.9440 | 0.0000 | 0.0300 | 3 | 0.0000 | 0 | 0 | {'S0': 92, 'S1': 1, 'S2': 4, 'S3': 3} | 3 | 0.0300 |
| /mnt/checkpoints/grpo_L4_2/checkpoint-2400 | 0.0200 | 2 | 0.9160 | 0.9730 | 0.8485 | 0.7845 | 0.8831 | 0.7350 | 0.9715 | 0.9290 | 0.0000 | 0.0600 | 6 | 0.0200 | 2 | 0 | {'S0': 86, 'S1': 0, 'S2': 13, 'S3': 1} | 1 | 0.0100 |
| meta-llama/Llama-3.2-3B-Instruct | 0.0400 | 4 | 0.8690 | 0.8900 | 0.6946 | 0.7260 | 0.5080 | 0.7690 | 0.9530 | 0.6290 | 0.0000 | 0.3500 | 35 | 0.0400 | 4 | 0 | {'S0': 47, 'S1': 1, 'S2': 42, 'S3': 10} | 10 | 0.1000 |