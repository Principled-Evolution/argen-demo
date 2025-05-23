# Model Evaluation Summary

Evaluation completed on: 2025-05-23 09:42:35

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.2-1B-Instruct | 0.0400 | 4 | 0.8293 | 0.8200 | 0.6179 | 0.5080 | 0.4652 | 0.6100 | 0.9050 | 0.5780 | 0.0000 | 0.3200 | 32 | 0.1800 | 18 | 0 | {'S0': 43, 'S1': 0, 'S2': 44, 'S3': 13} | 13 | 0.1300 |
| /mnt/checkpoints/grpo_L4_1 | 0.0400 | 4 | 0.9080 | 0.9750 | 0.8580 | 0.7690 | 0.8969 | 0.7560 | 0.9590 | 0.9250 | 0.0000 | 0.0800 | 8 | 0.0200 | 2 | 0 | {'S0': 89, 'S1': 0, 'S2': 7, 'S3': 4} | 4 | 0.0400 |
| meta-llama/Llama-3.2-3B-Instruct | 0.0300 | 3 | 0.8658 | 0.8930 | 0.7093 | 0.7195 | 0.5502 | 0.7650 | 0.9670 | 0.6590 | 0.0000 | 0.3000 | 30 | 0.0200 | 2 | 0 | {'S0': 50, 'S1': 1, 'S2': 42, 'S3': 7} | 7 | 0.0700 |
| medalpaca/medalpaca-7b | 0.2100 | 21 | 0.7472 | 0.9600 | 0.6442 | 0.5755 | 0.6218 | 0.5710 | 0.8550 | 0.6740 | 0.0000 | 0.3100 | 31 | 0.2800 | 28 | 0 | {'S0': 59, 'S1': 0, 'S2': 21, 'S3': 20} | 20 | 0.2000 |