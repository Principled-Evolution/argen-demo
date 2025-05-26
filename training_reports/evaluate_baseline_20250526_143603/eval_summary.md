# Model Evaluation Summary

Evaluation completed on: 2025-05-26 14:38:59

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /mnt/checkpoints/grpo_4 | 0.0200 | 2 | 0.9025 | 0.7280 | 0.8172 | 0.7190 | 0.8099 | 0.7415 | 0.9020 | 0.8590 | 0.0000 | 0.0900 | 9 | 0.0300 | 3 | 0 | {'S0': 78, 'S1': 2, 'S2': 15, 'S3': 5} | 5 | 0.0500 |
| meta-llama/Llama-3.2-3B-Instruct | 0.0200 | 2 | 0.8695 | 0.7830 | 0.7130 | 0.8130 | 0.5174 | 0.8172 | 0.9540 | 0.6290 | 0.0000 | 0.2900 | 29 | 0.0100 | 1 | 0 | {'S0': 48, 'S1': 1, 'S2': 41, 'S3': 10} | 10 | 0.1000 |