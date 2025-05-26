# Model Evaluation Summary

Evaluation completed on: 2025-05-26 14:46:48

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /mnt/checkpoints/grpo_4 | 0.0500 | 5 | 0.8890 | 0.7350 | 0.8358 | 0.7290 | 0.8639 | 0.7452 | 0.8970 | 0.9070 | 0.0000 | 0.0700 | 7 | 0.0300 | 3 | 0 | {'S0': 87, 'S1': 1, 'S2': 9, 'S3': 3} | 3 | 0.0300 |
| meta-llama/Llama-3.2-1B-Instruct | 0.0700 | 7 | 0.8275 | 0.6560 | 0.6693 | 0.6870 | 0.5285 | 0.6987 | 0.8660 | 0.6480 | 0.0000 | 0.3100 | 31 | 0.0700 | 7 | 0 | {'S0': 49, 'S1': 0, 'S2': 45, 'S3': 6} | 6 | 0.0600 |