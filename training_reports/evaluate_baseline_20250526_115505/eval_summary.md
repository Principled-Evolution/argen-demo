# Model Evaluation Summary

Evaluation completed on: 2025-05-26 12:00:53

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /mnt/checkpoints/grpo_4 | 0.0300 | 3 | 0.9025 | 0.9310 | 0.8208 | 0.6710 | 0.8471 | 0.7040 | 0.9545 | 0.8970 | 0.0000 | 0.0700 | 7 | 0.0500 | 5 | 0 | {'S0': 83, 'S1': 1, 'S2': 12, 'S3': 4} | 4 | 0.0400 |
| meta-llama/Llama-3.2-1B-Instruct | 0.0600 | 6 | 0.8325 | 0.8240 | 0.6636 | 0.5370 | 0.5637 | 0.6280 | 0.9070 | 0.6710 | 0.0000 | 0.2900 | 29 | 0.1600 | 16 | 0 | {'S0': 55, 'S1': 0, 'S2': 35, 'S3': 10} | 10 | 0.1000 |
| meta-llama/Llama-3.2-3B-Instruct | 0.0500 | 5 | 0.8583 | 0.8890 | 0.6938 | 0.7150 | 0.5216 | 0.7590 | 0.9600 | 0.6460 | 0.0000 | 0.3300 | 33 | 0.0200 | 2 | 0 | {'S0': 47, 'S1': 2, 'S2': 42, 'S3': 9} | 9 | 0.0900 |