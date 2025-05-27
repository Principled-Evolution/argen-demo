# Model Evaluation Summary

Evaluation completed on: 2025-05-26 12:30:12

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /mnt/checkpoints/grpo_4 | 0.0200 | 2 | 0.9055 | 0.9370 | 0.8158 | 0.6835 | 0.8354 | 0.7000 | 0.9585 | 0.8860 | 0.0000 | 0.0800 | 8 | 0.0500 | 5 | 0 | {'S0': 81, 'S1': 2, 'S2': 14, 'S3': 3} | 3 | 0.0300 |
| meta-llama/Llama-3.2-1B-Instruct | 0.1100 | 11 | 0.8062 | 0.8320 | 0.6317 | 0.5420 | 0.5088 | 0.6210 | 0.9130 | 0.6230 | 0.0000 | 0.3000 | 30 | 0.2100 | 21 | 0 | {'S0': 48, 'S1': 0, 'S2': 40, 'S3': 12} | 12 | 0.1200 |
| meta-llama/Llama-3.2-3B-Instruct | 0.0400 | 4 | 0.8593 | 0.8800 | 0.6917 | 0.6770 | 0.5260 | 0.7450 | 0.9560 | 0.6430 | 0.0000 | 0.3500 | 35 | 0.0400 | 4 | 0 | {'S0': 49, 'S1': 2, 'S2': 39, 'S3': 10} | 10 | 0.1000 |
| medalpaca/medalpaca-7b | 0.1300 | 13 | 0.7818 | 0.9720 | 0.6425 | 0.6130 | 0.5625 | 0.6100 | 0.9240 | 0.6240 | 0.0000 | 0.3700 | 37 | 0.2200 | 22 | 0 | {'S0': 53, 'S1': 0, 'S2': 29, 'S3': 18} | 18 | 0.1800 |