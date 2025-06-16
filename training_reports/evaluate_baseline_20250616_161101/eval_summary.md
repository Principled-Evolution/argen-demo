# Model Evaluation Summary

Evaluation completed on: 2025-06-16 16:32:59

Scenarios file: ./data/in_use_data/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-1000 | 0.0200 | 2 | 0.9005 | 0.7450 | 0.7996 | 0.7270 | 0.7564 | 0.7562 | 0.9050 | 0.8330 | 0.0000 | 0.1300 | 13 | 0.0500 | 5 | 0 | {'S0': 74, 'S1': 1, 'S2': 20, 'S3': 5} | 5 | 0.0500 |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-2000 | 0.0500 | 5 | 0.9105 | 0.7630 | 0.8574 | 0.7050 | 0.9064 | 0.7390 | 0.8890 | 0.9380 | 0.0000 | 0.0700 | 7 | 0.1000 | 10 | 0 | {'S0': 89, 'S1': 0, 'S2': 9, 'S3': 2} | 2 | 0.0200 |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-3000 | 0.0100 | 1 | 0.9340 | 0.7560 | 0.8784 | 0.6980 | 0.9449 | 0.7340 | 0.8880 | 0.9690 | 0.0000 | 0.0300 | 3 | 0.0800 | 8 | 0 | {'S0': 93, 'S1': 0, 'S2': 6, 'S3': 1} | 1 | 0.0100 |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-4000 | 0.0400 | 4 | 0.9198 | 0.7480 | 0.8666 | 0.6860 | 0.9340 | 0.7235 | 0.8830 | 0.9540 | 0.0000 | 0.0300 | 3 | 0.0700 | 7 | 0 | {'S0': 92, 'S1': 0, 'S2': 6, 'S3': 2} | 2 | 0.0200 |