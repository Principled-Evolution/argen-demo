# Model Evaluation Summary

Evaluation completed on: 2025-05-27 14:47:58

Scenarios file: ./data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.2-1B-Instruct | 0.0400 | 4 | 0.8353 | 0.6620 | 0.6594 | 0.6860 | 0.4982 | 0.6985 | 0.8650 | 0.6150 | 0.0000 | 0.3200 | 32 | 0.1000 | 10 | 0 | {'S0': 46, 'S1': 1, 'S2': 43, 'S3': 10} | 10 | 0.1000 |
| meta-llama/Llama-3.2-3B-Instruct | 0.0200 | 2 | 0.8735 | 0.7900 | 0.7309 | 0.8180 | 0.5566 | 0.8207 | 0.9590 | 0.6570 | 0.0000 | 0.2800 | 28 | 0.0000 | 0 | 0 | {'S0': 51, 'S1': 2, 'S2': 37, 'S3': 10} | 10 | 0.1000 |
| /mnt/checkpoints/grpo_L4_2 | 0.0200 | 2 | 0.9333 | 0.7500 | 0.8570 | 0.6920 | 0.8946 | 0.7307 | 0.8810 | 0.9260 | 0.0000 | 0.0300 | 3 | 0.1100 | 11 | 0 | {'S0': 87, 'S1': 0, 'S2': 11, 'S3': 2} | 2 | 0.0200 |
| /mnt/checkpoints/grpo_4 | 0.0300 | 3 | 0.8998 | 0.7430 | 0.8414 | 0.7250 | 0.8710 | 0.7438 | 0.8980 | 0.9110 | 0.0000 | 0.0700 | 7 | 0.0400 | 4 | 0 | {'S0': 87, 'S1': 0, 'S2': 9, 'S3': 4} | 4 | 0.0400 |
| /mnt/checkpoints/grpo_5 | 0.0400 | 4 | 0.9083 | 0.7730 | 0.8698 | 0.7110 | 0.9324 | 0.7480 | 0.8940 | 0.9640 | 0.0000 | 0.0300 | 3 | 0.0600 | 6 | 0 | {'S0': 92, 'S1': 0, 'S2': 7, 'S3': 1} | 1 | 0.0100 |
| /mnt/checkpoints/grpo_5/checkpoint-3000 | 0.0200 | 2 | 0.9120 | 0.7720 | 0.8492 | 0.7330 | 0.8685 | 0.7607 | 0.9020 | 0.9110 | 0.0000 | 0.0800 | 8 | 0.0500 | 5 | 0 | {'S0': 86, 'S1': 0, 'S2': 10, 'S3': 4} | 4 | 0.0400 |