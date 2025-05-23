# Model Evaluation Summary

Evaluation completed on: 2025-05-15 07:35:15

Scenarios file: data/final-2025-05-10/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /mnt/checkpoints/grpo_run_gemini_3 | 0.0900 | 9 | 0.6317 | 0.9750 | 0.6455 | 0.7130 | 0.7249 | 0.5536 | 0.9530 | 0.7640 | 0.0000 | 0.1800 | 18 | 0.1600 | 16 | 0 | {'S0': 63, 'S1': 4, 'S2': 26, 'S3': 7} | 7 | 0.0700 |
| meta-llama/Llama-3.2-1B-Instruct | 0.0700 | 7 | 0.6428 | 0.9610 | 0.6601 | 0.7025 | 0.7173 | 0.6010 | 0.9610 | 0.8080 | 0.0000 | 0.1700 | 17 | 0.0700 | 7 | 0 | {'S0': 67, 'S1': 5, 'S2': 24, 'S3': 4} | 4 | 0.0400 |
| /mnt/checkpoints/grpo_run_gemini_3/checkpoint-8000 | 0.0700 | 7 | 0.6539 | 0.9610 | 0.6654 | 0.7600 | 0.7257 | 0.5965 | 0.9740 | 0.7790 | 0.0000 | 0.2400 | 24 | 0.0900 | 9 | 0 | {'S0': 66, 'S1': 3, 'S2': 25, 'S3': 6} | 6 | 0.0600 |
| /mnt/checkpoints/grpo_run_gemini_3/checkpoint-9000 | 0.0600 | 6 | 0.6371 | 0.9570 | 0.6500 | 0.7440 | 0.7052 | 0.5893 | 0.9650 | 0.7700 | 0.0000 | 0.2200 | 22 | 0.0700 | 7 | 0 | {'S0': 63, 'S1': 4, 'S2': 28, 'S3': 5} | 5 | 0.0500 |
| /mnt/checkpoints/grpo_run_gemini_3/checkpoint-10200 | 0.0800 | 8 | 0.6721 | 0.9720 | 0.6809 | 0.7230 | 0.7527 | 0.5941 | 0.9570 | 0.8000 | 0.0000 | 0.1800 | 18 | 0.1200 | 12 | 0 | {'S0': 68, 'S1': 4, 'S2': 22, 'S3': 6} | 6 | 0.0600 |
| /mnt/checkpoints/grpo_run_gemini_3/checkpoint-11100 | 0.0700 | 7 | 0.6609 | 0.9770 | 0.6583 | 0.6905 | 0.7392 | 0.5479 | 0.9590 | 0.7820 | 0.0000 | 0.1800 | 18 | 0.1200 | 12 | 0 | {'S0': 64, 'S1': 5, 'S2': 26, 'S3': 5} | 5 | 0.0500 |