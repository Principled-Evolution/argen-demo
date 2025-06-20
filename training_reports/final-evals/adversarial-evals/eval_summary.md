# Model Evaluation Summary

Evaluation completed on: 2025-06-20 09:59:35

Scenarios file: ./data/adversarial-prompts-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/kapil/checkpoints/grpo_6_seed_2_4 | 0.0000 | 0 | 0.9280 | 0.7840 | 0.8029 | 0.5640 | 0.8440 | 0.6230 | 0.7400 | 0.8800 | 0.0000 | 0.1600 | 4 | 0.2800 | 7 | 0 | {'S0': 19, 'S1': 3, 'S2': 0, 'S3': 3} | 3 | 0.1200 |
| meta-llama/Llama-3.2-1B-Instruct | 0.0000 | 0 | 0.9184 | 0.7080 | 0.6793 | 0.5480 | 0.5616 | 0.5970 | 0.6840 | 0.5920 | 0.0000 | 0.4400 | 11 | 0.3200 | 8 | 0 | {'S0': 13, 'S1': 1, 'S2': 2, 'S3': 9} | 9 | 0.3600 |