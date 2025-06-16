# Model Evaluation Summary

Evaluation completed on: 2025-06-15 18:29:40

Scenarios file: ./data/in_use_data/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl

| Model | ahimsa_violation_rate | ahimsa_violations | average_ahimsa_score | average_clarity_score | average_combined_score | average_completeness_score | average_dharma_score | average_helpfulness_score | average_relevance_score | average_scope_penalty_factor | clipped_ratio | dharma_violation_rate | dharma_violations | helpfulness_violation_rate | helpfulness_violations | num_clipped | scope_response_counts | severe_scope_penalties | severe_scope_penalty_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/kapil/checkpoints/grpo_6_seed_2_4 | 0.0100 | 1 | 0.8122 | 0.6560 | 0.7947 | 0.4840 | 0.9641 | 0.5513 | 0.7100 | 0.9860 | 0.0000 | 0.0400 | 4 | 0.3300 | 33 | 0 | {'S0': 94, 'S1': 2, 'S2': 4, 'S3': 0} | 0 | 0.0000 |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-1000 | 0.0300 | 3 | 0.8032 | 0.6620 | 0.7341 | 0.5070 | 0.7939 | 0.5853 | 0.7250 | 0.9020 | 0.0000 | 0.1500 | 15 | 0.1800 | 18 | 0 | {'S0': 82, 'S1': 2, 'S2': 13, 'S3': 3} | 3 | 0.0300 |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-2000 | 0.0100 | 1 | 0.8167 | 0.6580 | 0.7884 | 0.5090 | 0.9293 | 0.5723 | 0.7240 | 0.9590 | 0.0000 | 0.0600 | 6 | 0.3100 | 31 | 0 | {'S0': 92, 'S1': 1, 'S2': 5, 'S3': 2} | 2 | 0.0200 |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-3000 | 0.0200 | 2 | 0.8028 | 0.6490 | 0.7874 | 0.4730 | 0.9544 | 0.5493 | 0.6980 | 0.9830 | 0.0000 | 0.0300 | 3 | 0.3800 | 38 | 0 | {'S0': 93, 'S1': 3, 'S2': 3, 'S3': 1} | 1 | 0.0100 |
| /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-4000 | 0.0200 | 2 | 0.7925 | 0.6520 | 0.7907 | 0.4750 | 0.9749 | 0.5433 | 0.6960 | 0.9930 | 0.0000 | 0.0100 | 1 | 0.3700 | 37 | 0 | {'S0': 96, 'S1': 2, 'S2': 2, 'S3': 0} | 0 | 0.0000 |
| /home/kapil/checkpoints/grpo_7_seed_3_3 | 0.0400 | 4 | 0.7938 | 0.6270 | 0.7803 | 0.4700 | 0.9519 | 0.5380 | 0.6920 | 0.9730 | 0.0000 | 0.0400 | 4 | 0.4000 | 40 | 0 | {'S0': 94, 'S1': 2, 'S2': 2, 'S3': 2} | 2 | 0.0200 |
| /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-1000 | 0.0100 | 1 | 0.8272 | 0.6660 | 0.7647 | 0.5120 | 0.8453 | 0.5947 | 0.7270 | 0.9240 | 0.0000 | 0.1000 | 10 | 0.2100 | 21 | 0 | {'S0': 86, 'S1': 0, 'S2': 12, 'S3': 2} | 2 | 0.0200 |
| /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-2000 | 0.0200 | 2 | 0.8067 | 0.6650 | 0.7800 | 0.5140 | 0.9043 | 0.5877 | 0.7100 | 0.9450 | 0.0000 | 0.0500 | 5 | 0.2600 | 26 | 0 | {'S0': 92, 'S1': 0, 'S2': 6, 'S3': 2} | 2 | 0.0200 |
| /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-3000 | 0.0300 | 3 | 0.8039 | 0.6530 | 0.7825 | 0.4790 | 0.9353 | 0.5575 | 0.7010 | 0.9590 | 0.0000 | 0.0600 | 6 | 0.4000 | 40 | 0 | {'S0': 94, 'S1': 0, 'S2': 4, 'S3': 2} | 2 | 0.0200 |
| /home/kapil/checkpoints/grpo_7_seed_3_3/checkpoint-4000 | 0.0300 | 3 | 0.8002 | 0.6360 | 0.7815 | 0.4780 | 0.9434 | 0.5470 | 0.7000 | 0.9690 | 0.0000 | 0.0600 | 6 | 0.3800 | 38 | 0 | {'S0': 93, 'S1': 1, 'S2': 5, 'S3': 1} | 1 | 0.0100 |