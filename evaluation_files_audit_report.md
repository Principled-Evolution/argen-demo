# Evaluation Files Audit Report

Generated: 2025-06-16 18:28:08
Total files audited: 59

## Summary by Evaluator Type

                file_path  combined_score  helpfulness_score
evaluator_type                                              
claude                 20          0.7608             0.5656
error                   4          0.0000             0.0000
gemini                 22          0.8367             0.7374
openai                 13          0.6184             0.7562

## Summary by Folder Category

evaluator_type         claude  error  gemini  openai
folder_category                                     
organized_baseline          1      0       0       0
organized_grpo5             5      0       0       0
organized_grpo6_grpo7       6      0       0       0
timestamped                 8      4      22      13

## Claude Evaluations (For Champion/Median/Helpful Selection)

Total Claude evaluations: 20

                                combined_score  helpfulness_score                                                                                                                                                 file_path
model_type seed     checkpoint                                                                                                                                                                                             
baseline   baseline final               0.7575             0.5250                                training_reports/evaluate_baseline_20250613_105444/eval_grpo_4_debug-garbled-chat-template-response-claude-3-5-sonnet.json
grpo5      1        1000                0.7440             0.5705                                                  training_reports/grpo5-evals/eval_checkpoint-1000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2000                0.7620             0.5665                                                  training_reports/grpo5-evals/eval_checkpoint-2000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2600                0.7690             0.5300                                                  training_reports/grpo5-evals/eval_checkpoint-2600_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    3000                0.7752             0.5468                                                  training_reports/grpo5-evals/eval_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    final               0.7631             0.5170                                                    training_reports/grpo5-evals/eval_grpo_5_seed_1_benchmarking_20250510_135534-cleanprep-hashprompt.json
grpo6      2        1000                0.7341             0.5852  training_reports/evaluate_baseline_20250616_164151/eval_grpo_6_seed_2_4_checkpoint-1000_anthropic_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2000                0.7884             0.5722  training_reports/evaluate_baseline_20250616_164151/eval_grpo_6_seed_2_4_checkpoint-2000_anthropic_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    3000                0.7874             0.5492  training_reports/evaluate_baseline_20250616_164151/eval_grpo_6_seed_2_4_checkpoint-3000_anthropic_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    4000                0.7907             0.5432  training_reports/evaluate_baseline_20250616_164151/eval_grpo_6_seed_2_4_checkpoint-4000_anthropic_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    final               0.7947             0.5513                                            training_reports/grpo6-grpo7-evals/eval_grpo_6_seed_2_4_benchmarking_20250510_135534-cleanprep-hashprompt.json
grpo7      3        1000                0.7647             0.5947                            training_reports/grpo6-grpo7-evals/eval_grpo_7_seed_3_3_checkpoint-1000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2000                0.7800             0.5877                            training_reports/grpo6-grpo7-evals/eval_grpo_7_seed_3_3_checkpoint-2000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    3000                0.7825             0.5575                            training_reports/grpo6-grpo7-evals/eval_grpo_7_seed_3_3_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    4000                0.7815             0.5470                            training_reports/grpo6-grpo7-evals/eval_grpo_7_seed_3_3_checkpoint-4000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    final               0.7803             0.5380                                            training_reports/grpo6-grpo7-evals/eval_grpo_7_seed_3_3_benchmarking_20250510_135534-cleanprep-hashprompt.json

## Gemini Evaluations (For Tabulation Only)

Total Gemini evaluations: 22

                                combined_score  helpfulness_score                                                                                                                                       file_path
model_type seed     checkpoint                                                                                                                                                                                   
baseline   baseline final               0.9175             0.8250                              training_reports/evaluate_baseline_20250613_101737/eval_grpo_4_debug-garbled-chat-template-response_from_json.json
grpo5      1        1000                0.8158             0.7290    training_reports/evaluate_baseline_20250616_155043/eval_grpo_5_seed_1_checkpoint-1000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2000                0.8463             0.7367    training_reports/evaluate_baseline_20250616_155043/eval_grpo_5_seed_1_checkpoint-2000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2600                0.8547             0.7322    training_reports/evaluate_baseline_20250616_124658/eval_grpo_5_seed_1_checkpoint-2600_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    3000                0.8400             0.7253    training_reports/evaluate_baseline_20250616_155043/eval_grpo_5_seed_1_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    final               0.8506             0.7133                    training_reports/evaluate_baseline_20250616_124658/eval_grpo_5_seed_1_benchmarking_20250510_135534-cleanprep-hashprompt.json
grpo6      2        1000                0.7996             0.7562  training_reports/evaluate_baseline_20250616_161101/eval_grpo_6_seed_2_4_checkpoint-1000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2000                0.8574             0.7390  training_reports/evaluate_baseline_20250616_161101/eval_grpo_6_seed_2_4_checkpoint-2000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    3000                0.8784             0.7340  training_reports/evaluate_baseline_20250616_161101/eval_grpo_6_seed_2_4_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    4000                0.8666             0.7235  training_reports/evaluate_baseline_20250616_161101/eval_grpo_6_seed_2_4_checkpoint-4000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    final               0.8619             0.7345                  training_reports/evaluate_baseline_20250616_124658/eval_grpo_6_seed_2_4_benchmarking_20250510_135534-cleanprep-hashprompt.json
grpo7      3        1000                0.8009             0.7592  training_reports/evaluate_baseline_20250616_124658/eval_grpo_7_seed_3_3_checkpoint-1000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    2000                0.8564             0.7632  training_reports/evaluate_baseline_20250616_124658/eval_grpo_7_seed_3_3_checkpoint-2000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    3000                0.8599             0.7178  training_reports/evaluate_baseline_20250616_124658/eval_grpo_7_seed_3_3_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    4000                0.8583             0.7240  training_reports/evaluate_baseline_20250616_124658/eval_grpo_7_seed_3_3_checkpoint-4000_benchmarking_20250510_135534-cleanprep-hashprompt.json
                    final               0.8524             0.7160                  training_reports/evaluate_baseline_20250616_124658/eval_grpo_7_seed_3_3_benchmarking_20250510_135534-cleanprep-hashprompt.json
unknown    unknown  final               0.8022             0.6425                                        training_reports/evaluate_baseline_20250529_132210/eval_grpo_4_debug-garbled-chat-template-response.json

## Missing Evaluations Analysis

### Missing Claude Evaluations:
- baseline seed baseline checkpoint baseline

