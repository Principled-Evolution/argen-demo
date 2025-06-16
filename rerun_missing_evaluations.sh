#!/bin/bash

# Re-run evaluations for the models that were overwritten
# This includes grpo_5 and grpo_6 checkpoints that were overwritten by grpo_7

echo "Re-running evaluations for missing grpo_5 and grpo_6 checkpoint models..."

python scripts/evaluate_multiple_models.py \
  --temperature 0.2 \
  --evaluator gemini \
  --scenarios ./data/in_use_data/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl \
  --models \
    /home/kapil/checkpoints/grpo_5_seed_1/checkpoint-1000 \
    /home/kapil/checkpoints/grpo_5_seed_1/checkpoint-2000 \
    /home/kapil/checkpoints/grpo_5_seed_1/checkpoint-3000 \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-1000 \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-2000 \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-3000 \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-4000

echo "Evaluation complete!"
