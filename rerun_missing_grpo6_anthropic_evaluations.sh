#!/bin/bash

# Re-run evaluations for grpo_6 checkpoint models that were overwritten by grpo_7
# This uses anthropic evaluator (claude-3-5-sonnet)

echo "Re-running evaluations for missing grpo_6 checkpoint models with anthropic evaluator..."

python scripts/evaluate_multiple_models.py \
  --temperature 0.2 \
  --evaluator anthropic \
  --scenarios ./data/in_use_data/benchmarking_20250510_135534-cleanprep-hashprompt.jsonl \
  --models \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-1000 \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-2000 \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-3000 \
    /home/kapil/checkpoints/grpo_6_seed_2_4/checkpoint-4000

echo "Evaluation complete!"
