### Here are the **four files** you can plug straight into the pipeline

| Purpose | File | Size (prompts) | When to use it |
|---------|------|----------------|----------------|
| **GRPO training** | `../../data/grpo_train_20250430.jsonl` | 150 | `train_grpo.py --scenarios data/grpo_train_20250430.jsonl` |
| **Baseline-20 (original)** | `sandbox:/mnt/data/combined_predibase_updated.jsonl` | 20 | Quick smoke-test / legacy comparability |
| **Eval-Core-60** | `../../data/eval_core_60_20250430.jsonl` | 60 | Main held-out evaluation: `evaluate_* --scenarios data/eval_core_60_20250430.jsonl` |
| **Eval-Adversarial-40** | `../../data/eval_adv_40_20250430.jsonl` | 40 | Stress-test: jailbreaks, misinformation traps |

All four files share the exact same schema (`prompt`, `role`, `patient_context`, `domain`, `completion`), so **no code changes** are needed—just point the CLI flags at the new paths.

* The 150-prompt training set is a non-overlapping subset of the 250 English-only scenarios you asked for.  
* The two evaluation sets are fully disjoint from training **and** from each other.  
* The original Baseline-20 stays untouched so your earlier numbers remain comparable.

