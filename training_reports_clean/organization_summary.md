# File Organization Summary

Generated: 2025-06-16 18:30:01
Base directory: training_reports_clean

## Summary Statistics

Total files processed: 55
Successfully organized: 55
Errors: 0

## Files by Evaluator Type

- claude: 20 files
- gemini: 22 files
- openai: 13 files

## Files by Model Type

model_type      baseline  grpo5  grpo6  grpo7  unknown
evaluator_type                                        
claude                 5      5      5      5        0
gemini                 5      5      5      5        2
openai                13      0      0      0        0

## Claude Evaluations (For Analysis)

Total Claude files: 20

### Claude Files by Model:
- baseline seed baseline checkpoint final: 5 file(s)
- grpo5 seed 1 checkpoint 1000: 1 file(s)
- grpo5 seed 1 checkpoint 2000: 1 file(s)
- grpo5 seed 1 checkpoint 2600: 1 file(s)
- grpo5 seed 1 checkpoint 3000: 1 file(s)
- grpo5 seed 1 checkpoint final: 1 file(s)
- grpo6 seed 2 checkpoint 1000: 1 file(s)
- grpo6 seed 2 checkpoint 2000: 1 file(s)
- grpo6 seed 2 checkpoint 3000: 1 file(s)
- grpo6 seed 2 checkpoint 4000: 1 file(s)
- grpo6 seed 2 checkpoint final: 1 file(s)
- grpo7 seed 3 checkpoint 1000: 1 file(s)
- grpo7 seed 3 checkpoint 2000: 1 file(s)
- grpo7 seed 3 checkpoint 3000: 1 file(s)
- grpo7 seed 3 checkpoint 4000: 1 file(s)
- grpo7 seed 3 checkpoint final: 1 file(s)

## Directory Structure

```
training_reports_clean/
├── claude_evaluations/          # For Champion/Median/Helpful selection
│   ├── grpo5_seed1/
│   ├── grpo6_seed2/
│   ├── grpo7_seed3/
│   └── baseline/
├── gemini_evaluations/          # For tabulation only
│   ├── grpo5_seed1/
│   ├── grpo6_seed2/
│   ├── grpo7_seed3/
│   └── baseline/
├── openai_evaluations/          # For tabulation only
│   ├── grpo5_seed1/
│   ├── grpo6_seed2/
│   ├── grpo7_seed3/
│   └── baseline/
└── other_evaluations/           # Other evaluators
```

## Next Steps

1. Update analysis scripts to use claude_evaluations/ for Champion/Median/Helpful selection
2. Update tabulation scripts to include all evaluator types
3. Re-run Champion/Median/Helpful analysis with clean Claude data
4. Generate comprehensive comparison tables with all evaluators

---
*File organization completed successfully*
