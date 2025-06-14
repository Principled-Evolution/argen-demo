# Ablation Mode Debug Logging

This document describes the enhanced debug logging system for ablation study modes, implemented to address GitHub issue #32.

## Overview

The ablation debug logging system provides detailed visibility into how different ablation modes (`full`, `reward_only`, `policy_only`) transform evaluation results during training. This helps verify that ablation modes are working correctly and provides clear evidence for research documentation.

## Features

### ✅ Comprehensive Mode Coverage
- **Combined Rewards Mode** (`use_separate_rewards=False`): Single `combined_reward_trl()` call
- **Separate Rewards Mode** (`use_separate_rewards=True`): Individual reward function calls
- **All Ablation Modes**: `full`, `reward_only`, `policy_only`

### ✅ Detailed Item-Level Logging
Shows for each prompt-response pair:
- Current ablation mode
- Raw LLM scores (before penalties)
- Policy penalty factors
- Mode-specific calculation logic
- Final selected score
- Cross-mode comparison (what other modes would give)

### ✅ Batch-Level Statistics
Provides batch summaries showing:
- Score distributions by component
- Average differences between modes
- Verification of expected patterns

## Usage

### Command Line

Enable debug logging with the `--ablation_debug` flag:

```bash
# Combined rewards mode with debug logging
python examples/train_grpo.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios data/scenarios.jsonl \
    --ablation_mode reward_only \
    --ablation_debug

# Separate rewards mode with debug logging
python examples/train_grpo.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios data/scenarios.jsonl \
    --ablation_mode policy_only \
    --ablation_debug \
    --use_separate_rewards
```

### Environment Variable

Alternatively, set the environment variable directly:

```bash
export ARGEN_ABLATION_DEBUG=true
python examples/train_grpo.py --ablation_mode full
```

### Programmatic

```python
import os
os.environ["ARGEN_ABLATION_DEBUG"] = "true"

from argen.config import get_ablation_debug
assert get_ablation_debug() == True
```

## Example Output

### Combined Rewards Mode

```
🧪 ABLATION DEBUG [Batch Item 3/10] Mode: reward_only | Combined Calculation
   Raw LLM Scores: {'A': 0.85, 'D': 0.75, 'H': 0.82}
   Policy Penalties: {'scope': 0.8, 'severity': 0.0}
   Calculation: (0.85×0.3 + 0.75×0.4 + 0.82×0.3) = 0.801
   ✅ SELECTED: 0.801 (LLM scores only, no penalties)
   Full Would Give: 0.641, Policy-Only Would Give: 0.800
```

### Separate Rewards Mode

```
🧪 ABLATION DEBUG [Batch Item 3/10] Mode: full | Component: Ahimsa
   Raw LLM Scores: {'harm_avoidance': 0.9, 'safety_context': 0.8, 'raw_average': 0.85}
   Policy Penalties: {'ahimsa_penalty': 0.8, 'severity': 0.2}
   Calculation: (0.9 + 0.8) / 2 × 0.8 = 0.68
   ✅ SELECTED: 0.68 (LLM scores × penalties)
   Final Weighted Contribution: 0.68 × 0.3 = 0.204

🧪 ABLATION SUMMARY [Batch Item 3/10] Mode: full
   Component Scores: A=0.68, D=0.60, H=0.82
   Weights: A=0.3, D=0.4, H=0.3
   Final Combined: (0.68×0.3 + 0.60×0.4 + 0.82×0.3) = 0.690
```

### Batch Summary

```
🧪 BATCH ABLATION SUMMARY [Batch Size: 10] Mode: reward_only
   Ahimsa: mean=0.850, std=0.120, range=[0.650, 0.950]
   Dharma: mean=0.780, std=0.090, range=[0.600, 0.900]
   Helpfulness: mean=0.820, std=0.080, range=[0.700, 0.950]
   Combined: mean=0.810, std=0.095, range=[0.680, 0.930]
   Mode Behavior: reward_only - LLM scores only
```

## Implementation Details

### Configuration

- **Config constant**: `ABLATION_DEBUG_LOGGING = False` in `argen/config.py`
- **Helper function**: `get_ablation_debug()` checks environment variable or config
- **Environment variable**: `ARGEN_ABLATION_DEBUG=true`

### Logging Functions

#### `log_ablation_debug()`
Item-level debug logging for individual evaluations.

#### `log_ablation_summary()`
Summary logging for separate rewards mode showing final combined calculation.

#### `log_batch_ablation_summary()`
Batch-level statistics and distribution analysis.

### Performance Impact

- **Zero overhead when disabled**: No performance impact in normal operation
- **Minimal overhead when enabled**: Only logging operations, no additional evaluations
- **Optional activation**: Only runs when explicitly requested

## Verification Examples

### Mode Differences

The debug logging clearly shows how different modes produce different behaviors:

| Mode | Behavior | Example Output |
|------|----------|----------------|
| `full` | LLM scores × policy penalties | `0.85 × 0.8 = 0.68` |
| `reward_only` | Raw LLM scores, bypass penalties | `0.85 (no penalties)` |
| `policy_only` | Policy compliance signals only | `0.8 (penalty factor)` |

### Cross-Mode Comparison

Each log entry shows what other modes would have produced:
- Helps verify mode logic is working correctly
- Enables easy comparison of mode behaviors
- Provides evidence for research documentation

## Demo Script

Run the demo script to see debug logging in action:

```bash
# Demo with debug logging enabled
python examples/demo_ablation_debug_logging.py --ablation_mode reward_only --ablation_debug

# Demo with separate rewards
python examples/demo_ablation_debug_logging.py --ablation_mode policy_only --ablation_debug --use_separate_rewards

# Demo without debug logging (normal operation)
python examples/demo_ablation_debug_logging.py --ablation_mode full
```

## Research Applications

This debug logging system enables:

1. **Verification**: Confirm ablation modes work as intended
2. **Documentation**: Generate examples for research papers
3. **Analysis**: Compare mode behaviors quantitatively
4. **Debugging**: Identify issues in ablation logic
5. **Validation**: Ensure consistent results across training runs

## Related Issues

- **GitHub Issue #32**: Enhanced Ablation Mode Debug Logging
- **GitHub Issue #27**: Master Ablation Study Implementation
- **Phases 1-3**: Core ablation functionality (completed)
- **Phase 4**: Comprehensive testing and validation
- **Phase 5**: Documentation and research validation
