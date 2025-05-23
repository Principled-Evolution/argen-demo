# GRPO Chat Template Debugging

This directory contains scripts and tools to debug issues with garbled or blank responses during GRPO training with Llama 3.2 1B model.

## Problem Description

The Llama 3.2 1B model works fine during evaluation with `evaluate_baseline.py`, but produces blank or garbled responses during training with `train_grpo.py`. This suggests a mismatch in how prompts are formatted or how generation is handled between these two contexts.

## Debugging Approach

We've created several scripts to diagnose and fix the issue:

1. **Debug Callback**: A custom callback to inspect the generation process during GRPO training
2. **Chat Template Test**: A script to test chat template handling for Llama 3.2 models
3. **TRL Version Comparison**: A script to compare the PyPI version of TRL with a local clone
4. **TRL Debug Patch**: A script to patch the TRL library with additional debugging information
5. **Debug GRPO Generation**: A minimal test script to isolate the issue

## Files

- `debug_callback.py`: Custom callback to debug generation issues in GRPO training
- `debug_grpo_generation.py`: Minimal test script to isolate the issue
- `compare_trl_versions.py`: Script to compare PyPI and local TRL implementations
- `test_chat_template.py`: Script to test chat template handling for Llama 3.2 models
- `trl_debug_patch.py`: Script to patch TRL with additional debugging information
- `run_debug_tests.sh`: Shell script to run all debug tests
- `DEBUG_README.md`: This file

## How to Use

1. Make sure you have the required dependencies installed:
   ```bash
   pip install trl==0.17.0 transformers datasets
   ```

2. Run the debug tests:
   ```bash
   chmod +x run_debug_tests.sh
   ./run_debug_tests.sh
   ```

3. Check the debug logs in the `debug_logs` directory:
   ```bash
   ls -la debug_logs/
   ```

## Potential Issues and Solutions

### 1. Chat Template Mismatch

The issue might be related to how the chat template is applied during GRPO training. The Llama 3.2 model has a specific chat template that might not be properly applied.

**Solution**: Ensure the model's native chat template is used consistently.

### 2. Generation Parameters

The generation parameters during GRPO training might be different from those used during evaluation.

**Solution**: Adjust the generation parameters to match those used during evaluation.

### 3. Dataset Format

The dataset format might not be compatible with the GRPO trainer's expectations.

**Solution**: Ensure the dataset is properly formatted for GRPO training.

### 4. System Prompt Handling

The system prompt might not be properly incorporated into the chat template.

**Solution**: Check how system prompts are handled and ensure they're properly formatted.

## Next Steps

After running the debug tests, analyze the logs to identify the root cause of the issue. Based on the findings, implement the appropriate fix and test it with a small dataset before applying it to the full training process.
