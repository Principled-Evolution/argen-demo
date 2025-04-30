# ArGen GRPO Implementation: Project Requirements Document

## 1. Project Overview

### 1.1 Introduction
This document outlines the requirements and implementation plan for integrating the General Reinforcement Policy Optimization (GRPO) training method into the ArGen codebase. The implementation will align with the existing baseline evaluation setup while transitioning from Unsloth to a direct integration with Hugging Face's TRL library.

### 1.2 Background
ArGen is a research project focused on developing AI safety approaches through the lens of Ahimsa (non-harm) and Dharma (adherence to right conduct) principles. The current implementation includes a baseline evaluation system using OpenAI for reward scoring. We're enhancing this system by implementing GRPO fine-tuning while maintaining consistent prompts, reward functions, and evaluation metrics.

### 1.3 Objectives
- Replace Unsloth dependency with direct TRL integration (v0.17.x)
- Maintain exact model parameters, prompts, and reward functions across evaluation and training
- Implement GRPO training while leveraging existing OpenAI-based reward functions
- Ensure real-time monitoring via Weights & Biases (wandb)
- Create a clean, maintainable implementation with minimal dependencies

## 2. Technical Requirements

### 2.1 Core Components
1. **Configuration Module**
   - Centralized configuration for model parameters, prompts, and reward weights
   - Used by both evaluation and training scripts

2. **Reward Functions**
   - TRL-compatible reward function adapters
   - Maintains OpenAI-based evaluation for Ahimsa and Dharma principles

3. **GRPO Training Script**
   - Integrates with TRL's GRPOTrainer
   - Uses shared configuration and reward functions

4. **Evaluation Scripts**
   - Baseline evaluation script (existing)
   - New script for evaluating GRPO-trained models

5. **Observability**
   - wandb integration for real-time monitoring
   - Consistent metrics between training and evaluation

### 2.2 System Architecture
```
┌───────────────────┐     ┌────────────────────┐
│  Shared Config    │◄────┤ Reward Functions   │
└─────────┬─────────┘     └──────────┬─────────┘
          │                          │
          ▼                          ▼
┌─────────────────────────────────────────────┐
│                TRL Integration              │
└─────────┬─────────────────────────┬─────────┘
          │                         │
┌─────────▼─────────┐     ┌─────────▼─────────┐
│  GRPO Training    │     │  Model Evaluation │
└─────────┬─────────┘     └──────────┬────────┘
          │                          │
          ▼                          ▼
┌─────────────────────────────────────────────┐
│             Wandb Monitoring                │
└─────────────────────────────────────────────┘
```

## 3. Implementation Plan

### 3.1 Phase 1: Preparation and Shared Components ✅
1. Create centralized configuration module
   - Extract model parameters, prompts, and settings from evaluate_baseline.py
   - Create configuration functions for GRPO parameters

2. Create TRL-compatible reward functions
   - Adapt existing OpenAI-based evaluators
   - Ensure tensor shapes match TRL requirements

### 3.2 Phase 2: GRPO Implementation ✅
3. Create GRPO training script
   - Integrate with TRL's GRPOTrainer
   - Use shared configuration and reward functions
   - Implement wandb monitoring

4. Create evaluation script for GRPO-trained models
   - Align with baseline evaluation metrics
   - Enable comparison between baseline and GRPO models

### 3.3 Phase 3: Testing and Refinement 🔄
5. Test components individually
   - Verify configuration consistency
   - Test reward functions
   - Test GRPO training with simple dataset

6. End-to-end testing
   - Run full training and evaluation pipeline
   - Verify metrics match between components

### 3.4 Phase 4: Documentation and Cleanup 🔄
7. Update documentation
   - Add detailed README and usage instructions
   - Document architecture and component interactions

8. Cleanup and optimization
   - Remove any unused code
   - Optimize for performance

## 4. Project Status

### 4.1 Completed Items ✅
- Created shared configuration module (`src/config.py`)
- Implemented TRL-compatible reward functions (`src/reward_functions/trl_rewards.py`)
- Updated baseline evaluation to use shared configuration
- Created GRPO training script (`examples/train_grpo.py`)
- Created evaluation script for GRPO-trained models (`examples/evaluate_trained_model.py`)

### 4.2 In Progress 🔄
- Testing and validation of reward function tensor shapes
- End-to-end testing of training pipeline
- Fine-tuning wandb integration

### 4.3 Pending Items ⏳
- Documentation updates
- Performance optimizations
- Cleanup of legacy code

## 5. Implementation Details

### 5.1 Configuration Module
The configuration module centralizes all parameters needed for both evaluation and training, ensuring consistency across components:

```python
# src/config.py
DEFAULT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_NEW_TOKENS = 512
REWARD_WEIGHTS = {"ahimsa": 0.5, "dharma": 0.5}
```

### 5.2 Reward Functions
TRL-compatible reward functions adapt the existing OpenAI evaluators:

```python
# src/reward_functions/trl_rewards.py
def trl_combined_reward_function(prompts, responses, model_ref=None):
    # Calculate ahimsa and dharma rewards using OpenAI evaluator
    # Return tensor of shape [batch_size] with combined rewards
```

### 5.3 GRPO Training Script
The training script integrates with TRL's GRPOTrainer:

```python
# examples/train_grpo.py
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_model=trl_combined_reward_function,
    args=GRPOConfig(**config),
)
trainer.train()
```

### 5.4 Evaluation Script
The evaluation script uses the same metrics as the baseline:

```python
# examples/evaluate_trained_model.py
results = evaluate_model_with_openai(
    model_name=trained_model_path,
    scenarios=scenarios,
    output_file=output_filename,
    temperature=temperature,
    use_basic_prompt=use_basic_prompt
)
```

## 6. Development Guidelines

### 6.1 Code Style
- Follow PEP 8 for Python code style
- Use descriptive variable names
- Add appropriate docstrings for all functions and classes

### 6.2 Testing
- Add unit tests for each component
- Ensure backward compatibility with existing scripts
- Test with representative scenarios

### 6.3 Documentation
- Update README.md with usage instructions
- Document configuration options
- Include example commands for training and evaluation

## 7. Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tensor shape mismatches | Training failures | Thorough testing of reward functions with TRL |
| Performance issues with real-time wandb | Slow training | Optimize update frequency and data sent to wandb |
| Inconsistent metrics between components | Invalid comparisons | Use shared calculation methods from config |
| API rate limits with OpenAI | Training interruptions | Implement robust retry and rate-limiting logic |

## 8. Timeline and Milestones

| Milestone | Estimated Completion | Status |
|-----------|----------------------|--------|
| Phase 1: Shared Components | Complete | ✅ |
| Phase 2: GRPO Implementation | Complete | ✅ |
| Phase 3: Testing and Refinement | In Progress | 🔄 |
| Phase 4: Documentation and Cleanup | In Progress | 🔄 |
| Final Release | 1 week | ⏳ |

## 9. Conclusion

The ArGen GRPO implementation provides a clean integration with Hugging Face's TRL library while maintaining consistency with the existing evaluation pipeline. By centralizing configuration and sharing reward functions, we ensure that metrics are comparable between baseline and GRPO-trained models, enabling rigorous scientific comparison of the approaches. 