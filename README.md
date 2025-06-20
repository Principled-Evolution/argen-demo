# ArGen - Auto-Regulation of Generative AI Systems

ArGen is a comprehensive framework for training and evaluating AI models using policy-driven alignment with custom reward functions. This repository demonstrates the implementation described in our research paper on auto-regulation of generative AI systems through principle-based automated reward scoring and Group Relative Policy Optimization (GRPO).

## Overview

ArGen enables the formal encoding and enforcement of diverse policies to guide reinforcement learning towards safer, more compliant, and demonstrably aligned AI behaviors. The framework integrates:

- **Principle-based automated reward scoring** using configurable reward functions
- **Group Relative Policy Optimization (GRPO)** for efficient policy learning
- **Open Policy Agent (OPA) inspired governance layer** for policy enforcement
- **Multi-faceted evaluation** across ethical, operational, and regulatory dimensions

## Core Components

1. **Scenario Generation** - Generate diverse scenarios for training and evaluation
2. **Data Processing Pipeline** - Enhance scenarios with tier and scope classification
3. **GRPO Training** - Train models using custom reward functions with TRL integration
4. **Model Evaluation** - Evaluate models against policy compliance and performance metrics
5. **Multi-Model Comparison** - Compare multiple models across scenarios with parallel processing

## Project Structure

```
argen-demo/
├── argen/                          # Core framework package
│   ├── config.py                   # Configuration and hyperparameters
│   ├── data/                       # Data processing and generation
│   ├── training/                   # GRPO training functionality
│   ├── evaluation/                 # Model evaluation utilities
│   ├── reward_functions/           # Custom reward function implementations
│   ├── opa/                        # Open Policy Agent integration
│   └── utils/                      # General utilities
├── commands/                       # CLI interface for core processes
│   ├── generate_scenarios.py       # Scenario generation
│   ├── process_data.py             # Data processing pipeline
│   ├── train_model.py              # Model training with GRPO
│   ├── evaluate_model.py           # Model evaluation
│   └── compare_models.py           # Multi-model comparison
├── examples/                       # Usage examples
├── tools/                          # Utility scripts
├── data/                           # Sample data and scenarios
├── gopal/                          # Governance policies (OPA-style)
└── docs/                           # Documentation
```

## Quick Start

### 1. Generate Training Scenarios

```bash
# Generate scenarios for training
python commands/generate_scenarios.py --datasets grpo_training --count 300

# Generate evaluation scenarios
python commands/generate_scenarios.py --datasets benchmarking --count 100
```

### 2. Process Data Through Pipeline

```bash
# Enhance scenarios with tier and scope evaluation
python commands/process_data.py grpo_training_*.jsonl
python commands/process_data.py benchmarking_*.jsonl
```

### 3. Train Model with GRPO

```bash
# Train model using custom reward functions
python commands/train_model.py \
    --scenarios grpo_training_*-hashprompt.jsonl \
    --eval_scenarios benchmarking_*-hashprompt.jsonl \
    --output_dir ./checkpoints/grpo_run_1
```

### 4. Evaluate Trained Model

```bash
# Evaluate model performance
python commands/evaluate_model.py \
    --model ./checkpoints/grpo_run_1 \
    --scenarios benchmarking_*-hashprompt.jsonl \
    --evaluator gemini
```

## Case Study: Dharmic Healthcare AI

This repository demonstrates ArGen's capabilities through a healthcare AI assistant case study guided by principles from Dharmic ethics:

- **Ahimsa (Non-harm)**: Reward function ensuring responses avoid potential harm
- **Dharma (Righteous duty)**: Reward function promoting ethical and appropriate responses
- **Helpfulness**: Reward function encouraging genuinely useful assistance

The implementation shows how ArGen can encode complex, culturally-specific ethical frameworks into machine-readable policies for AI alignment.

## Installation

### System Requirements
- Python 3.11+
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory for 1B models, 16GB+ for 7B models

### Install Dependencies
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install torch transformers trl openai google-generativeai wandb
```

### API Keys
Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export WANDB_API_KEY="your-wandb-api-key"  # Optional
```

## Key Features

- **Policy-Driven Alignment**: Encode complex ethical frameworks into machine-readable policies
- **Custom Reward Functions**: Implement domain-specific reward functions (Ahimsa, Dharma, Helpfulness)
- **GRPO Training**: Group Relative Policy Optimization for efficient policy learning
- **Multi-LLM Evaluation**: Support for OpenAI, Anthropic, and Gemini evaluators
- **Governance Layer**: OPA-inspired policy enforcement and compliance checking
- **Comprehensive Evaluation**: Multi-dimensional assessment across ethical and performance metrics

## Usage Examples

### Complete Workflow
```bash
# 1. Generate training scenarios
python commands/generate_scenarios.py --datasets grpo_training --count 300

# 2. Process data through pipeline
python commands/process_data.py grpo_training_*.jsonl

# 3. Train model with GRPO
python commands/train_model.py \
    --scenarios grpo_training_*-hashprompt.jsonl \
    --output_dir ./checkpoints/grpo_run_1

# 4. Evaluate trained model
python commands/evaluate_model.py \
    --model ./checkpoints/grpo_run_1 \
    --scenarios eval_scenarios-hashprompt.jsonl \
    --evaluator gemini
```

### Multi-Model Comparison
```bash
# Compare multiple models
python commands/compare_models.py \
    --models meta-llama/Llama-3.2-1B-Instruct ./checkpoints/grpo_run_1 \
    --scenarios eval_scenarios-hashprompt.jsonl \
    --evaluator gemini
```

### Quick Testing
```bash
# Test with small dataset
python commands/generate_scenarios.py --datasets smoke_test --count 5
python commands/process_data.py smoke_test_*.jsonl
python commands/evaluate_model.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios smoke_test_*-hashprompt.jsonl \
    --evaluator gemini
```

## Configuration

Key configuration files:
- `argen/config.py`: Main configuration including GRPO parameters and reward weights
- `pyproject.toml`: Dependencies and package configuration
- `argen/data/generator/config.py`: Scenario generation settings

### Customization

**Reward Function Weights** (`argen/config.py`):
```python
REWARD_WEIGHTS = {
    "ahimsa": 0.3,      # Non-harm principle
    "dharma": 0.4,      # Righteous duty
    "helpfulness": 0.3  # Practical assistance
}
```

**Training Parameters** (`argen/config.py`):
```python
GRPO_CONFIG = {
    "learning_rate": 3.2e-6,
    "num_train_epochs": 3,
    "beta": 0.10,  # KL penalty strength
    # ... other parameters
}
```

## Contributing

We welcome contributions to ArGen! Please see the documentation in `docs/` for detailed guidelines.

### Quick Development Setup
```bash
# Clone and install
git clone https://github.com/your-org/argen-demo.git
cd argen-demo
poetry install

# Test your setup
python commands/generate_scenarios.py --datasets smoke_test --count 5
```

### Code Organization
- Add new functionality to appropriate `argen/` submodules
- Use the `commands/` directory for user-facing CLI tools
- Put development utilities in `tools/`
- Follow existing code structure and naming conventions

## License

This project is licensed under the terms specified in the LICENSE file.

## Citation

If you use ArGen in your research, please cite our paper:

```bibtex
@article{argen2024,
  title={ArGen: Auto-Regulation of Generative AI Systems},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Support

For questions and support:
- Check the documentation in `docs/`
- Review existing GitHub issues
- Create a new issue for bugs or feature requests

---

*ArGen demonstrates a pathway toward "Governable AI" - systems that are technically proficient, ethically robust, and verifiably compliant for safe deployment in diverse global contexts.*
