# ArGen Demo - Healthcare AI Training and Evaluation Framework

ArGen is a comprehensive framework for training and evaluating healthcare AI models using GRPO (Generalized Reward Policy Optimization) with custom reward functions. This repository contains the complete pipeline from scenario generation to model training and evaluation.

## Overview

The ArGen framework consists of 5 key processes that work together to create, process, train, and evaluate healthcare AI models:

1. **Scenario Generation** - Generate diverse healthcare scenarios for training and evaluation
2. **Data Pipeline** - Process and enhance generated scenarios with tier and scope evaluation
3. **GRPO Training** - Train models using custom reward functions with TRL integration
4. **Baseline Evaluation** - Evaluate individual models against healthcare scenarios
5. **Multi-Model Evaluation** - Compare multiple models across scenarios with parallel processing

## Key Processes

### 1. Scenario Generation (`src/data_utils/scenario_generator/standalone_run.py`)

**Purpose**: Generate diverse healthcare scenarios for training and evaluation datasets using OpenAI or HuggingFace models.

**Key Features**:
- Supports multiple dataset types: `smoke_test`, `benchmarking`, `grpo_training`
- Configurable generation models (OpenAI GPT or HuggingFace models like MedAlpaca)
- Real-time progress tracking with progress bars
- Duplicate detection and filtering
- Medical content validation and tiering
- Concurrent API processing for improved performance

**Usage**:
```bash
# Basic usage with OpenAI model
python src/data_utils/scenario_generator/standalone_run.py --datasets benchmarking --count 100

# Using HuggingFace model
python src/data_utils/scenario_generator/standalone_run.py --datasets grpo_training --hf-model medalpaca/medalpaca-7b --count 300

# Advanced configuration
python src/data_utils/scenario_generator/standalone_run.py \
    --datasets smoke_test benchmarking \
    --model gpt-4o-mini \
    --temperature 1.3 \
    --batch-size 16 \
    --concurrent-eval-limit 30 \
    --exclude-from-file existing_scenarios.jsonl
```

**Key Arguments**:
- `--datasets`: Dataset types to generate (`smoke_test`, `benchmarking`, `grpo_training`)
- `--count`: Number of scenarios to generate (overrides dataset defaults)
- `--model`: OpenAI model for generation (default: `gpt-4o-mini`)
- `--hf-model`: HuggingFace model path (e.g., `medalpaca/medalpaca-7b`)
- `--temperature`: Generation temperature (higher = more diverse)
- `--batch-size`: Batch size for generation (default: 8)
- `--concurrent-eval-limit`: Max concurrent evaluations (default: 20)
- `--tiering-concurrency-limit`: Max concurrent tiering requests (default: 10)
- `--difficulty-ratio`: Ratio for difficulty banding (default: 0.8, -1 to disable)
- `--fail-threshold`: Threshold for accepting challenging prompts (0.0-1.0)
- `--exclude-from-file`: Path to JSONL file with scenarios to exclude

**Dependencies**:
- OpenAI API key (for OpenAI models)
- Gemini API key (for evaluation)
- HuggingFace transformers (for local models)
- Required Python packages: `openai`, `google-generativeai`, `transformers`, `torch`

**Output**: JSONL files with generated scenarios including prompts, tiers, and metadata.

### 2. Data Pipeline (`scripts/data_pipeline.py`)

**Purpose**: Process JSONL files through a multi-stage pipeline to enhance scenarios with tier evaluation, scope classification, and prompt hashing.

**Key Features**:
- Three-stage processing pipeline
- Automatic intermediate file management
- Gemini-based evaluation for consistency
- Final output with comprehensive metadata

**Pipeline Stages**:
1. **Tier Re-evaluation**: Uses `src/data_utils/reevaluate_tiers_cli.py` with Gemini evaluator
2. **Scope Classification**: Uses `src/data_utils/reevaluate_scope_cli.py` with Gemini evaluator
3. **Prompt Hashing**: Uses `scripts/add_prompt_hash_to_dataset_new.py` for deduplication

**Usage**:
```bash
# Process a dataset file
python scripts/data_pipeline.py data/my_scenarios.jsonl

# The output will be saved as data/my_scenarios-hashprompt.jsonl
```

**Dependencies**:
- Gemini API key for evaluation stages
- Input JSONL file with scenario data
- All pipeline scripts must be available in their respective locations

**Output**: Enhanced JSONL file with suffix `-hashprompt` containing tier, scope, and hash metadata.

### 3. GRPO Training (`examples/train_grpo.py`)

**Purpose**: Train language models using GRPO (Generalized Reward Policy Optimization) with custom healthcare reward functions.

**Key Features**:
- TRL (Transformers Reinforcement Learning) integration
- Custom reward functions: Ahimsa (safety), Dharma (ethics), Helpfulness
- Weights & Biases integration for experiment tracking
- Configurable training parameters and evaluation during training
- Support for both combined and separate reward functions
- Early stopping and adaptive learning rate scheduling

**Usage**:
```bash
# Basic GRPO training
python examples/train_grpo.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios data/grpo_training_scenarios-hashprompt.jsonl \
    --output_dir /path/to/checkpoints

# Advanced training with custom parameters
python examples/train_grpo.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios data/grpo_training_scenarios-hashprompt.jsonl \
    --eval_scenarios data/eval_scenarios-hashprompt.jsonl \
    --output_dir /path/to/checkpoints \
    --num_train_epochs 3 \
    --learning_rate 3.2e-6 \
    --use_separate_rewards \
    --wandb_project my-grpo-experiment \
    --evaluator gemini \
    --early_stopping
```

**Key Arguments**:
- `--model`: HuggingFace model identifier (default: `meta-llama/Llama-3.2-1B-Instruct`)
- `--scenarios`: Path to training scenarios file (must be processed with hashes)
- `--eval_scenarios`: Path to evaluation scenarios file (optional)
- `--output_dir`: Directory to save model checkpoints
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 3.2e-6)
- `--use_separate_rewards`: Use separate reward functions instead of combined
- `--wandb_project`: Weights & Biases project name
- `--evaluator`: Evaluation LLM (`openai` or `gemini`)
- `--early_stopping`: Enable early stopping based on evaluation metrics

**Dependencies**:
- CUDA-capable GPU for training
- TRL library (`pip install trl`)
- Transformers, PyTorch, Weights & Biases
- OpenAI or Gemini API key for evaluation
- Processed training data with prompt hashes

**Output**: Trained model checkpoints and training logs with W&B integration.

### 4. Baseline Evaluation (`examples/evaluate_baseline.py`)

**Purpose**: Evaluate individual models against healthcare scenarios using LLM evaluators (OpenAI or Gemini).

**Key Features**:
- Support for both local HuggingFace models and API-based models
- Configurable evaluation modes (batch vs individual for Gemini)
- Comprehensive reward function evaluation (Ahimsa, Dharma, Helpfulness)
- Comparison mode to analyze batch vs individual evaluation performance
- Configurable penalty systems for medical disclaimers and professional referrals

**Usage**:
```bash
# Basic evaluation with Gemini
python examples/evaluate_baseline.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios data/eval_scenarios-hashprompt.jsonl \
    --evaluator gemini

# Evaluation with custom parameters
python examples/evaluate_baseline.py \
    --model /path/to/trained/model \
    --scenarios data/eval_scenarios-hashprompt.jsonl \
    --evaluator gemini \
    --eval-mode batch \
    --temperature 0.9 \
    --system_prompt ENHANCED \
    --generation_batch_size 50

# Compare batch vs individual evaluation modes
python examples/evaluate_baseline.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios data/eval_scenarios-hashprompt.jsonl \
    --evaluator gemini \
    --compare-eval-modes
```

**Key Arguments**:
- `--model`: Model identifier or path to evaluate
- `--scenarios`: Path to evaluation scenarios file
- `--evaluator`: LLM evaluator (`openai` or `gemini`, default: `gemini`)
- `--eval-mode`: Evaluation mode for Gemini (`batch` or `individual`, default: `individual`)
- `--temperature`: Generation temperature (default: 0.9)
- `--system_prompt`: System prompt type (`BASIC`, `ENHANCED`, `MEDICAL`)
- `--generation_batch_size`: Batch size for local model generation (default: 50)
- `--compare-eval-modes`: Run comparison between batch and individual modes
- `--include-reasoning`: Include reasoning in evaluation responses

**Dependencies**:
- PyTorch and Transformers for local models
- OpenAI or Gemini API key depending on evaluator choice
- CUDA for GPU acceleration (recommended for local models)

**Output**: JSON file with detailed evaluation results including individual scores and combined metrics.

### 5. Multi-Model Evaluation (`scripts/evaluate_multiple_models.py`)

**Purpose**: Evaluate multiple models in parallel across available GPUs with comprehensive comparison reporting.

**Key Features**:
- Parallel evaluation across multiple GPUs
- Pipelined execution to maximize GPU utilization
- Automatic summary table generation with model comparisons
- Support for both sequential and parallel execution modes
- Comprehensive error handling and progress tracking

**Usage**:
```bash
# Evaluate multiple models in parallel
python scripts/evaluate_multiple_models.py \
    --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct medalpaca/medalpaca-7b \
    --scenarios data/eval_scenarios-hashprompt.jsonl \
    --evaluator gemini

# Sequential evaluation (no parallelization)
python scripts/evaluate_multiple_models.py \
    --models model1 model2 model3 \
    --scenarios data/eval_scenarios-hashprompt.jsonl \
    --evaluator gemini \
    --no_parallel

# Custom pipeline configuration
python scripts/evaluate_multiple_models.py \
    --models model1 model2 model3 model4 \
    --scenarios data/eval_scenarios-hashprompt.jsonl \
    --evaluator gemini \
    --pipeline_delay 15 \
    --max_concurrent_models 2 \
    --eval_mode batch
```

**Key Arguments**:
- `--models`: List of model names/paths to evaluate (space-separated)
- `--scenarios`: Path to evaluation scenarios file
- `--evaluator`: LLM evaluator (`openai` or `gemini`)
- `--no_parallel`: Disable parallel evaluation across GPUs
- `--no_pipeline`: Wait for each model to complete before starting the next
- `--pipeline_delay`: Delay in seconds before starting next model (default: 10)
- `--max_concurrent_models`: Maximum concurrent models (default: number of GPUs)
- `--eval_mode`: Evaluation mode for Gemini (`batch` or `individual`)
- `--generation_batch_size`: Batch size for local model generation

**Dependencies**:
- Multiple CUDA GPUs for parallel evaluation
- All dependencies from baseline evaluation
- Sufficient GPU memory for concurrent model loading

**Output**:
- Individual evaluation JSON files for each model
- Comprehensive summary markdown table (`eval_summary.md`)
- Timestamped results directory with all outputs

## Dependencies and Setup

### System Requirements
- Python 3.8+
- CUDA-capable GPU(s) for training and evaluation
- Sufficient GPU memory (8GB+ recommended for 1B models, 16GB+ for 7B models)

### Python Dependencies
Install using Poetry (recommended):
```bash
poetry install
```

Or using pip:
```bash
pip install torch transformers trl openai google-generativeai wandb datasets accelerate
```

### API Keys
Set up the following environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export WANDB_API_KEY="your-wandb-api-key"  # Optional, for experiment tracking
```

## Quick Start Guide

1. **Generate Training Data**:
```bash
python src/data_utils/scenario_generator/standalone_run.py --datasets grpo_training --count 300
```

2. **Process Data Through Pipeline**:
```bash
python scripts/data_pipeline.py grpo_training_*.jsonl
```

3. **Train Model with GRPO**:
```bash
python examples/train_grpo.py --scenarios grpo_training_*-hashprompt.jsonl --output_dir ./checkpoints
```

4. **Evaluate Trained Model**:
```bash
python examples/evaluate_baseline.py --model ./checkpoints --scenarios eval_scenarios-hashprompt.jsonl
```

5. **Compare Multiple Models**:
```bash
python scripts/evaluate_multiple_models.py --models ./checkpoints meta-llama/Llama-3.2-1B-Instruct --scenarios eval_scenarios-hashprompt.jsonl
```

## Workflow Examples

### Complete Training Pipeline
```bash
# 1. Generate diverse training scenarios
python src/data_utils/scenario_generator/standalone_run.py \
    --datasets grpo_training \
    --count 300 \
    --model gpt-4o-mini \
    --temperature 0.7

# 2. Generate evaluation scenarios
python src/data_utils/scenario_generator/standalone_run.py \
    --datasets benchmarking \
    --count 100 \
    --model gpt-4o-mini \
    --temperature 1.3

# 3. Process both datasets through pipeline
python scripts/data_pipeline.py grpo_training_*.jsonl
python scripts/data_pipeline.py benchmarking_*.jsonl

# 4. Train model with evaluation during training
python examples/train_grpo.py \
    --scenarios grpo_training_*-hashprompt.jsonl \
    --eval_scenarios benchmarking_*-hashprompt.jsonl \
    --output_dir ./checkpoints/grpo_run_1 \
    --wandb_project healthcare-grpo \
    --early_stopping \
    --evaluator gemini

# 5. Evaluate final model
python examples/evaluate_baseline.py \
    --model ./checkpoints/grpo_run_1 \
    --scenarios benchmarking_*-hashprompt.jsonl \
    --evaluator gemini \
    --eval-mode batch
```

### Multi-Model Comparison Workflow
```bash
# Generate evaluation dataset
python src/data_utils/scenario_generator/standalone_run.py \
    --datasets benchmarking \
    --count 100

# Process evaluation data
python scripts/data_pipeline.py benchmarking_*.jsonl

# Compare multiple models including trained checkpoints
python scripts/evaluate_multiple_models.py \
    --models \
        meta-llama/Llama-3.2-1B-Instruct \
        meta-llama/Llama-3.2-3B-Instruct \
        ./checkpoints/grpo_run_1 \
        medalpaca/medalpaca-7b \
    --scenarios benchmarking_*-hashprompt.jsonl \
    --evaluator gemini \
    --eval_mode batch \
    --pipeline_delay 10
```

### Development and Testing Workflow
```bash
# Quick smoke test with small dataset
python src/data_utils/scenario_generator/standalone_run.py \
    --datasets smoke_test \
    --count 5

# Process test data
python scripts/data_pipeline.py smoke_test_*.jsonl

# Quick evaluation test
python examples/evaluate_baseline.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scenarios smoke_test_*-hashprompt.jsonl \
    --evaluator gemini \
    --test
```

## Project Structure

```
argen-demo/
├── src/data_utils/scenario_generator/    # Scenario generation system
├── scripts/                             # Utility scripts
├── examples/                            # Training and evaluation examples
├── src/reward_functions/                # Custom reward function implementations
├── src/evaluation/                      # Evaluation utilities
├── data/                               # Data files and scenarios
└── docs/                               # Documentation
```

## Configuration Files

### Key Configuration Files
- `src/config.py`: Main configuration including model defaults, GRPO parameters, and reward weights
- `pyproject.toml`: Poetry configuration with all dependencies
- `src/data_utils/scenario_generator/config.py`: Scenario generation specific configuration

### Important Configuration Parameters

**GRPO Training Configuration** (`src/config.py`):
- `GRPO_CONFIG`: Contains training hyperparameters like learning rate, epochs, batch sizes
- `REWARD_WEIGHTS`: Weights for combining Ahimsa, Dharma, and Helpfulness rewards
- `DEFAULT_MODEL_ID`: Default model for training (`meta-llama/Llama-3.2-1B-Instruct`)

**Scenario Generation Configuration**:
- `DATASETS`: Defines dataset types and their default counts
- `DEFAULT_GENERATION_MODEL`: Default OpenAI model for generation
- `RISK_THRESHOLD`: Threshold for medical content filtering

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `generation_batch_size` in evaluation scripts
- Use smaller models or enable gradient checkpointing
- Reduce `per_device_train_batch_size` in GRPO training

**2. API Rate Limiting**
- Reduce `concurrent_eval_limit` and `tiering_concurrency_limit`
- Add delays between API calls
- Use batch evaluation mode for Gemini when possible

**3. Model Loading Issues**
- Ensure sufficient disk space for model downloads
- Check HuggingFace model permissions and access tokens
- Verify model compatibility with transformers version

**4. Training Convergence Issues**
- Adjust learning rate and KL penalty parameters
- Enable early stopping to prevent overfitting
- Monitor reward function outputs for debugging

### Performance Optimization

**For Scenario Generation**:
- Use batch processing with appropriate batch sizes
- Enable concurrent evaluation with optimal limits
- Use local HuggingFace models when possible to avoid API costs

**For Training**:
- Use multiple GPUs with data parallelism
- Enable gradient accumulation for effective larger batch sizes
- Use mixed precision training (fp16) to reduce memory usage

**For Evaluation**:
- Use batch evaluation mode for Gemini evaluator
- Leverage multiple GPUs for parallel model evaluation
- Cache model loading when evaluating multiple scenarios

## Advanced Usage

### Custom Reward Functions
To implement custom reward functions, see `src/reward_functions/` for examples. New reward functions should:
- Follow the TRL reward function interface
- Return scores between 0.0 and 1.0
- Handle batch processing for efficiency
- Include proper error handling and logging

### Custom System Prompts
System prompts can be customized in `src/config.py`. Available types:
- `BASIC`: Simple healthcare assistant prompt
- `ENHANCED`: Detailed prompt with safety guidelines
- `MEDICAL`: Specialized medical consultation prompt

### Extending Dataset Types
New dataset types can be added to `DATASETS` in scenario generator configuration:
```python
DATASETS = {
    "custom_dataset": {
        "count": 200,
        "temperature": 1.0,
        "description": "Custom scenarios for specific use case"
    }
}
```

## Contributing

When contributing to this project:
1. Follow the existing code structure and naming conventions
2. Add comprehensive logging for debugging
3. Include error handling for API calls and file operations
4. Update documentation for any new features or changes
5. Test with small datasets before running full experiments

## License

This project is licensed under the terms specified in the LICENSE file.

For more detailed information about specific components, see the documentation in the `docs/` directory.
