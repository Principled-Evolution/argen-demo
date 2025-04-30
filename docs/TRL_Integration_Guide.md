# TRL Integration Guide for ArGen GRPO Implementation

This document provides technical guidance on integrating Hugging Face's TRL library (v0.17.x) for General Reinforcement Policy Optimization (GRPO) training in the ArGen project.

## 1. TRL Overview

TRL (Transformer Reinforcement Learning) is a library for training transformer language models with reinforcement learning techniques. The GRPO trainer specifically implements a general framework for policy optimization that combines supervised fine-tuning with reinforcement learning.

## 2. Key Components

### 2.1 GRPOTrainer

The `GRPOTrainer` class is the main component we're integrating with, which implements the GRPO algorithm. Key parameters include:

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model=model,               # Model to train
    tokenizer=tokenizer,       # Tokenizer for the model
    reward_model=reward_fn,    # Function that returns rewards
    args=GRPOConfig(**config), # Configuration for GRPO 
)
```

### 2.2 GRPOConfig

The `GRPOConfig` class extends `TrainingArguments` and adds specific parameters for GRPO training:

```python
config = {
    "output_dir": "./grpo-training-output",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-5,
    "gamma": 0.99,                # Discount factor
    "policy_eps": 0.2,            # Clipping epsilon
    "value_eps": 0.2,             # Value clipping epsilon
    "beta": 0.04,                 # KL penalty coefficient
    "reference_model_path": None, # Path to reference model (if different from model)
    # TRL-specific parameters
    "report_to": "wandb",         # Logging integration
}
```

## 3. Reward Function Integration

### 3.1 Reward Function Format

The reward function must have the following signature:

```python
def reward_function(prompts, responses, model_ref=None):
    """
    Calculate rewards for a batch of model responses.
    
    Args:
        prompts: List of input prompts (strings)
        responses: List of model responses (strings)
        model_ref: Optional reference model for calculations
        
    Returns:
        torch.Tensor of shape [batch_size] containing reward values
    """
    # Calculate rewards
    rewards = calculate_rewards(prompts, responses)
    
    # Convert to tensor
    return torch.tensor(rewards)
```

### 3.2 ArGen Reward Adaptation

Our implementation adapts the existing OpenAI-based evaluators to match this format:

```python
import torch
import asyncio
from src.reward_functions.openai_rewards import (
    evaluate_ahimsa_with_openai,
    evaluate_dharma_with_openai
)
from src.config import REWARD_WEIGHTS

async def _evaluate_response(prompt, response, openai_api_key):
    """Evaluate a single response asynchronously."""
    ahimsa_results = await evaluate_ahimsa_with_openai(prompt, response, openai_api_key)
    dharma_results = await evaluate_dharma_with_openai(prompt, response, openai_api_key)
    
    # Calculate combined score
    ahimsa_weight = REWARD_WEIGHTS["ahimsa"]
    dharma_weight = REWARD_WEIGHTS["dharma"]
    combined_score = (
        ahimsa_results.get('ahimsa_score', 0.0) * ahimsa_weight + 
        dharma_results.get('dharma_score', 0.0) * dharma_weight
    ) / (ahimsa_weight + dharma_weight)
    
    return combined_score

def trl_combined_reward_function(prompts, responses, model_ref=None):
    """TRL-compatible reward function for GRPO training."""
    # Create an event loop to run async functions
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Evaluate each prompt-response pair
    tasks = [_evaluate_response(prompt, response, openai_api_key) 
             for prompt, response in zip(prompts, responses)]
    
    # Run evaluations concurrently
    rewards = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    
    # Convert to tensor
    return torch.tensor(rewards, dtype=torch.float32)
```

## 4. Training Implementation

### 4.1 Data Preparation

GRPO training requires a dataset of prompts:

```python
from datasets import Dataset

# Load prompts from scenarios
scenarios = load_scenarios(scenarios_path)
prompts = [scenario["prompt"] for scenario in scenarios]

# Create a simple Dataset
dataset = Dataset.from_dict({"prompt": prompts})
```

### 4.2 Model Loading

Load the model consistent with the baseline evaluation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "unsloth/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### 4.3 Training Loop

The complete training implementation:

```python
from trl import GRPOTrainer, GRPOConfig
import wandb

# Initialize wandb
wandb.init(project="argen-grpo", name=f"grpo-training-{timestamp}")

# Load GRPO configuration
grpo_config = get_grpo_config(output_dir)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_model=trl_combined_reward_function,
    args=GRPOConfig(**grpo_config),
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model(output_dir)
```

## 5. Tensor Shape Considerations

### 5.1 Input Formatting

The `GRPOTrainer` automatically formats inputs using the tokenizer's chat template, so we only need to provide prompts in the dataset. The provided reward function will receive:

- `prompts`: List of input prompts (raw strings) [batch_size]
- `responses`: List of model responses (raw strings) [batch_size]

### 5.2 Reward Output Format

The reward function must return a tensor of shape `[batch_size]` containing the reward values for each example. This is crucial for proper gradient calculation.

## 6. Wandb Integration

The TRL library natively supports wandb integration through the `report_to` parameter in `GRPOConfig`. Key metrics logged include:

- Loss values (policy loss, value loss, total loss)
- Learning rates
- Reward values (mean, min, max)
- KL divergence from reference model
- Training throughput

Additional custom metrics can be logged directly to wandb:

```python
wandb.log({
    "ahimsa_score_avg": avg_ahimsa_score,
    "dharma_score_avg": avg_dharma_score,
    "combined_score_avg": avg_combined_score,
})
```

## 7. Debugging and Troubleshooting

### 7.1 Common Issues

- **Tensor shape mismatch**: Ensure reward function returns tensor of shape `[batch_size]`
- **CUDA out-of-memory**: Reduce batch size or use gradient accumulation
- **API rate limiting**: Implement caching and rate-limit handling for OpenAI calls

### 7.2 Logging

Enable detailed logging to troubleshoot TRL integration issues:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trl")
logger.setLevel(logging.DEBUG)
```

## 8. Best Practices

1. **Start Small**: Begin with a small dataset to validate the integration
2. **Monitor GPU Memory**: Watch memory usage with tools like `nvidia-smi`
3. **API Rate Management**: Cache results and implement rate limiting for OpenAI calls
4. **Checkpointing**: Save checkpoints frequently to resume interrupted training
5. **Evaluation**: Regularly evaluate the model during training to track progress

## 9. References

- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [GRPO Paper](https://arxiv.org/abs/2306.13439)
- [TRL GitHub Repository](https://github.com/huggingface/trl) 