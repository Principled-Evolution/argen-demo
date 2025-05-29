"""
Configuration utilities for Predibase GRPO fine-tuning.
"""

from typing import Dict, List, Optional, Union, Callable
import os


def create_grpo_config(
    base_model: str = "llama-3-2-1b-instruct",
    reward_functions: Optional[Dict[str, Callable]] = None,
    learning_rate: float = 5e-5,
    epochs: int = 3,
    max_steps: Optional[int] = None,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    external_packages: Optional[List[str]] = None,
) -> Dict:
    """
    Create a configuration for Predibase GRPO fine-tuning.

    Args:
        base_model: The base model to fine-tune
        reward_functions: Dictionary mapping reward function names to callables
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs to train for
        max_steps: Maximum number of steps to train for (overrides epochs if provided)
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        external_packages: List of external packages required by reward functions

    Returns:
        Dictionary containing the Predibase GRPO configuration
    """
    try:
        from predibase import GRPOConfig, RewardFunctionsConfig, RewardFunctionsRuntimeConfig
    except ImportError:
        raise ImportError(
            "Predibase SDK not installed. Please install it with 'pip install predibase'."
        )

    # Set up reward functions configuration
    if reward_functions is None:
        # Import default reward functions if none provided
        from ..reward_functions.ahimsa_strict import ahimsa_strict_reward
        from ..reward_functions.dharma_domain import dharma_domain_reward
        reward_functions = {
            "ahimsa": ahimsa_strict_reward,
            "dharma": dharma_domain_reward,
        }

    # Set up runtime configuration if external packages are needed
    runtime_config = None
    if external_packages:
        runtime_config = RewardFunctionsRuntimeConfig(packages=external_packages)

    # Create reward functions configuration
    reward_fns_config = RewardFunctionsConfig(
        functions=reward_functions,
        runtime=runtime_config,
    )

    # Create GRPO configuration
    grpo_config = GRPOConfig(
        base_model=base_model,
        reward_fns=reward_fns_config,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    return grpo_config


def submit_grpo_job(
    config: Dict,
    dataset: str,
    repo: str,
    description: str = "ArGen GRPO fine-tuning",
) -> str:
    """
    Submit a GRPO fine-tuning job to Predibase.

    Args:
        config: Predibase GRPO configuration
        dataset: Name of the dataset to use for fine-tuning
        repo: Name of the repository to save the adapter to
        description: Description of the fine-tuning job

    Returns:
        ID of the submitted job
    """
    try:
        import predibase as pb
    except ImportError:
        raise ImportError(
            "Predibase SDK not installed. Please install it with 'pip install predibase'."
        )

    # Submit the job
    adapter = pb.adapters.create(
        config=config,
        dataset=dataset,
        repo=repo,
        description=description,
    )

    return adapter.id
