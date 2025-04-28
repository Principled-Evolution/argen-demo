"""
Configuration for Predibase GRPO fine-tuning with Gemini reward functions.
"""

import os
from typing import Dict, List, Optional, Callable, Any

from src.utils.env import load_env_vars


def create_gemini_grpo_config(
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
    Create a configuration for Predibase GRPO fine-tuning with Gemini reward functions.
    
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
    
    # Load environment variables
    load_env_vars()
    
    # Set up reward functions configuration
    if reward_functions is None:
        # Import Gemini reward functions
        from ..reward_functions.openai_rewards import gemini_ahimsa_reward, gemini_dharma_reward
        reward_functions = {
            "ahimsa": gemini_ahimsa_reward,
            "dharma": gemini_dharma_reward,
        }
    
    # Set up required packages
    if external_packages is None:
        external_packages = ["google-generativeai", "python-dotenv"]
    else:
        if "google-generativeai" not in external_packages:
            external_packages.append("google-generativeai")
        if "python-dotenv" not in external_packages:
            external_packages.append("python-dotenv")
    
    # Set up environment variables
    env_vars = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")
    }
    
    # Create runtime configuration
    runtime_config = RewardFunctionsRuntimeConfig(
        packages=external_packages,
        env_vars=env_vars
    )
    
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


def submit_gemini_grpo_job(
    config: Dict,
    dataset: str,
    repo: str,
    description: str = "ArGen GRPO fine-tuning with Gemini reward functions",
) -> str:
    """
    Submit a GRPO fine-tuning job to Predibase.
    
    Args:
        config: The GRPO configuration
        dataset: The name of the dataset in Predibase
        repo: The name of the repository to save the adapter to
        description: A description of the fine-tuning job
        
    Returns:
        The job ID
    """
    try:
        from predibase import Predibase
    except ImportError:
        raise ImportError(
            "Predibase SDK not installed. Please install it with 'pip install predibase'."
        )
    
    # Get API token from config file
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                raise ValueError("API token not found in config file")
    else:
        raise ValueError("Config file not found")
    
    # Initialize Predibase client
    pb = Predibase(api_token=api_token)
    
    # Submit the job
    job = pb.fine_tunings.create(
        config=config,
        dataset=dataset,
        repo=repo,
        description=description
    )
    
    return job.id
