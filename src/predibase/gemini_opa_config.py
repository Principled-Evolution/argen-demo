"""
Configuration for Predibase GRPO fine-tuning with Gemini → OPA → Reward flow.
Based on documentation at https://docs.predibase.com/user-guide/fine-tuning/grpo
"""

import os
import json
from typing import Dict, List, Optional, Callable, Any

from src.utils.env import load_env_vars


def create_gemini_opa_grpo_config(
    base_model: str = "llama-3-2-1b-instruct",
    learning_rate: float = 5e-5,
    epochs: int = 3,
    max_steps: Optional[int] = None,
    batch_size: int = 8,
) -> Dict:
    """
    Create a configuration for Predibase GRPO fine-tuning with Gemini → OPA → Reward flow.

    Args:
        base_model: The base model to fine-tune
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs to train for
        max_steps: Maximum number of steps to train for (overrides epochs if provided)
        batch_size: Batch size for training

    Returns:
        Dictionary containing the Predibase GRPO configuration
    """
    # Load environment variables
    load_env_vars()

    # Get Gemini API key
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Import reward function code as strings
    reward_functions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reward_functions")

    with open(os.path.join(reward_functions_dir, "gemini_rewards.py"), "r") as f:
        gemini_rewards_code = f.read()

    with open(os.path.join(reward_functions_dir, "gemini_opa_rewards.py"), "r") as f:
        gemini_opa_rewards_code = f.read()

    # Import the GRPOConfig class
    from predibase import GRPOConfig, RewardFunction, RewardFunctionsConfig, RewardFunctionsRuntimeConfig

    # Create reward functions
    reward_functions = RewardFunctionsConfig(
        code=gemini_rewards_code + "\n\n" + gemini_opa_rewards_code,
        functions={
            "ahimsa": RewardFunction(name="gemini_opa_ahimsa_reward"),
            "dharma": RewardFunction(name="gemini_opa_dharma_reward")
        },
        runtime=RewardFunctionsRuntimeConfig(
            packages=["google-generativeai", "python-dotenv"],
            env_vars={"GEMINI_API_KEY": gemini_api_key}
        )
    )

    # Create GRPO configuration using the Predibase GRPOConfig class
    config = GRPOConfig(
        base_model=base_model,
        learning_rate=learning_rate,
        epochs=epochs,
        effective_batch_size=batch_size
    )

    # Add max_steps if provided
    if max_steps is not None:
        config.max_steps = max_steps

    # Add reward functions
    config.reward_functions = reward_functions

    # Convert to dictionary for submission
    config_dict = config.model_dump()

    return config_dict


def submit_gemini_opa_grpo_job(
    config: Dict,
    dataset: str,
    repo: str,
    description: str = "ArGen GRPO fine-tuning with Gemini → OPA → Reward flow",
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
    # Updated to use the latest Predibase API
    job = pb.finetuning.jobs.create(
        config=config,
        dataset=dataset,
        repo=repo,
        description=description
    )

    return job.id
