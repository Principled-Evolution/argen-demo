#!/usr/bin/env python3
"""
Script to train a model using GRPO with TRL, matching the evaluation parameters
of the baseline model.
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional
import datetime
import logging
import wandb

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

from src.utils.env import load_env_vars, get_openai_api_key
from src.config import (
    DEFAULT_MODEL_ID, 
    DEFAULT_SCENARIOS_PATH,
    get_system_prompt,
    get_grpo_config,
    REWARD_WEIGHTS
)
from src.reward_functions.trl_rewards import (
    ahimsa_reward_trl,
    dharma_reward_trl,
    combined_reward_trl
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if necessary dependencies for GRPO training are installed."""
    try:
        import torch
        import transformers
        import trl
        print("torch, transformers, and trl libraries found.")
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, will use CPU (might be slow).")
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}. Please install required libraries.")
        print("Hint: pip install torch transformers trl")
        return False
        
def load_scenarios(file_path: str) -> List[Dict]:
    """
    Load scenarios from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of scenarios
    """
    scenarios = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    
    return scenarios

def prepare_dataset(scenarios_path: str) -> 'Dataset':
    """
    Prepare dataset for GRPO training.
    
    Args:
        scenarios_path: Path to the scenarios file
        
    Returns:
        Dataset ready for GRPO training
    """
    # Load scenarios
    scenarios = load_scenarios(scenarios_path)
    
    # Extract prompts from scenarios
    dataset_dict = {
        "prompt": [scenario["prompt"] for scenario in scenarios],
    }
    
    # Add any other fields that might be in the scenarios
    for key in scenarios[0].keys():
        if key != "prompt":
            dataset_dict[key] = [scenario.get(key) for scenario in scenarios]
    
    # Create and return the dataset
    return Dataset.from_dict(dataset_dict)

def main():
    """Run the GRPO training script."""
    parser = argparse.ArgumentParser(description="Train a model using GRPO with TRL.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Name/identifier of the model to fine-tune (e.g., HF identifier)"
    )
    parser.add_argument(
        "--scenarios", 
        type=str,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the scenarios file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="argen-grpo",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--use_basic_prompt",
        action="store_true",
        help="Use the basic system prompt instead of the enhanced one"
    )
    parser.add_argument(
        "--use_separate_rewards",
        action="store_true",
        help="Use separate Ahimsa and Dharma reward functions instead of the combined one"
    )
    
    args = parser.parse_args()
    
    # Load environment variables early
    load_env_vars()
    
    # Get the OpenAI API key
    openai_api_key = get_openai_api_key()
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file or environment.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Prepare the dataset
    print(f"Loading scenarios from {args.scenarios}...")
    if not os.path.exists(args.scenarios):
        print(f"Scenarios file {args.scenarios} not found. Preparing combined datasets...")
        os.system("python examples/prepare_combined_datasets.py")
    
    # Prepare the dataset for GRPO
    train_dataset = prepare_dataset(args.scenarios)
    
    # Get GRPO config with model name
    grpo_config = get_grpo_config(args.model)
    
    # Override config with command line arguments if provided
    if args.output_dir:
        grpo_config["output_dir"] = args.output_dir
    if args.num_train_epochs:
        grpo_config["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate:
        grpo_config["learning_rate"] = args.learning_rate
    
    # Create run name with timestamp if not provided
    if not args.wandb_run_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = args.model.split("/")[-1] if "/" in args.model else args.model
        args.wandb_run_name = f"{model_short_name}-grpo-{timestamp}"
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": args.model,
            "scenarios": args.scenarios,
            "use_basic_prompt": args.use_basic_prompt,
            "use_separate_rewards": args.use_separate_rewards,
            "ahimsa_weight": REWARD_WEIGHTS["ahimsa"],
            "dharma_weight": REWARD_WEIGHTS["dharma"],
            **grpo_config
        }
    )
    
    # Choose reward function based on argument
    if args.use_separate_rewards:
        reward_funcs = [ahimsa_reward_trl, dharma_reward_trl]
        reward_weights = [REWARD_WEIGHTS["ahimsa"], REWARD_WEIGHTS["dharma"]]
    else:
        reward_funcs = combined_reward_trl
        reward_weights = None
    
    # Create GRPO config
    trl_config = GRPOConfig(
        output_dir=grpo_config["output_dir"],
        num_train_epochs=grpo_config["num_train_epochs"],
        learning_rate=grpo_config["learning_rate"],
        gradient_accumulation_steps=grpo_config["gradient_accumulation_steps"],
        per_device_train_batch_size=grpo_config["per_device_train_batch_size"],
        max_prompt_length=grpo_config["max_prompt_length"],
        max_completion_length=grpo_config["max_completion_length"],
        num_generations=grpo_config["num_generations"],
        mini_repeat_count=grpo_config["mini_repeat_count"],
        beta=grpo_config["beta"],
        disable_dropout=grpo_config["disable_dropout"],
        warmup_steps=grpo_config["warmup_steps"],
        logging_steps=grpo_config["logging_steps"],
        fp16=grpo_config["fp16"],
        reward_weights=reward_weights,
        report_to=["wandb"]
    )
    
    # Add system prompt to model init kwargs
    model_init_kwargs = {
        "torch_dtype": "bfloat16",
        "system_prompt": get_system_prompt(args.use_basic_prompt)
    }
    trl_config.model_init_kwargs = model_init_kwargs
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for GRPO
    
    # Initialize trainer
    print(f"Initializing GRPOTrainer for model {args.model}...")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        args=trl_config,
        train_dataset=train_dataset,
        processing_class=tokenizer
    )
    
    # Training
    print("Starting GRPO training...")
    trainer.train()
    
    # Save the model
    print(f"Saving trained model to {trl_config.output_dir}...")
    trainer.save_model()
    
    print("GRPO training complete!")
    wandb.finish()

if __name__ == "__main__":
    main() 