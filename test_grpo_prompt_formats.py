#!/usr/bin/env python3
"""
Test script to compare GRPO trainer performance with instruct vs. chat templates.
This script loads the debug dataset and runs the GRPO trainer with both formats.

Usage:
    python test_grpo_prompt_formats.py
"""

import os
import sys
import json
import logging
import torch
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "test_grpo_prompt_formats.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import from project
from src.config import get_system_prompt
from fix_grpo_chat_template import CustomGRPOTrainer, extract_content_from_chat_response

# Constants
DEBUG_DATASET_PATH = "data/debug-data-set/debug-garbled-chat-template-response.jsonl"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR_BASE = "test_grpo_output"

def load_debug_dataset():
    """Load the debug dataset."""
    logger.info(f"Loading dataset from {DEBUG_DATASET_PATH}")
    
    try:
        dataset = load_dataset("json", data_files=DEBUG_DATASET_PATH)
        logger.info(f"Dataset loaded successfully with {len(dataset['train'])} examples")
        
        # Log a few examples
        for i, example in enumerate(dataset["train"][:2]):
            logger.info(f"Example {i}: {json.dumps(example, indent=2)}")
            
        return dataset["train"]
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def prepare_instruct_dataset(dataset, system_prompt):
    """
    Prepare dataset with instruct-style prompts.
    This prepends the system prompt to each user prompt.
    """
    logger.info("Preparing instruct-style dataset")
    
    instruct_data = {
        "prompt": [],
        "id": [],
        "tier": [],
        "scope": []
    }
    
    for item in dataset:
        # Create the full instruct prompt by prepending the system prompt
        user_question = item["prompt"]
        full_instruct_prompt = f"{system_prompt}\n\nUser question: {user_question}\n\nAnswer:"
        
        # Add to the new dataset
        instruct_data["prompt"].append(full_instruct_prompt)
        instruct_data["id"].append(item["id"])
        instruct_data["tier"].append(item["tier"])
        instruct_data["scope"].append(item["scope"])
    
    return Dataset.from_dict(instruct_data)

def prepare_chat_dataset(dataset, system_prompt):
    """
    Prepare dataset with chat-style prompts.
    This converts each prompt to a list of message dictionaries.
    """
    logger.info("Preparing chat-style dataset")
    
    chat_data = {
        "prompt": [],
        "id": [],
        "tier": [],
        "scope": []
    }
    
    for item in dataset:
        # Create the chat-style prompt as a list of message dictionaries
        chat_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["prompt"]}
        ]
        
        # Add to the new dataset
        chat_data["prompt"].append(chat_prompt)
        chat_data["id"].append(item["id"])
        chat_data["tier"].append(item["tier"])
        chat_data["scope"].append(item["scope"])
    
    return Dataset.from_dict(chat_data)

def simple_reward_func(completions, **kwargs):
    """Simple reward function that rewards longer completions."""
    logger.info(f"Completions type: {type(completions)}")
    logger.info(f"Completions: {completions[:2]}")  # Log first 2 completions
    
    # Process the completions
    processed_completions = []
    for completion_group in completions:
        processed_group = []
        for completion in completion_group:
            processed_completion = extract_content_from_chat_response(completion)
            processed_group.append(processed_completion)
        processed_completions.extend(processed_group)
    
    # Calculate rewards
    rewards = []
    for completion in processed_completions:
        # Simple reward based on length and coherence
        reward = min(len(completion.split()), 50) / 50.0
        rewards.append(reward)
    
    logger.info(f"Rewards calculated: {rewards}")
    return rewards

def run_grpo_trainer(dataset, output_dir, use_custom_trainer=True, is_chat_format=False):
    """
    Run the GRPO trainer on the given dataset.
    
    Args:
        dataset: The dataset to train on
        output_dir: The output directory for the trainer
        use_custom_trainer: Whether to use the CustomGRPOTrainer
        is_chat_format: Whether the dataset contains chat-formatted prompts
    """
    logger.info(f"Running GRPO trainer with {'chat' if is_chat_format else 'instruct'} format")
    logger.info(f"Using {'CustomGRPOTrainer' if use_custom_trainer else 'GRPOTrainer'}")
    
    # Initialize model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        logger.info(f"Model and tokenizer loaded successfully")
        logger.info(f"Tokenizer class: {tokenizer.__class__.__name__}")
        logger.info(f"Tokenizer chat template: {tokenizer.chat_template}")
        
        # Test basic chat template functionality
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        formatted = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        logger.info(f"Test formatted prompt:\n{formatted}")
        
    except Exception as e:
        logger.error(f"Error initializing model or tokenizer: {e}")
        raise
    
    # Create a minimal training configuration
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=5,
        logging_steps=1,
        save_steps=5,
        learning_rate=5e-5,
        seed=42,
        num_generations=2,  # Number of completions per prompt
        max_prompt_length=256,
        max_completion_length=256,
        temperature=0.7,
        log_completions=True,
        report_to=["none"],  # Disable wandb/tensorboard for debugging
    )
    
    # Initialize the trainer
    try:
        if use_custom_trainer:
            trainer_class = CustomGRPOTrainer
        else:
            trainer_class = GRPOTrainer
            
        logger.info(f"Initializing {trainer_class.__name__}")
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=dataset,
            reward_funcs=simple_reward_func,
            processing_class=tokenizer,  # Pass tokenizer as processing_class
        )
        
        # Run a minimal training loop
        logger.info("Starting minimal training run...")
        trainer.train()
        logger.info("Training completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("Starting GRPO prompt format test script")
    
    # Load the debug dataset
    dataset = load_debug_dataset()
    
    # Get the system prompt
    system_prompt = get_system_prompt(use_basic_prompt=False)  # Use enhanced prompt
    logger.info(f"System prompt: {system_prompt}")
    
    # Prepare datasets with different formats
    instruct_dataset = prepare_instruct_dataset(dataset, system_prompt)
    chat_dataset = prepare_chat_dataset(dataset, system_prompt)
    
    logger.info(f"Instruct dataset example: {instruct_dataset[0]}")
    logger.info(f"Chat dataset example: {chat_dataset[0]}")
    
    # Run GRPO trainer with instruct format
    instruct_output_dir = f"{OUTPUT_DIR_BASE}/instruct"
    instruct_success = run_grpo_trainer(
        instruct_dataset, 
        instruct_output_dir, 
        use_custom_trainer=True,
        is_chat_format=False
    )
    
    # Run GRPO trainer with chat format
    chat_output_dir = f"{OUTPUT_DIR_BASE}/chat"
    chat_success = run_grpo_trainer(
        chat_dataset, 
        chat_output_dir, 
        use_custom_trainer=True,
        is_chat_format=True
    )
    
    # Report results
    logger.info("=== Test Results ===")
    logger.info(f"Instruct format: {'SUCCESS' if instruct_success else 'FAILED'}")
    logger.info(f"Chat format: {'SUCCESS' if chat_success else 'FAILED'}")
    
    if instruct_success and chat_success:
        logger.info("Both formats completed successfully. Check logs for details.")
    elif instruct_success:
        logger.info("Only instruct format completed successfully.")
    elif chat_success:
        logger.info("Only chat format completed successfully.")
    else:
        logger.info("Both formats failed. Check logs for details.")

if __name__ == "__main__":
    main()
