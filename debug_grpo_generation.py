#!/usr/bin/env python3
"""
Debug script to diagnose issues with GRPO training and chat templates.
"""

import os
import sys
import logging
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Import our custom debug callback
from debug_callback import GRPOGenerationDebugCallback

# Configure logging
log_dir = Path("debug_logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "debug_grpo_main.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def simple_reward_func(completions, **kwargs):
    """Simple reward function that rewards longer completions."""
    logger.info(f"Completions type: {type(completions)}")
    logger.info(f"Completions: {completions[:2]}")  # Log first 2 completions

    # Handle different types of completions
    if isinstance(completions, list):
        if completions and isinstance(completions[0], list):
            # If completions is a list of lists, flatten it
            flat_completions = []
            for completion_group in completions:
                flat_completions.extend(completion_group)
            completions = flat_completions

    # Calculate rewards
    rewards = []
    for completion in completions:
        if isinstance(completion, str):
            reward = min(len(completion.split()), 50) / 50.0
        else:
            # If completion is not a string, give a default reward
            logger.warning(f"Non-string completion: {completion}")
            reward = 0.5
        rewards.append(reward)

    logger.info(f"Rewards calculated: {rewards}")
    return rewards

def main():
    logger.info("Starting GRPO generation debug script")

    # Load the debug dataset
    dataset_path = "data/debug-data-set/debug-garbled-chat-template-response.jsonl"
    logger.info(f"Loading dataset from {dataset_path}")

    try:
        dataset = load_dataset("json", data_files=dataset_path)
        logger.info(f"Dataset loaded successfully with {len(dataset['train'])} examples")

        # Log a few examples
        for i, example in enumerate(dataset["train"][:2]):
            logger.info(f"Example {i}: {json.dumps(example, indent=2)}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Use your model path
    logger.info(f"Loading model and tokenizer from {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
            device_map="auto"  # Automatically place model on available devices
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

        # Save the formatted prompt to a file for inspection
        with open(log_dir / "test_formatted_prompt.txt", "w") as f:
            f.write(formatted)

    except Exception as e:
        logger.error(f"Error initializing model or tokenizer: {e}")
        return

    # Prepare the dataset for GRPO training
    # Convert string prompts to message format if needed
    def preprocess_dataset(examples):
        processed_examples = {}
        for key, value in examples.items():
            if key == "prompt" and isinstance(value, str):
                processed_examples[key] = [{"role": "user", "content": value}]
            else:
                processed_examples[key] = value
        return processed_examples

    # Apply preprocessing if needed
    if isinstance(dataset["train"][0]["prompt"], str):
        logger.info("Converting string prompts to message format")
        dataset["train"] = dataset["train"].map(preprocess_dataset)

    # Create a minimal training configuration
    training_args = GRPOConfig(
        output_dir="debug_grpo_output",
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

    # Initialize the trainer with our debug callback
    debug_callback = GRPOGenerationDebugCallback(log_interval=1, num_samples=2, log_dir=log_dir)

    try:
        logger.info("Initializing GRPOTrainer")
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            reward_funcs=simple_reward_func,
            processing_class=tokenizer,  # Pass tokenizer as processing_class
            callbacks=[debug_callback],
        )

        # Run a minimal training loop
        logger.info("Starting minimal training run...")
        trainer.train()
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
