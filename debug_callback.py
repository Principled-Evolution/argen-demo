#!/usr/bin/env python3
"""
Debug callback for GRPO training to diagnose issues with chat templates and generation.
"""

import logging
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class GRPOGenerationDebugCallback(TrainerCallback):
    """Callback to debug generation issues in GRPO training."""
    
    def __init__(self, log_interval=1, num_samples=2, log_dir="debug_logs"):
        self.log_interval = log_interval
        self.num_samples = num_samples
        self.generation_count = 0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create a separate log file for the callback
        self.debug_log_file = self.log_dir / "grpo_generation_debug.log"
        self.file_handler = logging.FileHandler(self.debug_log_file)
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        logger.addHandler(self.file_handler)
        
        logger.info(f"GRPOGenerationDebugCallback initialized. Debug logs will be saved to {self.debug_log_file}")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Log input prompts before generation."""
        if state.global_step % self.log_interval != 0:
            return
            
        trainer = kwargs.get('trainer')
        if not trainer:
            return
            
        # Get tokenizer
        tokenizer = trainer.processing_class
        
        # Get a sample from the dataset
        if hasattr(trainer, 'train_dataset') and trainer.train_dataset is not None:
            try:
                # Get a few samples
                samples = trainer.train_dataset.select(range(min(self.num_samples, len(trainer.train_dataset))))
                
                logger.info(f"===== STEP {state.global_step} INPUT PROMPTS =====")
                
                for i, sample in enumerate(samples):
                    # Log the raw prompt
                    logger.info(f"Sample {i} raw prompt: {sample['prompt']}")
                    
                    # Try to format it as the model would see it
                    if isinstance(sample['prompt'], str):
                        # For string prompts, create a simple message format
                        messages = [{"role": "user", "content": sample['prompt']}]
                    else:
                        # Assume it's already in message format
                        messages = sample['prompt']
                    
                    # Add system prompt if not present
                    if not any(msg.get('role') == 'system' for msg in messages):
                        system_prompt = getattr(args, 'system_prompt', "You are a helpful assistant.")
                        messages = [{"role": "system", "content": system_prompt}] + messages
                    
                    # Format with chat template
                    try:
                        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        logger.info(f"Sample {i} formatted prompt:\n{formatted}")
                        
                        # Also log tokenized version
                        tokens = tokenizer.encode(formatted)
                        logger.info(f"Sample {i} token count: {len(tokens)}")
                        logger.info(f"Sample {i} first 10 tokens: {tokens[:10]}")
                        logger.info(f"Sample {i} last 10 tokens: {tokens[-10:]}")
                        
                        # Save the formatted prompt to a file for inspection
                        prompt_file = self.log_dir / f"step_{state.global_step}_sample_{i}_prompt.txt"
                        with open(prompt_file, 'w') as f:
                            f.write(formatted)
                        
                    except Exception as e:
                        logger.error(f"Error formatting prompt: {e}")
            except Exception as e:
                logger.error(f"Error accessing dataset: {e}")
    
    def on_generate(self, args, state, control, generated_outputs=None, **kwargs):
        """Capture generation outputs directly."""
        if state.global_step % self.log_interval != 0:
            return
            
        if generated_outputs:
            logger.info(f"===== STEP {state.global_step} DIRECT GENERATION OUTPUTS =====")
            try:
                # Log the raw generation outputs
                logger.info(f"Generated outputs type: {type(generated_outputs)}")
                logger.info(f"Generated outputs structure: {generated_outputs[:2] if isinstance(generated_outputs, list) else 'Not a list'}")
                
                # Save the outputs to a file
                output_file = self.log_dir / f"step_{state.global_step}_generation_outputs.json"
                try:
                    with open(output_file, 'w') as f:
                        json.dump(generated_outputs, f, indent=2, default=str)
                except:
                    logger.error("Could not serialize generation outputs to JSON")
            except Exception as e:
                logger.error(f"Error processing generation outputs: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log generation outputs from logs."""
        if not logs or state.global_step % self.log_interval != 0:
            return
            
        logger.info(f"===== STEP {state.global_step} LOG CONTENTS =====")
        logger.info(f"Log keys: {logs.keys()}")
        
        # Check if we have generation outputs in the logs
        if 'completions' in logs:
            logger.info(f"===== STEP {state.global_step} LOGGED COMPLETIONS =====")
            completions = logs['completions']
            for i, completion in enumerate(completions[:self.num_samples]):
                logger.info(f"Completion {i}: {completion}")
                
                # Save the completion to a file
                completion_file = self.log_dir / f"step_{state.global_step}_completion_{i}.txt"
                with open(completion_file, 'w') as f:
                    f.write(completion)
                
        # Log reward values if available
        if 'rewards' in logs:
            logger.info(f"===== STEP {state.global_step} REWARDS =====")
            rewards = logs['rewards']
            for i, reward in enumerate(rewards[:self.num_samples]):
                logger.info(f"Reward {i}: {reward}")
        
        # Save all logs to a file
        log_file = self.log_dir / f"step_{state.global_step}_logs.json"
        try:
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
        except:
            logger.error("Could not serialize logs to JSON")
