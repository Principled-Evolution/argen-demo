#!/usr/bin/env python3
"""
Train Model Command

This command trains language models using GRPO (Generalized Reward Policy Optimization) 
with custom healthcare reward functions.

Key features:
- TRL (Transformers Reinforcement Learning) integration
- Custom reward functions: Ahimsa (safety), Dharma (ethics), Helpfulness
- Weights & Biases integration for experiment tracking
- Configurable training parameters and evaluation during training
- Support for both combined and separate reward functions
- Early stopping and adaptive learning rate scheduling

Usage:
    python commands/train_model.py --model meta-llama/Llama-3.2-1B-Instruct --scenarios data/grpo_training_scenarios-hashprompt.jsonl --output_dir ./checkpoints

Example:
    python commands/train_model.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --scenarios data/grpo_training_scenarios-hashprompt.jsonl \
        --eval_scenarios data/eval_scenarios-hashprompt.jsonl \
        --output_dir ./checkpoints/grpo_run_1 \
        --num_train_epochs 3 \
        --learning_rate 3.2e-6 \
        --use_separate_rewards \
        --wandb_project my-grpo-experiment \
        --evaluator gemini \
        --early_stopping
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main():
    """
    Main entry point for the training command.
    This imports and runs the original train_grpo.py functionality.
    """
    # Import the original training script functionality
    # We need to temporarily modify sys.argv to make it look like we're running the original script
    original_argv = sys.argv.copy()
    
    # Replace the script name to match what the original script expects
    sys.argv[0] = os.path.join(project_root, 'examples', 'train_grpo.py')
    
    try:
        # Import and run the original training functionality
        # We'll import the main components from the original script
        from examples.train_grpo import main as train_grpo_main
        
        # Run the original main function
        return train_grpo_main()
        
    except ImportError:
        # If the import fails, we'll run the script directly
        import subprocess
        
        # Build the command to run the original script
        cmd = [sys.executable, os.path.join(project_root, 'examples', 'train_grpo.py')] + sys.argv[1:]
        
        # Run the original script
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
        
    finally:
        # Restore the original argv
        sys.argv = original_argv

if __name__ == "__main__":
    sys.exit(main())
