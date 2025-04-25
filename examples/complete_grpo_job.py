"""
Script to create a complete GRPO job in Predibase with Gemini-OPA reward functions.
"""

import os
import json
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.env import load_env_vars


def main():
    """Create a complete GRPO job in Predibase."""
    parser = argparse.ArgumentParser(description="Create a complete GRPO job in Predibase.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the base model")
    parser.add_argument("--dataset", type=str, default="argen_combined_dataset", help="Name of the dataset in Predibase")
    parser.add_argument("--repo", type=str, default="argen-gemini-opa", help="Name of the repository to save the adapter to")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    
    args = parser.parse_args()
    
    # Load environment variables
    print("Loading environment variables...")
    load_env_vars()
    
    # Get Gemini API key
    print("Getting Gemini API key...")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        print("GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    print("Gemini API key found.")
    
    # Get API token from config file
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                print("API token not found in config file")
                sys.exit(1)
    else:
        print("Config file not found")
        sys.exit(1)
    
    try:
        from predibase import Predibase
    except ImportError:
        print("Predibase SDK not installed. Please install it with 'pip install predibase'.")
        sys.exit(1)
    
    # Initialize Predibase client
    print("Initializing Predibase client...")
    pb = Predibase(api_token=api_token)
    print("Connected to Predibase.")
    
    # Import reward function code as strings
    print("Loading reward function code...")
    reward_functions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "reward_functions")
    
    with open(os.path.join(reward_functions_dir, "gemini_rewards.py"), "r") as f:
        gemini_rewards_code = f.read()
    
    with open(os.path.join(reward_functions_dir, "gemini_opa_rewards.py"), "r") as f:
        gemini_opa_rewards_code = f.read()
    
    # Create a complete GRPO configuration
    print("Creating GRPO configuration...")
    
    # Complete configuration with all necessary parameters
    config = {
        "method": "grpo",  # Specify GRPO method
        "base_model": args.model,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "reward_functions": {
            "code": gemini_rewards_code + "\n\n" + gemini_opa_rewards_code,
            "functions": {
                "ahimsa": "gemini_opa_ahimsa_reward",
                "dharma": "gemini_opa_dharma_reward"
            },
            "runtime": {
                "packages": ["google-generativeai", "python-dotenv"],
                "env_vars": {
                    "GEMINI_API_KEY": gemini_api_key
                }
            }
        }
    }
    
    # Submit the job
    print("Submitting GRPO job to Predibase...")
    try:
        job = pb.finetuning.jobs.create(
            config=config,
            dataset=args.dataset,
            repo=args.repo,
            description="ArGen GRPO fine-tuning with Gemini-OPA reward functions"
        )
        
        print("GRPO job submitted successfully!")
        print(f"Job details: {job}")
    except Exception as e:
        print(f"Error creating GRPO job: {e}")


if __name__ == "__main__":
    main()
