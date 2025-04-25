"""
Script to run GRPO fine-tuning with Gemini reward functions.
"""

import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.env import load_env_vars
from src.predibase.gemini_config import create_gemini_grpo_config, submit_gemini_grpo_job


def main():
    """Run the GRPO fine-tuning script with Gemini reward functions."""
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning with Gemini reward functions.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the base model")
    parser.add_argument("--dataset", type=str, default="combined_predibase", help="Name of the dataset in Predibase")
    parser.add_argument("--repo", type=str, default="argen-gemini", help="Name of the repository to save the adapter to")
    parser.add_argument("--description", type=str, default="ArGen GRPO fine-tuning with Gemini reward functions", help="Description of the fine-tuning job")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env_vars()
    
    # Check if GEMINI_API_KEY is set
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it in your environment or in a .env file."
        )
    
    print(f"Running GRPO fine-tuning with Gemini reward functions...")
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Repository: {args.repo}")
    
    # Prepare the combined datasets if they don't exist
    if not os.path.exists("data/combined_predibase.jsonl"):
        print("Preparing combined datasets...")
        os.system("python examples/prepare_combined_datasets.py")
    
    # Create the GRPO configuration
    config = create_gemini_grpo_config(
        base_model=args.model,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Submit the GRPO job
    try:
        job_id = submit_gemini_grpo_job(
            config=config,
            dataset=args.dataset,
            repo=args.repo,
            description=args.description
        )
        
        print(f"GRPO job submitted successfully!")
        print(f"Job ID: {job_id}")
        print(f"You can monitor the job progress in the Predibase UI.")
    except Exception as e:
        print(f"Error submitting GRPO job: {e}")
        print("Make sure you have set up the Predibase SDK and have the necessary credentials.")


if __name__ == "__main__":
    main()
