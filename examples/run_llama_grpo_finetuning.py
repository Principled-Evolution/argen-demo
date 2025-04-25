"""
Script to run GRPO fine-tuning with Predibase on the llama-3-2-1b-instruct model.
"""

import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predibase.config import create_grpo_config, submit_grpo_job


def main():
    """Run the GRPO fine-tuning script for llama-3-2-1b-instruct."""
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning with Predibase on the llama-3-2-1b-instruct model.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the base model")
    parser.add_argument("--dataset", type=str, default="challenging_predibase", help="Name of the dataset in Predibase")
    parser.add_argument("--repo", type=str, default="argen-ahimsa-llama", help="Name of the repository to save the adapter to")
    parser.add_argument("--description", type=str, default="ArGen GRPO fine-tuning with strict Ahimsa reward on llama-3-2-1b-instruct", help="Description of the fine-tuning job")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    
    args = parser.parse_args()
    
    print(f"Running GRPO fine-tuning with Predibase...")
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Repository: {args.repo}")
    
    # Prepare the challenging datasets if they don't exist
    if not os.path.exists("data/challenging_predibase.jsonl"):
        print("Preparing challenging datasets...")
        from src.data_utils.prepare_challenging_dataset import create_predibase_dataset
        create_predibase_dataset("data/challenging_predibase.jsonl")
    
    # Create the GRPO configuration
    config = create_grpo_config(
        base_model=args.model,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Submit the GRPO job
    try:
        job_id = submit_grpo_job(
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
