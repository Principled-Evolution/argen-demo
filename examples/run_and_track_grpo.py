"""
Script to run GRPO with Gemini-OPA reward functions and track progress.

This script:
1. Submits a GRPO job to Predibase
2. Tracks the progress of the job
3. Logs detailed metrics at each step
4. Generates a comparison report against baseline results
"""

import sys
import os
import json
import time
import argparse
import datetime
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.env import load_env_vars
from src.predibase.gemini_opa_config import create_gemini_opa_grpo_config


def setup_logging(log_dir: str = "logs") -> str:
    """
    Set up logging directory and file.

    Args:
        log_dir: Directory to store logs

    Returns:
        Path to the log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"grpo_run_{timestamp}.log")

    return log_file


def log_message(log_file: str, message: str) -> None:
    """
    Log a message to both console and log file.

    Args:
        log_file: Path to the log file
        message: Message to log
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format message with timestamp
    formatted_message = f"[{timestamp}] {message}"

    # Print to console
    print(formatted_message)

    # Write to log file
    with open(log_file, 'a') as f:
        f.write(formatted_message + "\n")


def submit_grpo_job(
    model: str,
    dataset: str,
    repo: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    log_file: str
) -> str:
    """
    Submit a GRPO job to Predibase.

    Args:
        model: Name of the base model
        dataset: Name of the dataset in Predibase
        repo: Name of the repository to save the adapter to
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        log_file: Path to the log file

    Returns:
        The job ID
    """
    try:
        from predibase import Predibase
    except ImportError:
        log_message(log_file, "Error: Predibase SDK not installed. Please install it with 'pip install predibase'.")
        sys.exit(1)

    # Log submission details
    log_message(log_file, f"Submitting GRPO job with the following parameters:")
    log_message(log_file, f"  Base model: {model}")
    log_message(log_file, f"  Dataset: {dataset}")
    log_message(log_file, f"  Repository: {repo}")
    log_message(log_file, f"  Learning rate: {learning_rate}")
    log_message(log_file, f"  Epochs: {epochs}")
    log_message(log_file, f"  Batch size: {batch_size}")

    # Create GRPO configuration
    config = create_gemini_opa_grpo_config(
        base_model=model,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    # Get API token from config file
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                log_message(log_file, "Error: API token not found in config file.")
                sys.exit(1)
    else:
        log_message(log_file, "Error: Predibase config file not found.")
        sys.exit(1)

    # Initialize Predibase client
    log_message(log_file, "Initializing Predibase client...")
    pb = Predibase(api_token=api_token)

    # Submit the job
    log_message(log_file, "Submitting GRPO job to Predibase...")
    # Updated to use the latest Predibase API
    job = pb.finetuning.jobs.create(
        config=config,
        dataset=dataset,
        repo=repo,
        description=f"ArGen GRPO fine-tuning with Gemini-OPA reward functions"
    )

    job_id = job.id
    log_message(log_file, f"GRPO job submitted successfully! Job ID: {job_id}")

    return job_id


def track_job_progress(job_id: str, log_file: str, poll_interval: int = 60) -> Dict[str, Any]:
    """
    Track the progress of a GRPO job.

    Args:
        job_id: The job ID to track
        log_file: Path to the log file
        poll_interval: Interval in seconds to poll for updates

    Returns:
        Dictionary containing job details
    """
    try:
        from predibase import Predibase
    except ImportError:
        log_message(log_file, "Error: Predibase SDK not installed. Please install it with 'pip install predibase'.")
        sys.exit(1)

    # Get API token from config file
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                log_message(log_file, "Error: API token not found in config file.")
                sys.exit(1)
    else:
        log_message(log_file, "Error: Predibase config file not found.")
        sys.exit(1)

    # Initialize Predibase client
    pb = Predibase(api_token=api_token)

    # Track job progress
    log_message(log_file, f"Tracking progress of job {job_id}...")

    job_complete = False
    last_status = None
    last_progress = -1

    while not job_complete:
        # Get job details
        # Updated to use the latest Predibase API
        job = pb.finetuning.jobs.get(job_id)

        # Check if status has changed
        if job.status != last_status:
            log_message(log_file, f"Job status: {job.status}")
            last_status = job.status

        # Check if progress has changed
        if hasattr(job, 'progress') and job.progress != last_progress and job.progress is not None:
            log_message(log_file, f"Job progress: {job.progress:.2%}")
            last_progress = job.progress

        # Check if job is complete
        if job.status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            job_complete = True

            if job.status == 'COMPLETED':
                log_message(log_file, "Job completed successfully!")

                # Get model details
                model_name = job.repo
                log_message(log_file, f"Fine-tuned model name: {model_name}")

                # Get metrics if available
                if hasattr(job, 'metrics') and job.metrics:
                    log_message(log_file, "Training metrics:")
                    for metric_name, metric_value in job.metrics.items():
                        log_message(log_file, f"  {metric_name}: {metric_value}")
            else:
                log_message(log_file, f"Job {job.status.lower()}.")
                if hasattr(job, 'error') and job.error:
                    log_message(log_file, f"Error: {job.error}")

        # Wait before polling again
        if not job_complete:
            time.sleep(poll_interval)

    # Return job details
    return {
        'job_id': job_id,
        'status': job.status,
        'model_name': job.repo if job.status == 'COMPLETED' else None,
        'metrics': job.metrics if hasattr(job, 'metrics') and job.metrics else None,
        'error': job.error if hasattr(job, 'error') and job.error else None
    }


def main():
    """Run the GRPO tracking script."""
    parser = argparse.ArgumentParser(description="Run GRPO with Gemini-OPA reward functions and track progress.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the base model")
    parser.add_argument("--dataset", type=str, default="argen_combined_dataset", help="Name of the dataset in Predibase")
    parser.add_argument("--repo", type=str, default="argen-gemini-opa", help="Name of the repository to save the adapter to")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--poll-interval", type=int, default=60, help="Interval in seconds to poll for updates")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to store logs")
    parser.add_argument("--job-id", type=str, help="Job ID to track (if already submitted)")

    args = parser.parse_args()

    # Load environment variables
    load_env_vars()

    # Check if GEMINI_API_KEY is set
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable not set. Please set it in your environment or in a .env file.")
        sys.exit(1)

    # Set up logging
    log_file = setup_logging(args.log_dir)
    log_message(log_file, "Starting GRPO tracking script...")

    # Submit job or use existing job ID
    job_id = args.job_id
    if job_id:
        log_message(log_file, f"Using existing job ID: {job_id}")
    else:
        # Prepare the combined datasets if they don't exist
        if not os.path.exists("data/combined_predibase.jsonl"):
            log_message(log_file, "Preparing combined datasets...")
            os.system("python examples/prepare_combined_datasets.py")

        # Submit GRPO job
        job_id = submit_grpo_job(
            args.model,
            args.dataset,
            args.repo,
            args.learning_rate,
            args.epochs,
            args.batch_size,
            log_file
        )

    # Track job progress
    job_details = track_job_progress(job_id, log_file, args.poll_interval)

    # Save job details
    job_details_file = os.path.join(args.log_dir, f"job_details_{job_id}.json")
    with open(job_details_file, 'w') as f:
        json.dump(job_details, f, indent=2)

    log_message(log_file, f"Job details saved to {job_details_file}")

    # If job completed successfully, prompt for evaluation
    if job_details['status'] == 'COMPLETED' and job_details['model_name']:
        log_message(log_file, "\nNext steps:")
        log_message(log_file, f"1. Evaluate the fine-tuned model with:")
        log_message(log_file, f"   python examples/evaluate_finetuned_with_gemini.py --model {job_details['model_name']}")
        log_message(log_file, f"2. Compare with baseline results:")
        log_message(log_file, f"   python examples/compare_results.py --baseline data/baseline_gemini_results.json --finetuned data/finetuned_gemini_results.json")

    log_message(log_file, "GRPO tracking script completed.")


if __name__ == "__main__":
    main()
