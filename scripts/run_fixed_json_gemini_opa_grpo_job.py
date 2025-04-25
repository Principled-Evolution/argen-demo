#!/usr/bin/env python3
"""
Script to run a GRPO job with Gemini-OPA reward functions that handle JSON code blocks.

This script:
1. Uses an API key defined directly inside the reward functions
2. Connects to Predibase using the API token
3. Defines the reward functions with the correct signature
4. Creates and submits a GRPO job with detailed configuration
5. Provides detailed logging for monitoring and troubleshooting

Usage:
    python scripts/run_fixed_json_gemini_opa_grpo_job.py --model llama-3-2-1b-instruct --repo argen-gemini-opa
"""

import os
import sys
import json
import time
import argparse
import logging
import re
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project utilities and SHARED reward functions
from src.utils.env import load_env_vars, get_gemini_api_key
from src.reward_functions.gemini_rewards import gemini_ahimsa_reward, gemini_dharma_reward
from predibase import RewardFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fixed_json_gemini_opa_grpo_job.log')
    ]
)
logger = logging.getLogger('fixed_json_gemini_opa_grpo_job')


def get_predibase_api_token() -> str:
    """
    Get the Predibase API token from the config file.

    Returns:
        str: The Predibase API token
    """
    logger.info("Getting Predibase API token...")

    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                pb_config = json.load(f)
                api_token = pb_config.get("api_key")
                if not api_token:
                    raise ValueError("API token not found in config file")
                logger.info("Predibase API token found")
                return api_token
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            raise
    else:
        logger.error(f"Config file not found: {config_path}")
        raise ValueError(f"Config file not found: {config_path}")


def create_grpo_config(
    model: str,
    learning_rate: float = 5e-5,
    epochs: int = 1,
    batch_size: int = 4,
    train_steps: int = 1000
):
    """
    Create a GRPO configuration for Predibase using shared reward functions.

    Args:
        model: The base model to use
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs (Less relevant for GRPO, use train_steps)
        batch_size: Batch size for training
        train_steps: Number of training steps for GRPO (default 1000)

    Returns:
        GRPOConfig object containing the GRPO configuration
    """
    logger.info(f"Creating GRPO configuration for model {model}...")

    try:
        from predibase import GRPOConfig, RewardFunction, RewardFunctionsConfig, RewardFunctionsRuntimeConfig
    except ImportError:
        logger.error("Required Predibase classes not found...")
        raise

    # Create the reward functions configuration USING THE IMPORTED FUNCTIONS and RewardFunction wrapper
    reward_fns_config = RewardFunctionsConfig(
        functions={
            # Wrap the imported functions
            "ahimsa": RewardFunction.from_callable(gemini_ahimsa_reward),
            "dharma": RewardFunction.from_callable(gemini_dharma_reward)
        },
        runtime=RewardFunctionsRuntimeConfig(
            # Keep explicit package list as gemini functions have external deps
            packages=["google-generativeai", "python-dotenv"]
        )
    )

    # Create the GRPO configuration
    config = GRPOConfig(
        base_model=model,
        learning_rate=learning_rate,
        train_steps=train_steps,
        per_device_train_batch_size=batch_size,
        reward_fns=reward_fns_config
    )

    logger.info(f"GRPO configuration created")
    return config


def submit_grpo_job(
    config,
    dataset: str,
    repo: str,
    description: str,
    api_token: str
):
    """
    Submit a GRPO job to Predibase.

    Args:
        config: The GRPO configuration
        dataset: The dataset to use
        repo: The repository to save the adapter to
        description: Description of the job
        api_token: The Predibase API token

    Returns:
        Tuple containing the job object and job UUID
    """
    logger.info(f"Submitting GRPO job to Predibase (dataset: {dataset}, repo: {repo})...")

    try:
        from predibase import Predibase
    except ImportError:
        logger.error("Predibase SDK not installed. Please install it with 'pip install predibase'.")
        raise

    # Initialize Predibase client
    try:
        pb = Predibase(api_token=api_token)
        logger.info(f"Connected to Predibase")
    except Exception as e:
        logger.error(f"Error connecting to Predibase: {e}")
        raise

    # Submit the job
    try:
        job = pb.finetuning.jobs.create(
            config=config,
            dataset=dataset,
            repo=repo,
            description=description
        )

        # Extract job UUID from the job object or its string representation
        job_uuid = None
        if hasattr(job, 'id'):
            job_uuid = job.id
        elif hasattr(job, 'uuid'):
            job_uuid = job.uuid
        else:
            # Try to extract UUID from string representation
            import re
            job_str = str(job)
            uuid_match = re.search(r'UUID: ([0-9a-f-]+)', job_str)
            if uuid_match:
                job_uuid = uuid_match.group(1)
        
        if job_uuid:
            logger.info(f"GRPO job submitted successfully! Job UUID: {job_uuid}")
        else:
            logger.info(f"GRPO job submitted successfully! Job details: {job}")
        
        return job, job_uuid
    except Exception as e:
        logger.error(f"Error creating GRPO job: {e}")
        raise


def monitor_job(job_uuid: str, api_token: str, check_interval: int = 60, max_checks: int = 60) -> None:
    """
    Monitor a Predibase job.

    Args:
        job_uuid: The job UUID to monitor
        api_token: The Predibase API token
        check_interval: Interval between checks in seconds
        max_checks: Maximum number of checks before giving up
    """
    logger.info(f"Monitoring job {job_uuid}...")

    try:
        from predibase import Predibase
    except ImportError:
        logger.error("Predibase SDK not installed. Please install it with 'pip install predibase'.")
        return

    # Initialize Predibase client
    try:
        pb = Predibase(api_token=api_token)
    except Exception as e:
        logger.error(f"Error connecting to Predibase: {e}")
        return

    # Monitor the job
    checks = 0
    while checks < max_checks:
        try:
            # Try to get the job by UUID
            try:
                job = pb.finetuning.jobs.get(job_uuid)
            except Exception as e:
                logger.warning(f"Error getting job by UUID: {e}")
                # Try to list all jobs and find the one with matching UUID
                jobs = pb.finetuning.jobs.list()
                job = None
                for j in jobs:
                    if hasattr(j, 'uuid') and j.uuid == job_uuid:
                        job = j
                        break
                    elif hasattr(j, 'id') and j.id == job_uuid:
                        job = j
                        break
                
                if not job:
                    logger.error(f"Could not find job with UUID {job_uuid}")
                    return
            
            # Get job status
            status = None
            if hasattr(job, 'status'):
                status = job.status
            else:
                # Try to extract status from string representation
                import re
                job_str = str(job)
                status_match = re.search(r'status=([A-Z]+)', job_str)
                if status_match:
                    status = status_match.group(1)
            
            if status:
                logger.info(f"Job {job_uuid} status: {status}")
                
                if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                    logger.info(f"Job {job_uuid} finished with status: {status}")
                    if status == "COMPLETED":
                        adapter_id = None
                        if hasattr(job, 'adapter_id'):
                            adapter_id = job.adapter_id
                        logger.info(f"Job completed successfully! Adapter ID: {adapter_id}")
                    elif status == "FAILED":
                        error = None
                        if hasattr(job, 'error'):
                            error = job.error
                        logger.error(f"Job failed: {error}")
                    return
            else:
                logger.warning(f"Could not determine status for job {job_uuid}")

            # Check logs
            try:
                logs = pb.finetuning.jobs.logs(job_uuid)
                if logs:
                    logger.info(f"Recent logs for job {job_uuid}:")
                    for log in logs[-5:]:  # Show last 5 log entries
                        logger.info(f"  {log}")
            except Exception as log_error:
                logger.warning(f"Error getting logs: {log_error}")

            # Wait for next check
            time.sleep(check_interval)
            checks += 1
        except Exception as e:
            logger.error(f"Error monitoring job: {e}")
            time.sleep(check_interval)
            checks += 1

    logger.warning(f"Stopped monitoring job {job_uuid} after {max_checks} checks")


def main():
    """Main function to run the GRPO job."""
    parser = argparse.ArgumentParser(description="Run a GRPO job with SHARED Gemini reward functions in Predibase.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the base model")
    parser.add_argument("--dataset", type=str, default="argen_combined_dataset", help="Name of the dataset in Predibase (ensure it contains formatted prompts)")
    parser.add_argument("--repo", type=str, default="argen-gemini-opa", help="Name of the repository to save the adapter to")
    parser.add_argument("--description", type=str, default="ArGen GRPO fine-tuning with SHARED Gemini reward functions", help="Description of the job")
    parser.add_argument("--train-steps", type=int, default=1000, help="Number of GRPO training steps (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--monitor", action="store_true", help="Monitor the job after submission")

    args = parser.parse_args()

    logger.info("Starting GRPO job script...")

    try:
        # Load environment variables (IMPORTANT for GEMINI_API_KEY)
        logger.info("Loading environment variables...")
        load_env_vars() # This should load .env if present

        # Verify Gemini API key is accessible (Optional but recommended check)
        try:
            gemini_key = get_gemini_api_key() # Use the shared utility
            if not gemini_key:
                 raise ValueError("GEMINI_API_KEY not found in environment.")
            logger.info("Gemini API key found in environment.")
            # DO NOT log the key itself
        except Exception as e:
            logger.error(f"Failed to get Gemini API key: {e}. Ensure GEMINI_API_KEY is set in the environment or .env file for the Predibase job runtime.")
            raise

        # Get Predibase API token
        api_token = get_predibase_api_token()

        # Create GRPO configuration
        config = create_grpo_config(
            model=args.model,
            learning_rate=args.learning_rate,
            train_steps=args.train_steps,
            batch_size=args.batch_size
        )

        # Submit the job
        _, job_uuid = submit_grpo_job(
            config=config,
            dataset=args.dataset,
            repo=args.repo,
            description=args.description,
            api_token=api_token
        )

        # Monitor the job if requested
        if args.monitor and job_uuid:
            monitor_job(job_uuid, api_token)

        logger.info("GRPO job script completed successfully")

    except Exception as e:
        logger.error(f"Error in GRPO job script: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
