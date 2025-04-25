#!/usr/bin/env python3
"""
Script to run a GRPO job with Gemini-OPA reward functions in Predibase.

This script:
1. Loads environment variables including the Gemini API key
2. Connects to Predibase using the API token
3. Loads the reward function code
4. Creates and submits a GRPO job with detailed configuration
5. Provides detailed logging for monitoring and troubleshooting

Usage:
    python scripts/run_gemini_opa_grpo_job.py --model llama-3-2-1b-instruct --repo argen-gemini-opa
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project utilities
from src.utils.env import load_env_vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gemini_opa_grpo_job.log')
    ]
)
logger = logging.getLogger('gemini_opa_grpo_job')


def load_reward_function_code() -> Dict[str, str]:
    """
    Load the reward function code from the source files.

    Returns:
        Dict containing the code for each reward function file
    """
    logger.info("Loading reward function code...")

    reward_functions_dir = Path(__file__).parent.parent / "src" / "reward_functions"
    code_files = {
        "gemini_opa_rewards.py": reward_functions_dir / "gemini_opa_rewards.py",
    }

    code = {}
    for name, path in code_files.items():
        try:
            with open(path, "r") as f:
                code[name] = f.read()
                logger.info(f"Loaded {name} ({len(code[name])} bytes)")
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
            raise

    return code


def load_opa_policy_code() -> Dict[str, str]:
    """
    Load the OPA policy code from the source files.

    Returns:
        Dict containing the code for each OPA policy file
    """
    logger.info("Loading OPA policy code...")

    # Check both custom/ and gopal/custom/ directories
    policy_files = {
        "custom/ahimsa.rego": Path(__file__).parent.parent / "custom" / "ahimsa.rego",
        "custom/dharma.rego": Path(__file__).parent.parent / "custom" / "dharma.rego",
    }

    code = {}
    for name, path in policy_files.items():
        if path.exists():
            try:
                with open(path, "r") as f:
                    code[name] = f.read()
                    logger.info(f"Loaded {name} ({len(code[name])} bytes)")
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")

    if not code:
        logger.warning("No OPA policy files found!")

    return code


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
    reward_function_code: Dict[str, str],
    opa_policy_code: Dict[str, str],
    gemini_api_key: str,
    learning_rate: float = 5e-5,
    epochs: int = 1,
    batch_size: int = 4,
):
    """
    Create a GRPO configuration for Predibase.

    Args:
        model: The base model to use
        reward_function_code: Dict containing the code for each reward function file
        opa_policy_code: Dict containing the code for each OPA policy file
        gemini_api_key: The Gemini API key
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs to train for
        batch_size: Batch size for training

    Returns:
        GRPOConfig object containing the GRPO configuration
    """
    logger.info(f"Creating GRPO configuration for model {model}...")

    # Import the required classes
    try:
        from predibase import GRPOConfig, RewardFunctionsConfig, RewardFunctionsRuntimeConfig
    except ImportError:
        logger.error("Required Predibase classes not found. Make sure you have the latest version of the Predibase SDK.")
        raise

    # Combine all reward function code
    combined_code = ""
    for name, code in reward_function_code.items():
        combined_code += f"# {name}\n{code}\n\n"

    # Add OPA policy code as comments for reference
    for name, code in opa_policy_code.items():
        combined_code += f"# {name} (included as reference)\n# {code.replace(chr(10), chr(10)+'# ')}\n\n"

    # Import the reward functions from the code
    namespace = {}
    exec(combined_code, namespace)

    # Create the reward functions configuration
    reward_fns_config = RewardFunctionsConfig(
        functions={
            "ahimsa": namespace["gemini_opa_ahimsa_reward"],
            "dharma": namespace["gemini_opa_dharma_reward"]
        },
        runtime=RewardFunctionsRuntimeConfig(
            packages=["google-generativeai", "python-dotenv"],
            env_vars={
                "GEMINI_API_KEY": gemini_api_key
            }
        )
    )

    # Create the GRPO configuration
    config = GRPOConfig(
        base_model=model,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        reward_fns=reward_fns_config
    )

    logger.info(f"GRPO configuration created with {len(combined_code)} bytes of code")
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
        Dict containing the job details
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

        logger.info(f"GRPO job submitted successfully! Job ID: {job.id}")
        logger.info(f"Job details: {job}")
        return job
    except Exception as e:
        logger.error(f"Error creating GRPO job: {e}")
        raise


def monitor_job(job_id: str, api_token: str, check_interval: int = 60, max_checks: int = 60) -> None:
    """
    Monitor a Predibase job.

    Args:
        job_id: The job ID to monitor
        api_token: The Predibase API token
        check_interval: Interval between checks in seconds
        max_checks: Maximum number of checks before giving up
    """
    logger.info(f"Monitoring job {job_id}...")

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
            job = pb.finetuning.jobs.get(job_id)
            status = job.status
            logger.info(f"Job {job_id} status: {status}")

            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                logger.info(f"Job {job_id} finished with status: {status}")
                if status == "COMPLETED":
                    logger.info(f"Job completed successfully! Adapter ID: {job.adapter_id}")
                elif status == "FAILED":
                    logger.error(f"Job failed: {job.error}")
                return

            # Check logs
            try:
                logs = pb.finetuning.jobs.logs(job_id)
                if logs:
                    logger.info(f"Recent logs for job {job_id}:")
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

    logger.warning(f"Stopped monitoring job {job_id} after {max_checks} checks")


def main():
    """Main function to run the GRPO job."""
    parser = argparse.ArgumentParser(description="Run a GRPO job with Gemini-OPA reward functions in Predibase.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the base model")
    parser.add_argument("--dataset", type=str, default="argen_combined_dataset", help="Name of the dataset in Predibase")
    parser.add_argument("--repo", type=str, default="argen-gemini-opa", help="Name of the repository to save the adapter to")
    parser.add_argument("--description", type=str, default="ArGen GRPO fine-tuning with Gemini-OPA reward functions", help="Description of the job")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--monitor", action="store_true", help="Monitor the job after submission")

    args = parser.parse_args()

    logger.info("Starting GRPO job script...")

    try:
        # Load environment variables
        logger.info("Loading environment variables...")
        load_env_vars()

        # Get Gemini API key
        logger.info("Getting Gemini API key...")
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            sys.exit(1)
        logger.info("Gemini API key found")

        # Get Predibase API token
        api_token = get_predibase_api_token()

        # Load reward function code
        reward_function_code = load_reward_function_code()

        # Load OPA policy code
        opa_policy_code = load_opa_policy_code()

        # Create GRPO configuration
        config = create_grpo_config(
            model=args.model,
            reward_function_code=reward_function_code,
            opa_policy_code=opa_policy_code,
            gemini_api_key=gemini_api_key,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Submit the job
        job = submit_grpo_job(
            config=config,
            dataset=args.dataset,
            repo=args.repo,
            description=args.description,
            api_token=api_token
        )

        # Monitor the job if requested
        if args.monitor:
            monitor_job(job.id, api_token)

        logger.info("GRPO job script completed successfully")

    except Exception as e:
        logger.error(f"Error in GRPO job script: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
