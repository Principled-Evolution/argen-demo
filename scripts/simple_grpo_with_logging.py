#!/usr/bin/env python3
"""
Simple script to run a GRPO job with detailed logging.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Configure logging to both console and file
log_file = "grpo_job.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('grpo_job')

# Log system information
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")

try:
    # Load environment variables
    logger.info("Loading environment variables...")
    load_dotenv()
    
    # Get Gemini API key
    logger.info("Getting Gemini API key...")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    logger.info("Gemini API key found")
    
    # Get API token from config file
    logger.info("Getting API token from config file...")
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                logger.error("API token not found in config file")
                sys.exit(1)
        logger.info("API token found")
    else:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Import Predibase
    logger.info("Importing Predibase...")
    try:
        from predibase import Predibase, GRPOConfig, RewardFunctionsConfig, RewardFunctionsRuntimeConfig
        logger.info("Predibase imported successfully")
    except ImportError as e:
        logger.error(f"Error importing Predibase: {e}")
        logger.error("Make sure Predibase is installed: pip install predibase")
        sys.exit(1)
    
    # Initialize Predibase client
    logger.info("Initializing Predibase client...")
    try:
        pb = Predibase(api_token=api_token)
        logger.info("Predibase client initialized")
    except Exception as e:
        logger.error(f"Error initializing Predibase client: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Define a simple reward function
    logger.info("Defining reward function...")
    reward_function_code = """
def simple_reward(prompt: str, completion: str, example: dict) -> float:
    # Very simple reward function that always returns 1.0
    print(f"Processing prompt: {prompt[:50]}...")
    print(f"Completion: {completion[:50]}...")
    return 1.0
"""
    
    # Create reward functions configuration
    logger.info("Creating reward functions configuration...")
    try:
        reward_fns_config = RewardFunctionsConfig(
            code=reward_function_code,
            functions={
                "simple": "simple_reward"
            },
            runtime=RewardFunctionsRuntimeConfig(
                packages=["python-dotenv"],
                env_vars={
                    "GEMINI_API_KEY": gemini_api_key
                }
            )
        )
        logger.info("Reward functions configuration created")
    except Exception as e:
        logger.error(f"Error creating reward functions configuration: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Create GRPO configuration
    logger.info("Creating GRPO configuration...")
    try:
        model = "llama-3-2-1b-instruct"
        config = GRPOConfig(
            base_model=model,
            learning_rate=5e-5,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            reward_fns=reward_fns_config
        )
        logger.info(f"GRPO configuration created for model {model}")
    except Exception as e:
        logger.error(f"Error creating GRPO configuration: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # List available datasets
    logger.info("Listing available datasets...")
    try:
        datasets = pb.datasets.list()
        logger.info(f"Found {len(datasets)} datasets:")
        for dataset in datasets:
            logger.info(f"  - {dataset.name}")
        
        # Check if our dataset exists
        dataset_name = "argen_combined_dataset"
        dataset_exists = any(d.name == dataset_name for d in datasets)
        if not dataset_exists:
            logger.warning(f"Dataset '{dataset_name}' not found in available datasets")
            # List the first dataset as fallback
            if datasets:
                dataset_name = datasets[0].name
                logger.info(f"Using '{dataset_name}' as fallback dataset")
            else:
                logger.error("No datasets available")
                sys.exit(1)
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # List available repositories
    logger.info("Listing available repositories...")
    try:
        repos = pb.repos.list()
        logger.info(f"Found {len(repos)} repositories:")
        for repo in repos:
            logger.info(f"  - {repo.name}")
        
        # Set repository name
        repo_name = "argen-gemini-opa-simple"
        logger.info(f"Using repository '{repo_name}'")
    except Exception as e:
        logger.error(f"Error listing repositories: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Submit the job
    logger.info(f"Submitting GRPO job (dataset: {dataset_name}, repo: {repo_name})...")
    try:
        job = pb.finetuning.jobs.create(
            config=config,
            dataset=dataset_name,
            repo=repo_name,
            description="Simple GRPO job with basic reward function"
        )
        
        logger.info(f"GRPO job submitted successfully! Job ID: {job.id}")
        logger.info(f"Job details: {job}")
    except Exception as e:
        logger.error(f"Error submitting GRPO job: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Script completed successfully")
    logger.info(f"Check the log file for details: {log_file}")

except Exception as e:
    logger.error(f"Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1)
