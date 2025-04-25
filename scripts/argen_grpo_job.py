#!/usr/bin/env python3
"""
Script to run a GRPO job for ArGen with Gemini-OPA reward functions.
Based on the Predibase Colab notebook example.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("argen_grpo_job.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('argen_grpo_job')

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
    from predibase import (
        Predibase, 
        GRPOConfig, 
        RewardFunctionsConfig, 
        RewardFunction,
        DeploymentConfig
    )
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
    sys.exit(1)

# Define reward functions
logger.info("Defining reward functions...")

def gemini_opa_ahimsa_reward(prompt: str, completion: str, example: dict) -> float:
    """
    Reward function that evaluates if the response follows Ahimsa principles.
    
    Args:
        prompt: The prompt given to the model
        completion: The model's response
        example: The example data from the dataset
        
    Returns:
        float: A score between 0 and 1 indicating adherence to Ahimsa principles
    """
    # Import packages inside the function
    import re
    import json
    
    # Simple implementation for testing
    print(f"Processing prompt: {prompt[:100]}...")
    print(f"Completion: {completion[:100]}...")
    
    # Check for harmful content (simplified)
    harmful_patterns = [
        r"harm", r"kill", r"hurt", r"injure", r"damage", 
        r"violent", r"dangerous", r"illegal", r"unethical"
    ]
    
    # Count harmful patterns
    harmful_count = 0
    for pattern in harmful_patterns:
        if re.search(pattern, completion.lower()):
            harmful_count += 1
            print(f"Found harmful pattern: {pattern}")
    
    # Calculate score (1.0 = no harmful content, 0.0 = very harmful)
    if harmful_count == 0:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (harmful_count / len(harmful_patterns)))
    
    print(f"Ahimsa reward score: {score}")
    return score

# Create GRPO configuration
logger.info("Creating GRPO configuration...")
try:
    model = "llama-3-2-1b-instruct"
    
    config = GRPOConfig(
        base_model=model,
        reward_fns=RewardFunctionsConfig(
            functions={
                "ahimsa": RewardFunction.from_callable(gemini_opa_ahimsa_reward)
            }
        ),
        # Optional parameters
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4
    )
    
    logger.info(f"GRPO configuration created for model {model}")
except Exception as e:
    logger.error(f"Error creating GRPO configuration: {e}")
    sys.exit(1)

# Set dataset and repository names
dataset_name = "argen_combined_dataset"
repo_name = "argen-gemini-opa"

# Submit the job
logger.info(f"Submitting GRPO job (dataset: {dataset_name}, repo: {repo_name})...")
try:
    adapter = pb.adapters.create(
        config=config,
        dataset=dataset_name,
        repo=repo_name,
        description="ArGen GRPO fine-tuning with Gemini-OPA reward functions"
    )
    
    logger.info(f"GRPO job submitted successfully! Adapter ID: {adapter.id}")
    logger.info(f"Adapter details: {adapter}")
except Exception as e:
    logger.error(f"Error submitting GRPO job: {e}")
    sys.exit(1)

logger.info("Script completed successfully")
