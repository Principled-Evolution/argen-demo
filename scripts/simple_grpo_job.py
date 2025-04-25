#!/usr/bin/env python3
"""
Simple script to run a GRPO job with Gemini-OPA reward functions in Predibase.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if not gemini_api_key:
    print("GEMINI_API_KEY environment variable not set")
    sys.exit(1)
print("Gemini API key found")

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
    from predibase import Predibase, GRPOConfig, RewardFunctionsConfig, RewardFunctionsRuntimeConfig
except ImportError:
    print("Predibase SDK not installed. Please install it with 'pip install predibase'.")
    sys.exit(1)

# Define a simple reward function
reward_function_code = """
def gemini_opa_ahimsa_reward(prompt: str, completion: str, example: dict) -> float:
    # Simple reward function that always returns 1.0
    print(f"Processing prompt: {prompt[:50]}...")
    print(f"Completion: {completion[:50]}...")
    return 1.0
"""

# Initialize Predibase client
print("Initializing Predibase client...")
pb = Predibase(api_token=api_token)
print("Connected to Predibase")

# Create reward functions configuration
reward_fns_config = RewardFunctionsConfig(
    code=reward_function_code,
    functions={
        "ahimsa": "gemini_opa_ahimsa_reward"
    },
    runtime=RewardFunctionsRuntimeConfig(
        packages=["google-generativeai", "python-dotenv"],
        env_vars={
            "GEMINI_API_KEY": gemini_api_key
        }
    )
)

# Create GRPO configuration
config = GRPOConfig(
    base_model="llama-3-2-1b-instruct",
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    reward_fns=reward_fns_config
)

# Submit the job
print("Submitting GRPO job to Predibase...")
try:
    job = pb.finetuning.jobs.create(
        config=config,
        dataset="argen_combined_dataset",
        repo="argen-gemini-opa",
        description="Simple GRPO job with Gemini-OPA reward functions"
    )
    
    print(f"GRPO job submitted successfully! Job ID: {job.id}")
    print(f"Job details: {job}")
except Exception as e:
    print(f"Error creating GRPO job: {e}")
    sys.exit(1)

print("Script completed successfully")
