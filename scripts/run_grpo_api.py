#!/usr/bin/env python3
"""
Script to run a GRPO job using the Predibase API directly.
"""

import os
import sys
import json
import requests
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

# Set variables
model = "llama-3-2-1b-instruct"
dataset = "argen_combined_dataset"
repo = "argen-gemini-opa"
description = "ArGen GRPO fine-tuning with Gemini-OPA reward functions"

# Create reward functions configuration
reward_fns = {
    "code": """def gemini_opa_ahimsa_reward(prompt: str, completion: str, example: dict) -> float:
    # Simple reward function that always returns 1.0
    print(f"Processing prompt: {prompt[:50]}...")
    print(f"Completion: {completion[:50]}...")
    return 1.0
""",
    "functions": {
        "ahimsa": "gemini_opa_ahimsa_reward"
    },
    "runtime": {
        "packages": ["google-generativeai", "python-dotenv"],
        "env_vars": {
            "GEMINI_API_KEY": gemini_api_key
        }
    }
}

# Create GRPO configuration
config = {
    "base_model": model,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "reward_fns": reward_fns
}

# Print the configuration
print("GRPO Configuration:")
print(json.dumps(config, indent=2))

# Set up the API request
url = "https://api.predibase.com/v1/finetuning/jobs"
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}
data = {
    "config": config,
    "dataset": dataset,
    "repo": repo,
    "description": description
}

# Submit the job
print("Submitting GRPO job...")
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    job = response.json()
    print(f"Job submitted successfully! Job ID: {job.get('id')}")
    print(f"Job details: {json.dumps(job, indent=2)}")
except requests.exceptions.RequestException as e:
    print(f"Error submitting job: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        try:
            error_data = e.response.json()
            print(f"Error details: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Response text: {e.response.text}")
    sys.exit(1)

print("Script completed successfully")
