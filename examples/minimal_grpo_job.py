"""
Minimal script to create a GRPO job in Predibase.
"""

import os
import json
import sys

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

# List datasets
print("Listing datasets...")
try:
    datasets = pb.datasets.list()
    print(f"Datasets: {datasets}")
except Exception as e:
    print(f"Error listing datasets: {e}")

# List repositories
print("Listing repositories...")
try:
    repos = pb.repos.list()
    print(f"Repositories: {repos}")
except Exception as e:
    print(f"Error listing repositories: {e}")

# Try to create a minimal GRPO job
print("Creating minimal GRPO job...")
try:
    config = {
        "base_model": "llama-3-2-1b-instruct"
    }
    
    job = pb.finetuning.jobs.create(
        config=config,
        dataset="argen_combined_dataset",
        repo="argen-gemini-opa",
        description="Minimal GRPO job"
    )
    
    print(f"GRPO job created successfully! Job ID: {job.id}")
except Exception as e:
    print(f"Error creating GRPO job: {e}")
