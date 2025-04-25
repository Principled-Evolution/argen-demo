"""
Test script to create a GRPO job in Predibase.
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
        "base_model": "llama-3-2-1b-instruct",
        "task": "grpo"
    }

    job = pb.finetuning.jobs.create(
        config=config,
        dataset="argen_combined_dataset",
        repo="argen-gemini-opa",
        description="Minimal GRPO job"
    )

    print(f"GRPO job created successfully!")
    print(f"Job details: {job}")

    # Try to get the job UUID from the output message
    import re
    job_uuid_match = re.search(r'Job UUID: ([0-9a-f-]+)', str(job))
    if job_uuid_match:
        job_uuid = job_uuid_match.group(1)
        print(f"Extracted job UUID: {job_uuid}")

        # Try to track the job
        print(f"Tracking job {job_uuid}...")
        try:
            tracked_job = pb.finetuning.jobs.get(job_uuid)
            print(f"Job status: {tracked_job.status}")
        except Exception as e:
            print(f"Error tracking job: {e}")
    else:
        print("Could not extract job UUID from output")
except Exception as e:
    print(f"Error creating GRPO job: {e}")
