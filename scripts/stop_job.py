"""
Script to stop a running Predibase job.
"""

import os
import json
import sys

try:
    from predibase import Predibase
except ImportError:
    print("Predibase SDK not installed. Please install it with 'pip install predibase'.")
    sys.exit(1)

def main():
    # Get job ID from command line
    if len(sys.argv) < 2:
        print("Usage: python stop_job.py <job_id>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    
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
    
    # Initialize Predibase client
    print(f"Initializing Predibase client...")
    pb = Predibase(api_token=api_token)
    
    # Stop the job
    print(f"Stopping job {job_id}...")
    try:
        pb.finetuning.jobs.cancel(job_id)
        print(f"Job {job_id} stopped successfully.")
    except Exception as e:
        print(f"Error stopping job {job_id}: {e}")

if __name__ == "__main__":
    main()
