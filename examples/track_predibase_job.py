"""
Script to track a Predibase job.
"""

import os
import json
import sys
import time
import argparse

def main():
    """Track a Predibase job."""
    parser = argparse.ArgumentParser(description="Track a Predibase job.")
    parser.add_argument("--job-id", type=str, required=True, help="The job ID to track")
    parser.add_argument("--poll-interval", type=int, default=60, help="Interval in seconds to poll for updates")
    
    args = parser.parse_args()
    
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
    print(f"Initializing Predibase client...")
    pb = Predibase(api_token=api_token)
    print("Connected to Predibase.")
    
    # Track job progress
    print(f"Tracking progress of job {args.job_id}...")
    
    job_complete = False
    last_status = None
    last_progress = -1
    
    while not job_complete:
        # Get job details
        try:
            job = pb.finetuning.jobs.get(args.job_id)
            print(f"Job details: {job}")
            
            # Check if status has changed
            if hasattr(job, 'status') and job.status != last_status:
                print(f"Job status: {job.status}")
                last_status = job.status
            
            # Check if progress has changed
            if hasattr(job, 'progress') and job.progress != last_progress and job.progress is not None:
                print(f"Job progress: {job.progress:.2%}")
                last_progress = job.progress
            
            # Check if job is complete
            if hasattr(job, 'status') and job.status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                job_complete = True
                
                if job.status == 'COMPLETED':
                    print("Job completed successfully!")
                    
                    # Get model details
                    if hasattr(job, 'repo'):
                        model_name = job.repo
                        print(f"Fine-tuned model name: {model_name}")
                    
                    # Get metrics if available
                    if hasattr(job, 'metrics') and job.metrics:
                        print("Training metrics:")
                        for metric_name, metric_value in job.metrics.items():
                            print(f"  {metric_name}: {metric_value}")
                else:
                    print(f"Job {job.status.lower()}.")
                    if hasattr(job, 'error') and job.error:
                        print(f"Error: {job.error}")
            
        except Exception as e:
            print(f"Error getting job details: {e}")
        
        # Wait before polling again
        if not job_complete:
            print(f"Waiting {args.poll_interval} seconds before polling again...")
            time.sleep(args.poll_interval)

if __name__ == "__main__":
    main()
