#!/usr/bin/env python3
"""
Script to monitor a GRPO job in Predibase.
"""

import os
import sys
import json
import time
import logging
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitor_grpo_job.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('monitor_grpo_job')

def get_api_token():
    """Get the Predibase API token from the config file."""
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                logger.error("API token not found in config file")
                sys.exit(1)
            return api_token
    else:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

def monitor_job(adapter_id, check_interval=60, max_checks=60):
    """
    Monitor a GRPO job in Predibase.
    
    Args:
        adapter_id: The ID of the adapter (e.g., "argen-gemini-opa/4")
        check_interval: Interval between checks in seconds
        max_checks: Maximum number of checks before giving up
    """
    logger.info(f"Monitoring GRPO job for adapter {adapter_id}...")
    
    # Get API token
    api_token = get_api_token()
    
    # Import Predibase
    try:
        from predibase import Predibase
    except ImportError as e:
        logger.error(f"Error importing Predibase: {e}")
        sys.exit(1)
    
    # Initialize Predibase client
    try:
        pb = Predibase(api_token=api_token)
        logger.info("Predibase client initialized")
    except Exception as e:
        logger.error(f"Error initializing Predibase client: {e}")
        sys.exit(1)
    
    # Monitor the job
    checks = 0
    while checks < max_checks:
        try:
            # Get adapter details
            adapter = pb.adapters.get(adapter_id)
            status = adapter.status
            logger.info(f"Adapter {adapter_id} status: {status}")
            
            # Get job details
            job_id = adapter.job_id
            if job_id:
                job = pb.finetuning.jobs.get(job_id)
                logger.info(f"Job {job_id} status: {job.status}")
                
                # Check if job is completed
                if job.status in ["COMPLETED", "FAILED", "CANCELLED"]:
                    logger.info(f"Job {job_id} finished with status: {job.status}")
                    if job.status == "COMPLETED":
                        logger.info(f"Job completed successfully!")
                    elif job.status == "FAILED":
                        logger.error(f"Job failed: {job.error}")
                    return
                
                # Get job logs
                try:
                    logs = pb.finetuning.jobs.logs(job_id)
                    if logs:
                        logger.info(f"Recent logs for job {job_id}:")
                        for log in logs[-5:]:  # Show last 5 log entries
                            logger.info(f"  {log}")
                except Exception as log_error:
                    logger.warning(f"Error getting logs: {log_error}")
                
                # Get reward logs
                try:
                    reward_logs = pb.finetuning.jobs.reward_logs(job_id)
                    if reward_logs:
                        logger.info(f"Recent reward logs for job {job_id}:")
                        for log in reward_logs[-5:]:  # Show last 5 reward log entries
                            logger.info(f"  {log}")
                except Exception as reward_log_error:
                    logger.warning(f"Error getting reward logs: {reward_log_error}")
            
            # Wait for next check
            logger.info(f"Waiting {check_interval} seconds for next check...")
            time.sleep(check_interval)
            checks += 1
        except Exception as e:
            logger.error(f"Error monitoring job: {e}")
            time.sleep(check_interval)
            checks += 1
    
    logger.warning(f"Stopped monitoring job after {max_checks} checks")

def main():
    """Main function to monitor a GRPO job."""
    parser = argparse.ArgumentParser(description="Monitor a GRPO job in Predibase.")
    parser.add_argument("--adapter-id", type=str, required=True, help="ID of the adapter (e.g., 'argen-gemini-opa/4')")
    parser.add_argument("--check-interval", type=int, default=60, help="Interval between checks in seconds")
    parser.add_argument("--max-checks", type=int, default=60, help="Maximum number of checks before giving up")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Monitor the job
    monitor_job(
        adapter_id=args.adapter_id,
        check_interval=args.check_interval,
        max_checks=args.max_checks
    )

if __name__ == "__main__":
    main()
