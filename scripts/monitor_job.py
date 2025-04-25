#!/usr/bin/env python3
"""
Script to monitor a Predibase job.

This script:
1. Connects to Predibase using the API token
2. Monitors the status of a job
3. Displays logs and other information

Usage:
    python scripts/monitor_job.py --job-id argen-gemini-opa/3
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('job_monitor.log')
    ]
)
logger = logging.getLogger('job_monitor')


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
        logger.info(f"Connected to Predibase")
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
                    for log in logs[-10:]:  # Show last 10 log entries
                        logger.info(f"  {log}")
            except Exception as log_error:
                logger.warning(f"Error getting logs: {log_error}")

            # Check reward logs if available
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

    logger.warning(f"Stopped monitoring job {job_id} after {max_checks} checks")


def main():
    """Main function to monitor a job."""
    parser = argparse.ArgumentParser(description="Monitor a Predibase job.")
    parser.add_argument("--job-id", type=str, required=True, help="ID of the job to monitor")
    parser.add_argument("--check-interval", type=int, default=60, help="Interval between checks in seconds")
    parser.add_argument("--max-checks", type=int, default=60, help="Maximum number of checks before giving up")

    args = parser.parse_args()

    logger.info("Starting job monitor script...")

    try:
        # Get Predibase API token
        api_token = get_predibase_api_token()

        # Monitor the job
        monitor_job(
            job_id=args.job_id,
            api_token=api_token,
            check_interval=args.check_interval,
            max_checks=args.max_checks
        )

        logger.info("Job monitor script completed successfully")

    except Exception as e:
        logger.error(f"Error in job monitor script: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
