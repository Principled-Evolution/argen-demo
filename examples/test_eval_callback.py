#!/usr/bin/env python3
"""
Script to test the run_core60_eval function used in the training callback.
"""

import sys
import os
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


try:
    from examples.train_grpo import run_core60_eval
    from src.utils.env import load_env_vars, get_openai_api_key
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error("Ensure you are running this script from the 'examples' directory or that the project root is correctly added to PYTHONPATH.")
    sys.exit(1)

if __name__ == "__main__":
    # --- Configuration ---
    # Use the specific checkpoint directory you want to test
    # IMPORTANT: Replace with the actual full path if needed, or ensure it's relative to where you run the script.
    # Assuming the script is run from the project root (argen-demo)
    checkpoint_to_test = "output/Llama-3.2-1B-Instruct-grpo/checkpoint-6500"
    # --------------------

    logger.info(f"Attempting to test evaluation callback function for checkpoint: {checkpoint_to_test}")

    # Verify checkpoint exists
    if not os.path.isdir(checkpoint_to_test):
        logger.error(f"Checkpoint directory not found: {checkpoint_to_test}")
        logger.error("Please ensure the path is correct and relative to your current working directory if not absolute.")
        sys.exit(1)

    # Load environment variables and check API key
    logger.info("Loading environment variables...")
    load_env_vars()
    if not get_openai_api_key():
        logger.error("OPENAI_API_KEY not found. Please set it in your .env file or environment.")
        sys.exit(1)
    else:
        logger.info("OpenAI API Key found.")

    # Call the function
    logger.info("Calling run_core60_eval...")
    try:
        score = run_core60_eval(checkpoint_dir=checkpoint_to_test)

        if score is not None:
            logger.info(f"\n--- TEST COMPLETE ---")
            logger.info(f"Evaluation function returned score: {score:.4f}")
            logger.info(f"-------------------")
        else:
            logger.warning(f"\n--- TEST COMPLETE (Evaluation returned None) ---")
            logger.warning("The evaluation function completed but did not return a score. Check logs above for errors (e.g., API key issues, file not found, evaluation failures).")
            logger.warning(f"---------------------------------------------")

    except Exception as e:
        logger.error(f"\n--- TEST FAILED --- ")
        logger.error(f"An unexpected error occurred during the call to run_core60_eval: {e}", exc_info=True)
        logger.error(f"-----------------") 