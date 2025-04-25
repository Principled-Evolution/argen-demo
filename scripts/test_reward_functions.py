#!/usr/bin/env python3
"""
Script to test the Gemini-OPA reward functions locally before running on Predibase.

This script:
1. Loads a few sample scenarios from the evaluation dataset
2. Calls the reward functions directly with sample completions
3. Verifies that both Ahimsa and Dharma reward functions are working
4. Checks that Gemini API is being called and OPA rules are being applied
5. Provides detailed logging for debugging

Usage:
    python scripts/test_reward_functions.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the reward functions
from src.reward_functions.gemini_opa_rewards import (
    gemini_opa_ahimsa_reward,
    gemini_opa_dharma_reward,
    get_gemini_evaluation,
    apply_ahimsa_rules,
    apply_dharma_rules,
    configure_gemini
)

# Import environment utilities
from src.utils.env import load_env_vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reward_functions_test.log')
    ]
)
logger = logging.getLogger('reward_functions_test')


def load_sample_scenarios(file_path: str = "data/combined_evaluation.jsonl", num_samples: int = 3) -> List[Dict]:
    """
    Load sample scenarios from the evaluation dataset.
    
    Args:
        file_path: Path to the evaluation dataset
        num_samples: Number of samples to load
        
    Returns:
        List of sample scenarios
    """
    logger.info(f"Loading {num_samples} sample scenarios from {file_path}...")
    
    scenarios = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                if line.strip():
                    scenarios.append(json.loads(line))
        
        logger.info(f"Loaded {len(scenarios)} scenarios")
        return scenarios
    except Exception as e:
        logger.error(f"Error loading scenarios: {e}")
        return []


def load_sample_responses(file_path: str = "data/baseline_gemini_results.json", num_samples: int = 3) -> List[Dict]:
    """
    Load sample responses from the baseline results.
    
    Args:
        file_path: Path to the baseline results
        num_samples: Number of samples to load
        
    Returns:
        List of sample responses
    """
    logger.info(f"Loading {num_samples} sample responses from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            responses = data.get("results", [])[:num_samples]
        
        logger.info(f"Loaded {len(responses)} responses")
        return responses
    except Exception as e:
        logger.error(f"Error loading responses: {e}")
        return []


def test_gemini_api_connection():
    """Test the connection to the Gemini API."""
    logger.info("Testing Gemini API connection...")
    
    # Load environment variables
    load_env_vars()
    
    # Get Gemini API key
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return False
    
    # Configure Gemini
    if configure_gemini(gemini_api_key):
        logger.info("Gemini API connection successful")
        return True
    else:
        logger.error("Gemini API connection failed")
        return False


def test_reward_functions(responses: List[Dict]):
    """
    Test the reward functions with sample responses.
    
    Args:
        responses: List of sample responses
    """
    logger.info("Testing reward functions...")
    
    for i, response in enumerate(responses):
        prompt = response.get("prompt", "")
        completion = response.get("response", "")
        
        logger.info(f"Testing scenario {i+1}:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Completion: {completion}")
        
        # Test Ahimsa reward function
        logger.info("Testing Ahimsa reward function...")
        ahimsa_score = gemini_opa_ahimsa_reward(prompt, completion, {})
        logger.info(f"Ahimsa score: {ahimsa_score}")
        
        # Test Dharma reward function
        logger.info("Testing Dharma reward function...")
        dharma_score = gemini_opa_dharma_reward(prompt, completion, {})
        logger.info(f"Dharma score: {dharma_score}")
        
        # Test direct Gemini evaluation
        logger.info("Testing direct Gemini evaluation...")
        ahimsa_eval = get_gemini_evaluation(prompt, completion, "Ahimsa")
        logger.info(f"Ahimsa evaluation: {ahimsa_eval}")
        
        dharma_eval = get_gemini_evaluation(prompt, completion, "Dharma")
        logger.info(f"Dharma evaluation: {dharma_eval}")
        
        # Test OPA rules application
        if ahimsa_eval:
            logger.info("Testing Ahimsa OPA rules...")
            ahimsa_opa_score = apply_ahimsa_rules(ahimsa_eval)
            logger.info(f"Ahimsa OPA score: {ahimsa_opa_score}")
        
        if dharma_eval:
            logger.info("Testing Dharma OPA rules...")
            dharma_opa_score = apply_dharma_rules(dharma_eval)
            logger.info(f"Dharma OPA score: {dharma_opa_score}")
        
        logger.info(f"Completed testing scenario {i+1}\n")


def main():
    """Main function to test the reward functions."""
    logger.info("Starting reward functions test...")
    
    # Test Gemini API connection
    if not test_gemini_api_connection():
        logger.error("Gemini API connection test failed. Exiting.")
        sys.exit(1)
    
    # Load sample responses
    responses = load_sample_responses()
    if not responses:
        logger.error("No sample responses loaded. Exiting.")
        sys.exit(1)
    
    # Test reward functions
    test_reward_functions(responses)
    
    logger.info("Reward functions test completed successfully")


if __name__ == "__main__":
    main()
