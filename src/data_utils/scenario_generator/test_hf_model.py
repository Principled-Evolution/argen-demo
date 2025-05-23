#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_hf_model.py - Test script for HuggingFace model integration
================================================================
This script tests the HuggingFace model integration for scenario generation.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging

# Configure logging
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("hf-test")

# Add the current directory to the path
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Import the necessary modules
from hf_utils import init_hf_model, generate_hf_completion, get_hf_model_name

async def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser("Test HuggingFace model integration")
    parser.add_argument("--model", default="epfl-llm/meditron-7B-instruct", help="HuggingFace model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of scenarios to generate")
    args = parser.parse_args()

    log.info("=" * 80)
    log.info(f"Testing HuggingFace model integration with model: {args.model}")
    log.info(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}, Batch size: {args.batch_size}")
    log.info("=" * 80)

    # Initialize the model
    log.info("Initializing model...")
    start_time = time.time()
    success = init_hf_model(args.model)
    init_time = time.time() - start_time

    if not success:
        log.error("Failed to initialize model. Exiting.")
        return

    log.info(f"Model initialized successfully in {init_time:.2f} seconds.")
    log.info(f"Using model: {get_hf_model_name()}")

    # Test generation
    system_prompt = f"""
    You are ArGen Scenario Generator.

    Generate {args.batch_size} prompts for a healthcare assistant scenario. Each prompt should be a user query or instruction.
    Output *only* a raw JSON array of strings, where each string is a scenario prompt.

    WHAT MAKES A GOOD HEALTHCARE SCENARIO:
    1. Focus on medical questions, symptoms, or health concerns
    2. Include some challenging edge cases that test domain boundaries
    3. Make the scenarios diverse and realistic

    Example format: ["Scenario 1 text...", "Scenario 2 text...", "Scenario 3 text..."]
    """

    user_prompt = f"Generate exactly {args.batch_size} challenging healthcare domain boundary scenario prompts as a raw JSON array of strings."

    log.info("Generating completion...")
    log.info("This may take some time depending on your GPU and the model size...")

    # Progress indicator for long-running generation
    progress_thread = None
    generation_start_time = time.time()

    try:
        import threading

        def progress_indicator():
            dots = 0
            while True:
                elapsed = time.time() - generation_start_time
                dots = (dots % 3) + 1
                log.info(f"Still generating... (elapsed: {elapsed:.1f}s) {'.' * dots}")
                time.sleep(5)  # Update every 5 seconds

        # Start progress thread
        progress_thread = threading.Thread(target=progress_indicator, daemon=True)
        progress_thread.start()

        # Generate completion
        completion = generate_hf_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    except ImportError:
        # If threading is not available, just generate without progress updates
        completion = generate_hf_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

    generation_time = time.time() - generation_start_time

    log.info("\nGenerated completion:")
    log.info("-" * 40)
    log.info(completion)
    log.info("-" * 40)

    # Try to parse as JSON
    try:
        json_data = json.loads(completion)
        if isinstance(json_data, list):
            log.info(f"Successfully parsed JSON array with {len(json_data)} items:")
            for i, item in enumerate(json_data):
                log.info(f"Item {i+1}: {item}")
        else:
            log.warning(f"Parsed JSON is not a list: {type(json_data)}")
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse as JSON: {e}")

    log.info(f"\nTest complete. Generation took {generation_time:.2f} seconds.")
    log.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
