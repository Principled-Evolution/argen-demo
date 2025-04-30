#!/usr/bin/env python3
"""
Script to evaluate the baseline model using an LLM evaluator (e.g., OpenAI).
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional
import datetime
import subprocess
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.env import load_env_vars, get_openai_api_key
from src.evaluation.openai_evaluator import evaluate_model_with_openai
from src.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_SCENARIOS_PATH,
    DEFAULT_OUTPUT_BASE,
    DEFAULT_TEMPERATURE,
    get_system_prompt
)


def check_dependencies():
    """Check if necessary dependencies for local inference are installed."""
    try:
        import torch
        import transformers
        print("torch and transformers libraries found.")
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
            # Consider adding accelerate and bitsandbytes checks if quantization is used later
        else:
            print("CUDA not available, will use CPU (might be slow).")
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}. Please install required libraries.")
        print("Hint: pip install torch transformers accelerate") # Add bitsandbytes if needed
        return False


def load_scenarios(file_path: str) -> List[Dict]:
    """
    Load scenarios from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of scenarios
    """
    scenarios = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    
    return scenarios


def main():
    """Run the baseline model evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a baseline model using an LLM evaluator (default: OpenAI).")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Name/identifier of the model to evaluate (e.g., HF identifier like 'unsloth/Llama-3.2-1B-Instruct', or local path)"
    )
    parser.add_argument(
        "--scenarios", 
        type=str,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the scenarios file"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help="Base path and filename for the evaluation results (timestamp and .json will be added)"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run in test mode with mock responses"
    )
    parser.add_argument(
        "--temperature", 
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for generation (higher = more random)"
    )
    parser.add_argument(
        "--use_basic_prompt", 
        action="store_true", 
        help="Use the original basic system prompt instead of the enhanced 'fairer' one."
    )
    
    args = parser.parse_args()
    
    # Load environment variables early
    load_env_vars()
    # Get the API key directly after loading
    openai_api_key = get_openai_api_key()
    
    # Check if the key was loaded successfully
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file or environment.")
        sys.exit(1)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct the final output filename
    output_filename = f"{args.output_base}_{timestamp}.json"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check for local inference dependencies if using a local model
    is_local_model = args.model.startswith(('/', './')) or args.model == DEFAULT_MODEL_ID
    if is_local_model:
        print("Local model specified, checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("Note: First run might take time to download the model.")
    else:
        print("Non-local model specified. Ensure necessary API keys (e.g., OpenAI) are configured.")

    print(f"Evaluating {args.model} using OpenAI evaluator...")
    if args.use_basic_prompt:
        print("Using basic system prompt for baseline model generation.")
    else:
        print("Using enhanced (fairer) system prompt.")
    print(f"Results will be saved to: {output_filename}")
    
    # Prepare the combined datasets if they don't exist
    if not os.path.exists(args.scenarios):
        print(f"Scenarios file {args.scenarios} not found. Preparing combined datasets...")
        os.system("python examples/prepare_combined_datasets.py")
    
    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)
    
    # For testing, use only the first scenario if --test flag is provided
    if args.test:
        print("Running in test mode with only the first scenario...")
        scenarios = scenarios[:1]
    
    # Run the async function using asyncio.run()
    asyncio.run(evaluate_model_with_openai(
        model_name=args.model,
        scenarios=scenarios,
        output_file=output_filename,
        temperature=args.temperature,
        openai_api_key=openai_api_key,
        test_mode=args.test,
        use_basic_prompt=args.use_basic_prompt
    ))
    
    print("Baseline model evaluation complete!")


if __name__ == "__main__":
    main()
