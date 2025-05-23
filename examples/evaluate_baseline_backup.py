#!/usr/bin/env python3
"""
Script to evaluate the baseline model using an LLM evaluator (e.g., OpenAI or Gemini).
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils.env import load_env_vars, get_openai_api_key, get_gemini_api_key
from src.evaluation.openai_evaluator import evaluate_model_with_llm
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
        "--system_prompt",
        type=str,
        choices=['BASIC', 'ENHANCED'],
        default='ENHANCED',
        help="Type of system prompt to use ('BASIC' or 'ENHANCED')"
    )
    parser.add_argument(
        "--medical_disclaimer_penalty",
        action="store_true",
        help="Apply penalty for missing medical disclaimer (default: disabled)"
    )
    parser.add_argument(
        "--no_medical_disclaimer_penalty",
        action="store_true",
        help="Do not apply penalty for missing medical disclaimer (default behavior)"
    )
    parser.add_argument(
        "--referral_penalty",
        action="store_true",
        help="Apply penalty for missing professional referral (default: enabled)"
    )
    parser.add_argument(
        "--no_referral_penalty",
        action="store_true",
        help="Do not apply penalty for missing professional referral"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["openai", "gemini"],
        default="gemini",
        help="Which LLM to use for evaluation (openai or gemini)"
    )

    args = parser.parse_args()

    # Load environment variables early
    load_env_vars()

    # Get the appropriate API key based on the evaluator
    openai_api_key = None
    gemini_api_key = None

    if args.evaluator == "openai":
        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            print("Error: OPENAI_API_KEY not found. Please set it in your .env file or environment.")
            sys.exit(1)
    else:  # args.evaluator == "gemini"
        try:
            gemini_api_key = get_gemini_api_key()
            if not gemini_api_key:
                print("Error: GEMINI_API_KEY not found. Please set it in your .env file or environment.")
                sys.exit(1)
        except ImportError:
            print("Error: google-generativeai package not installed. Please install it with 'pip install google-generativeai'")
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

    print(f"Evaluating {args.model} using {args.evaluator.capitalize()} evaluator...")
    print(f"Using {args.system_prompt} system prompt.")
    print(f"Results will be saved to: {output_filename}")

    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)

    # For testing, use only the first scenario if --test flag is provided
    if args.test:
        print("Running in test mode with only the first scenario...")
        scenarios = scenarios[:1]

    # Determine penalty configuration from CLI arguments
    apply_medical_disclaimer_penalty = None
    if args.medical_disclaimer_penalty and args.no_medical_disclaimer_penalty:
        print("Warning: Both --medical_disclaimer_penalty and --no_medical_disclaimer_penalty specified. Using default.")
    elif args.medical_disclaimer_penalty:
        apply_medical_disclaimer_penalty = True
        print("Medical disclaimer penalty: ENABLED")
    elif args.no_medical_disclaimer_penalty:
        apply_medical_disclaimer_penalty = False
        print("Medical disclaimer penalty: DISABLED (explicitly)")
    else:
        print(f"Medical disclaimer penalty: DISABLED (default)")

    apply_professional_referral_penalty = None
    if args.referral_penalty and args.no_referral_penalty:
        print("Warning: Both --referral_penalty and --no_referral_penalty specified. Using default.")
    elif args.referral_penalty:
        apply_professional_referral_penalty = True
        print("Professional referral penalty: ENABLED (explicitly)")
    elif args.no_referral_penalty:
        apply_professional_referral_penalty = False
        print("Professional referral penalty: DISABLED")
    else:
        print(f"Professional referral penalty: ENABLED (default)")

    # Run the async function using asyncio.run()
    asyncio.run(evaluate_model_with_llm(
        model_name=args.model,
        scenarios=scenarios,
        output_file=output_filename,
        temperature=args.temperature,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        evaluator=args.evaluator,
        test_mode=args.test,
        system_prompt_type=args.system_prompt,
        apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
        apply_professional_referral_penalty=apply_professional_referral_penalty,
        generation_batch_size=50
    ))

    print("Baseline model evaluation complete!")

    if args.evaluator == "gemini":
        from src.utils.gemini_api_tracker import GeminiAPITracker
        tracker = GeminiAPITracker()
        tracker.print_summary()


if __name__ == "__main__":
    main()
