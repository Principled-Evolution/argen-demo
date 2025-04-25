"""
Script to evaluate the baseline model using Gemini AI.
"""

import sys
import os
import json
import argparse
from typing import Dict, List
import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.env import load_env_vars
from src.evaluation.gemini_evaluator import evaluate_model_with_gemini


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
    """Run the baseline model evaluation script with Gemini."""
    parser = argparse.ArgumentParser(description="Evaluate the baseline model using Gemini AI.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the model to evaluate")
    parser.add_argument("--scenarios", type=str, default="data/combined_predibase_updated.jsonl", help="Path to the scenarios file")
    parser.add_argument("--output_base", type=str, default="data/baseline_gemini_results", help="Base path and filename for the evaluation results (timestamp and .json will be added)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock responses")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation (higher = more random)")
    parser.add_argument("--use_basic_prompt", action="store_true", help="Use the original basic system prompt instead of the enhanced 'fairer' one.")
    
    args = parser.parse_args()
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct the final output filename
    output_filename = f"{args.output_base}_{timestamp}.json"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load environment variables
    load_env_vars()
    
    print(f"Evaluating {args.model} using Gemini AI...")
    if args.use_basic_prompt:
        print("Using basic system prompt.")
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
    
    # Evaluate the model
    evaluate_model_with_gemini(
        args.model,
        scenarios,
        output_filename,
        args.temperature,
        test_mode=args.test,
        use_basic_prompt=args.use_basic_prompt
    )
    
    print("Baseline model evaluation with Gemini complete!")


if __name__ == "__main__":
    main()
