"""
Script to evaluate the baseline model on both Ahimsa and Dharma principles.
"""

import sys
import os
import json
import argparse
from typing import Dict, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.comprehensive_evaluator import evaluate_model_comprehensive


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
    parser = argparse.ArgumentParser(description="Evaluate the baseline model on both Ahimsa and Dharma principles.")
    parser.add_argument("--model", type=str, default="llama-3-2-1b-instruct", help="Name of the model to evaluate")
    parser.add_argument("--scenarios", type=str, default="data/combined_evaluation.jsonl", help="Path to the scenarios file")
    parser.add_argument("--output", type=str, default="data/baseline_comprehensive_results.json", help="Path to save the evaluation results")
    parser.add_argument("--use-opa", action="store_true", help="Whether to use OPA for policy checking")
    parser.add_argument("--ahimsa-policy", type=str, default="policies/ahimsa_strict.rego", help="Path to the Ahimsa OPA policy file")
    parser.add_argument("--dharma-policy", type=str, default="policies/dharma_domain.rego", help="Path to the Dharma OPA policy file")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock responses")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation (higher = more random)")
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model} on both Ahimsa and Dharma principles...")
    
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
    evaluate_model_comprehensive(
        args.model,
        scenarios,
        args.output,
        args.use_opa,
        args.ahimsa_policy,
        args.dharma_policy,
        args.temperature,
        test_mode=args.test
    )
    
    print("Baseline model evaluation complete!")


if __name__ == "__main__":
    main()
