"""
Script to evaluate a base model on the healthcare scenarios dataset.
"""

import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluate_model import evaluate_model, load_scenarios


def main():
    """Run the base model evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a base model on the healthcare scenarios dataset.")
    parser.add_argument("--model", type=str, default="microsoft/phi-3-mini-4k-instruct", help="Name of the model to evaluate")
    parser.add_argument("--scenarios", type=str, default="data/healthcare_evaluation.jsonl", help="Path to the scenarios file")
    parser.add_argument("--output", type=str, default="data/baseline_results.json", help="Path to save the evaluation results")
    parser.add_argument("--use-opa", action="store_true", help="Whether to use OPA for policy checking")
    parser.add_argument("--policy", type=str, default="gopal/custom/dharmic/v1/ahimsa/ahimsa_simplified.rego", help="Path to the OPA policy file")
    parser.add_argument("--use-predibase", action="store_true", help="Whether to use Predibase for model inference")
    parser.add_argument("--test", action="store_true", help="Run in test mode with only the first scenario")

    args = parser.parse_args()

    print(f"Evaluating base model {args.model} on healthcare scenarios...")

    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)

    # For testing, use only the first scenario if --test flag is provided
    if args.test:
        print("Running in test mode with only the first scenario...")
        scenarios = scenarios[:1]

    # Evaluate the model
    evaluate_model(
        args.model,
        scenarios,
        args.output,
        args.use_opa,
        args.policy,
        args.use_predibase
    )

    print("Base model evaluation complete!")


if __name__ == "__main__":
    main()
