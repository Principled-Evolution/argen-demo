"""
Script to evaluate a fine-tuned model on the healthcare scenarios dataset.
"""

import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluate_model import evaluate_model, load_scenarios


def main():
    """Run the fine-tuned model evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on the healthcare scenarios dataset.")
    parser.add_argument("--model", type=str, required=True, help="Name of the fine-tuned model in Predibase")
    parser.add_argument("--scenarios", type=str, default="data/healthcare_evaluation.jsonl", help="Path to the scenarios file")
    parser.add_argument("--output", type=str, default="data/finetuned_results.json", help="Path to save the evaluation results")
    parser.add_argument("--use-opa", action="store_true", help="Whether to use OPA for policy checking")
    parser.add_argument("--policy", type=str, default="gopal/custom/dharmic/v1/ahimsa/ahimsa_simplified.rego", help="Path to the OPA policy file")
    parser.add_argument("--baseline", type=str, default="data/baseline_results.json", help="Path to the baseline results for comparison")
    
    args = parser.parse_args()
    
    print(f"Evaluating fine-tuned model {args.model} on healthcare scenarios...")
    
    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)
    
    # Evaluate the model
    finetuned_results = evaluate_model(
        args.model,
        scenarios,
        args.output,
        args.use_opa,
        args.policy
    )
    
    # Compare with baseline if available
    if os.path.exists(args.baseline):
        import json
        
        with open(args.baseline, 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        
        print("\nComparison with baseline:")
        print(f"Baseline model: {baseline_results['model_name']}")
        print(f"Fine-tuned model: {finetuned_results['model_name']}")
        print(f"Harmful rate - Baseline: {baseline_results['harmful_rate']:.2%}, Fine-tuned: {finetuned_results['harmful_rate']:.2%}")
        print(f"Average Ahimsa score - Baseline: {baseline_results['average_ahimsa_score']:.4f}, Fine-tuned: {finetuned_results['average_ahimsa_score']:.4f}")
        
        # Calculate improvement
        harmful_rate_improvement = baseline_results['harmful_rate'] - finetuned_results['harmful_rate']
        score_improvement = finetuned_results['average_ahimsa_score'] - baseline_results['average_ahimsa_score']
        
        print(f"Improvement - Harmful rate: {harmful_rate_improvement:.2%}, Ahimsa score: {score_improvement:.4f}")
    
    print("Fine-tuned model evaluation complete!")


if __name__ == "__main__":
    main()
