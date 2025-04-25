"""
Script to evaluate the fine-tuned llama-3-2-1b-instruct model on challenging healthcare scenarios.
"""

import sys
import os
import json
import argparse
from typing import Dict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.evaluate_llama_baseline import evaluate_predibase_model, load_scenarios


def main():
    """Run the fine-tuned llama-3-2-1b-instruct model evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned llama-3-2-1b-instruct model on challenging healthcare scenarios.")
    parser.add_argument("--model", type=str, required=True, help="Name of the fine-tuned model in Predibase")
    parser.add_argument("--scenarios", type=str, default="data/challenging_evaluation.jsonl", help="Path to the scenarios file")
    parser.add_argument("--output", type=str, default="data/llama_finetuned_results.json", help="Path to save the evaluation results")
    parser.add_argument("--use-opa", action="store_true", help="Whether to use OPA for policy checking")
    parser.add_argument("--policy", type=str, default="gopal/custom/dharmic/v1/ahimsa/ahimsa_strict.rego", help="Path to the OPA policy file")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock responses")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation (higher = more random)")
    parser.add_argument("--baseline", type=str, default="data/llama_baseline_results.json", help="Path to the baseline results for comparison")
    
    args = parser.parse_args()
    
    print(f"Evaluating fine-tuned model {args.model} on challenging healthcare scenarios...")
    
    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)
    
    # For testing, use only the first scenario if --test flag is provided
    if args.test:
        print("Running in test mode with only the first scenario...")
        scenarios = scenarios[:1]
    
    # Evaluate the model
    finetuned_results = evaluate_predibase_model(
        args.model,
        scenarios,
        args.output,
        args.use_opa,
        args.policy,
        args.test,
        args.temperature
    )
    
    # Compare with baseline if available
    if os.path.exists(args.baseline):
        with open(args.baseline, 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        
        print("\nComparison with baseline:")
        print(f"Baseline model: {baseline_results['model_name']}")
        print(f"Fine-tuned model: {finetuned_results['model_name']}")
        print(f"Temperature: {args.temperature}")
        
        # Calculate metrics
        baseline_harmful_rate = baseline_results.get('harmful_rate', 0)
        finetuned_harmful_rate = finetuned_results.get('harmful_rate', 0)
        
        baseline_ahimsa_score = baseline_results.get('average_ahimsa_score', 0)
        finetuned_ahimsa_score = finetuned_results.get('average_ahimsa_score', 0)
        
        # Calculate improvements
        harmful_rate_improvement = baseline_harmful_rate - finetuned_harmful_rate
        ahimsa_score_improvement = finetuned_ahimsa_score - baseline_ahimsa_score
        
        # Print comparison
        print(f"Harmful rate - Baseline: {baseline_harmful_rate:.2%}, Fine-tuned: {finetuned_harmful_rate:.2%}")
        print(f"Harmful rate improvement: {harmful_rate_improvement:.2%}")
        print(f"Average Ahimsa score - Baseline: {baseline_ahimsa_score:.4f}, Fine-tuned: {finetuned_ahimsa_score:.4f}")
        print(f"Ahimsa score improvement: {ahimsa_score_improvement:.4f}")
        
        # Save comparison results
        comparison_results = {
            "baseline_model": baseline_results['model_name'],
            "finetuned_model": finetuned_results['model_name'],
            "temperature": args.temperature,
            "baseline_harmful_rate": baseline_harmful_rate,
            "finetuned_harmful_rate": finetuned_harmful_rate,
            "harmful_rate_improvement": harmful_rate_improvement,
            "baseline_ahimsa_score": baseline_ahimsa_score,
            "finetuned_ahimsa_score": finetuned_ahimsa_score,
            "ahimsa_score_improvement": ahimsa_score_improvement
        }
        
        comparison_path = os.path.join(os.path.dirname(args.output), "comparison_results.json")
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Comparison results saved to {comparison_path}")
    
    print("Fine-tuned model evaluation complete!")


if __name__ == "__main__":
    main()
