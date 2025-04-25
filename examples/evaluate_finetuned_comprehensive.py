"""
Script to evaluate the fine-tuned model on both Ahimsa and Dharma principles.
"""

import sys
import os
import json
import argparse
from typing import Dict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.evaluate_baseline_comprehensive import load_scenarios
from src.evaluation.comprehensive_evaluator import evaluate_model_comprehensive


def main():
    """Run the fine-tuned model evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model on both Ahimsa and Dharma principles.")
    parser.add_argument("--model", type=str, required=True, help="Name of the fine-tuned model in Predibase")
    parser.add_argument("--scenarios", type=str, default="data/combined_evaluation.jsonl", help="Path to the scenarios file")
    parser.add_argument("--output", type=str, default="data/finetuned_comprehensive_results.json", help="Path to save the evaluation results")
    parser.add_argument("--use-opa", action="store_true", help="Whether to use OPA for policy checking")
    parser.add_argument("--ahimsa-policy", type=str, default="policies/ahimsa_strict.rego", help="Path to the Ahimsa OPA policy file")
    parser.add_argument("--dharma-policy", type=str, default="policies/dharma_domain.rego", help="Path to the Dharma OPA policy file")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock responses")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation (higher = more random)")
    parser.add_argument("--baseline", type=str, default="data/baseline_comprehensive_results.json", help="Path to the baseline results for comparison")
    
    args = parser.parse_args()
    
    print(f"Evaluating fine-tuned model {args.model} on both Ahimsa and Dharma principles...")
    
    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)
    
    # For testing, use only the first scenario if --test flag is provided
    if args.test:
        print("Running in test mode with only the first scenario...")
        scenarios = scenarios[:1]
    
    # Evaluate the model
    finetuned_results = evaluate_model_comprehensive(
        args.model,
        scenarios,
        args.output,
        args.use_opa,
        args.ahimsa_policy,
        args.dharma_policy,
        args.temperature,
        test_mode=args.test
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
        baseline_ahimsa_violation_rate = baseline_results.get('ahimsa_violation_rate', 0)
        finetuned_ahimsa_violation_rate = finetuned_results.get('ahimsa_violation_rate', 0)
        
        baseline_dharma_violation_rate = baseline_results.get('dharma_violation_rate', 0)
        finetuned_dharma_violation_rate = finetuned_results.get('dharma_violation_rate', 0)
        
        baseline_ahimsa_score = baseline_results.get('average_ahimsa_score', 0)
        finetuned_ahimsa_score = finetuned_results.get('average_ahimsa_score', 0)
        
        baseline_dharma_score = baseline_results.get('average_dharma_score', 0)
        finetuned_dharma_score = finetuned_results.get('average_dharma_score', 0)
        
        baseline_combined_score = baseline_results.get('average_combined_score', 0)
        finetuned_combined_score = finetuned_results.get('average_combined_score', 0)
        
        # Calculate improvements
        ahimsa_violation_improvement = baseline_ahimsa_violation_rate - finetuned_ahimsa_violation_rate
        dharma_violation_improvement = baseline_dharma_violation_rate - finetuned_dharma_violation_rate
        
        ahimsa_score_improvement = finetuned_ahimsa_score - baseline_ahimsa_score
        dharma_score_improvement = finetuned_dharma_score - baseline_dharma_score
        combined_score_improvement = finetuned_combined_score - baseline_combined_score
        
        # Print comparison
        print(f"Ahimsa violation rate - Baseline: {baseline_ahimsa_violation_rate:.2%}, Fine-tuned: {finetuned_ahimsa_violation_rate:.2%}")
        print(f"Ahimsa violation rate improvement: {ahimsa_violation_improvement:.2%}")
        print(f"Average Ahimsa score - Baseline: {baseline_ahimsa_score:.4f}, Fine-tuned: {finetuned_ahimsa_score:.4f}")
        print(f"Ahimsa score improvement: {ahimsa_score_improvement:.4f}")
        
        print(f"Dharma violation rate - Baseline: {baseline_dharma_violation_rate:.2%}, Fine-tuned: {finetuned_dharma_violation_rate:.2%}")
        print(f"Dharma violation rate improvement: {dharma_violation_improvement:.2%}")
        print(f"Average Dharma score - Baseline: {baseline_dharma_score:.4f}, Fine-tuned: {finetuned_dharma_score:.4f}")
        print(f"Dharma score improvement: {dharma_score_improvement:.4f}")
        
        print(f"Average combined score - Baseline: {baseline_combined_score:.4f}, Fine-tuned: {finetuned_combined_score:.4f}")
        print(f"Combined score improvement: {combined_score_improvement:.4f}")
        
        # Save comparison results
        comparison_results = {
            "baseline_model": baseline_results['model_name'],
            "finetuned_model": finetuned_results['model_name'],
            "temperature": args.temperature,
            "baseline_ahimsa_violation_rate": baseline_ahimsa_violation_rate,
            "finetuned_ahimsa_violation_rate": finetuned_ahimsa_violation_rate,
            "ahimsa_violation_improvement": ahimsa_violation_improvement,
            "baseline_ahimsa_score": baseline_ahimsa_score,
            "finetuned_ahimsa_score": finetuned_ahimsa_score,
            "ahimsa_score_improvement": ahimsa_score_improvement,
            "baseline_dharma_violation_rate": baseline_dharma_violation_rate,
            "finetuned_dharma_violation_rate": finetuned_dharma_violation_rate,
            "dharma_violation_improvement": dharma_violation_improvement,
            "baseline_dharma_score": baseline_dharma_score,
            "finetuned_dharma_score": finetuned_dharma_score,
            "dharma_score_improvement": dharma_score_improvement,
            "baseline_combined_score": baseline_combined_score,
            "finetuned_combined_score": finetuned_combined_score,
            "combined_score_improvement": combined_score_improvement
        }
        
        comparison_path = os.path.join(os.path.dirname(args.output), "comprehensive_comparison_results.json")
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Comparison results saved to {comparison_path}")
    
    print("Fine-tuned model evaluation complete!")


if __name__ == "__main__":
    main()
