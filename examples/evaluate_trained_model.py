#!/usr/bin/env python3
"""
Script to evaluate a GRPO-trained model using the same metrics as the baseline.
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional
import datetime
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.env import load_env_vars, get_openai_api_key
from src.evaluation.openai_evaluator import evaluate_model_with_openai
from src.config import DEFAULT_SCENARIOS_PATH, DEFAULT_TEMPERATURE

def check_dependencies():
    """Check if necessary dependencies for local inference are installed."""
    try:
        import torch
        import transformers
        print("torch and transformers libraries found.")
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, will use CPU (might be slow).")
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}. Please install required libraries.")
        print("Hint: pip install torch transformers")
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

def load_baseline_results(file_path: str) -> Dict:
    """
    Load baseline evaluation results from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing baseline results
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_metrics(baseline_metrics: Dict, trained_metrics: Dict) -> Dict:
    """
    Compare metrics between baseline and trained models.
    
    Args:
        baseline_metrics: Metrics from baseline model
        trained_metrics: Metrics from trained model
        
    Returns:
        Dict containing comparison results
    """
    comparison = {
        "ahimsa_score": {
            "baseline": baseline_metrics["average_ahimsa_score"],
            "trained": trained_metrics["average_ahimsa_score"],
            "improvement": trained_metrics["average_ahimsa_score"] - baseline_metrics["average_ahimsa_score"],
            "relative_improvement": (
                (trained_metrics["average_ahimsa_score"] - baseline_metrics["average_ahimsa_score"]) / 
                max(baseline_metrics["average_ahimsa_score"], 0.001) * 100
            ),
        },
        "dharma_score": {
            "baseline": baseline_metrics["average_dharma_score"],
            "trained": trained_metrics["average_dharma_score"],
            "improvement": trained_metrics["average_dharma_score"] - baseline_metrics["average_dharma_score"],
            "relative_improvement": (
                (trained_metrics["average_dharma_score"] - baseline_metrics["average_dharma_score"]) / 
                max(baseline_metrics["average_dharma_score"], 0.001) * 100
            ),
        },
        "combined_score": {
            "baseline": baseline_metrics["average_combined_score"],
            "trained": trained_metrics["average_combined_score"],
            "improvement": trained_metrics["average_combined_score"] - baseline_metrics["average_combined_score"],
            "relative_improvement": (
                (trained_metrics["average_combined_score"] - baseline_metrics["average_combined_score"]) / 
                max(baseline_metrics["average_combined_score"], 0.001) * 100
            ),
        },
        "ahimsa_violations": {
            "baseline": baseline_metrics["ahimsa_violations"],
            "trained": trained_metrics["ahimsa_violations"],
            "improvement": baseline_metrics["ahimsa_violations"] - trained_metrics["ahimsa_violations"],
            "relative_improvement": (
                (baseline_metrics["ahimsa_violations"] - trained_metrics["ahimsa_violations"]) / 
                max(baseline_metrics["ahimsa_violations"], 0.001) * 100
            ),
        },
        "dharma_violations": {
            "baseline": baseline_metrics["dharma_violations"],
            "trained": trained_metrics["dharma_violations"],
            "improvement": baseline_metrics["dharma_violations"] - trained_metrics["dharma_violations"],
            "relative_improvement": (
                (baseline_metrics["dharma_violations"] - trained_metrics["dharma_violations"]) / 
                max(baseline_metrics["dharma_violations"], 0.001) * 100
            ),
        },
    }
    
    return comparison

def main():
    """Run the evaluation script for GRPO-trained model."""
    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained model against the baseline.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--scenarios", 
        type=str,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the scenarios file (same as used for baseline)"
    )
    parser.add_argument(
        "--baseline_results",
        type=str,
        required=True,
        help="Path to the baseline evaluation results JSON file"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="data/trained_openai_results",
        help="Base path for the evaluation results (timestamp and .json will be added)"
    )
    parser.add_argument(
        "--temperature", 
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for generation (should match baseline)"
    )
    parser.add_argument(
        "--use_basic_prompt",
        action="store_true",
        help="Use the original basic system prompt instead of the enhanced one (match baseline)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with mock responses"
    )
    
    args = parser.parse_args()
    
    # Load environment variables early
    load_env_vars()
    
    # Get the API key
    openai_api_key = get_openai_api_key()
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
    
    # Check dependencies
    print("Checking dependencies for local model evaluation...")
    if not check_dependencies():
        sys.exit(1)
    
    print(f"Evaluating trained model {args.model} using OpenAI evaluator...")
    print(f"Will compare against baseline results from {args.baseline_results}")
    print(f"Results will be saved to: {output_filename}")
    
    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)
    
    # For testing, use only the first scenario if --test flag is provided
    if args.test:
        print("Running in test mode with only the first scenario...")
        scenarios = scenarios[:1]
    
    # Evaluate the trained model
    asyncio.run(evaluate_model_with_openai(
        model_name=args.model,
        scenarios=scenarios,
        output_file=output_filename,
        temperature=args.temperature,
        openai_api_key=openai_api_key,
        test_mode=args.test,
        use_basic_prompt=args.use_basic_prompt
    ))
    
    # Load the baseline and trained results
    baseline_results = load_baseline_results(args.baseline_results)
    with open(output_filename, 'r', encoding='utf-8') as f:
        trained_results = json.load(f)
    
    # Compare metrics
    comparison = compare_metrics(
        baseline_results["summary_metrics"],
        trained_results["summary_metrics"]
    )
    
    # Update the trained results with comparison
    trained_results["comparison_to_baseline"] = comparison
    
    # Save the updated results
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(trained_results, f, indent=4)
    
    # Print comparison summary
    print("\n===== COMPARISON TO BASELINE =====")
    print(f"Ahimsa Score: {comparison['ahimsa_score']['baseline']:.3f} → {comparison['ahimsa_score']['trained']:.3f} ({comparison['ahimsa_score']['relative_improvement']:.1f}% change)")
    print(f"Dharma Score: {comparison['dharma_score']['baseline']:.3f} → {comparison['dharma_score']['trained']:.3f} ({comparison['dharma_score']['relative_improvement']:.1f}% change)")
    print(f"Combined Score: {comparison['combined_score']['baseline']:.3f} → {comparison['combined_score']['trained']:.3f} ({comparison['combined_score']['relative_improvement']:.1f}% change)")
    print(f"Ahimsa Violations: {comparison['ahimsa_violations']['baseline']} → {comparison['ahimsa_violations']['trained']} ({comparison['ahimsa_violations']['improvement']} fewer)")
    print(f"Dharma Violations: {comparison['dharma_violations']['baseline']} → {comparison['dharma_violations']['trained']} ({comparison['dharma_violations']['improvement']} fewer)")
    print("===============================")
    
    print("Trained model evaluation complete!")

if __name__ == "__main__":
    main() 