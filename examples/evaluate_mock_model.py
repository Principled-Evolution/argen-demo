"""
Script to evaluate a mock model on the healthcare scenarios dataset.

This script is for testing purposes only. It uses a mock model that returns
predefined responses for each scenario.
"""

import sys
import os
import json
import argparse
from typing import Dict, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import evaluate_model_response


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


def evaluate_mock_model(
    scenarios: List[Dict],
    output_path: str,
    use_opa: bool = False,
    policy_path: str = "gopal/custom/dharmic/v1/ahimsa/ahimsa_simplified.rego",
    use_harmful_responses: bool = False
) -> Dict:
    """
    Evaluate a mock model on the healthcare scenarios dataset.
    
    Args:
        scenarios: List of scenarios to evaluate the model on
        output_path: Path to save the evaluation results
        use_opa: Whether to use OPA for policy checking
        policy_path: Path to the OPA policy file
        use_harmful_responses: Whether to use harmful responses for testing
        
    Returns:
        Dictionary containing evaluation results
    """
    # Evaluate the model on each scenario
    results = []
    harmful_count = 0
    total_score = 0.0
    
    for i, scenario in enumerate(scenarios):
        print(f"Evaluating scenario {i+1}/{len(scenarios)}...")
        
        # Use the predefined harmful or safe response
        if use_harmful_responses and "harmful_response" in scenario:
            response = scenario["harmful_response"]
        elif "safe_response" in scenario:
            response = scenario["safe_response"]
        else:
            # Generate a generic safe response if none is provided
            response = (
                "I understand your concern. While I can provide some general information, "
                "I recommend consulting with a healthcare professional for personalized advice. "
                "They can provide guidance specific to your situation and help ensure you receive "
                "appropriate care."
            )
        
        # Evaluate the response
        evaluation = evaluate_model_response(
            response,
            scenario,
            policy_path=policy_path,
            use_opa=use_opa
        )
        
        # Update statistics
        if evaluation["is_harmful"]:
            harmful_count += 1
        
        total_score += evaluation["ahimsa_score"]
        
        # Add to results
        result = {
            "scenario_id": i,
            "prompt": scenario["prompt"],
            "response": response,
            "is_harmful": evaluation["is_harmful"],
            "reason": evaluation["reason"],
            "ahimsa_score": evaluation["ahimsa_score"],
        }
        
        results.append(result)
    
    # Calculate overall statistics
    harmful_rate = harmful_count / len(scenarios)
    average_score = total_score / len(scenarios)
    
    # Prepare the evaluation summary
    evaluation_summary = {
        "model_name": "mock_model_harmful" if use_harmful_responses else "mock_model_safe",
        "num_scenarios": len(scenarios),
        "harmful_count": harmful_count,
        "harmful_rate": harmful_rate,
        "average_ahimsa_score": average_score,
        "results": results
    }
    
    # Save the evaluation results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")
    
    # Print the evaluation summary
    print("\nEvaluation Summary:")
    print(f"Model: {'mock_model_harmful' if use_harmful_responses else 'mock_model_safe'}")
    print(f"Number of scenarios: {len(scenarios)}")
    print(f"Harmful responses: {harmful_count}/{len(scenarios)} ({harmful_rate:.2%})")
    print(f"Average Ahimsa score: {average_score:.4f}")
    
    return evaluation_summary


def main():
    """Run the mock model evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a mock model on the healthcare scenarios dataset.")
    parser.add_argument("--scenarios", type=str, default="data/healthcare_evaluation.jsonl", help="Path to the scenarios file")
    parser.add_argument("--output", type=str, default="data/mock_baseline_results.json", help="Path to save the evaluation results")
    parser.add_argument("--use-opa", action="store_true", help="Whether to use OPA for policy checking")
    parser.add_argument("--policy", type=str, default="gopal/custom/dharmic/v1/ahimsa/ahimsa_simplified.rego", help="Path to the OPA policy file")
    parser.add_argument("--harmful", action="store_true", help="Whether to use harmful responses for testing")
    
    args = parser.parse_args()
    
    print(f"Evaluating mock model on healthcare scenarios...")
    
    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)
    
    # Evaluate the model
    evaluate_mock_model(
        scenarios,
        args.output,
        args.use_opa,
        args.policy,
        args.harmful
    )
    
    print("Mock model evaluation complete!")


if __name__ == "__main__":
    main()
