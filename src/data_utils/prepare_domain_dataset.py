"""
Script to prepare a dataset of domain violation scenarios for ArGen GRPO fine-tuning.
"""

import json
import os
from typing import Dict, List, Optional

# Import the domain violation scenarios
from data.domain_scenarios import DOMAIN_VIOLATION_SCENARIOS


def create_dataset(output_path: str, scenarios: Optional[List[Dict]] = None) -> None:
    """
    Create a dataset of domain violation scenarios for ArGen GRPO fine-tuning.
    
    Args:
        output_path: Path to save the dataset
        scenarios: List of scenarios to include in the dataset
    """
    if scenarios is None:
        scenarios = DOMAIN_VIOLATION_SCENARIOS
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the scenarios to a JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + '\n')
    
    print(f"Created dataset with {len(scenarios)} scenarios at {output_path}")


def create_evaluation_dataset(
    output_path: str,
    scenarios: Optional[List[Dict]] = None
) -> None:
    """
    Create an evaluation dataset of domain violation scenarios.
    
    Args:
        output_path: Path to save the dataset
        scenarios: List of scenarios to include in the dataset
    """
    if scenarios is None:
        scenarios = DOMAIN_VIOLATION_SCENARIOS
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare the evaluation scenarios
    eval_scenarios = []
    for scenario in scenarios:
        eval_scenario = {
            "prompt": scenario["prompt"],
            "role": scenario["role"],
            "patient_context": scenario["patient_context"],
            "domain": scenario["domain"]
        }
        
        eval_scenarios.append(eval_scenario)
    
    # Write the evaluation scenarios to a JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for scenario in eval_scenarios:
            f.write(json.dumps(scenario) + '\n')
    
    print(f"Created evaluation dataset with {len(eval_scenarios)} scenarios at {output_path}")


def create_predibase_dataset(
    output_path: str,
    scenarios: Optional[List[Dict]] = None
) -> None:
    """
    Create a dataset for Predibase GRPO fine-tuning.
    
    Args:
        output_path: Path to save the dataset
        scenarios: List of scenarios to include in the dataset
    """
    if scenarios is None:
        scenarios = DOMAIN_VIOLATION_SCENARIOS
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare the Predibase scenarios
    predibase_scenarios = []
    for scenario in scenarios:
        # Create a scenario with the prompt only
        predibase_scenario = {
            "prompt": scenario["prompt"],
            "role": scenario["role"],
            "patient_context": scenario["patient_context"],
            "domain": scenario["domain"]
        }
        
        predibase_scenarios.append(predibase_scenario)
    
    # Write the Predibase scenarios to a JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for scenario in predibase_scenarios:
            f.write(json.dumps(scenario) + '\n')
    
    print(f"Created Predibase dataset with {len(predibase_scenarios)} scenarios at {output_path}")


if __name__ == "__main__":
    # Create the datasets
    create_dataset("data/domain_scenarios.jsonl")
    create_evaluation_dataset("data/domain_evaluation.jsonl")
    create_predibase_dataset("data/domain_predibase.jsonl")
