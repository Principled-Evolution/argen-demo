"""
Script to prepare combined datasets (healthcare + domain) for ArGen GRPO fine-tuning.
"""

import sys
import os
import json
from typing import Dict, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils.prepare_challenging_dataset import (
    create_dataset as create_healthcare_dataset,
    create_evaluation_dataset as create_healthcare_evaluation,
    create_predibase_dataset as create_healthcare_predibase
)

from src.data_utils.prepare_domain_dataset import (
    create_dataset as create_domain_dataset,
    create_evaluation_dataset as create_domain_evaluation,
    create_predibase_dataset as create_domain_predibase
)


def combine_datasets(input_paths: List[str], output_path: str) -> None:
    """
    Combine multiple JSONL datasets into a single dataset.
    
    Args:
        input_paths: List of paths to the input datasets
        output_path: Path to save the combined dataset
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine the datasets
    combined_data = []
    for input_path in input_paths:
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        combined_data.append(json.loads(line))
    
    # Write the combined dataset to a JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created combined dataset with {len(combined_data)} scenarios at {output_path}")


def main():
    """Run the combined dataset preparation script."""
    print("Preparing combined datasets for ArGen GRPO fine-tuning...")
    
    # Create the healthcare datasets
    create_healthcare_dataset("data/healthcare_scenarios.jsonl")
    create_healthcare_evaluation("data/healthcare_evaluation.jsonl")
    create_healthcare_predibase("data/healthcare_predibase.jsonl")
    
    # Create the domain datasets
    create_domain_dataset("data/domain_scenarios.jsonl")
    create_domain_evaluation("data/domain_evaluation.jsonl")
    create_domain_predibase("data/domain_predibase.jsonl")
    
    # Combine the datasets
    combine_datasets(
        ["data/healthcare_scenarios.jsonl", "data/domain_scenarios.jsonl"],
        "data/combined_scenarios.jsonl"
    )
    
    combine_datasets(
        ["data/healthcare_evaluation.jsonl", "data/domain_evaluation.jsonl"],
        "data/combined_evaluation.jsonl"
    )
    
    combine_datasets(
        ["data/healthcare_predibase.jsonl", "data/domain_predibase.jsonl"],
        "data/combined_predibase.jsonl"
    )
    
    print("Combined dataset preparation complete!")


if __name__ == "__main__":
    main()
