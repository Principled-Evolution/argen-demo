"""
Script to prepare datasets for ArGen GRPO fine-tuning.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils.prepare_dataset import (
    create_dataset,
    create_evaluation_dataset,
    create_predibase_dataset
)


def main():
    """Run the dataset preparation script."""
    print("Preparing datasets for ArGen GRPO fine-tuning...")
    
    # Create the datasets
    create_dataset("data/healthcare_scenarios.jsonl")
    create_evaluation_dataset("data/healthcare_evaluation.jsonl")
    create_predibase_dataset("data/healthcare_predibase.jsonl")
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
