"""
Dataset utilities for ArGen GRPO fine-tuning.
"""

import json
import os
from typing import Dict, List, Optional, Union


def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """
    Load a dataset from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, each representing an example
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    return examples


def validate_dataset(examples: List[Dict], required_fields: Optional[List[str]] = None) -> bool:
    """
    Validate that a dataset contains the required fields.
    
    Args:
        examples: List of examples to validate
        required_fields: List of field names that must be present in each example
            Defaults to ["prompt"]
            
    Returns:
        True if the dataset is valid, False otherwise
    """
    if not examples:
        return False
    
    if required_fields is None:
        required_fields = ["prompt"]
    
    for example in examples:
        for field in required_fields:
            if field not in example:
                return False
    
    return True





def split_dataset(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        examples: List of examples to split
        train_ratio: Proportion of examples to use for training
        val_ratio: Proportion of examples to use for validation
        test_ratio: Proportion of examples to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing the split datasets
    """
    import random
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Split ratios must sum to 1.0")
    
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    train_size = int(len(shuffled) * train_ratio)
    val_size = int(len(shuffled) * val_ratio)
    
    train_set = shuffled[:train_size]
    val_set = shuffled[train_size:train_size + val_size]
    test_set = shuffled[train_size + val_size:]
    
    return {
        "train": train_set,
        "validation": val_set,
        "test": test_set
    }
