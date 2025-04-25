"""
Script to check and download a dataset from Predibase.
"""

import sys
import os
import json
import argparse
import pandas as pd
from typing import Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_predibase_client():
    """
    Get a Predibase client.
    
    Returns:
        Predibase client
    """
    try:
        from predibase import Predibase
    except ImportError:
        raise ImportError(
            "Predibase SDK not installed. Please install it with 'pip install predibase'."
        )
    
    # Get API token from config file
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                raise ValueError("API token not found in config file")
    else:
        raise ValueError("Config file not found")
    
    # Initialize Predibase client
    print(f"Initializing Predibase client...")
    pb = Predibase(api_token=api_token)
    
    return pb


def check_dataset(dataset_name: str) -> Optional[str]:
    """
    Check if a dataset exists in Predibase.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset ID if the dataset exists, None otherwise
    """
    pb = get_predibase_client()
    
    # Try to get the dataset
    try:
        dataset = pb.datasets.get(dataset_name)
        print(f"Dataset '{dataset_name}' exists with ID: {dataset.id}")
        return dataset.id
    except Exception as e:
        print(f"Dataset '{dataset_name}' does not exist or cannot be accessed: {e}")
        return None


def download_dataset(dataset_name: str, output_path: str) -> bool:
    """
    Download a dataset from Predibase.
    
    Args:
        dataset_name: Name of the dataset
        output_path: Path to save the dataset
        
    Returns:
        True if the dataset was downloaded successfully, False otherwise
    """
    pb = get_predibase_client()
    
    # Try to get the dataset
    try:
        dataset = pb.datasets.get(dataset_name)
        print(f"Dataset '{dataset_name}' found with ID: {dataset.id}")
        
        # Download the dataset
        print(f"Downloading dataset to {output_path}...")
        dataset.download(output_path)
        
        print(f"Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def create_dataset_with_completion(input_path: str, output_path: str) -> None:
    """
    Create a dataset with a completion column.
    
    Args:
        input_path: Path to the input dataset
        output_path: Path to save the output dataset
    """
    # Load the dataset
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Check if the dataset has the required columns
    if 'prompt' not in df.columns:
        raise ValueError("Dataset does not have 'prompt' column")
    
    # Create completion column if it doesn't exist
    if 'completion' not in df.columns:
        print("Creating 'completion' column...")
        
        # If we have 'role' column, use it to create completion
        if 'role' in df.columns:
            print("Creating 'completion' from 'role' column...")
            df['completion'] = df['role'].apply(lambda x: f"You are a {x}.")
        else:
            # Default completion
            print("Creating default 'completion' column...")
            df['completion'] = "You are a helpful assistant."
    
    # Save the dataset
    print(f"Saving dataset with completion column to {output_path}...")
    df.to_json(output_path, orient='records', lines=True)
    
    print(f"Dataset saved successfully!")


def main():
    """Run the dataset check script."""
    parser = argparse.ArgumentParser(description="Check and download a dataset from Predibase.")
    parser.add_argument("--dataset-name", type=str, default="combined_predibase", help="Name of the dataset in Predibase")
    parser.add_argument("--output-path", type=str, default="data/downloaded_dataset.jsonl", help="Path to save the downloaded dataset")
    parser.add_argument("--create-completion", action="store_true", help="Create a dataset with a completion column")
    parser.add_argument("--input-path", type=str, default="data/combined_predibase.jsonl", help="Path to the input dataset (for --create-completion)")
    
    args = parser.parse_args()
    
    if args.create_completion:
        # Create a dataset with a completion column
        output_path = args.input_path.replace('.jsonl', '_with_completion.jsonl')
        create_dataset_with_completion(args.input_path, output_path)
    else:
        # Check if the dataset exists
        dataset_id = check_dataset(args.dataset_name)
        
        if dataset_id:
            # Download the dataset
            download_dataset(args.dataset_name, args.output_path)


if __name__ == "__main__":
    main()
