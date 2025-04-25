"""
Script to upload a dataset to Predibase.
"""

import sys
import os
import json
import argparse
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def upload_dataset_to_predibase(
    dataset_path: str,
    dataset_name: str,
    description: str = "Dataset for ArGen GRPO fine-tuning"
) -> str:
    """
    Upload a dataset to Predibase.

    Args:
        dataset_path: Path to the dataset file
        dataset_name: Name to give the dataset in Predibase
        description: Description of the dataset

    Returns:
        The dataset ID
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

    # Check if dataset already exists
    print(f"Checking if dataset {dataset_name} already exists...")
    try:
        # Try to get the dataset by name
        dataset = pb.datasets.get(dataset_name)
        print(f"Dataset {dataset_name} already exists with ID: {dataset.id}")
        return dataset.id
    except Exception as e:
        print(f"Dataset {dataset_name} does not exist. Creating...")

    # Upload the dataset
    print(f"Uploading dataset {dataset_path} to Predibase as {dataset_name}...")

    # Based on the available methods in the Predibase SDK
    try:
        # First, create a dataset from a local file
        dataset = pb.datasets.from_file(
            file_path=dataset_path,
            name=dataset_name
        )

        print(f"Dataset uploaded successfully! Dataset ID: {dataset.id}")
        return dataset.id
    except Exception as e:
        print(f"Error using from_file method: {e}")

        # Try alternative method - read the file and use from_pandas_dataframe
        import pandas as pd

        print("Trying to upload using from_pandas_dataframe method...")
        # Read the JSONL file into a pandas DataFrame
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        df = pd.DataFrame(data)

        # Upload the DataFrame
        dataset = pb.datasets.from_pandas_dataframe(
            df=df,
            name=dataset_name
        )

        print(f"Dataset uploaded successfully using from_pandas! Dataset ID: {dataset.id}")
        return dataset.id


def main():
    """Run the dataset upload script."""
    parser = argparse.ArgumentParser(description="Upload a dataset to Predibase.")
    parser.add_argument("--dataset-path", type=str, default="data/combined_predibase_with_completion.jsonl", help="Path to the dataset file")
    parser.add_argument("--dataset-name", type=str, default="argen_dataset", help="Name to give the dataset in Predibase")
    parser.add_argument("--description", type=str, default="Dataset for ArGen GRPO fine-tuning", help="Description of the dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Enable debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Check if dataset file exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        print("Please run examples/check_predibase_dataset.py --create-completion first.")
        sys.exit(1)

    # Upload the dataset
    try:
        # Print debug information
        print(f"Dataset path: {args.dataset_path}")
        print(f"Dataset content preview:")
        with open(args.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 2:  # Print first 2 lines
                    print(line.strip())
                else:
                    break

        # Check if the dataset has the required columns
        import pandas as pd
        data = []
        with open(args.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        df = pd.DataFrame(data)
        print(f"Dataset columns: {df.columns.tolist()}")

        # Verify that the dataset has the required columns
        if 'prompt' not in df.columns or 'completion' not in df.columns:
            print("Error: Dataset does not have 'prompt' and 'completion' columns.")
            print("Predibase requires these columns for training.")
            sys.exit(1)

        dataset_id = upload_dataset_to_predibase(
            args.dataset_path,
            args.dataset_name,
            args.description
        )

        print(f"Dataset uploaded with ID: {dataset_id}")
        print(f"You can now use this dataset for GRPO fine-tuning with the name: {args.dataset_name}")
    except Exception as e:
        import traceback
        print(f"Error uploading dataset: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
