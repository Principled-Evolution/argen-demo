#!/usr/bin/env python3
"""
Processes a JSONL dataset file to add a hash of the 'prompt' field
to the 'tier' field for data integrity checks.

Example:
python scripts/add_prompt_hash_to_dataset.py \\
    --input data/my_scenarios.jsonl \\
    --output data/my_scenarios_hashed.jsonl
"""

import argparse
import json
import sys
import os
import logging
from tqdm import tqdm

# Add project root to path to allow importing 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from src.utils.data_integrity import (
        calculate_prompt_hash,
        create_compound_tier,
        _DELIMITER # Import delimiter for checking if already processed
    )
except ImportError as e:
    print(f"Error importing hashing utilities: {e}")
    print("Ensure you are running this script from the project root directory or have the 'src' directory in your PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_dataset(input_path: str, output_path: str):
    """
    Reads a JSONL file, adds prompt hash to the tier field, and writes
    to a new JSONL file.

    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to the output JSONL file.
    """
    logger.info(f"Starting processing of dataset: {input_path}")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Get total lines for progress bar
        with open(input_path, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for line in infile if line.strip())

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line in tqdm(infile, total=total_lines, desc="Processing lines"):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    if 'prompt' not in data or 'tier' not in data:
                        logger.warning(f"Skipping line due to missing 'prompt' or 'tier': {line[:100]}...")
                        skipped_count += 1
                        # Write the original line if skipping? Or omit? Omit for now.
                        continue

                    prompt = data['prompt']
                    original_tier = str(data['tier']) # Ensure it's a string

                    # Check if tier seems already processed
                    if _DELIMITER in original_tier:
                         logger.debug(f"Tier '{original_tier}' appears already processed. Skipping hashing.")
                         # Just write the original data
                         outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                         skipped_count += 1 # Count as skipped for hashing
                         continue


                    # Calculate hash and create compound tier
                    prompt_hash = calculate_prompt_hash(prompt)
                    compound_tier = create_compound_tier(original_tier, prompt_hash)

                    # Update the data dictionary
                    data['tier'] = compound_tier

                    # Write the modified data to the output file
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed_count += 1

                except json.JSONDecodeError:
                    logger.error(f"Skipping invalid JSON line: {line[:100]}...")
                    error_count += 1
                except Exception as e:
                    logger.error(f"Error processing line: {line[:100]}... Error: {e}", exc_info=True)
                    error_count += 1

    except IOError as e:
        logger.error(f"File I/O error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

    logger.info("Processing complete.")
    logger.info(f"Lines processed and hashed: {processed_count}")
    logger.info(f"Lines skipped (missing fields or already processed): {skipped_count}")
    logger.info(f"Lines with errors: {error_count}")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add prompt hashes to tier field in a JSONL dataset.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL dataset file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL dataset file.")
    args = parser.parse_args()

    process_dataset(args.input, args.output) 