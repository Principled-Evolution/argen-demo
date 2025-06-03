#!/usr/bin/env python3
"""
Processes a JSONL dataset file to add a hash of the 'prompt' field
to the 'tier' field and 'scope' field (if it exists) for data integrity checks.

Example:
python scripts/add_prompt_hash_to_dataset_new.py \\
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
    from argen.utils.data_integrity import (
        calculate_prompt_hash,
        create_compound_tier,
        _DELIMITER # Import delimiter for checking if already processed
    )
except ImportError as e:
    print(f"Error importing hashing utilities: {e}")
    print("Ensure you are running this script from the project root directory or have the 'argen' directory in your PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_dataset(input_path: str, output_path: str):
    """
    Reads a JSONL file, adds prompt hash to the tier field and scope field (if it exists),
    and writes to a new JSONL file.

    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to the output JSONL file.
    """
    logger.info(f"Starting processing of dataset: {input_path}")
    processed_count = 0
    skipped_count = 0
    error_count = 0
    scope_processed_count = 0

    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read all lines from the input file
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = [line.strip() for line in infile if line.strip()]

    # Process each line
    processed_lines = []
    for line in tqdm(lines, desc="Processing lines"):
        try:
            data = json.loads(line)

            if 'prompt' not in data or 'tier' not in data:
                logger.warning(f"Skipping line due to missing 'prompt' or 'tier': {line[:100]}...")
                skipped_count += 1
                continue

            prompt = data['prompt']
            original_tier = str(data['tier'])  # Ensure it's a string

            # Check if tier seems already processed
            tier_already_processed = _DELIMITER in original_tier

            # Check if scope exists and if it's already processed
            has_scope = 'scope' in data
            scope_already_processed = False
            if has_scope:
                original_scope = str(data['scope'])  # Ensure it's a string
                scope_already_processed = _DELIMITER in original_scope

            # If both tier and scope (if it exists) are already processed, skip
            if tier_already_processed and (not has_scope or scope_already_processed):
                logger.debug(f"Fields already processed. Skipping hashing.")
                processed_lines.append(line)
                skipped_count += 1
                continue

            # Calculate hash once for both fields
            prompt_hash = calculate_prompt_hash(prompt)

            # Process tier if not already processed
            if not tier_already_processed:
                compound_tier = create_compound_tier(original_tier, prompt_hash)
                data['tier'] = compound_tier

            # Process scope if it exists and not already processed
            if has_scope and not scope_already_processed:
                original_scope = str(data['scope'])  # Ensure it's a string
                compound_scope = create_compound_tier(original_scope, prompt_hash)
                data['scope'] = compound_scope
                scope_processed_count += 1

            # Add the processed data to the list
            processed_lines.append(json.dumps(data, ensure_ascii=False))
            processed_count += 1

        except json.JSONDecodeError:
            logger.error(f"Skipping invalid JSON line: {line[:100]}...")
            error_count += 1
        except Exception as e:
            logger.error(f"Error processing line: {line[:100]}... Error: {e}", exc_info=True)
            error_count += 1

    # Write all processed lines to the output file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in processed_lines:
            outfile.write(line + '\n')

    logger.info("Processing complete.")
    logger.info(f"Lines processed and hashed: {processed_count}")
    logger.info(f"Scope fields processed and hashed: {scope_processed_count}")
    logger.info(f"Lines skipped (missing fields or already processed): {skipped_count}")
    logger.info(f"Lines with errors: {error_count}")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add prompt hashes to tier and scope fields in a JSONL dataset.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL dataset file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL dataset file.")
    args = parser.parse_args()

    process_dataset(args.input, args.output)
