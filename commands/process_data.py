#!/usr/bin/env python3
"""
Data Pipeline Command

This command processes a JSONL file through multiple stages:
1. Reevaluate tiers using argen/data/utils/reevaluate_tiers_cli.py
2. Reevaluate scope using argen/data/utils/reevaluate_scope_cli.py
3. Add prompt hash using tools/add_prompt_hash.py

The final output file is suffixed with '-hashprompt' and placed in the same location as the input file.
No intermediate files are retained.

Usage:
  python commands/process_data.py input_file.jsonl

Example:
  python commands/process_data.py data/combined_predibase_updated.jsonl
"""

import argparse
import os
import sys
import tempfile
import subprocess
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path to allow importing from argen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def process_file(input_file):
    """
    Process the input file through the pipeline.

    Args:
        input_file: Path to the input JSONL file

    Returns:
        Path to the final output file or None if processing failed
    """
    input_file = os.path.abspath(input_file)

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")

        # Define intermediate file paths
        input_path = Path(input_file)
        base_name = input_path.stem

        # Step 1: Process through reevaluate_tiers_cli.py
        tiers_output = os.path.join(temp_dir, f"{base_name}-reevaluated-tiers-gemini.jsonl")
        logger.info(f"Running tier evaluation...")

        tiers_cmd = [
            sys.executable,
            os.path.join(project_root, "argen/data/utils/reevaluate_tiers_cli.py"),
            input_file,
            "--evaluator", "gemini"
        ]

        try:
            subprocess.run(tiers_cmd, check=True)

            # The script creates an output file with a specific naming pattern
            expected_tiers_output = f"{str(input_path.parent / base_name)}-reevaluated-tiers-gemini.jsonl"

            if os.path.exists(expected_tiers_output):
                # Move the file to our temp directory
                shutil.move(expected_tiers_output, tiers_output)
                logger.info(f"Tier evaluation completed successfully")
            else:
                logger.error(f"Expected tier output file not found: {expected_tiers_output}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during tier evaluation: {e}")
            return None

        # Step 2: Process through reevaluate_scope_cli.py
        scope_output = os.path.join(temp_dir, f"{base_name}-with-scope-gemini.jsonl")
        logger.info(f"Running scope evaluation...")

        scope_cmd = [
            sys.executable,
            os.path.join(project_root, "argen/data/utils/reevaluate_scope_cli.py"),
            tiers_output,
            "--evaluator", "gemini"
        ]

        try:
            subprocess.run(scope_cmd, check=True)

            # The script creates an output file with a specific naming pattern
            expected_scope_output = f"{tiers_output.replace('.jsonl', '')}-with-scope-gemini.jsonl"

            if os.path.exists(expected_scope_output):
                # Move the file to our temp directory
                shutil.move(expected_scope_output, scope_output)
                logger.info(f"Scope evaluation completed successfully")
            else:
                logger.error(f"Expected scope output file not found: {expected_scope_output}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during scope evaluation: {e}")
            return None

        # Step 3: Process through add_prompt_hash.py
        final_output = str(input_path.parent / f"{base_name}-hashprompt.jsonl")
        logger.info(f"Adding prompt hash...")

        hash_cmd = [
            sys.executable,
            os.path.join(project_root, "tools/add_prompt_hash.py"),
            "--input", scope_output,
            "--output", final_output
        ]

        try:
            subprocess.run(hash_cmd, check=True)

            if os.path.exists(final_output):
                logger.info(f"Prompt hash addition completed successfully")
            else:
                logger.error(f"Expected final output file not found: {final_output}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during prompt hash addition: {e}")
            return None

        logger.info(f"Pipeline completed successfully")
        logger.info(f"Final output saved to: {final_output}")

        return final_output

def main():
    parser = argparse.ArgumentParser(
        description="Process a JSONL file through the data pipeline."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file"
    )
    args = parser.parse_args()

    output_file = process_file(args.input_file)

    if output_file:
        logger.info(f"Pipeline completed successfully. Output: {output_file}")
        return 0
    else:
        logger.error("Pipeline failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
