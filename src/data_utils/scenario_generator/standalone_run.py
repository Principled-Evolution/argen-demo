#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standalone_run.py - Standalone script to run the scenario generator
==================================================================
This script provides a completely standalone way to run the scenario generator.
The generator now includes progress bars to track the generation, tiering, and
sampling stages in real-time.

Example usage:
    python standalone_run.py --datasets benchmarking --hf-model medalpaca/medalpaca-7b --count 20
    python standalone_run.py --datasets benchmarking --model gpt-4 --count 20
    python standalone_run.py --datasets smoke_test --batch-size 16 --concurrent-eval-limit 30
    python standalone_run.py --datasets grpo_training --exclude-from-file path/to/benchmarking.jsonl

Common parameters:
    --datasets: Space-separated list of datasets to generate (smoke_test, benchmarking, grpo_training)
    --count: Number of scenarios to generate (overrides dataset default)
    --model: OpenAI model to use for generation (e.g., gpt-4o-mini). Overrides default if --hf-model is not specified
    --hf-model: HuggingFace model to use for generation (e.g., medalpaca/medalpaca-7b)
    --temperature: Temperature for generation (higher = more diverse)
    --hf-max-new-tokens: Maximum number of tokens to generate (512 recommended for Medalpaca)
    --difficulty-ratio: Ratio for difficulty banding (-1 to disable)
    --fail-threshold: Threshold for accepting challenging prompts (0.0-1.0)
    --batch-size: Batch size for generation (default: 8)
    --concurrent-eval-limit: Maximum number of concurrent evaluations (default: 20)
    --tiering-concurrency-limit: Maximum number of concurrent tiering requests (default: 10)
    --exclude-from-file: Path to a JSONL file containing previously generated scenarios to exclude
"""

import os
import sys
import subprocess

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False

def run_generator():
    """Run the scenario generator as a subprocess."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory (argen-demo)
    parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))

    # Load the .env file from the parent project root
    env_path = os.path.join(parent_dir, '.env')
    if os.path.exists(env_path):
        print(f"Loading .env file from {env_path}")
        if has_dotenv:
            load_dotenv(env_path)
        else:
            print("WARNING: python-dotenv not installed. Install with: pip install python-dotenv")
            # Manually load the .env file
            with open(env_path) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

        if os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY loaded successfully")
        else:
            print("WARNING: OPENAI_API_KEY not found in .env file")
    else:
        print(f"WARNING: .env file not found at {env_path}")

    # Check for the scenario generator virtual environment (Poetry .venv first, then legacy .venv-scenario-generator)
    poetry_venv = os.path.join(current_dir, '.venv')
    legacy_venv = os.path.join(current_dir, '.venv-scenario-generator')
    python_executable = sys.executable

    if os.path.exists(poetry_venv):
        # Use the Poetry virtual environment
        if os.name == 'nt':
            python_executable = os.path.join(poetry_venv, 'Scripts', 'python.exe')
        else:
            python_executable = os.path.join(poetry_venv, 'bin', 'python')
        print(f"Using Poetry virtual environment: {poetry_venv}")
    elif os.path.exists(legacy_venv):
        # Use the legacy scenario generator virtual environment
        if os.name == 'nt':
            python_executable = os.path.join(legacy_venv, 'Scripts', 'python.exe')
        else:
            python_executable = os.path.join(legacy_venv, 'bin', 'python')
        print(f"Using legacy virtual environment: {legacy_venv}")
    else:
        print("No scenario generator virtual environment found; using system Python.")
        print(f"Using system Python: {python_executable}")

    # Build the command
    cmd = [python_executable, '-m', 'src.data_utils.generate_scenarios_v2_cli']

    # Add any command line arguments
    # Process arguments to handle temperature and max_new_tokens
    args = sys.argv[1:]

    # Check if we need to pass temperature and max_new_tokens
    cmd.extend(args)

    # Set the environment
    env = os.environ.copy()
    env['PYTHONPATH'] = parent_dir

    # Add the scenario_generator directory to PYTHONPATH
    scenario_generator_dir = os.path.dirname(os.path.abspath(__file__))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{scenario_generator_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = scenario_generator_dir

    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=parent_dir)

if __name__ == "__main__":
    run_generator()
