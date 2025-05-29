#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scenarios.py - Command to generate healthcare scenarios
==============================================================
This command provides a clean interface to run the scenario generator.
The generator includes progress bars to track the generation, tiering, and
sampling stages in real-time.

Example usage:
    python commands/generate_scenarios.py --datasets benchmarking --hf-model medalpaca/medalpaca-7b --count 20
    python commands/generate_scenarios.py --datasets benchmarking --model gpt-4 --count 20
    python commands/generate_scenarios.py --datasets smoke_test --batch-size 16 --concurrent-eval-limit 30
    python commands/generate_scenarios.py --datasets grpo_training --exclude-from-file path/to/benchmarking.jsonl

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

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False

def setup_generator_venv(generator_dir):
    """Set up the virtual environment for the scenario generator."""
    try:
        print(f"Setting up Poetry virtual environment in {generator_dir}...")

        # Change to the generator directory
        original_cwd = os.getcwd()
        os.chdir(generator_dir)

        # Run poetry install
        result = subprocess.run(['poetry', 'install'],
                              capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("Poetry virtual environment set up successfully!")
            return True
        else:
            print(f"Poetry install failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Poetry install timed out after 5 minutes")
        return False
    except FileNotFoundError:
        print("Poetry not found. Please install Poetry first: https://python-poetry.org/docs/#installation")
        return False
    except Exception as e:
        print(f"Error setting up virtual environment: {e}")
        return False
    finally:
        # Always change back to original directory
        os.chdir(original_cwd)

def run_generator():
    """Run the scenario generator as a subprocess."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory (argen-demo)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

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

    # Check for the scenario generator virtual environment in the new location
    generator_dir = os.path.join(parent_dir, 'argen', 'data', 'generator')
    poetry_venv = os.path.join(generator_dir, '.venv')
    legacy_venv = os.path.join(generator_dir, '.venv-scenario-generator')
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
        print("No scenario generator virtual environment found. Setting up...")
        # Try to set up the virtual environment
        setup_success = setup_generator_venv(generator_dir)
        if setup_success:
            # Check again for the newly created venv
            if os.path.exists(poetry_venv):
                if os.name == 'nt':
                    python_executable = os.path.join(poetry_venv, 'Scripts', 'python.exe')
                else:
                    python_executable = os.path.join(poetry_venv, 'bin', 'python')
                print(f"Using newly created Poetry virtual environment: {poetry_venv}")
            else:
                print("Failed to create virtual environment; using system Python.")
                print(f"Using system Python: {python_executable}")
        else:
            print("Failed to set up virtual environment; using system Python.")
            print(f"Using system Python: {python_executable}")

    # Build the command - now using the new module path
    cmd = [python_executable, os.path.join(parent_dir, 'argen', 'data', 'utils', 'generate_scenarios_v2_cli.py')]

    # Add any command line arguments
    args = sys.argv[1:]
    cmd.extend(args)

    # Set the environment
    env = os.environ.copy()
    env['PYTHONPATH'] = parent_dir

    # Add the scenario_generator directory to PYTHONPATH
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{generator_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = generator_dir

    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=parent_dir)

if __name__ == "__main__":
    run_generator()
