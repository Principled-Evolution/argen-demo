#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_with_env.py - Run the scenario generator with environment variables from parent project
=========================================================================================
This script loads the .env file from the parent project root and then runs the scenario generator.
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

def run_with_env():
    """Load the .env file from the parent project root and run the scenario generator."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory (argen-demo)
    parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))

    # Path to the .env file
    env_path = os.path.join(parent_dir, '.env')

    # Load the .env file
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

        # Check if the API key was loaded
        if os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY loaded successfully")
        else:
            print("WARNING: OPENAI_API_KEY not found in .env file")
    else:
        print(f"WARNING: .env file not found at {env_path}")

    # Check for the scenario generator virtual environment
    venv_path = os.path.join(current_dir, '.venv-scenario-generator')
    # Also check for the old .venv as fallback
    old_venv_path = os.path.join(current_dir, '.venv')
    python_executable = sys.executable

    if os.path.exists(venv_path):
        # Use the scenario generator's virtual environment
        if os.name == 'nt':  # Windows
            python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:  # Unix/Linux/Mac
            python_executable = os.path.join(venv_path, 'bin', 'python')
        print(f"Using scenario generator virtual environment: {venv_path}")
    elif os.path.exists(old_venv_path):
        # Use the old virtual environment as fallback
        if os.name == 'nt':  # Windows
            python_executable = os.path.join(old_venv_path, 'Scripts', 'python.exe')
        else:  # Unix/Linux/Mac
            python_executable = os.path.join(old_venv_path, 'bin', 'python')
        print(f"Using old virtual environment: {old_venv_path}")
        print(f"Consider migrating to the new virtual environment with ./migrate_venv.sh")
    else:
        print(f"Scenario generator virtual environment not found at {venv_path}")
        print(f"Using system Python: {python_executable}")

    # Build the command to run standalone_run.py
    standalone_script = os.path.join(current_dir, 'standalone_run.py')
    cmd = [python_executable, standalone_script]

    # Add any command line arguments
    cmd.extend(sys.argv[1:])

    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    # Print the API key status (just whether it's set, not the actual key)
    print(f"OPENAI_API_KEY is {'set' if os.getenv('OPENAI_API_KEY') else 'NOT set'}")
    subprocess.run(cmd, env=os.environ)

if __name__ == "__main__":
    run_with_env()
