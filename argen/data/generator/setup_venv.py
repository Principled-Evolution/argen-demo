#!/usr/bin/env python3
"""
Setup script for the scenario generator virtual environment.
This script sets up the Poetry virtual environment for the scenario generator.
"""

import os
import sys
import subprocess

def setup_venv():
    """Set up the Poetry virtual environment."""
    try:
        print("Setting up Poetry virtual environment for scenario generator...")
        
        # Get the current directory (should be the generator directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Working in directory: {current_dir}")
        
        # Check if pyproject.toml exists
        pyproject_path = os.path.join(current_dir, 'pyproject.toml')
        if not os.path.exists(pyproject_path):
            print(f"ERROR: pyproject.toml not found at {pyproject_path}")
            return False
        
        # Change to the generator directory
        original_cwd = os.getcwd()
        os.chdir(current_dir)
        
        # Run poetry install
        print("Running 'poetry install'...")
        result = subprocess.run(['poetry', 'install'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Poetry virtual environment set up successfully!")
            print("Output:", result.stdout)
            
            # Check if .venv directory was created
            venv_path = os.path.join(current_dir, '.venv')
            if os.path.exists(venv_path):
                print(f"✅ Virtual environment created at: {venv_path}")
            else:
                print("ℹ️  Virtual environment created (location managed by Poetry)")
            
            return True
        else:
            print(f"❌ Poetry install failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Poetry install timed out after 5 minutes")
        return False
    except FileNotFoundError:
        print("❌ Poetry not found. Please install Poetry first:")
        print("   https://python-poetry.org/docs/#installation")
        return False
    except Exception as e:
        print(f"❌ Error setting up virtual environment: {e}")
        return False
    finally:
        # Always change back to original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = setup_venv()
    sys.exit(0 if success else 1)
