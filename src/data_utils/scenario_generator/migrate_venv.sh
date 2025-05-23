#!/bin/bash
# migrate_venv.sh - Script to migrate from the old .venv to the new .venv-scenario-generator

set -e  # Exit on error

echo "Migrating from old .venv to new .venv-scenario-generator..."

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if the old virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Old virtual environment (.venv) not found. Nothing to migrate."
    echo "You can create a new virtual environment with ./install.sh"
    exit 0
fi

# Check if the new virtual environment already exists
if [ -d ".venv-scenario-generator" ]; then
    echo "New virtual environment (.venv-scenario-generator) already exists."
    read -p "Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Migration aborted."
        exit 0
    fi
    echo "Removing existing .venv-scenario-generator..."
    rm -rf .venv-scenario-generator
fi

# Configure Poetry to create the virtual environment in the project directory with a custom name
echo "Configuring Poetry virtual environment settings..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.path ".venv-scenario-generator"

# Create a new virtual environment and install dependencies
echo "Creating a new virtual environment and installing dependencies..."
cd "${SCRIPT_DIR}"

# Regenerate the lock file if needed
echo "Regenerating the Poetry lock file..."
poetry lock 

# Install dependencies
echo "Installing dependencies..."
poetry install -v --no-root

# Install additional dependencies for HuggingFace models
echo "Installing additional dependencies for HuggingFace models..."
poetry run pip install protobuf
poetry run pip install "transformers[torch]"

# Install python-dotenv for .env file loading
echo "Installing python-dotenv for .env file loading..."
poetry run pip install python-dotenv

# Install scispaCy if it was installed in the old environment
if [ -d ".venv/lib/python3.8/site-packages/scispacy" ] || [ -d ".venv/lib/python3.9/site-packages/scispacy" ] || [ -d ".venv/lib/python3.10/site-packages/scispacy" ]; then
    echo "Installing scispaCy and en_core_sci_sm model..."
    poetry run pip install "spacy>=3.7.0,<3.8.0"
    poetry run pip install "scispacy==0.5.5"
    poetry run pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
fi

echo ""
echo "Migration complete!"
echo "You can now use the new virtual environment with:"
echo "  source .venv-scenario-generator/bin/activate  # On Linux/Mac"
echo "  .venv-scenario-generator\\Scripts\\activate  # On Windows"
echo ""
echo "To run the scenario generator, use:"
echo "  python standalone_run.py --datasets smoke_test"
echo ""
echo "Do you want to remove the old virtual environment (.venv)? (y/n)"
read -r remove_old
if [[ "$remove_old" =~ ^[Yy]$ ]]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
    echo "Old virtual environment removed."
fi
