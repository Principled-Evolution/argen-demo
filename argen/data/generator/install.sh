#!/bin/bash
# Installation script for setting up the isolated Poetry environment

set -e  # Exit on error

echo "Installing ArGen Scenario Generator with isolated Poetry environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Configure Poetry to create the virtual environment in the project directory with a custom name
echo "Configuring Poetry virtual environment settings..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.path ".venv-scenario-generator"

# Use a simple structure without unnecessary files
echo "Setting up simple package structure..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create an __init__.py file in the root directory if it doesn't exist
if [ ! -f "${SCRIPT_DIR}/__init__.py" ]; then
    echo "Creating __init__.py in root directory"
    touch "${SCRIPT_DIR}/__init__.py"
fi

# Create a new virtual environment and install dependencies
echo "Creating a new virtual environment and installing dependencies..."
cd "${SCRIPT_DIR}"

# Regenerate the lock file if needed
echo "Regenerating the Poetry lock file..."
poetry lock 

# Install dependencies
echo "Installing dependencies..."
poetry install -v --no-root

# Install optional dependencies
echo "Do you want to install optional scispaCy components for better medical entity detection? (y/n)"
read -r install_scispacy
if [[ "$install_scispacy" =~ ^[Yy]$ ]]; then
    echo "Installing scispaCy and en_core_sci_sm model..."
    # From:
    poetry run pip install "spacy>=3.7.0,<3.8.0"
    poetry run pip install "scispacy==0.5.5"
    # Update the URL to the current version
    poetry run pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
fi

# Install additional dependencies for HuggingFace models
echo "Installing additional dependencies for HuggingFace models..."
poetry run pip install protobuf
poetry run pip install "transformers[torch]"

# Install python-dotenv for .env file loading
echo "Installing python-dotenv for .env file loading..."
poetry run pip install python-dotenv

echo ""
echo "Installation complete!"
echo "To run the scenario generator, use:"
echo "  python standalone_run.py --datasets smoke_test"
echo "  # OR"
echo "  poetry run python standalone_run.py --datasets smoke_test"
echo ""
echo "You can also activate the virtual environment with:"
echo "  source .venv-scenario-generator/bin/activate  # On Linux/Mac"
echo "  .venv-scenario-generator\\Scripts\\activate  # On Windows"
echo ""
echo "Make sure to set your OpenAI API key:"
echo "  export OPENAI_API_KEY=your_api_key_here"
