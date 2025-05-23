#!/bin/bash
# Simple installation script for the scenario generator

set -e  # Exit on error

echo "Installing ArGen Scenario Generator with a simple virtual environment..."

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a virtual environment
echo "Creating a virtual environment..."
python -m venv .venv-scenario-generator

# Activate the virtual environment
if [ -f ".venv-scenario-generator/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv-scenario-generator/bin/activate
else
    echo "Virtual environment activation script not found. Installation failed."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e .
pip install protobuf
pip install "transformers[torch]"
pip install python-dotenv

# Install optional dependencies
echo "Do you want to install optional scispaCy components for better medical entity detection? (y/n)"
read -r install_scispacy
if [[ "$install_scispacy" =~ ^[Yy]$ ]]; then
    echo "Installing scispaCy and en_core_sci_sm model..."
    pip install "spacy>=3.7.0,<3.8.0"
    pip install "scispacy==0.5.5"
    pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
fi

echo ""
echo "Installation complete!"
echo "To run the scenario generator, use:"
echo "  source .venv-scenario-generator/bin/activate"
echo "  python standalone_run.py --datasets smoke_test"
echo ""
echo "Make sure to set your OpenAI API key:"
echo "  export OPENAI_API_KEY=your_api_key_here"
