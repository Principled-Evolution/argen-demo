#!/bin/bash
# Setup script to create symbolic links to parent project files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "Setting up symbolic links from parent project..."
echo "Script directory: ${SCRIPT_DIR}"
echo "Project root: ${PROJECT_ROOT}"

# Create data directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/data"

# Create src/reward_functions directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/src/reward_functions"

# Link UMLS terms file
if [ -f "${PROJECT_ROOT}/data/umls_5k_terms.txt" ]; then
    echo "Linking UMLS terms file..."
    ln -sf "${PROJECT_ROOT}/data/umls_5k_terms.txt" "${SCRIPT_DIR}/data/umls_5k_terms.txt"
else
    echo "Warning: UMLS terms file not found at ${PROJECT_ROOT}/data/umls_5k_terms.txt"
fi

# Link eval_core_60.jsonl file
if [ -f "${PROJECT_ROOT}/data/eval_core_60.jsonl" ]; then
    echo "Linking eval_core_60.jsonl file..."
    ln -sf "${PROJECT_ROOT}/data/eval_core_60.jsonl" "${SCRIPT_DIR}/data/eval_core_60.jsonl"
else
    echo "Warning: eval_core_60.jsonl file not found at ${PROJECT_ROOT}/data/eval_core_60.jsonl"
fi

# Link OpenAI reward functions
if [ -f "${PROJECT_ROOT}/src/reward_functions/openai_rewards.py" ]; then
    echo "Linking OpenAI reward functions..."
    ln -sf "${PROJECT_ROOT}/src/reward_functions/openai_rewards.py" "${SCRIPT_DIR}/src/reward_functions/openai_rewards.py"
    
    # Create an __init__.py file in the reward_functions directory
    touch "${SCRIPT_DIR}/src/reward_functions/__init__.py"
else
    echo "Warning: OpenAI reward functions not found at ${PROJECT_ROOT}/src/reward_functions/openai_rewards.py"
fi

# Create an __init__.py file in the src directory
mkdir -p "${SCRIPT_DIR}/src"
touch "${SCRIPT_DIR}/src/__init__.py"

echo "Setup complete!"
