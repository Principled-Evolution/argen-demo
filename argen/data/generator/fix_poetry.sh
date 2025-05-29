#!/bin/bash
# Script to fix Poetry configuration

set -e  # Exit on error

echo "Fixing Poetry configuration..."

# Backup the current pyproject.toml file
if [ -f "pyproject.toml" ]; then
    echo "Backing up current pyproject.toml to pyproject.toml.bak"
    cp pyproject.toml pyproject.toml.bak
fi

# Replace with the new pyproject.toml file
if [ -f "pyproject.toml.new" ]; then
    echo "Replacing pyproject.toml with pyproject.toml.new"
    cp pyproject.toml.new pyproject.toml
else
    echo "pyproject.toml.new not found. Cannot proceed."
    exit 1
fi

# Remove the existing lock file
if [ -f "poetry.lock" ]; then
    echo "Removing existing poetry.lock file"
    rm poetry.lock
fi

# Generate a new lock file
echo "Generating a new lock file"
poetry lock

echo "Poetry configuration fixed!"
echo "You can now run ./install.sh to install the package"
