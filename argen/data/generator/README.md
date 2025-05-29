# ArGen Scenario Generator

A modular implementation of the ArGen dataset generator that creates healthcare scenario prompts for testing and training language models.

## Overview

This package provides functionality to generate diverse, challenging prompts that test an AI's ability to stay within healthcare domain boundaries. It implements the same functionality as the original `generate_scenarios_v2.py` script but in a modularized, maintainable structure.

## Package Structure

The package uses a flat structure with clear package boundaries. It is organized into several modules:

- **config.py**: Constants, argument parser, and configuration setup
- **openai_utils.py**: OpenAI API interactions and backoff strategies
- **embedding_utils.py**: Functions for embeddings and similarity detection
- **medical_terms.py**: Medical term trie and detection functions
- **baseline_model.py**: Functions for baseline model initialization and generation
- **evaluation.py**: Functions for evaluating prompts and responses
- **generation.py**: Main scenario generation logic
- **main.py**: Main entry point script that ties everything together
- **import_helper.py**: Helper module for consistent imports
- **standalone_run.py**: Standalone script to run the generator

The project uses a dedicated virtual environment named `.venv-scenario-generator` to avoid confusion with the main project's virtual environment.

## Installation

This package can be used in two ways:

### 1. As part of the parent project

When used as part of the parent ArGen project, no additional installation is needed.

### 2. As a standalone package with its own isolated environment

The package includes Poetry configuration for creating an isolated environment with all required dependencies. The virtual environment is named `.venv-scenario-generator` to avoid confusion with the main project's virtual environment.

```bash
# Navigate to the scenario_generator directory
cd argen-demo/src/data_utils/scenario_generator

# Run the installation script
./install.sh

# Alternatively, if you already have Poetry installed, you can do it manually:
# Configure Poetry to create the virtual environment in the project directory with a custom name
poetry config virtualenvs.in-project true
poetry config virtualenvs.path ".venv-scenario-generator"

# Install dependencies with Poetry
poetry install

# Install the package in development mode
poetry run pip install -e .
```

### Testing the Installation

To verify that the installation is working correctly:

```bash
# Run the test script
poetry run python test_install.py
```

All modules should show as importable if the installation is correct.

## Usage

### Integrated with parent project

Use the CLI wrapper script:

```bash
python src/data_utils/generate_scenarios_v2_cli.py --datasets smoke_test benchmarking grpo_training \
       --use-synthetic-negatives
```

### Standalone with isolated environment

After installing with Poetry:

```bash
# Run using the standalone script
python standalone_run.py --datasets smoke_test

# Or with Poetry's environment
poetry run python standalone_run.py --datasets smoke_test

# Or activate the Poetry shell first
source .venv-scenario-generator/bin/activate  # On Linux/Mac
# OR
.venv-scenario-generator\Scripts\activate  # On Windows
python standalone_run.py --datasets smoke_test

# You can also use the entry point script defined in pyproject.toml
poetry run generate --datasets smoke_test
```

The standalone_run.py script will automatically detect and use the scenario generator's virtual environment (`.venv-scenario-generator`) if it exists.

## Troubleshooting

### Import Errors

If you see errors like `ImportError: attempted relative import with no known parent package`, check:

1. That you've installed the package correctly with `poetry install` and `poetry run pip install -e .`
2. Run the test script to verify imports: `poetry run python test_install.py`
3. Use the `standalone_run.py` script which handles all the import paths correctly

### Module Not Found Errors

If specific modules cannot be found:

```bash
# Make sure the scenario_generator package is in your Python path
PYTHONPATH=$PYTHONPATH:/path/to/argen-demo poetry run python standalone_run.py
```

### Virtual Environment Issues

If you're having issues with the virtual environment:

```bash
# Check which virtual environment is active
which python  # On Linux/Mac
where python  # On Windows

# Recreate the virtual environment with the correct name
rm -rf .venv-scenario-generator  # Remove old environment (Linux/Mac)
# OR
rmdir /s /q .venv-scenario-generator  # Remove old environment (Windows)

# Configure Poetry and reinstall
poetry config virtualenvs.in-project true
poetry config virtualenvs.path ".venv-scenario-generator"
poetry install
```

### OpenAI API Key

If you get authentication errors:

```bash
# Set your API key
export OPENAI_API_KEY=your_api_key_here
```

### Missing Dependencies

If you encounter errors about missing dependencies like protobuf when using HuggingFace models:

```bash
# Activate the virtual environment
source .venv-scenario-generator/bin/activate  # On Linux/Mac
# OR
.venv-scenario-generator\Scripts\activate  # On Windows

# Install the missing dependencies
pip install protobuf
pip install "transformers[torch]"
```

## Command Line Options

The generator supports various command line options:

- `--datasets`: List of datasets to generate (smoke_test, benchmarking, grpo_training)
- `--model`: Model to use for generation (default: gpt-4o-mini)
- `--baseline`: Baseline model for evaluation (default: meta-llama/Llama-3.2-1B-Instruct)
- `--adv-baselines`: List of baseline models to stress-test
- `--embedding-model`: Model for embedding generation (default: all-MiniLM-L6-v2)
- `--difficulty-ratio`: Difficulty ratio for scenario generation (default: 1.3)
- `--duplicate-threshold`: Threshold for duplicate detection (default: 0.8)
- `--max-retries`: Maximum retries for API calls (default: 5)
- `--initial-delay`: Initial delay for backoff strategy (default: 1.0)
- `--tfidf-core60`: Path to core60 dataset (default: eval_core_60.jsonl)
- `--use-synthetic-negatives`: Use synthetic negatives for training
- `--fail-threshold`: Threshold for accepting prompts (default: 1.0)
- `--dry-run`: Run in dry-run mode with mock evaluations (for testing only)
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Requirements

When installed with Poetry, all dependencies are automatically managed. The key dependencies are:

- Python 3.8+
- OpenAI API key set as environment variable `OPENAI_API_KEY`
- PyTorch
- SentenceTransformers
- Transformers
- scikit-learn
- tiktoken

For GPU acceleration, ensure you have the appropriate CUDA setup for PyTorch.