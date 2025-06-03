#!/usr/bin/env python3
"""
Compare Models Command

This command evaluates multiple models in parallel across available GPUs 
with comprehensive comparison reporting.

Key features:
- Parallel evaluation across multiple GPUs
- Pipelined execution to maximize GPU utilization
- Automatic summary table generation with model comparisons
- Support for both sequential and parallel execution modes
- Comprehensive error handling and progress tracking

Usage:
    python commands/compare_models.py --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct medalpaca/medalpaca-7b --scenarios data/eval_scenarios-hashprompt.jsonl --evaluator gemini

Example:
    python commands/compare_models.py \
        --models model1 model2 model3 model4 \
        --scenarios data/eval_scenarios-hashprompt.jsonl \
        --evaluator gemini \
        --pipeline_delay 15 \
        --max_concurrent_models 2 \
        --eval_mode batch
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main():
    """
    Main entry point for the multi-model comparison command.
    This imports and runs the original evaluate_multiple_models.py functionality.
    """
    # Import the original evaluation script functionality
    # We need to temporarily modify sys.argv to make it look like we're running the original script
    original_argv = sys.argv.copy()
    
    # Replace the script name to match what the original script expects
    sys.argv[0] = os.path.join(project_root, 'scripts', 'evaluate_multiple_models.py')
    
    try:
        # Import and run the original evaluation functionality
        from scripts.evaluate_multiple_models import main as evaluate_multiple_models_main
        
        # Run the original main function
        return evaluate_multiple_models_main()
        
    except ImportError:
        # If the import fails, we'll run the script directly
        import subprocess
        
        # Build the command to run the original script
        cmd = [sys.executable, os.path.join(project_root, 'scripts', 'evaluate_multiple_models.py')] + sys.argv[1:]
        
        # Run the original script
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
        
    finally:
        # Restore the original argv
        sys.argv = original_argv

if __name__ == "__main__":
    sys.exit(main())
