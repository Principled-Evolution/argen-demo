#!/usr/bin/env python3
"""
Evaluate Model Command

This command evaluates individual models against healthcare scenarios using 
LLM evaluators (OpenAI or Gemini).

Key features:
- Support for both local HuggingFace models and API-based models
- Configurable evaluation modes (batch vs individual for Gemini)
- Comprehensive reward function evaluation (Ahimsa, Dharma, Helpfulness)
- Comparison mode to analyze batch vs individual evaluation performance
- Configurable penalty systems for medical disclaimers and professional referrals

Usage:
    python commands/evaluate_model.py --model meta-llama/Llama-3.2-1B-Instruct --scenarios data/eval_scenarios-hashprompt.jsonl --evaluator gemini

Example:
    python commands/evaluate_model.py \
        --model /path/to/trained/model \
        --scenarios data/eval_scenarios-hashprompt.jsonl \
        --evaluator gemini \
        --eval-mode batch \
        --temperature 0.9 \
        --system_prompt ENHANCED \
        --generation_batch_size 50
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main():
    """
    Main entry point for the evaluation command.
    This imports and runs the original evaluate_baseline.py functionality.
    """
    # Import the original evaluation script functionality
    # We need to temporarily modify sys.argv to make it look like we're running the original script
    original_argv = sys.argv.copy()
    
    # Replace the script name to match what the original script expects
    sys.argv[0] = os.path.join(project_root, 'examples', 'evaluate_baseline.py')
    
    try:
        # Import and run the original evaluation functionality
        from examples.evaluate_baseline import main as evaluate_baseline_main
        
        # Run the original main function
        return evaluate_baseline_main()
        
    except ImportError:
        # If the import fails, we'll run the script directly
        import subprocess
        
        # Build the command to run the original script
        cmd = [sys.executable, os.path.join(project_root, 'examples', 'evaluate_baseline.py')] + sys.argv[1:]
        
        # Run the original script
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
        
    finally:
        # Restore the original argv
        sys.argv = original_argv

if __name__ == "__main__":
    sys.exit(main())
