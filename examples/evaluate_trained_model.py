#!/usr/bin/env python3
"""
Script to evaluate a GRPO-trained model using the same metrics as the baseline.
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional
import datetime
import asyncio
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from argen.utils.env import load_env_vars, get_openai_api_key, get_gemini_api_key
from argen.evaluation.openai_evaluator import evaluate_model_with_llm, evaluate_model_with_openai
from argen.config import DEFAULT_SCENARIOS_PATH, DEFAULT_TEMPERATURE, PENALTY_CONFIG

# Set up logger for this module (optional but good practice)
logger = logging.getLogger(__name__)
# Basic logging config if running standalone and not configured elsewhere
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def check_dependencies():
    """Check if necessary dependencies for local inference are installed."""
    try:
        import torch
        import transformers
        print("torch and transformers libraries found.")
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, will use CPU (might be slow).")
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}. Please install required libraries.")
        print("Hint: pip install torch transformers")
        return False

def load_scenarios(file_path: str) -> List[Dict]:
    """
    Load scenarios from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of scenarios
    """
    scenarios = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    return scenarios

def load_baseline_results(file_path: str) -> Dict:
    """
    Load baseline evaluation results from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dict containing baseline results
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_metrics(baseline_metrics: Dict, trained_metrics: Dict) -> Dict:
    """
    Compare metrics between baseline and trained models.

    Args:
        baseline_metrics: Metrics from baseline model
        trained_metrics: Metrics from trained model

    Returns:
        Dict containing comparison results
    """
    comparison = {
        "helpfulness_score": {
            "baseline": baseline_metrics.get("average_helpfulness_score", 0.0),
            "trained": trained_metrics.get("average_helpfulness_score", 0.0),
            "relative_improvement": (
                (trained_metrics.get("average_helpfulness_score", 0.0) - baseline_metrics.get("average_helpfulness_score", 0.0)) /
                max(abs(baseline_metrics.get("average_helpfulness_score", 0.0)), 0.001) * 100
            ),
        },
        "ahimsa_score": {
            "baseline": baseline_metrics["average_ahimsa_score"],
            "trained": trained_metrics["average_ahimsa_score"],
            "improvement": trained_metrics["average_ahimsa_score"] - baseline_metrics["average_ahimsa_score"],
            "relative_improvement": (
                (trained_metrics["average_ahimsa_score"] - baseline_metrics["average_ahimsa_score"]) /
                max(baseline_metrics["average_ahimsa_score"], 0.001) * 100
            ),
        },
        "dharma_score": {
            "baseline": baseline_metrics["average_dharma_score"],
            "trained": trained_metrics["average_dharma_score"],
            "improvement": trained_metrics["average_dharma_score"] - baseline_metrics["average_dharma_score"],
            "relative_improvement": (
                (trained_metrics["average_dharma_score"] - baseline_metrics["average_dharma_score"]) /
                max(baseline_metrics["average_dharma_score"], 0.001) * 100
            ),
        },
        "combined_score": {
            "baseline": baseline_metrics["average_combined_score"],
            "trained": trained_metrics["average_combined_score"],
            "improvement": trained_metrics["average_combined_score"] - baseline_metrics["average_combined_score"],
            "relative_improvement": (
                (trained_metrics["average_combined_score"] - baseline_metrics["average_combined_score"]) /
                max(baseline_metrics["average_combined_score"], 0.001) * 100
            ),
        },
        "ahimsa_violations": {
            "baseline": baseline_metrics["ahimsa_violations"],
            "trained": trained_metrics["ahimsa_violations"],
            "improvement": baseline_metrics["ahimsa_violations"] - trained_metrics["ahimsa_violations"],
            "relative_improvement": (
                (baseline_metrics["ahimsa_violations"] - trained_metrics["ahimsa_violations"]) /
                max(baseline_metrics["ahimsa_violations"], 0.001) * 100
            ),
        },
        "dharma_violations": {
            "baseline": baseline_metrics["dharma_violations"],
            "trained": trained_metrics["dharma_violations"],
            "improvement": baseline_metrics["dharma_violations"] - trained_metrics["dharma_violations"],
            "relative_improvement": (
                (baseline_metrics["dharma_violations"] - trained_metrics["dharma_violations"]) /
                max(baseline_metrics["dharma_violations"], 0.001) * 100
            ),
        },
        "helpfulness_violations": {
            "baseline": baseline_metrics.get("helpfulness_violations", 0),
            "trained": trained_metrics.get("helpfulness_violations", 0),
            "improvement": baseline_metrics.get("helpfulness_violations", 0) - trained_metrics.get("helpfulness_violations", 0),
            "relative_improvement": (
                (baseline_metrics.get("helpfulness_violations", 0) - trained_metrics.get("helpfulness_violations", 0)) /
                max(baseline_metrics.get("helpfulness_violations", 0), 0.001) * 100
            ),
        },
    }

    return comparison

async def perform_evaluation(
    model_path: str,
    scenarios_path: str,
    output_file: str,
    temperature: float,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    evaluator: str = "openai",
    system_prompt_type: str = 'ENHANCED',
    test_mode: bool = False,
    apply_medical_disclaimer_penalty: Optional[bool] = None,
    apply_professional_referral_penalty: Optional[bool] = None
) -> Optional[Dict]:
    """
    Performs model evaluation using either OpenAI or Gemini and returns results dictionary or None on failure.

    Args:
        model_path: Path to the model to evaluate
        scenarios_path: Path to the scenarios file
        output_file: Path to save the evaluation results
        temperature: Temperature for generation
        openai_api_key: OpenAI API key (required if evaluator="openai")
        gemini_api_key: Gemini API key (required if evaluator="gemini")
        evaluator: Which LLM to use for evaluation ("openai" or "gemini")
        system_prompt_type: Type of system prompt to use
        test_mode: Whether to run in test mode
        apply_medical_disclaimer_penalty: Whether to apply penalty for missing medical disclaimer
        apply_professional_referral_penalty: Whether to apply penalty for missing professional referral

    Returns:
        Dictionary containing evaluation results or None on failure
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        logger.info(f"Loading scenarios from: {scenarios_path}")
        scenarios = load_scenarios(scenarios_path)
        if not scenarios:
            logger.error("No scenarios loaded.")
            return None

        if test_mode:
            logger.info("Running in test mode with only the first scenario.")
            scenarios = scenarios[:1]

        logger.info(f"Starting evaluation of model {model_path} using {evaluator} evaluator...")
        # Log penalty configuration
        if apply_medical_disclaimer_penalty is not None:
            logger.info(f"Medical disclaimer penalty: {'ENABLED' if apply_medical_disclaimer_penalty else 'DISABLED'}")
        else:
            logger.info(f"Medical disclaimer penalty: {'ENABLED' if PENALTY_CONFIG['apply_medical_disclaimer_penalty'] else 'DISABLED'} (default)")

        if apply_professional_referral_penalty is not None:
            logger.info(f"Professional referral penalty: {'ENABLED' if apply_professional_referral_penalty else 'DISABLED'}")
        else:
            logger.info(f"Professional referral penalty: {'ENABLED' if PENALTY_CONFIG['apply_professional_referral_penalty'] else 'DISABLED'} (default)")

        await evaluate_model_with_llm(
            model_name=model_path,
            scenarios=scenarios,
            output_file=output_file,
            temperature=temperature,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            evaluator=evaluator,
            test_mode=test_mode,
            system_prompt_type=system_prompt_type,
            apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
            apply_professional_referral_penalty=apply_professional_referral_penalty
        )

        # Load results
        if os.path.exists(output_file):
            logger.info(f"Loading evaluation results from: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return results
        else:
            logger.error(f"Evaluation output file not found after execution: {output_file}")
            return None
    except Exception as e:
        logger.error(f"Error during evaluation of {model_path}: {e}", exc_info=True)
        return None

# Backward compatibility function
async def perform_evaluation_with_openai(
    model_path: str,
    scenarios_path: str,
    output_file: str,
    temperature: float,
    openai_api_key: str,
    system_prompt_type: str = 'ENHANCED',
    test_mode: bool = False,
    apply_medical_disclaimer_penalty: Optional[bool] = None,
    apply_professional_referral_penalty: Optional[bool] = None
) -> Optional[Dict]:
    """
    Backward compatibility function that calls perform_evaluation with OpenAI evaluator.
    """
    return await perform_evaluation(
        model_path=model_path,
        scenarios_path=scenarios_path,
        output_file=output_file,
        temperature=temperature,
        openai_api_key=openai_api_key,
        evaluator="openai",
        system_prompt_type=system_prompt_type,
        test_mode=test_mode,
        apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
        apply_professional_referral_penalty=apply_professional_referral_penalty
    )

def main():
    """Run the evaluation script for GRPO-trained model."""
    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained model, optionally comparing against baseline.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model directory or model identifier"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the scenarios file"
    )
    parser.add_argument(
        "--baseline_results",
        type=str,
        default=None,
        help="Optional: Path to the baseline evaluation results JSON file for comparison"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="data/trained_openai_results",
        help="Base path for the evaluation results (timestamp and model name will be added)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        choices=['BASIC', 'ENHANCED'],
        default='ENHANCED',
        help="Type of system prompt to use ('BASIC' or 'ENHANCED')"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with mock responses and only the first scenario"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["openai", "gemini"],
        default="openai",
        help="Which LLM to use for evaluation (openai or gemini)"
    )
    parser.add_argument(
        "--medical_disclaimer_penalty",
        action="store_true",
        help="Apply penalty for missing medical disclaimer (default: disabled)"
    )
    parser.add_argument(
        "--no_medical_disclaimer_penalty",
        action="store_true",
        help="Do not apply penalty for missing medical disclaimer (default behavior)"
    )
    parser.add_argument(
        "--referral_penalty",
        action="store_true",
        help="Apply penalty for missing professional referral (default: enabled)"
    )
    parser.add_argument(
        "--no_referral_penalty",
        action="store_true",
        help="Do not apply penalty for missing professional referral"
    )

    args = parser.parse_args()

    # Load environment variables early
    load_env_vars()

    # Get the appropriate API key based on the evaluator
    openai_api_key = None
    gemini_api_key = None

    if args.evaluator == "openai":
        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            print("Error: OPENAI_API_KEY not found. Please set it in your .env file or environment.")
            sys.exit(1)
    else:  # args.evaluator == "gemini"
        try:
            gemini_api_key = get_gemini_api_key()
            if not gemini_api_key:
                print("Error: GEMINI_API_KEY not found. Please set it in your .env file or environment.")
                sys.exit(1)
        except ImportError:
            print("Error: google-generativeai package not installed. Please install it with 'pip install google-generativeai'")
            sys.exit(1)

    # Check dependencies (only relevant if using local models potentially, but good practice)
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    # Generate timestamp and construct output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_part = os.path.basename(args.model.rstrip('/'))
    output_filename = f"{args.output_base}_{args.evaluator}_{model_name_part}_{timestamp}.json"
    print(f"Results will be saved to: {output_filename}")

    print(f"\nEvaluating model: {args.model}")
    print(f"Using scenarios: {args.scenarios}")
    print(f"Using system prompt: {args.system_prompt}")
    print(f"Using temperature: {args.temperature}")
    print(f"Using evaluator: {args.evaluator}")
    if args.test:
        print("Running in TEST mode.")

    # Determine penalty configuration from CLI arguments
    apply_medical_disclaimer_penalty = None
    if args.medical_disclaimer_penalty and args.no_medical_disclaimer_penalty:
        print("Warning: Both --medical_disclaimer_penalty and --no_medical_disclaimer_penalty specified. Using default.")
    elif args.medical_disclaimer_penalty:
        apply_medical_disclaimer_penalty = True
        print("Medical disclaimer penalty: ENABLED")
    elif args.no_medical_disclaimer_penalty:
        apply_medical_disclaimer_penalty = False
        print("Medical disclaimer penalty: DISABLED (explicitly)")
    else:
        print(f"Medical disclaimer penalty: DISABLED (default)")

    apply_professional_referral_penalty = None
    if args.referral_penalty and args.no_referral_penalty:
        print("Warning: Both --referral_penalty and --no_referral_penalty specified. Using default.")
    elif args.referral_penalty:
        apply_professional_referral_penalty = True
        print("Professional referral penalty: ENABLED (explicitly)")
    elif args.no_referral_penalty:
        apply_professional_referral_penalty = False
        print("Professional referral penalty: DISABLED")
    else:
        print(f"Professional referral penalty: ENABLED (default)")

    # Perform evaluation using the refactored function
    evaluation_results = asyncio.run(perform_evaluation(
        model_path=args.model,
        scenarios_path=args.scenarios,
        output_file=output_filename,
        temperature=args.temperature,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        evaluator=args.evaluator,
        system_prompt_type=args.system_prompt,
        test_mode=args.test,
        apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
        apply_professional_referral_penalty=apply_professional_referral_penalty
    ))

    if not evaluation_results:
        print("\nEvaluation failed. Check logs for details.")
        sys.exit(1)

    print(f"\nEvaluation results successfully saved to {output_filename}")

    # --- Comparison Logic ---
    if args.baseline_results:
        print(f"\nComparing against baseline results from: {args.baseline_results}")
        try:
            baseline_results = load_baseline_results(args.baseline_results)
            comparison = compare_metrics(
                baseline_results["summary_metrics"],
                evaluation_results["summary_metrics"]
            )
            # Add comparison to the results dict
            evaluation_results["comparison_to_baseline"] = comparison

            # Save the updated results file including the comparison
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=4)
            print("Comparison results added to the output file.")

            # Print comparison summary
            print("\n===== COMPARISON TO BASELINE =====")
            print(f"Ahimsa Score: {comparison['ahimsa_score']['baseline']:.3f} -> {comparison['ahimsa_score']['trained']:.3f} ({comparison['ahimsa_score']['relative_improvement']:.1f}% change)")
            print(f"Dharma Score: {comparison['dharma_score']['baseline']:.3f} -> {comparison['dharma_score']['trained']:.3f} ({comparison['dharma_score']['relative_improvement']:.1f}% change)")
            print(f"Helpfulness Score: {comparison['helpfulness_score']['baseline']:.3f} -> {comparison['helpfulness_score']['trained']:.3f} ({comparison['helpfulness_score']['relative_improvement']:.1f}% change)")
            print(f"Combined Score: {comparison['combined_score']['baseline']:.3f} -> {comparison['combined_score']['trained']:.3f} ({comparison['combined_score']['relative_improvement']:.1f}% change)")
            print(f"Ahimsa Violations: {comparison['ahimsa_violations']['baseline']} -> {comparison['ahimsa_violations']['trained']} ({comparison['ahimsa_violations']['improvement']} fewer)")
            print(f"Dharma Violations: {comparison['dharma_violations']['baseline']} -> {comparison['dharma_violations']['trained']} ({comparison['dharma_violations']['improvement']} fewer)")
            print(f"Helpfulness Violations: {comparison['helpfulness_violations']['baseline']} -> {comparison['helpfulness_violations']['trained']} ({comparison['helpfulness_violations']['improvement']} fewer)")
            print("===============================")

        except FileNotFoundError:
            print(f"Warning: Baseline results file not found at {args.baseline_results}. Skipping comparison.")
        except KeyError as e:
             print(f"Warning: Missing key {e} in baseline or evaluation results JSON. Skipping comparison.")
        except Exception as e:
             print(f"Error during comparison: {e}")
    else:
         # Just print summary metrics if no baseline comparison requested
         summary = evaluation_results.get("summary_metrics", {})
         print("\n===== EVALUATION SUMMARY (NO BASELINE) =====")
         print(f"Ahimsa Score: {summary.get('average_ahimsa_score', 'N/A'):.3f}")
         print(f"Dharma Score: {summary.get('average_dharma_score', 'N/A'):.3f}")
         print(f"Helpfulness Score: {summary.get('average_helpfulness_score', 'N/A'):.3f}")
         print(f"Combined Score: {summary.get('average_combined_score', 'N/A'):.3f}")
         print(f"Ahimsa Violations: {summary.get('ahimsa_violations', 'N/A')}")
         print(f"Dharma Violations: {summary.get('dharma_violations', 'N/A')}")
         print(f"Helpfulness Violations: {summary.get('helpfulness_violations', 'N/A')}")
         print("============================================")

    print("\nEvaluation script finished!")

if __name__ == "__main__":
    main()