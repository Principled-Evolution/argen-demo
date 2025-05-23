#!/usr/bin/env python3
"""
Script to evaluate the baseline model using an LLM evaluator (e.g., OpenAI or Gemini).
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional
import datetime
import subprocess
import asyncio
import numpy as np
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils.env import load_env_vars, get_openai_api_key, get_gemini_api_key
from src.evaluation.openai_evaluator import evaluate_model_with_llm
from src.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_SCENARIOS_PATH,
    DEFAULT_OUTPUT_BASE,
    DEFAULT_TEMPERATURE,
    get_system_prompt
)


def check_dependencies():
    """Check if necessary dependencies for local inference are installed."""
    try:
        import torch
        import transformers
        print("torch and transformers libraries found.")
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
            # Consider adding accelerate and bitsandbytes checks if quantization is used later
        else:
            print("CUDA not available, will use CPU (might be slow).")
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}. Please install required libraries.")
        print("Hint: pip install torch transformers accelerate") # Add bitsandbytes if needed
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


async def generate_comparison_report(
    batch_output_file: str,
    individual_output_file: str,
    batch_duration: float,
    individual_duration: float,
    batch_api_calls: int,
    individual_api_calls: int,
    num_scenarios: int
) -> Dict:
    """
    Generate a detailed comparison report between batch and individual evaluation modes.
    """
    import json

    # Load results from both files
    with open(batch_output_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)

    with open(individual_output_file, 'r', encoding='utf-8') as f:
        individual_data = json.load(f)

    batch_results = batch_data.get('individual_results', [])
    individual_results = individual_data.get('individual_results', [])

    # Performance comparison
    performance_comparison = {
        "execution_time": {
            "batch_mode_seconds": batch_duration,
            "individual_mode_seconds": individual_duration,
            "speedup_factor": individual_duration / batch_duration if batch_duration > 0 else 0,
            "time_saved_seconds": individual_duration - batch_duration
        },
        "api_calls": {
            "batch_mode_calls": batch_api_calls,
            "individual_mode_calls": individual_api_calls,
            "reduction_factor": individual_api_calls / batch_api_calls if batch_api_calls > 0 else 0,
            "calls_saved": individual_api_calls - batch_api_calls
        },
        "efficiency": {
            "batch_mode_seconds_per_scenario": batch_duration / num_scenarios if num_scenarios > 0 else 0,
            "individual_mode_seconds_per_scenario": individual_duration / num_scenarios if num_scenarios > 0 else 0,
            "batch_mode_calls_per_scenario": batch_api_calls / num_scenarios if num_scenarios > 0 else 0,
            "individual_mode_calls_per_scenario": individual_api_calls / num_scenarios if num_scenarios > 0 else 0
        }
    }

    # Score consistency analysis
    score_comparison = analyze_score_consistency(batch_results, individual_results)

    # Generate summary
    summary = {
        "evaluation_date": datetime.datetime.now().isoformat(),
        "scenarios_evaluated": num_scenarios,
        "performance_winner": "batch" if batch_duration < individual_duration else "individual",
        "api_efficiency_winner": "batch" if batch_api_calls < individual_api_calls else "individual",
        "score_consistency": score_comparison.get("overall_consistency", "unknown"),
        "recommendation": generate_recommendation(performance_comparison, score_comparison)
    }

    return {
        "summary": summary,
        "performance_comparison": performance_comparison,
        "score_consistency_analysis": score_comparison,
        "batch_results_file": batch_output_file,
        "individual_results_file": individual_output_file
    }


def analyze_score_consistency(batch_results: List[Dict], individual_results: List[Dict]) -> Dict:
    """
    Analyze consistency between batch and individual evaluation scores.
    """
    if len(batch_results) != len(individual_results):
        return {
            "error": f"Mismatched result counts: batch={len(batch_results)}, individual={len(individual_results)}",
            "overall_consistency": "error"
        }

    score_fields = ["ahimsa_score", "dharma_score", "helpfulness_score", "combined_score"]
    consistency_analysis = {}

    for field in score_fields:
        batch_scores = []
        individual_scores = []

        for batch_res, individual_res in zip(batch_results, individual_results):
            if field in batch_res and field in individual_res:
                batch_scores.append(batch_res[field])
                individual_scores.append(individual_res[field])

        if batch_scores and individual_scores:
            batch_scores = np.array(batch_scores)
            individual_scores = np.array(individual_scores)

            # Calculate metrics
            mean_absolute_diff = np.mean(np.abs(batch_scores - individual_scores))
            max_diff = np.max(np.abs(batch_scores - individual_scores))

            # Calculate correlation with proper handling of edge cases
            if len(batch_scores) <= 1:
                correlation = 1.0  # Perfect correlation for single point
            elif np.std(batch_scores) == 0 and np.std(individual_scores) == 0:
                # Both arrays are constant - perfect correlation if same values, 0 if different
                correlation = 1.0 if np.allclose(batch_scores, individual_scores) else 0.0
            elif np.std(batch_scores) == 0 or np.std(individual_scores) == 0:
                # One array is constant - no meaningful correlation
                correlation = 0.0
            else:
                # Normal case - calculate correlation
                with np.errstate(invalid='ignore'):  # Suppress warnings
                    corr_matrix = np.corrcoef(batch_scores, individual_scores)
                    correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0

            consistency_analysis[field] = {
                "mean_absolute_difference": float(mean_absolute_diff),
                "correlation_coefficient": float(correlation),
                "max_difference": float(max_diff),
                "batch_mean": float(np.mean(batch_scores)),
                "individual_mean": float(np.mean(individual_scores)),
                "consistency_rating": get_consistency_rating(mean_absolute_diff, correlation)
            }

    # Overall consistency rating
    if consistency_analysis:
        correlations = [analysis["correlation_coefficient"] for analysis in consistency_analysis.values()]
        mean_diffs = [analysis["mean_absolute_difference"] for analysis in consistency_analysis.values()]

        # Filter out NaN values for averaging
        valid_correlations = [c for c in correlations if not np.isnan(c)]
        valid_mean_diffs = [d for d in mean_diffs if not np.isnan(d)]

        avg_correlation = np.mean(valid_correlations) if valid_correlations else 0.0
        avg_mean_diff = np.mean(valid_mean_diffs) if valid_mean_diffs else 1.0
        overall_consistency = get_consistency_rating(avg_mean_diff, avg_correlation)
    else:
        overall_consistency = "no_data"

    consistency_analysis["overall_consistency"] = overall_consistency
    return consistency_analysis


def get_consistency_rating(mean_diff: float, correlation: float) -> str:
    """
    Get a consistency rating based on mean difference and correlation.
    """
    if correlation > 0.95 and mean_diff < 0.05:
        return "excellent"
    elif correlation > 0.90 and mean_diff < 0.10:
        return "good"
    elif correlation > 0.80 and mean_diff < 0.15:
        return "fair"
    else:
        return "poor"


def generate_recommendation(performance_comparison: Dict, score_comparison: Dict) -> str:
    """
    Generate a recommendation based on performance and consistency analysis.
    """
    speedup = performance_comparison["execution_time"]["speedup_factor"]
    api_reduction = performance_comparison["api_calls"]["reduction_factor"]
    consistency = score_comparison.get("overall_consistency", "unknown")

    if consistency in ["excellent", "good"] and speedup > 1.5 and api_reduction > 2:
        return "Strongly recommend batch mode: significant performance gains with excellent score consistency"
    elif consistency in ["excellent", "good"] and speedup > 1.2:
        return "Recommend batch mode: good performance gains with reliable score consistency"
    elif consistency == "fair" and speedup > 2:
        return "Consider batch mode: significant performance gains but monitor score consistency"
    elif consistency == "poor":
        return "Use individual mode: batch mode shows poor score consistency"
    else:
        return "Evaluate based on specific requirements: modest performance differences"


async def evaluate_responses_with_gemini(
    original_prompts: List[str],
    generated_responses: List[str],
    scenario_metadata: List[Dict],
    output_file: str,
    eval_mode: str,
    test_mode: bool = False,
    apply_medical_disclaimer_penalty: Optional[bool] = None,
    apply_professional_referral_penalty: Optional[bool] = None,
    temperature: Optional[float] = None,
) -> None:
    """
    Evaluate pre-generated responses using Gemini in either batch or individual mode.
    """
    from src.evaluation.openai_evaluator import (
        REWARD_WEIGHTS, PENALTY_CONFIG,
        calculate_metrics, GEMINI_EVAL_MODEL
    )
    from src.reward_functions.gemini_rewards import DEFAULT_EVAL_RESPONSE
    from src.reward_functions.gemini.ahimsa import (
        evaluate_ahimsa_with_gemini, batch_process_ahimsa_evaluations_concurrently
    )
    from src.reward_functions.gemini.dharma import (
        evaluate_dharma_with_gemini, batch_process_dharma_evaluations_concurrently
    )
    from src.reward_functions.gemini.helpfulness import (
        evaluate_helpfulness_with_gemini, batch_process_helpfulness_evaluations_concurrently
    )

    from src.config import GRPO_CONFIG
    import datetime

    evaluation_results = []

    if eval_mode == "batch":
        # Batch evaluation mode
        ahimsa_payload = []
        dharma_payload = []
        helpfulness_payload = []

        for i, (prompt, response, metadata) in enumerate(zip(original_prompts, generated_responses, scenario_metadata)):
            current_prompt_meta = {"tier": metadata.get("tier", "C"), "scope": metadata.get("scope", "S0")}

            ahimsa_payload.append({
                "prompt": prompt,
                "model_response": response,
                "original_prompt_meta": current_prompt_meta
            })
            dharma_payload.append({
                "prompt": prompt,
                "model_response": response,
                "original_prompt_meta": current_prompt_meta
            })
            helpfulness_payload.append({
                "prompt": prompt,
                "model_response": response,
                "original_prompt_meta": {}
            })

        # Run batch evaluations
        evaluation_tasks = []

        if ahimsa_payload:
            items_per_call = GRPO_CONFIG.get("gemini_ahimsa_items_per_call_eval", 10)
            max_concurrent = GRPO_CONFIG.get("gemini_ahimsa_max_concurrent_eval", 5)
            evaluation_tasks.append(
                batch_process_ahimsa_evaluations_concurrently(
                    ahimsa_payload, items_per_gemini_call=items_per_call, max_concurrent_calls=max_concurrent, temperature=temperature
                )
            )

        if dharma_payload:
            items_per_call = GRPO_CONFIG.get("gemini_dharma_items_per_call_eval", 10)
            max_concurrent = GRPO_CONFIG.get("gemini_dharma_max_concurrent_eval", 5)
            evaluation_tasks.append(
                batch_process_dharma_evaluations_concurrently(
                    dharma_payload, items_per_gemini_call=items_per_call, max_concurrent_calls=max_concurrent, temperature=temperature
                )
            )

        if helpfulness_payload:
            items_per_call = GRPO_CONFIG.get("gemini_helpfulness_items_per_call_eval", 10)
            max_concurrent = GRPO_CONFIG.get("gemini_helpfulness_max_concurrent_eval", 5)
            evaluation_tasks.append(
                batch_process_helpfulness_evaluations_concurrently(
                    helpfulness_payload, items_per_gemini_call=items_per_call, max_concurrent_calls=max_concurrent, temperature=temperature
                )
            )

        # Execute batch evaluations
        gathered_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Process results
        ahimsa_results = gathered_results[0] if len(gathered_results) > 0 else []
        dharma_results = gathered_results[1] if len(gathered_results) > 1 else []
        helpfulness_results = gathered_results[2] if len(gathered_results) > 2 else []

    elif eval_mode == "individual":
        # Individual evaluation mode
        individual_tasks = []

        for prompt, response, metadata in zip(original_prompts, generated_responses, scenario_metadata):
            current_prompt_meta = {"tier": metadata.get("tier", "C"), "scope": metadata.get("scope", "S0")}

            individual_tasks.append([
                evaluate_ahimsa_with_gemini(prompt, response, current_prompt_meta, temperature=temperature),
                evaluate_dharma_with_gemini(prompt, response, current_prompt_meta, temperature=temperature),
                evaluate_helpfulness_with_gemini(prompt, response, temperature=temperature)
            ])

        # Execute individual evaluations
        all_individual_results = []
        for task_group in individual_tasks:
            group_results = await asyncio.gather(*task_group, return_exceptions=True)
            all_individual_results.append(group_results)

        # Reorganize results by evaluation type
        ahimsa_results = [results[0] for results in all_individual_results]
        dharma_results = [results[1] for results in all_individual_results]
        helpfulness_results = [results[2] for results in all_individual_results]

    else:
        raise ValueError(f"Invalid eval_mode: {eval_mode}. Must be 'batch' or 'individual'.")

    # Combine results
    for i in range(len(original_prompts)):
        ahimsa_res = ahimsa_results[i] if i < len(ahimsa_results) else DEFAULT_EVAL_RESPONSE.copy()
        dharma_res = dharma_results[i] if i < len(dharma_results) else DEFAULT_EVAL_RESPONSE.copy()
        help_res = helpfulness_results[i] if i < len(helpfulness_results) else DEFAULT_EVAL_RESPONSE.copy()

        # Handle exceptions
        if isinstance(ahimsa_res, Exception):
            ahimsa_res = {"error": f"Ahimsa Eval Exception: {str(ahimsa_res)}", **DEFAULT_EVAL_RESPONSE}
        if isinstance(dharma_res, Exception):
            dharma_res = {"error": f"Dharma Eval Exception: {str(dharma_res)}", **DEFAULT_EVAL_RESPONSE}
        if isinstance(help_res, Exception):
            help_res = {"error": f"Helpfulness Eval Exception: {str(help_res)}", **DEFAULT_EVAL_RESPONSE}

        # Combine results
        combined_result = DEFAULT_EVAL_RESPONSE.copy()
        combined_result.update(ahimsa_res)

        # Add dharma fields
        for key, value in dharma_res.items():
            if key not in ["ahimsa_score", "ahimsa_violation"]:
                combined_result[key] = value

        # Add helpfulness fields
        for key, value in help_res.items():
            if key not in ["ahimsa_score", "ahimsa_violation", "dharma_score", "dharma_violation"]:
                combined_result[key] = value

        # Remove error field if all evaluations were successful (no exceptions and no error fields in individual results)
        has_errors = (
            isinstance(ahimsa_res, Exception) or
            isinstance(dharma_res, Exception) or
            isinstance(help_res, Exception) or
            ahimsa_res.get("error") or
            dharma_res.get("error") or
            help_res.get("error")
        )
        if not has_errors and "error" in combined_result:
            del combined_result["error"]

        # Calculate combined score
        ahimsa_s = combined_result.get("ahimsa_score", 0.0)
        dharma_s = combined_result.get("dharma_score", 0.0)
        helpfulness_s = combined_result.get("helpfulness_score", 0.0)

        combined_result["combined_score"] = (
            (ahimsa_s * REWARD_WEIGHTS.get("ahimsa", 0.0)) +
            (dharma_s * REWARD_WEIGHTS.get("dharma", 0.0)) +
            (helpfulness_s * REWARD_WEIGHTS.get("helpfulness", 0.0))
        )

        # Add prompt, response, and scenario info
        combined_result["prompt"] = original_prompts[i]
        combined_result["response"] = generated_responses[i]
        combined_result["scenario_id"] = scenario_metadata[i].get("scenario_id", f"item_{i}")

        evaluation_results.append(combined_result)

    # Calculate final metrics
    final_metrics = calculate_metrics(evaluation_results)

    # Create output data
    output_data = {
        "evaluation_config": {
            "evaluator": f"gemini ({GEMINI_EVAL_MODEL})",
            "eval_mode": eval_mode,
            "num_scenarios": len(original_prompts),
            "test_mode": test_mode,
            "ahimsa_weight": REWARD_WEIGHTS["ahimsa"],
            "dharma_weight": REWARD_WEIGHTS["dharma"],
            "helpfulness_weight": REWARD_WEIGHTS["helpfulness"],
            "apply_medical_disclaimer_penalty": PENALTY_CONFIG["apply_medical_disclaimer_penalty"],
            "referral_policy": PENALTY_CONFIG["referral_policy"],
            "timestamp": datetime.datetime.now().isoformat()
        },
        "summary_metrics": final_metrics,
        "individual_results": evaluation_results
    }

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


async def run_comparison_mode(
    model_name: str,
    scenarios: List[Dict],
    output_base: str,
    timestamp: str,
    temperature: float,
    openai_api_key: Optional[str],
    gemini_api_key: Optional[str],
    test_mode: bool,
    system_prompt_type: str,
    apply_medical_disclaimer_penalty: Optional[bool],
    apply_professional_referral_penalty: Optional[bool],
    generation_batch_size: int
) -> None:
    """
    Run comparison between batch and individual Gemini evaluation modes.
    Generates responses once, then evaluates the same responses with both methods.
    """
    import time
    import json

    print("\n" + "="*80)
    print("RUNNING GEMINI EVALUATION MODE COMPARISON")
    print("="*80)

    # Prepare output filenames
    batch_output = f"{output_base}_{timestamp}_batch.json"
    individual_output = f"{output_base}_{timestamp}_individual.json"
    comparison_output = f"{output_base}_{timestamp}_comparison.json"

    # Ensure output directories exist
    for output_file in [batch_output, individual_output, comparison_output]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print(f"Batch mode results will be saved to: {batch_output}")
    print(f"Individual mode results will be saved to: {individual_output}")
    print(f"Comparison report will be saved to: {comparison_output}")

    # PHASE 0: Generate responses once (not timed)
    print("\n" + "-"*60)
    print("PHASE 0: Generating model responses (shared for both evaluations)...")
    print("-"*60)

    # Generate responses using a temporary output file
    temp_output = f"{output_base}_{timestamp}_temp_responses.json"

    print("Generating responses for all scenarios...")
    generation_start_time = time.time()

    await evaluate_model_with_llm(
        model_name=model_name,
        scenarios=scenarios,
        output_file=temp_output,
        temperature=temperature,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        evaluator="gemini",
        test_mode=test_mode,
        system_prompt_type=system_prompt_type,
        apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
        apply_professional_referral_penalty=apply_professional_referral_penalty,
        generation_batch_size=generation_batch_size,
        gemini_eval_mode="batch",  # Use batch mode for generation
        skip_evaluation=True  # New parameter to skip evaluation phase
    )

    generation_end_time = time.time()
    generation_duration = generation_end_time - generation_start_time

    print(f"Response generation completed in {generation_duration:.2f} seconds")

    # Load the generated responses
    with open(temp_output, 'r', encoding='utf-8') as f:
        temp_data = json.load(f)

    # Extract the generated responses
    generated_responses = []
    original_prompts = []
    scenario_metadata = []

    for result in temp_data.get('individual_results', []):
        generated_responses.append(result.get('response', ''))
        original_prompts.append(result.get('prompt', ''))
        # Extract metadata from scenarios
        scenario_id = result.get('scenario_id', '')
        matching_scenario = next((s for s in scenarios if s.get('scenario_id') == scenario_id), {})
        scenario_metadata.append({
            'tier': matching_scenario.get('tier', 'C'),
            'scope': matching_scenario.get('scope', 'S0'),
            'scenario_id': scenario_id
        })

    print(f"Extracted {len(generated_responses)} responses for evaluation comparison")

    # Clean up temp file
    if os.path.exists(temp_output):
        os.remove(temp_output)

    # Reset API tracker for batch mode
    from src.utils.gemini_api_tracker import GeminiAPITracker
    tracker = GeminiAPITracker()
    tracker.reset()

    # PHASE 1: Batch evaluation mode (timed)
    print("\n" + "-"*60)
    print("PHASE 1: Running BATCH evaluation mode...")
    print("-"*60)
    batch_start_time = time.time()

    await evaluate_responses_with_gemini(
        original_prompts=original_prompts,
        generated_responses=generated_responses,
        scenario_metadata=scenario_metadata,
        output_file=batch_output,
        eval_mode="batch",
        test_mode=test_mode,
        apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
        apply_professional_referral_penalty=apply_professional_referral_penalty,
        temperature=temperature
    )

    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    batch_api_calls = tracker.get_total_calls()

    print(f"Batch evaluation completed in {batch_duration:.2f} seconds")
    print(f"Batch mode API calls: {batch_api_calls}")

    # Reset API tracker for individual mode
    tracker.reset()

    # PHASE 2: Individual evaluation mode (timed)
    print("\n" + "-"*60)
    print("PHASE 2: Running INDIVIDUAL evaluation mode...")
    print("-"*60)
    individual_start_time = time.time()

    await evaluate_responses_with_gemini(
        original_prompts=original_prompts,
        generated_responses=generated_responses,
        scenario_metadata=scenario_metadata,
        output_file=individual_output,
        eval_mode="individual",
        test_mode=test_mode,
        apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
        apply_professional_referral_penalty=apply_professional_referral_penalty,
        temperature=temperature
    )

    individual_end_time = time.time()
    individual_duration = individual_end_time - individual_start_time
    individual_api_calls = tracker.get_total_calls()

    print(f"Individual evaluation completed in {individual_duration:.2f} seconds")
    print(f"Individual mode API calls: {individual_api_calls}")

    # PHASE 3: Generate comparison report
    print("\n" + "-"*60)
    print("PHASE 3: Generating comparison report...")
    print("-"*60)

    comparison_report = await generate_comparison_report(
        batch_output, individual_output, batch_duration, individual_duration,
        batch_api_calls, individual_api_calls, len(scenarios)
    )

    # Add generation timing info to the report
    comparison_report["generation_info"] = {
        "generation_duration_seconds": generation_duration,
        "responses_generated": len(generated_responses),
        "note": "Generation time excluded from evaluation timing comparison"
    }

    # Save comparison report
    with open(comparison_output, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=4, ensure_ascii=False)

    print(f"Comparison report saved to: {comparison_output}")

    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Scenarios evaluated: {len(scenarios)}")
    print(f"Response generation: {generation_duration:.2f}s (excluded from comparison)")
    print(f"Batch evaluation: {batch_duration:.2f}s, {batch_api_calls} API calls")
    print(f"Individual evaluation: {individual_duration:.2f}s, {individual_api_calls} API calls")
    print(f"Speed improvement: {individual_duration/batch_duration:.2f}x faster with batch mode")
    print(f"API call reduction: {individual_api_calls/batch_api_calls:.2f}x fewer calls with batch mode")
    print("="*80)


def main():
    """Run the baseline model evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline model using an LLM evaluator (default: OpenAI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Name/identifier of the model to evaluate (e.g., HF identifier like 'unsloth/Llama-3.2-1B-Instruct', or local path)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the scenarios file"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help="Base path and filename for the evaluation results (timestamp and .json will be added)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with mock responses"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for generation (higher = more random)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        choices=['BASIC', 'ENHANCED'],
        default='ENHANCED',
        help="Type of system prompt to use ('BASIC' or 'ENHANCED')"
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
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["openai", "gemini"],
        default="gemini",
        help="Which LLM to use for evaluation (openai or gemini)"
    )
    parser.add_argument(
        "--compare-eval-modes",
        action="store_true",
        help="Run comparison between batch and individual Gemini evaluation modes (only works with --evaluator gemini)"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["batch", "individual"],
        default="individual",
        help="Evaluation mode for Gemini evaluator: 'individual' (default, one API call per evaluation) or 'batch' (faster)"
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include reasoning field in evaluation responses (default: disabled)"
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=50,
        help="Batch size for local model generation (higher = faster but more GPU memory required)"
    )

    args = parser.parse_args()

    # Validate comparison mode
    if args.compare_eval_modes and args.evaluator != "gemini":
        print("Error: --compare-eval-modes can only be used with --evaluator gemini")
        sys.exit(1)

    # Validate eval mode
    if args.eval_mode != "batch" and args.evaluator != "gemini":
        print("Error: --eval-mode can only be used with --evaluator gemini")
        sys.exit(1)

    # Load environment variables early
    load_env_vars()

    # Set reasoning flag if requested
    if args.include_reasoning:
        from src.reward_functions.gemini_rewards import set_include_reasoning
        set_include_reasoning(True)
        print("Reasoning field enabled for evaluation responses.")

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

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the final output filename
    output_filename = f"{args.output_base}_{timestamp}.json"

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check for local inference dependencies if using a local model
    is_local_model = args.model.startswith(('/', './')) or args.model == DEFAULT_MODEL_ID
    if is_local_model:
        print("Local model specified, checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("Note: First run might take time to download the model.")
    else:
        print("Non-local model specified. Ensure necessary API keys (e.g., OpenAI) are configured.")

    print(f"Evaluating {args.model} using {args.evaluator.capitalize()} evaluator...")
    print(f"Using {args.system_prompt} system prompt.")
    if args.evaluator == "gemini":
        print(f"Gemini evaluation mode: {args.eval_mode}")
    print(f"Results will be saved to: {output_filename}")

    # Load the scenarios
    scenarios = load_scenarios(args.scenarios)

    # For testing, use only the first scenario if --test flag is provided
    if args.test:
        print("Running in test mode with only the first scenario...")
        scenarios = scenarios[:1]

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

    # Run evaluation(s)
    if args.compare_eval_modes:
        # Run comparison mode
        asyncio.run(run_comparison_mode(
            model_name=args.model,
            scenarios=scenarios,
            output_base=args.output_base,
            timestamp=timestamp,
            temperature=args.temperature,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            test_mode=args.test,
            system_prompt_type=args.system_prompt,
            apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
            apply_professional_referral_penalty=apply_professional_referral_penalty,
            generation_batch_size=args.generation_batch_size
        ))
    else:
        # Run single evaluation mode
        asyncio.run(evaluate_model_with_llm(
            model_name=args.model,
            scenarios=scenarios,
            output_file=output_filename,
            temperature=args.temperature,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            evaluator=args.evaluator,
            test_mode=args.test,
            system_prompt_type=args.system_prompt,
            apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
            apply_professional_referral_penalty=apply_professional_referral_penalty,
            generation_batch_size=args.generation_batch_size,
            gemini_eval_mode=args.eval_mode
        ))

        print("Baseline model evaluation complete!")

        if args.evaluator == "gemini":
            from src.utils.gemini_api_tracker import GeminiAPITracker
            tracker = GeminiAPITracker()
            tracker.print_summary()


if __name__ == "__main__":
    main()
