#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py - Evaluation functions for ArGen dataset generator
===============================================================
Contains functions for evaluating prompts and responses against policy criteria
"""

import asyncio
from typing import Dict, Optional, List

from .config import log

# Import OpenAI rewards for evaluation
import os
import sys
import importlib.util

# Try to import from the local symbolic link first
reward_functions_found = False
local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/reward_functions/openai_rewards.py")

if os.path.exists(local_path):
    log.info(f"Found OpenAI reward functions at {local_path}")
    try:
        # Add the src directory to sys.path
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Try to import directly
        try:
            from reward_functions.openai_rewards import (
                evaluate_ahimsa_with_openai,
                evaluate_dharma_with_openai,
                evaluate_helpfulness_with_openai,
                batch_evaluate_with_openai
            )
            reward_functions_found = True
            log.info("Successfully imported OpenAI reward functions from symbolic link")
        except ImportError:
            # If direct import fails, try using importlib
            spec = importlib.util.spec_from_file_location("openai_rewards", local_path)
            if spec and spec.loader:
                openai_rewards = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(openai_rewards)

                # Get the functions from the module
                evaluate_ahimsa_with_openai = openai_rewards.evaluate_ahimsa_with_openai
                evaluate_dharma_with_openai = openai_rewards.evaluate_dharma_with_openai
                evaluate_helpfulness_with_openai = openai_rewards.evaluate_helpfulness_with_openai
                batch_evaluate_with_openai = getattr(openai_rewards, 'batch_evaluate_with_openai', None)

                reward_functions_found = True
                log.info("Successfully imported OpenAI reward functions from symbolic link using importlib")
    except Exception as e:
        log.warning(f"Error importing from symbolic link {local_path}: {e}")

# If symbolic link approach failed, try importing from parent project
if not reward_functions_found:
    # Add the project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Try to import directly from parent project
    try:
        from argen.reward_functions.openai_rewards import (
            evaluate_ahimsa_with_openai,
            evaluate_dharma_with_openai,
            evaluate_helpfulness_with_openai,
            batch_evaluate_with_openai
        )
        reward_functions_found = True
        log.info("Successfully imported OpenAI reward functions from parent project")
    except ImportError:
        # If direct import fails, try using importlib
        parent_path = os.path.join(project_root, "src/reward_functions/openai_rewards.py")
        if os.path.exists(parent_path):
            try:
                spec = importlib.util.spec_from_file_location("openai_rewards", parent_path)
                if spec and spec.loader:
                    openai_rewards = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(openai_rewards)

                    # Get the functions from the module
                    evaluate_ahimsa_with_openai = openai_rewards.evaluate_ahimsa_with_openai
                    evaluate_dharma_with_openai = openai_rewards.evaluate_dharma_with_openai
                    evaluate_helpfulness_with_openai = openai_rewards.evaluate_helpfulness_with_openai
                    batch_evaluate_with_openai = getattr(openai_rewards, 'batch_evaluate_with_openai', None)

                    reward_functions_found = True
                    log.info("Successfully imported OpenAI reward functions from parent project using importlib")
            except Exception as e:
                log.warning(f"Error importing from parent project {parent_path}: {e}")
        else:
            log.warning(f"OpenAI reward functions not found in parent project at {parent_path}")

if not reward_functions_found:
    log.error("Could not import OpenAI reward functions. This is required for proper evaluation.")
    log.error("Please ensure the reward functions are properly installed.")

    # Import args to check if we're in dry-run mode
    import sys
    dry_run = "--dry-run" in sys.argv

    if not dry_run:
        raise RuntimeError("Policy reward functions not found. Use --dry-run flag for testing without reward functions.")
    else:
        log.warning("Running in dry-run mode with mock reward functions.")

        # For development/testing only with --dry-run flag
        # These mock functions should not be used in production
        async def evaluate_ahimsa_with_openai(prompt: str, response: str, api_key: str) -> Dict:
            log.warning("Using MOCK ahimsa evaluation - results will not be reliable!")
            return {"ahimsa_score": 0.5, "ahimsa_violation": False}

        async def evaluate_dharma_with_openai(prompt: str, response: str, api_key: str) -> Dict:
            log.warning("Using MOCK dharma evaluation - results will not be reliable!")
            return {"dharma_score": 0.5, "dharma_violation": False}

        async def evaluate_helpfulness_with_openai(prompt: str, response: str, api_key: str) -> Dict:
            log.warning("Using MOCK helpfulness evaluation - results will not be reliable!")
            return {"helpfulness_score": 0.5, "helpfulness_violation": False}

        # Mock batch evaluation function
        async def batch_evaluate_with_openai(prompts: List[str], responses: List[str], api_key: str, max_concurrency: int = 20) -> List[Dict]:
            log.warning("Using MOCK batch evaluation - results will not be reliable!")
            return [{"ahimsa_score": 0.5, "dharma_score": 0.5, "helpfulness_score": 0.5,
                     "ahimsa_violation": False, "dharma_violation": False, "helpfulness_violation": False,
                     "overall_risk": 1.5, "violation": False} for _ in range(len(prompts))]

async def evaluate_interaction(
    prompt: str,
    response: str,
    api_key: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Evaluate a user prompt and model response using OpenAI API.

    Args:
        prompt: The user prompt.
        response: The model's response.
        api_key: Optional OpenAI API key.
        metadata: Optional metadata containing tier information.

    Returns:
        Dictionary containing evaluation metrics.
    """
    if not api_key:
        log.error("evaluate_interaction: OpenAI API Key not available.")
        return {
            "ahimsa_score": 0.0, "dharma_score": 0.0, "helpfulness_score": 0.0,
            "ahimsa_violation": True, "dharma_violation": True, "helpfulness_violation": True,
            "error": "API Key missing"
        }

    log.debug("evaluate_interaction: evaluating prompt-response pair...")
    try:
        # Create all evaluation tasks concurrently
        # This allows the API calls to be made in parallel
        tasks = [
            evaluate_ahimsa_with_openai(prompt, response, api_key, metadata),
            evaluate_dharma_with_openai(prompt, response, api_key, metadata),
            evaluate_helpfulness_with_openai(prompt, response, api_key)
        ]

        # Gather results from concurrent evaluations
        results = await asyncio.gather(*tasks, return_exceptions=True)

        eval_results = {}
        keys = ["ahimsa", "dharma", "helpfulness"]
        all_successful = True

        for i, res in enumerate(results):
            key_base = keys[i]
            if isinstance(res, Exception):
                log.error(f"evaluate_interaction: Evaluation failed for {key_base}: {res}")
                eval_results[f"{key_base}_score"] = 0.0
                eval_results[f"{key_base}_violation"] = True
                eval_results[f"{key_base}_error"] = str(res)
                all_successful = False
            elif isinstance(res, dict):
                # For Ahimsa, capture raw scores before penalties
                if key_base == "ahimsa":
                    # Get raw scores without penalty for disclaimers/referrals
                    harm_avoidance = res.get("harm_avoidance_score", 0.0)
                    safety_context = res.get("safety_context_score", 0.0)
                    raw_ahimsa_score = (harm_avoidance + safety_context) / 2.0
                    eval_results[f"{key_base}_raw_score"] = raw_ahimsa_score

                    # Store whether medical disclaimer and referral were present
                    eval_results["has_medical_disclaimer"] = res.get("has_medical_disclaimer", False)
                    eval_results["has_professional_referral"] = res.get("has_professional_referral", False)

                    # Still store the penalized score for violation check
                    eval_results[f"{key_base}_score"] = res.get(f"{key_base}_score", 0.0)
                else:
                    # For other metrics, just use the score as provided
                    eval_results[f"{key_base}_score"] = res.get(f"{key_base}_score", 0.0)

                eval_results[f"{key_base}_violation"] = res.get(f"{key_base}_violation", True)
            else:
                log.error(f"evaluate_interaction: Unexpected result type for {key_base}: {type(res)}")
                eval_results[f"{key_base}_score"] = 0.0
                eval_results[f"{key_base}_violation"] = True
                all_successful = False

        if all_successful:
            # Calculate overall_risk based on summing individual risks (1-score)
            # Use ahimsa_raw_score (before penalties) if available, otherwise use regular score
            ahimsa_score = eval_results.get("ahimsa_raw_score", eval_results.get("ahimsa_score", 0.0))
            ahimsa_risk = 1.0 - ahimsa_score

            # Add dharma risk (1-score)
            dharma_score = eval_results.get("dharma_score", 0.0)
            dharma_risk = 1.0 - dharma_score

            # Add helpfulness risk (1-score)
            helpfulness_score = eval_results.get("helpfulness_score", 0.0)
            helpfulness_risk = 1.0 - helpfulness_score

            # Sum the individual risks
            total_risk = ahimsa_risk + dharma_risk + helpfulness_risk

            # Store the individual risks and total risk
            eval_results["ahimsa_risk"] = ahimsa_risk
            eval_results["dharma_risk"] = dharma_risk
            eval_results["helpfulness_risk"] = helpfulness_risk
            eval_results["overall_risk"] = total_risk

            # Add a single violation flag for easier downstream checks
            eval_results["violation"] = (
                eval_results.get("ahimsa_violation", False) or
                eval_results.get("dharma_violation", False) or
                eval_results.get("helpfulness_violation", False)
            )

            log.debug(f"evaluate_interaction: Risks AH={ahimsa_risk:.2f}, "
                     f"DH={dharma_risk:.2f}, "
                     f"HP={helpfulness_risk:.2f}, "
                     f"Total Risk={total_risk:.2f}")

        return eval_results

    except Exception as e:
        log.error(f"evaluate_interaction: Unexpected error: {e}")
        return {
            "ahimsa_score": 0.0, "dharma_score": 0.0, "helpfulness_score": 0.0,
            "ahimsa_violation": True, "dharma_violation": True, "helpfulness_violation": True,
            "error": f"Evaluation error: {e}"
        }

async def evaluate_batch_interactions(
    prompts: List[str],
    responses: List[str],
    api_key: str,
    concurrent_limit: int = 20,
    metadata_list: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Evaluate a batch of prompt-response pairs with high concurrency.
    Uses the batch_evaluate_with_openai function if available, otherwise falls back to individual evaluations.

    Args:
        prompts: List of prompt strings.
        responses: List of response strings.
        api_key: OpenAI API key.
        concurrent_limit: Maximum number of concurrent evaluations (default: 20).
        metadata_list: Optional list of metadata dicts corresponding to each prompt-response pair.

    Returns:
        List of evaluation result dictionaries.
    """
    if len(prompts) != len(responses):
        log.error(f"Mismatch between prompts ({len(prompts)}) and responses ({len(responses)})")
        return []

    # Check if batch evaluation function is available
    if 'batch_evaluate_with_openai' in globals():
        log.info(f"Using batch evaluation for {len(prompts)} prompt-response pairs with concurrency {concurrent_limit}...")
        start_time = asyncio.get_event_loop().time()

        # Use the batch evaluation function
        results = await batch_evaluate_with_openai(
            prompts=prompts,
            responses=responses,
            openai_api_key=api_key,
            max_concurrency=concurrent_limit,
            metadata_list=metadata_list  # Pass the metadata list
        )

        end_time = asyncio.get_event_loop().time()
        log.info(f"Completed batch evaluation of {len(prompts)} pairs in {end_time - start_time:.2f} seconds "
                f"(avg: {(end_time - start_time) / max(1, len(prompts)):.2f}s per pair)")

        return results
    else:
        # Fall back to individual evaluations with semaphore
        log.info(f"Batch evaluation function not available, falling back to individual evaluations...")

        # Create a semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def evaluate_with_semaphore(prompt, response, metadata=None):
            async with semaphore:
                # Pass metadata to evaluate_interaction if available
                return await evaluate_interaction(prompt, response, api_key, metadata)

        # Create all evaluation tasks at once with semaphore control
        eval_tasks = [
            evaluate_with_semaphore(p, r, metadata_list[i] if metadata_list and i < len(metadata_list) else None)
            for i, (p, r) in enumerate(zip(prompts, responses))
        ]

        # Run all evaluations concurrently with semaphore control
        log.info(f"Evaluating {len(prompts)} prompt-response pairs with concurrency limit {concurrent_limit}...")
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*eval_tasks)
        end_time = asyncio.get_event_loop().time()

        log.info(f"Completed {len(prompts)} evaluations in {end_time - start_time:.2f} seconds "
                f"(avg: {(end_time - start_time) / max(1, len(prompts)):.2f}s per evaluation)")

        return results

def is_challenging_prompt(eval_dict: Dict, risk_threshold: float) -> bool:
    """
    Determine if a prompt represents a challenge for the model based on evaluation results.

    Args:
        eval_dict: Evaluation results dictionary.
        risk_threshold: Threshold for considering a prompt challenging. Configurable via --fail-threshold CLI flag.

    Returns:
        Boolean indicating if the prompt is challenging.
    """
    # Log the threshold being used for debugging
    log.debug(f"Using risk_threshold: {risk_threshold} for prompt evaluation")

    # Use the single violation flag if available, otherwise fall back to individual checks
    return (
        eval_dict.get("overall_risk", 0.0) > risk_threshold or
        eval_dict.get("violation", False) or
        (
            eval_dict.get("ahimsa_violation", False) or
            eval_dict.get("dharma_violation", False) or
            eval_dict.get("helpfulness_violation", False)
        )
    )