"""
TRL-compatible reward functions for ArGen GRPO training.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Union, Any
import random # Import random for sampling

from openai import AsyncOpenAI
import wandb
import torch
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
VERBOSE_LOGGING = False

# Default error response
DEFAULT_EVAL_RESPONSE = {
    "error": "Evaluation failed",
    "reasoning": "Could not get response from API.",
    "ahimsa_score": 0.0,
    "harm_avoidance_score": 0.0,
    "safety_consideration_score": 0.0,
    "has_medical_disclaimer": False,
    "has_professional_referral": False,
    "ahimsa_violation": True,  # Default to violation on error
    "dharma_score": 0.0,
    "domain_adherence_score": 0.0,
    "out_of_domain_advice": True,  # Default to violation on error
    "response_scope": "S3",  # Default to most restrictive scope on error
    "scope_risk": "high",  # Default to highest risk on error
    "scope_penalty_factor": 0.0,  # Default to maximum penalty on error
    "dharma_violation": True
}

# Import the async evaluation functions from openai_rewards
from src.reward_functions.openai_rewards import (
    evaluate_ahimsa_with_openai,
    evaluate_dharma_with_openai,
    evaluate_helpfulness_with_openai
)

# Import Gemini reward functions if available
from src.reward_functions.gemini.ahimsa import evaluate_ahimsa_with_gemini
from src.reward_functions.gemini.ahimsa import evaluate_ahimsa_multi_with_gemini
from src.reward_functions.gemini.ahimsa import batch_process_ahimsa_evaluations_concurrently
from src.reward_functions.gemini.dharma import evaluate_dharma_with_gemini
from src.reward_functions.gemini.dharma import batch_process_dharma_evaluations_concurrently
from src.reward_functions.gemini.helpfulness import evaluate_helpfulness_with_gemini
from src.reward_functions.gemini.helpfulness import batch_process_helpfulness_evaluations_concurrently
from src.reward_functions.gemini_rewards import GEMINI_EVAL_MODEL

# Import data integrity utilities
from src.utils.data_integrity import (
    verify_prompt_tier_hash,
    extract_tier_from_compound
)

# Import configurations
from src.config import GRPO_CONFIG, REWARD_WEIGHTS

# Module-level storage for audit data with step-based tracking
# This prevents race conditions between reward function execution and callback reading
_audit_log_data = []
_audit_log_data_step = None  # Track which step the current audit data belongs to
_audit_log_data_by_step = {}  # Store audit data by step number for timing independence

# Configuration for the check (can be moved to a config file)
VERIFY_HASH_SAMPLE_RATE = 0.1 # Check 10% of items per batch

def populate_audit_log_data(
    prompts: List[str],
    processed_completions: List[str],
    compound_tiers: List[str],
    ahimsa_results: List[Dict],
    dharma_results: List[Dict],
    helpfulness_results: List[Dict],
    **kwargs
) -> None:
    """
    Populate _audit_log_data with evaluation results for callback logging.

    This function extracts the audit data population logic from combined_reward_trl
    so it can be reused by SharedEvaluationCoordinator in separate rewards mode.

    Args:
        prompts: List of user prompts
        processed_completions: List of processed completions
        compound_tiers: List of compound tier values
        ahimsa_results: List of Ahimsa evaluation results
        dharma_results: List of Dharma evaluation results
        helpfulness_results: List of Helpfulness evaluation results
        **kwargs: Additional arguments (scope, etc.)
    """
    global _audit_log_data

    # Only main process should modify shared state
    if os.getenv("RANK", "0") != "0":
        return

    # CRITICAL FIX: Don't clear existing data here - let the callback handle clearing
    # This prevents race conditions where we clear data before the callback reads it
    # The callback will clear the data after it processes it

    # If we already have data for this exact batch, don't duplicate it
    if _audit_log_data and len(_audit_log_data) == len(prompts):
        # Check if this looks like the same batch (same number of items)
        logger.info(f"🔍 DEBUG: Skipping audit data population - already have {len(_audit_log_data)} entries for {len(prompts)} prompts")
        return

    # Clear only if we're starting fresh or have mismatched data
    if _audit_log_data:
        logger.info(f"🔍 DEBUG: Clearing {len(_audit_log_data)} existing audit entries to populate new batch")
    _audit_log_data.clear()

    # Apply weights from config (default to equal)
    weights = REWARD_WEIGHTS

    # Check if severity penalty should be applied during training
    apply_severity_penalty_in_training = GRPO_CONFIG.get("apply_training_severity_penalty", False)

    logger.info(f"🔍 DEBUG: populate_audit_log_data called with {len(prompts)} items")
    logger.info(f"🔍 DEBUG: _audit_log_data id = {id(_audit_log_data)}, cleared length = {len(_audit_log_data)}")

    for i in range(len(prompts)):
        # Extract scores from results
        a_score = ahimsa_results[i].get("ahimsa_score", 0.0)
        d_score = dharma_results[i].get("dharma_score", 0.0)
        h_score = helpfulness_results[i].get("helpfulness_score", 0.0)

        # Calculate severity penalty (same logic as combined_reward_trl)
        severity_penalty = 0.0
        scope_penalty_factor = 1.0

        if apply_severity_penalty_in_training:
            # Extract scope from kwargs if available
            compound_scopes = kwargs.get("scope", [])
            if i < len(compound_scopes):
                compound_scope = compound_scopes[i]
                if compound_scope:
                    # Extract scope value from compound format if needed
                    scope_value = extract_tier_from_compound(compound_scope) if ":" in str(compound_scope) else str(compound_scope)

                    # Apply scope-based penalty
                    if scope_value == "S0":
                        scope_penalty_factor = 0.5  # 50% penalty for S0 (most severe)
                    elif scope_value == "S1":
                        scope_penalty_factor = 0.75  # 25% penalty for S1
                    elif scope_value == "S2":
                        scope_penalty_factor = 0.9   # 10% penalty for S2
                    # S3 gets no penalty (factor = 1.0)

                    severity_penalty = 1.0 - scope_penalty_factor

        # Calculate combined reward with weights and penalties
        combined = (
            weights["ahimsa"] * a_score +
            weights["dharma"] * d_score +
            weights["helpfulness"] * h_score
        ) * scope_penalty_factor

        # Create audit entry
        audit_entry = {
            "prompt": prompts[i],
            "completion": processed_completions[i],
            "tier_raw": compound_tiers[i],
            "ahimsa_score": a_score,
            "dharma_score": d_score,
            "helpfulness_score": h_score,
            "combined_reward": combined,
            "severity_penalty": severity_penalty,
            "scope_penalty_factor": scope_penalty_factor
        }
        _audit_log_data.append(audit_entry)

        # DEBUG: Log first entry
        if i == 0:
            logger.info(f"🔍 DEBUG: First audit entry appended: {audit_entry}")
            logger.info(f"🔍 DEBUG: _audit_log_data length after first append = {len(_audit_log_data)}")

    logger.info(f"🔍 DEBUG: populate_audit_log_data completed. Final _audit_log_data length = {len(_audit_log_data)}")
    logger.info(f"🔍 DEBUG: Final _audit_log_data id = {id(_audit_log_data)}")

def configure_gemini_reasoning(include_reasoning: bool):
    """
    Configure whether to include reasoning in Gemini evaluations.

    Args:
        include_reasoning: Whether to include reasoning in Gemini evaluations
    """
    try:
        # This import should be at the top level, or set_include_reasoning needs to be imported from gemini_rewards
        # Assuming set_include_reasoning is available via `from src.reward_functions.gemini_rewards import ...`
        # If not, this line will cause an error. For now, assuming it's imported.
        from src.reward_functions.gemini_rewards import set_include_reasoning
        set_include_reasoning(include_reasoning)
        logger.info(f"TRL: Set include_reasoning to {include_reasoning} for Gemini evaluations")
    except NameError: # If set_include_reasoning is not defined (e.g. gemini_rewards itself failed to import)
        logger.error(f"Failed to set include_reasoning: gemini_rewards components might not be available.")
    except Exception as e:
        logger.error(f"Failed to set include_reasoning for Gemini evaluations: {e}")

# Helper function to safely run async functions
def run_async_safely(coro):
    """Run an async coroutine in a way that's safe regardless of the current event loop state."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # If the loop is closed, create a new one
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)

# --- REMOVED: _wandb_log_safe helper function is no longer needed here ---

from src.reward_functions.chat_response_helper import process_completions # Import the helper function

def ahimsa_reward_trl(prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> List[float]:
    """
    TRL-compatible reward function for Ahimsa evaluation.
    Includes random hash verification.
    Supports both OpenAI and Gemini evaluators.

    Args:
        prompts: List of user prompts
        completions: List of model completions
        **kwargs: Additional columns from the dataset

    Returns:
        List of Ahimsa reward scores
    """
    # Check if we're in separate rewards mode and should use shared coordinator
    from src.reward_functions.shared_evaluation_coordinator import is_separate_rewards_mode, get_shared_coordinator

    if is_separate_rewards_mode():
        logger.info("ahimsa_reward_trl: Using shared evaluation coordinator for concurrent evaluation")
        coordinator = get_shared_coordinator()

        # Use the coordinator to get results, triggering concurrent evaluation if needed
        results = coordinator.get_or_evaluate_batch_sync(
            reward_type="ahimsa",
            prompts=prompts,
            completions=completions,
            **kwargs
        )

        # Extract scores and return
        rewards = [result.get("ahimsa_score", 0.0) for result in results]
        logger.info(f"ahimsa_reward_trl: Returning {len(rewards)} scores from shared coordinator")
        return rewards

    # ── sanity-check that we really got G completions per prompt ──
    G = GRPO_CONFIG["num_generations"]                                     # or read from cfg
    assert len(prompts) == len(completions)
    assert len(completions) % G == 0, \
        f"Batch has {len(completions)} rows; not divisible by {G}"
    # ─────────────────────────────────────────────────────────────


    # Process completions to extract content from chat responses
    processed_completions = process_completions(completions)

    # Determine which evaluator to use
    evaluator = os.environ.get("ARGEN_EVALUATOR", "openai")

    # Get the appropriate API key
    if evaluator == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment")
            return [0.0] * len(prompts)  # Return zeros on error
    else:  # evaluator == "gemini"
        # For Gemini, we don't need to store the API key here since the configure_gemini()
        # function in gemini_rewards.py will handle it, but we still check if it's available
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("Gemini API key not found in environment")
            return [0.0] * len(prompts)  # Return zeros on error
        api_key = None  # Not used for Gemini

    # --- ADDED: Log the first fully formatted prompt in the batch (Ahimsa) ---
    if prompts:
        if VERBOSE_LOGGING:
            logger.info(f"GRPO_PROMPT_DEBUG (ahimsa_reward_trl): First prompt in batch (total {len(prompts)}):\n{prompts[0]}")
    # --- END ADDED ---
    # --- ADDED: Log the first processed completion in the batch (Ahimsa) ---
    if processed_completions:
        if VERBOSE_LOGGING:
            logger.info(f"GRPO_COMPLETION_DEBUG (ahimsa_reward_trl): First completion in batch (total {len(processed_completions)}):\n{processed_completions[0]}")
    # --- END ADDED ---

    # --- Perform Verification ---
    compound_tiers = kwargs.get("tier", [])
    num_items = len(prompts)
    if len(compound_tiers) != num_items:
        logger.error(f"[Ahimsa TRL] Mismatch: Prompts ({num_items}) vs Tiers ({len(compound_tiers)})")
        return [0.0] * num_items

    num_to_sample = max(1, int(num_items * VERIFY_HASH_SAMPLE_RATE))
    sample_indices = random.sample(range(num_items), num_to_sample)
    for i in sample_indices:
        if not verify_prompt_tier_hash(prompts[i], compound_tiers[i]):
            # Error logged within verify function. Decide action.
            # For now, just log and continue, but could return zeros.
            logger.error(f"[Ahimsa TRL] Hash verification failed for sample index {i}.")
            # return [0.0] * num_items # Optional: Fail entire batch on mismatch
    # --- End Verification ---

    # Create a list of coroutines to run concurrently
    async def run_all_evaluations():
        tasks = []
        if evaluator == "openai":
            for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
                actual_tier = extract_tier_from_compound(compound_tiers[idx])
                tasks.append(
                    evaluate_ahimsa_with_openai(
                        prompt,
                        completion,
                        api_key,
                        original_prompt_meta={"tier": actual_tier}
                    )
                )
            raw_results = await asyncio.gather(*tasks)
            results = raw_results # For OpenAI, gather directly gives the list of dicts
        else:  # evaluator == "gemini"
            # Check if we should use single calls or batch calls for training
            use_single_calls = GRPO_CONFIG.get("use_single_gemini_calls_for_training", True)

            if use_single_calls:
                # Single-call approach: one Gemini API call per prompt+completion pair
                logger.info(f"ahimsa_reward_trl: Using single-call approach for {len(prompts)} items")

                # Import the single evaluation function
                from src.reward_functions.gemini.ahimsa import evaluate_ahimsa_with_gemini

                # Set up semaphore for concurrency control
                max_concurrent = GRPO_CONFIG.get("gemini_single_call_max_concurrent", 200)
                semaphore = asyncio.Semaphore(max_concurrent)

                async def evaluate_single_with_semaphore(prompt, completion, tier):
                    async with semaphore:
                        return await evaluate_ahimsa_with_gemini(
                            prompt,
                            completion,
                            original_prompt_meta={"tier": tier}
                        )

                # Create tasks for all prompt+completion pairs
                single_tasks = []
                for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
                    actual_tier = extract_tier_from_compound(compound_tiers[idx])
                    single_tasks.append(evaluate_single_with_semaphore(prompt, completion, actual_tier))

                results = await asyncio.gather(*single_tasks)

                if VERBOSE_LOGGING:
                    logger.info(f"[ahimsa_reward_trl DEBUG] Results from single-call approach: {json.dumps(results, indent=2)}")
            else:
                # Batch approach: multiple prompt+completion pairs per Gemini API call
                logger.info(f"ahimsa_reward_trl: Using batch-call approach for {len(prompts)} items")

                all_ahimsa_items_to_evaluate = []
                for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
                    actual_tier = extract_tier_from_compound(compound_tiers[idx])
                    all_ahimsa_items_to_evaluate.append({
                        "prompt": prompt,
                        "model_response": completion,
                        "original_prompt_meta": {"tier": actual_tier}
                    })

                if VERBOSE_LOGGING:
                    logger.info(f"[ahimsa_reward_trl DEBUG] All Ahimsa items to evaluate with Gemini: {json.dumps(all_ahimsa_items_to_evaluate, indent=2)}")

                items_per_gemini_call_ahimsa = GRPO_CONFIG.get("gemini_ahimsa_items_per_call_combined", 10)
                max_concurrent_calls_ahimsa = GRPO_CONFIG.get("gemini_ahimsa_max_concurrent_combined", 50)

                results = await batch_process_ahimsa_evaluations_concurrently(
                    all_ahimsa_items_to_evaluate,
                    items_per_gemini_call=items_per_gemini_call_ahimsa,
                    max_concurrent_calls=max_concurrent_calls_ahimsa
                )
                if VERBOSE_LOGGING:
                    logger.info(f"[ahimsa_reward_trl DEBUG] Results from batch_process_ahimsa_evaluations_concurrently: {json.dumps(results, indent=2)}")

        logger.info(f"ahimsa_reward_trl: Completed {len(results) if results else 0} evaluation tasks/items.")
        return results

    # Run all evaluations in one go
    results = run_async_safely(run_all_evaluations())

    # Extract scores from results
    rewards = [result.get("ahimsa_score", 0.0) for result in results]
    if VERBOSE_LOGGING and evaluator == "gemini":
        logger.info(f"[ahimsa_reward_trl DEBUG] Extracted Ahimsa scores (rewards): {rewards}")
    return rewards

def dharma_reward_trl(prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> List[float]:
    """
    TRL-compatible reward function for Dharma evaluation.
    Includes random hash verification.
    Supports both OpenAI and Gemini evaluators.

    Args:
        prompts: List of user prompts
        completions: List of model completions
        **kwargs: Additional columns from the dataset

    Returns:
        List of Dharma reward scores
    """
    # Check if we're in separate rewards mode and should use shared coordinator
    from src.reward_functions.shared_evaluation_coordinator import is_separate_rewards_mode, get_shared_coordinator

    if is_separate_rewards_mode():
        logger.info("dharma_reward_trl: Using shared evaluation coordinator for concurrent evaluation")
        coordinator = get_shared_coordinator()

        # Use the coordinator to get results, triggering concurrent evaluation if needed
        results = coordinator.get_or_evaluate_batch_sync(
            reward_type="dharma",
            prompts=prompts,
            completions=completions,
            **kwargs
        )

        # Extract scores and return
        rewards = [result.get("dharma_score", 0.0) for result in results]
        logger.info(f"dharma_reward_trl: Returning {len(rewards)} scores from shared coordinator")
        return rewards

    # Process completions to extract content from chat responses
    processed_completions = process_completions(completions)

    # Determine which evaluator to use
    evaluator = os.environ.get("ARGEN_EVALUATOR", "openai")

    # Get the appropriate API key
    if evaluator == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment")
            return [0.0] * len(prompts)  # Return zeros on error
    else:  # evaluator == "gemini"
        # For Gemini, we don't need to store the API key here since the configure_gemini()
        # function in gemini_rewards.py will handle it, but we still check if it's available
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("Gemini API key not found in environment")
            return [0.0] * len(prompts)  # Return zeros on error
        api_key = None  # Not used for Gemini

    # --- ADDED: Log the first fully formatted prompt in the batch (Dharma) ---
    if prompts:
        if VERBOSE_LOGGING:
            logger.info(f"GRPO_PROMPT_DEBUG (dharma_reward_trl): First prompt in batch (total {len(prompts)}):\n{prompts[0]}")
    # --- END ADDED ---
    # --- ADDED: Log the first processed completion in the batch (Dharma) ---
    if processed_completions:
        if VERBOSE_LOGGING:
            logger.info(f"GRPO_COMPLETION_DEBUG (dharma_reward_trl): First completion in batch (total {len(processed_completions)}):\n{processed_completions[0]}")
    # --- END ADDED ---

    # --- Perform Verification ---
    compound_tiers = kwargs.get("tier", []) # Still need tiers for verification
    num_items = len(prompts)
    if len(compound_tiers) != num_items:
        logger.error(f"[Dharma TRL] Mismatch: Prompts ({num_items}) vs Tiers ({len(compound_tiers)})")
        return [0.0] * num_items

    num_to_sample = max(1, int(num_items * VERIFY_HASH_SAMPLE_RATE))
    sample_indices = random.sample(range(num_items), num_to_sample)
    for i in sample_indices:
        if not verify_prompt_tier_hash(prompts[i], compound_tiers[i]):
            logger.error(f"[Dharma TRL] Hash verification failed for sample index {i}.")
            # return [0.0] * num_items # Optional: Fail entire batch
    # --- End Verification ---

    # Create a list of coroutines to run concurrently
    async def run_all_evaluations():
        tasks = []
        if evaluator == "openai":
            for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
                 # Extract actual tier before passing (needed for consistency, even if dharma doesn't use it directly yet)
                actual_tier = extract_tier_from_compound(compound_tiers[idx]) # Ensure compound_tiers is defined and populated
                current_prompt_meta = {"tier": actual_tier}
                # Dharma also needs scope information
                compound_scope = kwargs.get("scope", [])[idx] if "scope" in kwargs and idx < len(kwargs["scope"]) else None
                if compound_scope:
                    current_prompt_meta["scope"] = compound_scope


                tasks.append(
                    evaluate_dharma_with_openai(
                        prompt,
                        completion,
                        api_key,
                        original_prompt_meta=current_prompt_meta
                    )
                )
            raw_results = await asyncio.gather(*tasks)
            results = raw_results # For OpenAI, gather directly gives the list of dicts
        else:  # evaluator == "gemini"
            # Check if we should use single calls or batch calls for training
            use_single_calls = GRPO_CONFIG.get("use_single_gemini_calls_for_training", True)

            if use_single_calls:
                # Single-call approach: one Gemini API call per prompt+completion pair
                logger.info(f"dharma_reward_trl: Using single-call approach for {len(prompts)} items")

                # Import the single evaluation function
                from src.reward_functions.gemini.dharma import evaluate_dharma_with_gemini

                # Set up semaphore for concurrency control
                max_concurrent = GRPO_CONFIG.get("gemini_single_call_max_concurrent", 200)
                semaphore = asyncio.Semaphore(max_concurrent)

                async def evaluate_single_with_semaphore(prompt, completion, prompt_meta):
                    async with semaphore:
                        return await evaluate_dharma_with_gemini(
                            prompt,
                            completion,
                            original_prompt_meta=prompt_meta
                        )

                # Create tasks for all prompt+completion pairs
                single_tasks = []
                for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
                    actual_tier = extract_tier_from_compound(compound_tiers[idx])
                    current_prompt_meta = {"tier": actual_tier}
                    compound_scope = kwargs.get("scope", [])[idx] if "scope" in kwargs and idx < len(kwargs["scope"]) else None
                    if compound_scope:
                        current_prompt_meta["scope"] = compound_scope
                    single_tasks.append(evaluate_single_with_semaphore(prompt, completion, current_prompt_meta))

                results = await asyncio.gather(*single_tasks)

                if VERBOSE_LOGGING:
                    logger.info(f"[dharma_reward_trl DEBUG] Results from single-call approach: {json.dumps(results, indent=2)}")
            else:
                # Batch approach: multiple prompt+completion pairs per Gemini API call
                logger.info(f"dharma_reward_trl: Using batch-call approach for {len(prompts)} items")

                all_dharma_items_to_evaluate = []
                for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
                    actual_tier = extract_tier_from_compound(compound_tiers[idx])
                    current_prompt_meta = {"tier": actual_tier}
                    compound_scope = kwargs.get("scope", [])[idx] if "scope" in kwargs and idx < len(kwargs["scope"]) else None
                    if compound_scope:
                        current_prompt_meta["scope"] = compound_scope
                    # For batch Dharma
                    all_dharma_items_to_evaluate.append({
                        "prompt": prompt,
                        "model_response": completion,
                        "original_prompt_meta": current_prompt_meta
                    })

                if VERBOSE_LOGGING:
                    logger.info(f"[dharma_reward_trl DEBUG] All Dharma items to evaluate with Gemini: {json.dumps(all_dharma_items_to_evaluate, indent=2)}")

                # Use specific config keys for Dharma TRL batching, similar to Ahimsa
                items_per_gemini_call_dharma = GRPO_CONFIG.get("gemini_dharma_items_per_call_combined", 10)
                max_concurrent_calls_dharma = GRPO_CONFIG.get("gemini_dharma_max_concurrent_combined", 50)

                results = await batch_process_dharma_evaluations_concurrently(
                    all_dharma_items_to_evaluate,
                    items_per_gemini_call=items_per_gemini_call_dharma,
                    max_concurrent_calls=max_concurrent_calls_dharma
                )
                if VERBOSE_LOGGING:
                    logger.info(f"[dharma_reward_trl DEBUG] Results from batch_process_dharma_evaluations_concurrently: {json.dumps(results, indent=2)}")

        logger.info(f"dharma_reward_trl: Completed {len(results) if results else 0} evaluation tasks/items.")
        return results

    # Run all evaluations in one go
    results = run_async_safely(run_all_evaluations())

    # Extract scores from results
    rewards = [result.get("dharma_score", 0.0) for result in results]
    if VERBOSE_LOGGING and evaluator == "gemini":
        logger.info(f"[dharma_reward_trl DEBUG] Extracted Dharma scores (rewards): {rewards}")
    return rewards

def helpfulness_reward_trl(prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> List[float]:
    """
    TRL-compatible reward function for Helpfulness evaluation.
    Includes random hash verification (for consistency, even if helpfulness doesn't use tier).
    Supports both OpenAI and Gemini evaluators.

    Args:
        prompts: List of user prompts
        completions: List of model completions
        **kwargs: Additional columns from the dataset (currently unused for helpfulness but good for consistency)

    Returns:
        List of Helpfulness reward scores
    """
    # Check if we're in separate rewards mode and should use shared coordinator
    from src.reward_functions.shared_evaluation_coordinator import is_separate_rewards_mode, get_shared_coordinator

    if is_separate_rewards_mode():
        logger.info("helpfulness_reward_trl: Using shared evaluation coordinator for concurrent evaluation")
        coordinator = get_shared_coordinator()

        # Use the coordinator to get results, triggering concurrent evaluation if needed
        results = coordinator.get_or_evaluate_batch_sync(
            reward_type="helpfulness",
            prompts=prompts,
            completions=completions,
            **kwargs
        )

        # Extract scores and return
        rewards = [result.get("helpfulness_score", 0.0) for result in results]
        logger.info(f"helpfulness_reward_trl: Returning {len(rewards)} scores from shared coordinator")
        return rewards

    # Process completions to extract content from chat responses
    processed_completions = process_completions(completions)

    # Determine which evaluator to use
    evaluator = os.environ.get("ARGEN_EVALUATOR", "openai")

    # Get the appropriate API key
    if evaluator == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment for helpfulness_reward_trl")
            return [0.0] * len(prompts)  # Return zeros on error
    else:  # evaluator == "gemini"
        # For Gemini, we don't need to store the API key here since the configure_gemini()
        # function in gemini_rewards.py will handle it, but we still check if it's available
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("Gemini API key not found in environment for helpfulness_reward_trl")
            return [0.0] * len(prompts)  # Return zeros on error
        api_key = None  # Not used for Gemini

    # --- ADDED: Log the first fully formatted prompt in the batch (Helpfulness) ---
    if prompts:
        if VERBOSE_LOGGING:
            logger.info(f"GRPO_PROMPT_DEBUG (helpfulness_reward_trl): First prompt in batch (total {len(prompts)}):\n{prompts[0]}")
    # --- END ADDED ---
    # --- ADDED: Log the first processed completion in the batch (Helpfulness) ---
    if processed_completions:
        if VERBOSE_LOGGING:
            logger.info(f"GRPO_COMPLETION_DEBUG (helpfulness_reward_trl): First completion in batch (total {len(processed_completions)}):\n{processed_completions[0]}")
    # --- END ADDED ---

    # --- Perform Verification ---
    compound_tiers = kwargs.get("tier", []) # Still need tiers for verification
    num_items = len(prompts)
    if len(compound_tiers) != num_items:
        logger.error(f"[Helpfulness TRL] Mismatch: Prompts ({num_items}) vs Tiers ({len(compound_tiers)})")
        return [0.0] * num_items

    num_to_sample = max(1, int(num_items * VERIFY_HASH_SAMPLE_RATE))
    sample_indices = random.sample(range(num_items), num_to_sample)
    for i in sample_indices:
        if not verify_prompt_tier_hash(prompts[i], compound_tiers[i]):
            logger.error(f"[Helpfulness TRL] Hash verification failed for sample index {i}.")
            # return [0.0] * num_items # Optional: Fail entire batch
    # --- End Verification ---

    # Create a list of coroutines to run concurrently
    async def run_all_evaluations():
        tasks = []
        if evaluator == "openai":
            for prompt, completion in zip(prompts, processed_completions):
                # Helpfulness doesn't use tier, so no need to extract/pass metadata
                tasks.append(evaluate_helpfulness_with_openai(prompt, completion, api_key))
            results = await asyncio.gather(*tasks)
        else:  # evaluator == "gemini"
            # Check if we should use single calls or batch calls for training
            use_single_calls = GRPO_CONFIG.get("use_single_gemini_calls_for_training", True)

            if use_single_calls:
                # Single-call approach: one Gemini API call per prompt+completion pair
                logger.info(f"helpfulness_reward_trl: Using single-call approach for {len(prompts)} items")

                # Import the single evaluation function
                from src.reward_functions.gemini.helpfulness import evaluate_helpfulness_with_gemini

                # Set up semaphore for concurrency control
                max_concurrent = GRPO_CONFIG.get("gemini_single_call_max_concurrent", 200)
                semaphore = asyncio.Semaphore(max_concurrent)

                async def evaluate_single_with_semaphore(prompt, completion):
                    async with semaphore:
                        return await evaluate_helpfulness_with_gemini(prompt, completion)

                # Create tasks for all prompt+completion pairs
                single_tasks = []
                for prompt, completion in zip(prompts, processed_completions):
                    single_tasks.append(evaluate_single_with_semaphore(prompt, completion))

                results = await asyncio.gather(*single_tasks)

                if VERBOSE_LOGGING:
                    logger.info(f"[helpfulness_reward_trl DEBUG] Results from single-call approach: {json.dumps(results, indent=2)}")
            else:
                # Batch approach: multiple prompt+completion pairs per Gemini API call
                logger.info(f"helpfulness_reward_trl: Using batch-call approach for {len(prompts)} items")

                all_helpfulness_items_to_evaluate = []
                for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
                    # Helpfulness doesn't use original_prompt_meta in its current single/multi eval signatures
                    all_helpfulness_items_to_evaluate.append({
                        "prompt": prompt,
                        "model_response": completion,
                        "original_prompt_meta": {} # Pass empty meta for consistency if helpfulness_multi expects it
                    })

                if VERBOSE_LOGGING:
                    logger.info(f"[helpfulness_reward_trl DEBUG] All Helpfulness items to evaluate with Gemini: {json.dumps(all_helpfulness_items_to_evaluate, indent=2)}")

                items_per_gemini_call_helpfulness = GRPO_CONFIG.get("gemini_helpfulness_items_per_call_combined", 10)
                max_concurrent_calls_helpfulness = GRPO_CONFIG.get("gemini_helpfulness_max_concurrent_combined", 50)

                results = await batch_process_helpfulness_evaluations_concurrently(
                    all_helpfulness_items_to_evaluate,
                    items_per_gemini_call=items_per_gemini_call_helpfulness,
                    max_concurrent_calls=max_concurrent_calls_helpfulness
                )
                if VERBOSE_LOGGING:
                    logger.info(f"[helpfulness_reward_trl DEBUG] Results from batch_process_helpfulness_evaluations_concurrently: {json.dumps(results, indent=2)}")

        logger.info(f"helpfulness_reward_trl: Completed {len(results) if results else 0} evaluation tasks/items.")
        return results

    # Run all evaluations in one go
    results = run_async_safely(run_all_evaluations())

    # Extract scores from results
    rewards = [result.get("helpfulness_score", 0.0) for result in results]
    return rewards

def combined_reward_trl(prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> List[float]:
    """
    TRL-compatible reward function combining Ahimsa, Dharma, and Helpfulness scores.
    Populates _audit_log_data for logging by the callback.
    Includes random hash verification.
    Supports both OpenAI and Gemini evaluators.

    Args:
        prompts: List of user prompts
        completions: List of model completions
        **kwargs: Additional columns from the dataset

    Returns:
        List of combined reward scores
    """
    # Process completions to extract content from chat responses
    processed_completions = process_completions(completions)

    # Determine which evaluator to use
    evaluator = os.environ.get("ARGEN_EVALUATOR", "openai")

    # Get the appropriate API key
    if evaluator == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment")
            return [0.0] * len(prompts)  # Return zeros on error
    else:  # evaluator == "gemini"
        # For Gemini, we don't need to store the API key here since the configure_gemini()
        # function in gemini_rewards.py will handle it, but we still check if it's available
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("Gemini API key not found in environment")
            return [0.0] * len(prompts)  # Return zeros on error
        api_key = None  # Not used for Gemini

    # --- ADDED: Log the first fully formatted prompt in the batch ---
    if prompts:
        if VERBOSE_LOGGING:
            logger.info(f"GRPO_PROMPT_DEBUG (combined_reward_trl): First prompt in batch (total {len(prompts)}):\n{prompts[0]}")
    # --- END ADDED ---

    # --- Perform Verification ---
    compound_tiers = kwargs.get("tier", []) # Get compound tiers from kwargs
    num_items = len(prompts)
    if len(compound_tiers) != num_items:
        logger.error(f"[Combined TRL] Mismatch: Prompts ({num_items}) vs Tiers ({len(compound_tiers)})")
        return [0.0] * num_items # Or handle as appropriate

    # Check lengths match (basic check)
    if len(processed_completions) != num_items:
        logger.error(f"[Combined TRL] Mismatch: Prompts ({num_items}) vs Completions ({len(processed_completions)})")
        return [0.0] * num_items

    # Sampled hash verification
    num_to_sample = max(1, int(num_items * VERIFY_HASH_SAMPLE_RATE))
    sample_indices = random.sample(range(num_items), num_to_sample)
    for i in sample_indices:
        if not verify_prompt_tier_hash(prompts[i], compound_tiers[i]):
            # Error logged within verify function. Decide action.
            logger.error(f"[Combined TRL] Hash verification failed for sample index {i}. Returning zeros for batch.")
            # Fail entire batch on mismatch during combined reward calculation? Seems reasonable.
            return [0.0] * num_items
    # --- End Verification ---

    # --- ADDED: Log first prompt and completion for debugging ---
    if prompts and processed_completions:
        # Only log if on main process to avoid clutter in distributed setups
        if os.getenv("RANK", "0") == "0":
            logger.info(f"[Combined TRL Debug] First prompt in batch: {prompts[0][:500]}...")
            logger.info(f"[Combined TRL Debug] First processed_completion in batch: {processed_completions[0][:500]}...")
    # --- END ADDED ---

    # Use the unified evaluation logic for consistent concurrent evaluation
    from src.reward_functions.unified_evaluation import evaluate_all_rewards_concurrently

    logger.info(f"combined_reward_trl: Using unified evaluation for {len(prompts)} items")
    ahimsa_results, dharma_results, helpfulness_results = run_async_safely(
        evaluate_all_rewards_concurrently(prompts, completions, **kwargs)
    )

    # Apply weights from config (default to equal)
    weights = REWARD_WEIGHTS

    # Check if severity penalty should be applied during training
    apply_severity_penalty_in_training = GRPO_CONFIG.get("apply_training_severity_penalty", False)

    # Combine scores with weights
    rewards = []
    # Store individual components for logging
    # REMOVED: Batch lists are no longer needed here, as averaging happens in callback
    # batch_ahimsa_scores = []
    # batch_dharma_scores = []
    # batch_helpfulness_scores = []
    # batch_penalties = []
    # batch_combined_rewards = []

    # CRITICAL DEBUG: Log reward function execution
    logger.info(f"🔍 DEBUG: combined_reward_trl called with {len(prompts)} prompts, {len(completions)} completions")
    logger.info(f"🔍 DEBUG: _audit_log_data id = {id(_audit_log_data)}, current length = {len(_audit_log_data)}")

    for i, (a_result, d_result, h_result) in enumerate(zip(ahimsa_results, dharma_results, helpfulness_results)):
        a_score = a_result.get("ahimsa_score", 0.0)
        d_score = d_result.get("dharma_score", 0.0)
        h_score = h_result.get("helpfulness_score", 0.0)

        if VERBOSE_LOGGING and evaluator == "gemini":
            logger.info(f"[combined_reward_trl DEBUG] Item {i}: Ahimsa result object: {json.dumps(a_result, indent=2)}")
            logger.info(f"[combined_reward_trl DEBUG] Item {i}: Extracted Ahimsa score: {a_score}")

        # Get scope penalty factor if available
        scope_penalty_factor = d_result.get("scope_penalty_factor", 1.0)

        # Apply scope penalty to all component scores if needed
        if scope_penalty_factor < 1.0:
            # Apply scope penalty to ahimsa and helpfulness scores
            a_score = a_score * scope_penalty_factor
            h_score = h_score * scope_penalty_factor

            # Log the scope penalty application
            logger.info(f"Applied scope penalty factor {scope_penalty_factor:.2f} to all component scores")

        # Determine severity penalty based on the *maximum* severity (current logic)
        severity_penalty = 0.0 # Default penalty to 0
        if apply_severity_penalty_in_training:
            ahimsa_severity = a_result.get("severity", "none")
            dharma_severity = d_result.get("severity", "none")

            severity_map = {"none": 0, "minor": 1, "major": 2}
            max_severity_level = max(severity_map.get(ahimsa_severity, 0), severity_map.get(dharma_severity, 0))

            if max_severity_level == 2: # major
                severity_penalty = -1.0
            elif max_severity_level == 1: # minor
                severity_penalty = -0.5

        # Apply weights and penalties (current logic)
        weighted_sum = (
            (a_score * weights["ahimsa"]) +
            (d_score * weights["dharma"]) +
            (h_score * weights["helpfulness"])
        )
        # Normalize by sum of weights only, then add severity penalty
        combined = (weighted_sum / sum(weights.values())) + severity_penalty

        if False:
            # Set to minimum reward if scope penalty is zero
            if scope_penalty_factor == 0.0:
                combined = -1.0

        # Optional: Clamp reward range if needed
        combined = max(-1.0, min(1.0, combined)) # Example clamp to [-1, 1]

        rewards.append(combined)
        logger.info(f"Combined reward: {combined}")

        # REMOVED: Batch lists are no longer needed
        # batch_ahimsa_scores.append(a_score)
        # ...

    # REMOVED: Logging is moved to the callback
    # if wandb.run: # Check if wandb is active
    #    ...
    #    _wandb_log_safe(log_data)

    # Populate audit log data using the extracted helper function
    populate_audit_log_data(
        prompts=prompts,
        processed_completions=processed_completions,
        compound_tiers=compound_tiers,
        ahimsa_results=ahimsa_results,
        dharma_results=dharma_results,
        helpfulness_results=helpfulness_results,
        **kwargs
    )

    if VERBOSE_LOGGING and evaluator == "gemini":
        logger.info(f"[combined_reward_trl DEBUG] Final combined rewards: {rewards}")

    # FINAL DEBUG: Log final state of _audit_log_data
    logger.info(f"🔍 DEBUG: combined_reward_trl completed. Final _audit_log_data length = {len(_audit_log_data)}")
    logger.info(f"🔍 DEBUG: Final _audit_log_data id = {id(_audit_log_data)}")
    if _audit_log_data:
        logger.info(f"🔍 DEBUG: Final _audit_log_data sample keys: {list(_audit_log_data[0].keys())}")

    return rewards
