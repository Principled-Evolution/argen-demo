"""
Unified Evaluation Logic for Reward Functions

This module contains the core concurrent evaluation logic extracted from combined_reward_trl,
allowing both combined and separate reward modes to use the same efficient evaluation engine.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Tuple, Union

from src.reward_functions.chat_response_helper import process_completions
from src.utils.data_integrity import extract_tier_from_compound
from src.config import GRPO_CONFIG

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

logger = logging.getLogger(__name__)

# Import evaluation functions
try:
    from src.reward_functions.openai_rewards import (
        evaluate_ahimsa_with_openai,
        evaluate_dharma_with_openai,
        evaluate_helpfulness_with_openai
    )
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI reward functions not available")
    OPENAI_AVAILABLE = False

try:
    from src.reward_functions.gemini.ahimsa import evaluate_ahimsa_with_gemini
    from src.reward_functions.gemini.dharma import evaluate_dharma_with_gemini
    from src.reward_functions.gemini.helpfulness import evaluate_helpfulness_with_gemini
    from src.reward_functions.gemini.ahimsa import batch_process_ahimsa_evaluations_concurrently
    from src.reward_functions.gemini.dharma import batch_process_dharma_evaluations_concurrently
    from src.reward_functions.gemini.helpfulness import batch_process_helpfulness_evaluations_concurrently
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Gemini reward functions not available")
    GEMINI_AVAILABLE = False

# Configuration
VERBOSE_LOGGING = os.environ.get("ARGEN_VERBOSE_LOGGING", "false").lower() == "true"

async def evaluate_all_rewards_concurrently(
    prompts: List[str],
    completions: List[Union[str, List[Dict]]],
    **kwargs
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform concurrent evaluation for all reward types (Ahimsa, Dharma, Helpfulness).

    This function contains the core evaluation logic extracted from combined_reward_trl,
    ensuring that both combined and separate reward modes use the same efficient
    concurrent evaluation patterns.

    Args:
        prompts: List of user prompts
        completions: List of model completions (can be strings or chat format)
        **kwargs: Additional arguments including tier, scope, etc.

    Returns:
        Tuple of (ahimsa_results, dharma_results, helpfulness_results)
        Each is a list of evaluation result dictionaries.
    """
    # Process completions to extract content from chat responses
    processed_completions = process_completions(completions)

    # Determine which evaluator to use
    evaluator = os.environ.get("ARGEN_EVALUATOR", "openai")

    if evaluator == "openai" and not OPENAI_AVAILABLE:
        logger.warning("OpenAI evaluator requested but not available, falling back to gemini")
        evaluator = "gemini"
    elif evaluator == "gemini" and not GEMINI_AVAILABLE:
        logger.warning("Gemini evaluator requested but not available, falling back to openai")
        evaluator = "openai"

    # Get API key for OpenAI if needed
    api_key = None
    if evaluator == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI evaluator")

    # Extract compound tiers and validate
    compound_tiers = kwargs.get("tier", [])
    num_items = len(prompts)

    if len(compound_tiers) != num_items:
        logger.error(f"[Unified Evaluation] Mismatch: Prompts ({num_items}) vs Tiers ({len(compound_tiers)})")
        # Return empty results with correct structure
        empty_results = [{"ahimsa_score": 0.0, "dharma_score": 0.0, "helpfulness_score": 0.0} for _ in range(num_items)]
        return empty_results, empty_results, empty_results

    if len(processed_completions) != num_items:
        logger.error(f"[Unified Evaluation] Mismatch: Prompts ({num_items}) vs Completions ({len(processed_completions)})")
        empty_results = [{"ahimsa_score": 0.0, "dharma_score": 0.0, "helpfulness_score": 0.0} for _ in range(num_items)]
        return empty_results, empty_results, empty_results

    # Create the main evaluation coroutine
    async def run_all_evaluations():
        # Extract actual tiers before creating tasks
        actual_tiers_for_eval = [extract_tier_from_compound(t) for t in compound_tiers]

        if evaluator == "openai":
            return await _evaluate_with_openai(
                prompts, processed_completions, actual_tiers_for_eval, api_key, **kwargs
            )
        else:  # evaluator == "gemini"
            return await _evaluate_with_gemini(
                prompts, processed_completions, actual_tiers_for_eval, **kwargs
            )

    # Run all evaluations concurrently
    logger.info(f"Unified Evaluation: Starting concurrent evaluation for {len(prompts)} items using {evaluator}")
    ahimsa_results, dharma_results, helpfulness_results = await run_all_evaluations()
    logger.info(f"Unified Evaluation: Completed concurrent evaluation for {len(prompts)} items")

    return ahimsa_results, dharma_results, helpfulness_results

async def _evaluate_with_openai(
    prompts: List[str],
    processed_completions: List[str],
    actual_tiers_for_eval: List[str],
    api_key: str,
    **kwargs
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Evaluate using OpenAI with concurrent API calls."""
    openai_tasks = []

    for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
        actual_tier = actual_tiers_for_eval[idx]
        current_prompt_meta = {"tier": actual_tier}
        compound_scope = kwargs.get("scope", [])[idx] if "scope" in kwargs and idx < len(kwargs["scope"]) else None
        if compound_scope:
            current_prompt_meta["scope"] = compound_scope

        ahimsa_task = evaluate_ahimsa_with_openai(
            prompt, completion, api_key, original_prompt_meta=current_prompt_meta
        )
        dharma_task = evaluate_dharma_with_openai(
            prompt, completion, api_key, original_prompt_meta=current_prompt_meta
        )
        helpfulness_task = evaluate_helpfulness_with_openai(prompt, completion, api_key)
        openai_tasks.append((ahimsa_task, dharma_task, helpfulness_task))

    # Flatten all tasks and run concurrently
    all_tasks = [task for task_group in openai_tasks for task in task_group]
    all_results = await asyncio.gather(*all_tasks)

    # Reshape results back into separate lists
    ahimsa_results = all_results[0::3]  # Every 3rd result starting from 0
    dharma_results = all_results[1::3]  # Every 3rd result starting from 1
    helpfulness_results = all_results[2::3]  # Every 3rd result starting from 2

    return ahimsa_results, dharma_results, helpfulness_results

async def _evaluate_with_gemini(
    prompts: List[str],
    processed_completions: List[str],
    actual_tiers_for_eval: List[str],
    **kwargs
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Evaluate using Gemini with concurrent API calls."""
    # Check if we should use single calls or batch calls for training
    use_single_calls = GRPO_CONFIG.get("use_single_gemini_calls_for_training", True)

    if use_single_calls:
        return await _evaluate_gemini_single_calls(
            prompts, processed_completions, actual_tiers_for_eval, **kwargs
        )
    else:
        return await _evaluate_gemini_batch_calls(
            prompts, processed_completions, actual_tiers_for_eval, **kwargs
        )

async def _evaluate_gemini_single_calls(
    prompts: List[str],
    processed_completions: List[str],
    actual_tiers_for_eval: List[str],
    **kwargs
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Evaluate using Gemini single-call approach with semaphore."""
    logger.info(f"Unified Evaluation: Using Gemini single-call approach for {len(prompts)} items")

    # Set up semaphore for concurrency control
    max_concurrent = GRPO_CONFIG.get("gemini_single_call_max_concurrent", 200)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_ahimsa_single_with_semaphore(prompt, completion, prompt_meta):
        async with semaphore:
            return await evaluate_ahimsa_with_gemini(
                prompt, completion, original_prompt_meta=prompt_meta
            )

    async def evaluate_dharma_single_with_semaphore(prompt, completion, prompt_meta):
        async with semaphore:
            return await evaluate_dharma_with_gemini(
                prompt, completion, original_prompt_meta=prompt_meta
            )

    async def evaluate_helpfulness_single_with_semaphore(prompt, completion):
        async with semaphore:
            return await evaluate_helpfulness_with_gemini(prompt, completion)

    # Create tasks for all prompt+completion pairs
    ahimsa_tasks = []
    dharma_tasks = []
    helpfulness_tasks = []

    for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
        actual_tier = actual_tiers_for_eval[idx]
        current_prompt_meta = {"tier": actual_tier}
        compound_scope = kwargs.get("scope", [])[idx] if "scope" in kwargs and idx < len(kwargs["scope"]) else None
        if compound_scope:
            current_prompt_meta["scope"] = compound_scope

        ahimsa_tasks.append(evaluate_ahimsa_single_with_semaphore(prompt, completion, current_prompt_meta))
        dharma_tasks.append(evaluate_dharma_single_with_semaphore(prompt, completion, current_prompt_meta))
        helpfulness_tasks.append(evaluate_helpfulness_single_with_semaphore(prompt, completion))

    # Run all tasks concurrently
    ahimsa_results, dharma_results, helpfulness_results = await asyncio.gather(
        asyncio.gather(*ahimsa_tasks),
        asyncio.gather(*dharma_tasks),
        asyncio.gather(*helpfulness_tasks)
    )

    if VERBOSE_LOGGING:
        logger.info(f"[Unified Evaluation DEBUG] Results from single-call approach - Ahimsa: {json.dumps(ahimsa_results, indent=2)}")
        logger.info(f"[Unified Evaluation DEBUG] Results from single-call approach - Dharma: {json.dumps(dharma_results, indent=2)}")
        logger.info(f"[Unified Evaluation DEBUG] Results from single-call approach - Helpfulness: {json.dumps(helpfulness_results, indent=2)}")

    return ahimsa_results, dharma_results, helpfulness_results

async def _evaluate_gemini_batch_calls(
    prompts: List[str],
    processed_completions: List[str],
    actual_tiers_for_eval: List[str],
    **kwargs
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Evaluate using Gemini batch-call approach."""
    logger.info(f"Unified Evaluation: Using Gemini batch-call approach for {len(prompts)} items")

    # Prepare items for each reward type
    all_ahimsa_items_for_gemini = []
    all_dharma_items_for_gemini = []
    all_helpfulness_items_for_gemini = []

    for idx, (prompt, completion) in enumerate(zip(prompts, processed_completions)):
        actual_tier = actual_tiers_for_eval[idx]
        current_prompt_meta = {"tier": actual_tier}
        compound_scope = kwargs.get("scope", [])[idx] if "scope" in kwargs and idx < len(kwargs["scope"]) else None
        if compound_scope:
            current_prompt_meta["scope"] = compound_scope

        # Ahimsa items
        all_ahimsa_items_for_gemini.append({
            "prompt": prompt,
            "model_response": completion,
            "original_prompt_meta": current_prompt_meta
        })

        # Dharma items
        all_dharma_items_for_gemini.append({
            "prompt": prompt,
            "model_response": completion,
            "original_prompt_meta": current_prompt_meta
        })

        # Helpfulness items
        all_helpfulness_items_for_gemini.append({
            "prompt": prompt,
            "model_response": completion
        })

    # Create concurrent batch tasks
    concurrent_ahimsa_task = batch_process_ahimsa_evaluations_concurrently(all_ahimsa_items_for_gemini)
    concurrent_dharma_task = batch_process_dharma_evaluations_concurrently(all_dharma_items_for_gemini)
    concurrent_helpfulness_task = batch_process_helpfulness_evaluations_concurrently(all_helpfulness_items_for_gemini)

    # Gather all Gemini tasks concurrently
    gathered_results = await asyncio.gather(
        concurrent_ahimsa_task,
        concurrent_dharma_task,
        concurrent_helpfulness_task,
        return_exceptions=True
    )

    ahimsa_results = gathered_results[0]
    dharma_results = gathered_results[1]
    helpfulness_results = gathered_results[2]

    if VERBOSE_LOGGING:
        logger.info(f"[Unified Evaluation DEBUG] Ahimsa items for Gemini: {json.dumps(all_ahimsa_items_for_gemini, indent=2)}")
        logger.info(f"[Unified Evaluation DEBUG] Ahimsa results from concurrent batch: {json.dumps(ahimsa_results, indent=2)}")
        logger.info(f"[Unified Evaluation DEBUG] Dharma items for Gemini: {json.dumps(all_dharma_items_for_gemini, indent=2)}")
        logger.info(f"[Unified Evaluation DEBUG] Dharma results from concurrent batch: {json.dumps(dharma_results, indent=2)}")
        logger.info(f"[Unified Evaluation DEBUG] Helpfulness items for Gemini: {json.dumps(all_helpfulness_items_for_gemini, indent=2)}")
        logger.info(f"[Unified Evaluation DEBUG] Helpfulness results from concurrent batch: {json.dumps(helpfulness_results, indent=2)}")

    return ahimsa_results, dharma_results, helpfulness_results