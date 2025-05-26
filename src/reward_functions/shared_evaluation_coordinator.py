"""
Shared Evaluation Coordinator for Concurrent Reward Function Evaluation

This module provides a coordination mechanism to ensure that when using separate
reward functions (ahimsa, dharma, helpfulness), all evaluations are performed
concurrently rather than sequentially, achieving the same performance as the
combined reward function.

The coordinator uses a batch identification system to detect when multiple reward
functions are called with the same prompts/completions, and triggers a single
concurrent evaluation for all reward types.
"""

import asyncio
import hashlib
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import threading

logger = logging.getLogger(__name__)

class SharedEvaluationCoordinator:
    """
    Coordinates concurrent evaluation of all reward functions when using separate rewards mode.

    This class ensures that when multiple reward functions are called with the same
    prompts/completions (which happens in separate rewards mode), only one concurrent
    evaluation is performed for all reward types, with results cached for subsequent calls.
    """

    def __init__(self):
        self._current_batch_id: Optional[str] = None
        self._evaluation_results: Dict[str, Dict[str, List[Dict]]] = {}
        self._evaluation_lock = asyncio.Lock()
        self._evaluation_future: Optional[asyncio.Future] = None
        self._batch_call_count: Dict[str, int] = {}
        self._expected_calls_per_batch = 3  # ahimsa, dharma, helpfulness

    def _generate_batch_id(self, prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> str:
        """
        Generate a unique batch ID based on prompts, completions, and relevant kwargs.

        This ensures that the same batch of data gets the same ID regardless of
        which reward function calls it first.
        """
        # Convert completions to strings if they're chat format
        completion_strs = []
        for comp in completions:
            if isinstance(comp, list):
                # Extract content from chat format
                completion_strs.append(str(comp))
            else:
                completion_strs.append(str(comp))

        # Create hash from prompts and completions
        batch_data = {
            "prompts": prompts,
            "completions": completion_strs,
            # Include relevant kwargs that affect evaluation
            "tier": kwargs.get("tier", []),
            "scope": kwargs.get("scope", [])
        }

        batch_str = json.dumps(batch_data, sort_keys=True)
        batch_hash = hashlib.sha256(batch_str.encode()).hexdigest()[:16]
        return f"batch_{batch_hash}"

    def get_or_evaluate_batch_sync(
        self,
        reward_type: str,
        prompts: List[str],
        completions: List[Union[str, List[Dict]]],
        **kwargs
    ) -> List[Dict]:
        """
        Synchronous wrapper for get_or_evaluate_batch that handles event loop properly.

        Args:
            reward_type: One of "ahimsa", "dharma", "helpfulness"
            prompts: List of user prompts
            completions: List of model completions
            **kwargs: Additional arguments (tier, scope, etc.)

        Returns:
            List of evaluation result dictionaries for the specified reward type
        """
        batch_id = self._generate_batch_id(prompts, completions, **kwargs)

        # Check if we already have results for this batch and reward type
        if batch_id in self._evaluation_results and reward_type in self._evaluation_results[batch_id]:
            logger.info(f"SharedEvaluationCoordinator: Returning cached results for {reward_type} (batch {batch_id})")

            # CRITICAL FIX: Always populate audit data on EVERY call, regardless of caching
            # This ensures timing independence and consistency (KISS principle)
            # Use a flag to ensure we only populate once per batch to avoid overwriting
            if not hasattr(self, '_audit_populated_batches'):
                self._audit_populated_batches = set()

            if batch_id not in self._audit_populated_batches:
                self._populate_audit_data_if_needed(batch_id, prompts, completions, **kwargs)
                self._audit_populated_batches.add(batch_id)

            return self._evaluation_results[batch_id][reward_type]

        # If this is the first call for this batch, trigger evaluation
        if batch_id not in self._evaluation_results:
            logger.info(f"SharedEvaluationCoordinator: Starting concurrent evaluation for batch {batch_id}")

            # Initialize results storage for this batch
            self._evaluation_results[batch_id] = {}

            # Track how many times this batch has been called
            if batch_id not in self._batch_call_count:
                self._batch_call_count[batch_id] = 0

            # Perform the evaluation synchronously
            try:
                self._evaluate_all_rewards_sync(batch_id, prompts, completions, **kwargs)
            except Exception as e:
                logger.error(f"SharedEvaluationCoordinator: Evaluation failed for batch {batch_id}: {e}")
                # Clean up failed batch
                if batch_id in self._evaluation_results:
                    del self._evaluation_results[batch_id]
                if batch_id in self._batch_call_count:
                    del self._batch_call_count[batch_id]
                raise

        # Increment call count
        self._batch_call_count[batch_id] += 1

        # Return the results for the requested reward type
        if batch_id in self._evaluation_results and reward_type in self._evaluation_results[batch_id]:
            logger.info(f"SharedEvaluationCoordinator: Evaluation completed, returning {reward_type} results")
            return self._evaluation_results[batch_id][reward_type]
        else:
            logger.error(f"SharedEvaluationCoordinator: No results found for {reward_type} after evaluation")
            raise RuntimeError(f"Evaluation failed to produce results for {reward_type}")

    def _evaluate_all_rewards_sync(
        self,
        batch_id: str,
        prompts: List[str],
        completions: List[Union[str, List[Dict]]],
        **kwargs
    ) -> None:
        """
        Perform concurrent evaluation for all reward types synchronously.

        This method contains the core evaluation logic extracted from combined_reward_trl,
        but runs it in a way that's compatible with existing event loops.
        """
        try:
            # Import the unified evaluation function
            from src.reward_functions.unified_evaluation import evaluate_all_rewards_concurrently

            logger.info(f"SharedEvaluationCoordinator: Running concurrent evaluation for {len(prompts)} items")

            # Check if we're in an async context
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # If we get here, we're in an async context, so we need to run in a thread
                import concurrent.futures
                import threading

                def run_evaluation():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            evaluate_all_rewards_concurrently(prompts, completions, **kwargs)
                        )
                    finally:
                        new_loop.close()
                        # Clean up the event loop reference
                        asyncio.set_event_loop(None)

                # Run the evaluation in a separate thread with its own event loop
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_evaluation)
                    ahimsa_results, dharma_results, helpfulness_results = future.result()

            except RuntimeError as e:
                # Check if it's specifically "no running event loop" error
                if "no running event loop" in str(e):
                    # No event loop running, we can run directly
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        ahimsa_results, dharma_results, helpfulness_results = loop.run_until_complete(
                            evaluate_all_rewards_concurrently(prompts, completions, **kwargs)
                        )
                    finally:
                        loop.close()
                        # Clean up the event loop reference
                        asyncio.set_event_loop(None)
                else:
                    # Some other RuntimeError, re-raise it
                    raise

            # Store results
            self._evaluation_results[batch_id] = {
                "ahimsa": ahimsa_results,
                "dharma": dharma_results,
                "helpfulness": helpfulness_results
            }

            # Populate audit data after successful evaluation
            self._populate_audit_data_if_needed(batch_id, prompts, completions, **kwargs)

            # Mark this batch as having audit data populated
            if not hasattr(self, '_audit_populated_batches'):
                self._audit_populated_batches = set()
            self._audit_populated_batches.add(batch_id)

            logger.info(f"SharedEvaluationCoordinator: Concurrent evaluation completed for batch {batch_id}")

        except Exception as e:
            logger.error(f"SharedEvaluationCoordinator: Error in concurrent evaluation: {e}")
            raise

    def _populate_audit_data_if_needed(
        self,
        batch_id: str,
        prompts: List[str],
        completions: List[Union[str, List[Dict]]],
        **kwargs
    ) -> None:
        """
        Populate audit log data for callback logging.

        This method ensures that audit data is populated on EVERY reward function call,
        regardless of caching, ensuring timing independence (KISS principle).
        """
        try:
            # Only populate if we have evaluation results for this batch
            if batch_id not in self._evaluation_results:
                return

            from src.reward_functions.trl_rewards import populate_audit_log_data
            from src.reward_functions.chat_response_helper import process_completions

            # Get the cached results
            results = self._evaluation_results[batch_id]
            ahimsa_results = results["ahimsa"]
            dharma_results = results["dharma"]
            helpfulness_results = results["helpfulness"]

            # Process completions to match the format expected by populate_audit_log_data
            processed_completions = process_completions(completions)

            # Extract compound tiers from kwargs
            compound_tiers = kwargs.get("tier", [])
            if not compound_tiers:
                # Fallback to empty strings if no tiers provided
                compound_tiers = [""] * len(prompts)

            # Ensure we have the right number of tiers
            if len(compound_tiers) < len(prompts):
                compound_tiers.extend([""] * (len(prompts) - len(compound_tiers)))

            logger.info(f"SharedEvaluationCoordinator: Populating audit log data for {len(prompts)} items (batch {batch_id})")
            populate_audit_log_data(
                prompts=prompts,
                processed_completions=processed_completions,
                compound_tiers=compound_tiers,
                ahimsa_results=ahimsa_results,
                dharma_results=dharma_results,
                helpfulness_results=helpfulness_results,
                **kwargs
            )
            logger.info("SharedEvaluationCoordinator: Successfully populated audit log data")

        except Exception as e:
            logger.error(f"SharedEvaluationCoordinator: Failed to populate audit log data: {e}")
            # Don't fail the entire evaluation if audit data population fails
            pass

    def cleanup_batch(self, batch_id: str) -> None:
        """
        Clean up cached results for a batch after all reward functions have been called.
        """
        if batch_id in self._evaluation_results:
            del self._evaluation_results[batch_id]
        if batch_id in self._batch_call_count:
            del self._batch_call_count[batch_id]
        logger.debug(f"SharedEvaluationCoordinator: Cleaned up batch {batch_id}")

    def should_cleanup_batch(self, batch_id: str) -> bool:
        """
        Check if a batch should be cleaned up (all expected calls have been made).
        """
        return (batch_id in self._batch_call_count and
                self._batch_call_count[batch_id] >= self._expected_calls_per_batch)


# Global coordinator instance
_global_coordinator: Optional[SharedEvaluationCoordinator] = None
_coordinator_lock = threading.Lock()

def get_shared_coordinator() -> SharedEvaluationCoordinator:
    """
    Get the global shared evaluation coordinator instance.

    Uses thread-safe singleton pattern to ensure only one coordinator exists.
    """
    global _global_coordinator

    if _global_coordinator is None:
        with _coordinator_lock:
            if _global_coordinator is None:
                _global_coordinator = SharedEvaluationCoordinator()
                logger.info("SharedEvaluationCoordinator: Created global coordinator instance")

    return _global_coordinator

def is_separate_rewards_mode() -> bool:
    """
    Check if we're currently in separate rewards mode.

    This is determined by an environment variable set by the training script.
    """
    return os.environ.get("ARGEN_USE_SEPARATE_REWARDS", "false").lower() == "true"

def set_separate_rewards_mode(enabled: bool) -> None:
    """
    Set the separate rewards mode flag.

    This should be called by the training script when using separate rewards.
    """
    os.environ["ARGEN_USE_SEPARATE_REWARDS"] = "true" if enabled else "false"
    logger.info(f"SharedEvaluationCoordinator: Separate rewards mode {'enabled' if enabled else 'disabled'}")
