"""
TRL-compatible reward functions for ArGen GRPO training.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Union, Any

from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Default error response
DEFAULT_EVAL_RESPONSE = {
    "error": "Evaluation failed",
    "reasoning": "Could not get response from OpenAI API.",
    "ahimsa_score": 0.0,
    "harm_avoidance_score": 0.0,
    "safety_consideration_score": 0.0,
    "has_medical_disclaimer": False,
    "has_professional_referral": False,
    "ahimsa_violation": True,  # Default to violation on error
    "dharma_score": 0.0,
    "domain_adherence_score": 0.0,
    "out_of_domain_advice": True,  # Default to violation on error
    "dharma_violation": True
}

# Import the async evaluation functions from openai_rewards
from src.reward_functions.openai_rewards import (
    evaluate_ahimsa_with_openai,
    evaluate_dharma_with_openai
)

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

def ahimsa_reward_trl(prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> List[float]:
    """
    TRL-compatible reward function for Ahimsa evaluation.
    
    Args:
        prompts: List of user prompts 
        completions: List of model completions
        **kwargs: Additional columns from the dataset
        
    Returns:
        List of Ahimsa reward scores
    """
    # For TRL, we need to handle both string and conversational completions
    processed_completions = []
    for completion in completions:
        if isinstance(completion, list):  # Conversational format
            processed_completions.append(completion[0]["content"])
        else:  # String format
            processed_completions.append(completion)
            
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found in environment")
        return [0.0] * len(prompts)  # Return zeros on error
    
    # Create a list of coroutines to run concurrently
    async def run_all_evaluations():
        tasks = []
        for prompt, completion in zip(prompts, processed_completions):
            tasks.append(evaluate_ahimsa_with_openai(prompt, completion, openai_api_key))
        return await asyncio.gather(*tasks)
    
    # Run all evaluations in one go
    results = run_async_safely(run_all_evaluations())
    
    # Extract scores from results
    rewards = [result.get("ahimsa_score", 0.0) for result in results]
    return rewards

def dharma_reward_trl(prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> List[float]:
    """
    TRL-compatible reward function for Dharma evaluation.
    
    Args:
        prompts: List of user prompts 
        completions: List of model completions
        **kwargs: Additional columns from the dataset
        
    Returns:
        List of Dharma reward scores
    """
    # For TRL, we need to handle both string and conversational completions
    processed_completions = []
    for completion in completions:
        if isinstance(completion, list):  # Conversational format
            processed_completions.append(completion[0]["content"])
        else:  # String format
            processed_completions.append(completion)
            
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found in environment")
        return [0.0] * len(prompts)  # Return zeros on error
    
    # Create a list of coroutines to run concurrently
    async def run_all_evaluations():
        tasks = []
        for prompt, completion in zip(prompts, processed_completions):
            tasks.append(evaluate_dharma_with_openai(prompt, completion, openai_api_key))
        return await asyncio.gather(*tasks)
    
    # Run all evaluations in one go
    results = run_async_safely(run_all_evaluations())
    
    # Extract scores from results
    rewards = [result.get("dharma_score", 0.0) for result in results]
    return rewards

def combined_reward_trl(prompts: List[str], completions: List[Union[str, List[Dict]]], **kwargs) -> List[float]:
    """
    TRL-compatible reward function combining Ahimsa and Dharma scores.
    
    Args:
        prompts: List of user prompts 
        completions: List of model completions
        **kwargs: Additional columns from the dataset
        
    Returns:
        List of combined reward scores
    """
    # Process completions only once
    processed_completions = []
    for completion in completions:
        if isinstance(completion, list):  # Conversational format
            processed_completions.append(completion[0]["content"])
        else:  # String format
            processed_completions.append(completion)
            
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found in environment")
        return [0.0] * len(prompts)  # Return zeros on error
    
    # Create a list of coroutines to run concurrently
    async def run_all_evaluations():
        all_tasks = []
        for prompt, completion in zip(prompts, processed_completions):
            # Create tasks for both ahimsa and dharma evaluations
            ahimsa_task = evaluate_ahimsa_with_openai(prompt, completion, openai_api_key)
            dharma_task = evaluate_dharma_with_openai(prompt, completion, openai_api_key)
            all_tasks.append((ahimsa_task, dharma_task))
        
        # Unpack and gather all tasks
        ahimsa_tasks, dharma_tasks = zip(*all_tasks)
        ahimsa_results = await asyncio.gather(*ahimsa_tasks)
        dharma_results = await asyncio.gather(*dharma_tasks)
        
        return ahimsa_results, dharma_results
    
    # Run all evaluations in one go
    ahimsa_results, dharma_results = run_async_safely(run_all_evaluations())
    
    # Apply weights from config (default to equal)
    from src.config import REWARD_WEIGHTS
    weights = REWARD_WEIGHTS
    
    # Combine scores with weights
    rewards = []
    for a_result, d_result in zip(ahimsa_results, dharma_results):
        a_score = a_result.get("ahimsa_score", 0.0)
        d_score = d_result.get("dharma_score", 0.0)
        
        combined = (
            a_score * weights["ahimsa"] + 
            d_score * weights["dharma"]
        ) / sum(weights.values())
        rewards.append(combined)
    
    return rewards 