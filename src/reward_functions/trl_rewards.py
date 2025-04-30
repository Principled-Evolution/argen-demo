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
    
    # Use asyncio to run the evaluation for each prompt/completion pair
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        rewards = []
        for prompt, completion in zip(prompts, processed_completions):
            # Run the async function in the event loop
            result = loop.run_until_complete(
                evaluate_ahimsa_with_openai(prompt, completion, openai_api_key)
            )
            rewards.append(result.get("ahimsa_score", 0.0))
        return rewards
    finally:
        loop.close()

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
    
    # Use asyncio to run the evaluation for each prompt/completion pair
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        rewards = []
        for prompt, completion in zip(prompts, processed_completions):
            # Run the async function in the event loop
            result = loop.run_until_complete(
                evaluate_dharma_with_openai(prompt, completion, openai_api_key)
            )
            rewards.append(result.get("dharma_score", 0.0))
        return rewards
    finally:
        loop.close()

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
    # Get both reward components
    ahimsa_scores = ahimsa_reward_trl(prompts, completions, **kwargs)
    dharma_scores = dharma_reward_trl(prompts, completions, **kwargs)
    
    # Apply weights from config (default to equal)
    from src.config import REWARD_WEIGHTS
    weights = REWARD_WEIGHTS
    
    # Combine scores with weights
    rewards = []
    for a_score, d_score in zip(ahimsa_scores, dharma_scores):
        combined = (
            a_score * weights["ahimsa"] + 
            d_score * weights["dharma"]
        ) / sum(weights.values())
        rewards.append(combined)
    
    return rewards 