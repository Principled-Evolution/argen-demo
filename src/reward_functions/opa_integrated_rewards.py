"""
OPA-integrated reward functions for ArGen GRPO fine-tuning.

This module provides reward functions that use OPA policies to evaluate responses
based on Dharmic principles.
"""

import os
import json
import subprocess
from typing import Dict, Any, Optional, List, Union, Callable

# Import OPA evaluator
from src.opa.evaluator import evaluate_with_opa


def opa_ahimsa_reward(prompt: str, completion: str, example: Dict[str, str]) -> float:
    """
    Calculate Ahimsa (non-maleficence) reward using OPA policy.
    
    This function evaluates whether the completion adheres to the principle of
    non-maleficence (Ahimsa) in a healthcare context using OPA policy.
    
    Args:
        prompt: The user prompt
        completion: The model's completion
        example: Dictionary containing additional context
        
    Returns:
        Float between 0.0 and 1.0 representing the Ahimsa reward score
    """
    try:
        # Evaluate with OPA
        result = evaluate_with_opa(prompt, completion, policy_name="ahimsa")
        
        # Extract score from result
        if "score" in result:
            return float(result["score"])
        elif "error" in result:
            print(f"Error in OPA evaluation: {result['error']}")
            return 0.0
        else:
            print(f"Unexpected OPA result format: {result}")
            return 0.0
    
    except Exception as e:
        print(f"Error in opa_ahimsa_reward: {e}")
        return 0.0


def opa_dharma_reward(prompt: str, completion: str, example: Dict[str, str]) -> float:
    """
    Calculate domain adherence reward using OPA policy.
    
    This function evaluates whether the completion adheres to the principle of
    professional duty (Dharma) by staying within the healthcare domain using OPA policy.
    
    Args:
        prompt: The user prompt
        completion: The model's completion
        example: Dictionary containing additional context
        
    Returns:
        Float between 0.0 and 1.0 representing the domain adherence reward score
    """
    try:
        # Evaluate with OPA
        result = evaluate_with_opa(prompt, completion, policy_name="dharma")
        
        # Extract score from result
        if "score" in result:
            return float(result["score"])
        elif "error" in result:
            print(f"Error in OPA evaluation: {result['error']}")
            return 0.0
        else:
            print(f"Unexpected OPA result format: {result}")
            return 0.0
    
    except Exception as e:
        print(f"Error in opa_dharma_reward: {e}")
        return 0.0


def combined_ahimsa_reward(prompt: str, completion: str, example: Dict[str, str]) -> float:
    """
    Calculate Ahimsa reward using both Gemini and OPA.
    
    This function combines evaluations from Gemini and OPA for a more robust assessment.
    
    Args:
        prompt: The user prompt
        completion: The model's completion
        example: Dictionary containing additional context
        
    Returns:
        Float between 0.0 and 1.0 representing the combined Ahimsa reward score
    """
    try:
        # Import Gemini reward function
        from src.reward_functions.gemini_rewards import gemini_ahimsa_reward
        
        # Get scores from both evaluators
        gemini_score = gemini_ahimsa_reward(prompt, completion, example)
        opa_score = opa_ahimsa_reward(prompt, completion, example)
        
        # If either score is very low (policy violation), return the lower score
        if gemini_score < 0.2 or opa_score < 0.2:
            return min(gemini_score, opa_score)
        
        # Otherwise, return the average
        return (gemini_score + opa_score) / 2.0
    
    except Exception as e:
        print(f"Error in combined_ahimsa_reward: {e}")
        return 0.0


def combined_dharma_reward(prompt: str, completion: str, example: Dict[str, str]) -> float:
    """
    Calculate Dharma reward using both Gemini and OPA.
    
    This function combines evaluations from Gemini and OPA for a more robust assessment.
    
    Args:
        prompt: The user prompt
        completion: The model's completion
        example: Dictionary containing additional context
        
    Returns:
        Float between 0.0 and 1.0 representing the combined Dharma reward score
    """
    try:
        # Import Gemini reward function
        from src.reward_functions.gemini_rewards import gemini_dharma_reward
        
        # Get scores from both evaluators
        gemini_score = gemini_dharma_reward(prompt, completion, example)
        opa_score = opa_dharma_reward(prompt, completion, example)
        
        # If either score is very low (policy violation), return the lower score
        if gemini_score < 0.2 or opa_score < 0.2:
            return min(gemini_score, opa_score)
        
        # Otherwise, return the average
        return (gemini_score + opa_score) / 2.0
    
    except Exception as e:
        print(f"Error in combined_dharma_reward: {e}")
        return 0.0
