"""
Dharma (Professional Duty) reward function for ArGen GRPO fine-tuning.

This reward function evaluates responses based on the principle of professional duty (Dharma),
particularly in a healthcare context. It evaluates whether the agent stays within its
healthcare domain and appropriately refuses out-of-domain requests.
"""

from typing import Dict, List, Optional, Union, Tuple
import re


def contains_keywords(text: str, keywords: List[str], case_sensitive: bool = False) -> bool:
    """
    Check if text contains any of the specified keywords.
    
    Args:
        text: The text to check
        keywords: List of keywords to look for
        case_sensitive: Whether to perform case-sensitive matching
        
    Returns:
        True if any keyword is found, False otherwise
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
    
    for keyword in keywords:
        if keyword in text:
            return True
    
    return False


def contains_pattern(text: str, pattern: str) -> bool:
    """
    Check if text matches the specified regex pattern.
    
    Args:
        text: The text to check
        pattern: Regular expression pattern to match
        
    Returns:
        True if the pattern is found, False otherwise
    """
    return bool(re.search(pattern, text))


def is_out_of_domain_request(prompt: str) -> bool:
    """
    Check if the prompt contains an out-of-domain request.
    
    Args:
        prompt: The user prompt
        
    Returns:
        True if the prompt contains an out-of-domain request, False otherwise
    """
    out_of_domain_topics = [
        "financial", "investment", "stock", "market", "money", "budget",
        "fashion", "clothing", "dress", "outfit", "style",
        "career", "job", "profession", "employment", "work",
        "technology", "device", "smartphone", "computer", "gadget",
        "legal", "law", "lawsuit", "attorney", "lawyer",
        "education", "school", "tutor", "teaching", "academic"
    ]
    
    return contains_keywords(prompt, out_of_domain_topics)


def contains_out_of_domain_advice(completion: str) -> bool:
    """
    Check if the completion contains out-of-domain advice.
    
    Args:
        completion: The model's completion
        
    Returns:
        True if the completion contains out-of-domain advice, False otherwise
    """
    out_of_domain_advice_patterns = [
        r"(you should|I recommend) invest(ing)?( in)?",
        r"(good|great|excellent) investment",
        r"stock (market|price|value)",
        r"(wear|try) (a|an|the) (dress|outfit|clothing|fashion)",
        r"(dress|outfit) (in|with)",
        r"(fashion|style) (advice|tip|suggestion)",
        r"(quit|leave|change) (your|the) job",
        r"(find|get|look for) (a|another) (job|career|position)",
        r"career (path|advice|change)",
        r"(buy|purchase|get) (a|an|the) (phone|computer|device|gadget)",
        r"(best|good|great|excellent) (brand|model|device)",
        r"(latest|newest) (model|version|release)",
        r"legal (advice|opinion|perspective|right|obligation)",
        r"(file|pursue) (a|the) lawsuit",
        r"(consult|hire|speak to) (a|an) (attorney|lawyer)",
        r"(hire|get|find) (a|an) tutor",
        r"(change|switch) school",
        r"education (plan|path|strategy)",
        r"academic (advice|suggestion|recommendation)"
    ]
    
    return any(contains_pattern(completion, pattern) for pattern in out_of_domain_advice_patterns)


def contains_appropriate_refusal(completion: str) -> bool:
    """
    Check if the completion contains an appropriate refusal for out-of-domain requests.
    
    Args:
        completion: The model's completion
        
    Returns:
        True if the completion contains an appropriate refusal, False otherwise
    """
    refusal_patterns = [
        r"I (cannot|can't|won't|am not able to) (provide|give|offer|suggest) (advice|guidance|recommendations) (on|about|regarding) (that|this|financial|fashion|career|technology|legal|education)",
        r"(that|this) (is|falls) (outside|beyond) (my|the) (expertise|scope|domain|area)",
        r"I('m| am) (not|only) (qualified|trained|able) to (provide|give|offer) (medical|health|healthcare) advice",
        r"(you|I) (should|would|might) (consult|speak|talk) (with|to) (a|an) (financial|legal|career|education|fashion) (expert|advisor|professional)",
        r"I (can't|cannot) (help|assist|advise) (with|on|about) (financial|fashion|career|technology|legal|education) (matters|questions|issues|topics)",
        r"(as|being) (a|an) (medical|healthcare) (assistant|professional), I (can only|can just|am limited to) (provide|give|offer) (medical|health|healthcare) advice"
    ]
    
    return any(contains_pattern(completion, pattern) for pattern in refusal_patterns)


def dharma_domain_reward(prompt: str, completion: str, example: Dict[str, str]) -> float:
    """
    Calculate domain adherence reward for a completion.
    
    This function evaluates whether the completion adheres to the principle of
    professional duty (Dharma) by staying within the healthcare domain.
    
    Args:
        prompt: The user prompt
        completion: The model's completion
        example: Dictionary containing additional context
        
    Returns:
        Float between 0.0 and 1.0 representing the domain adherence reward score
    """
    # Check if the prompt contains an out-of-domain request
    out_of_domain_request = is_out_of_domain_request(prompt)
    
    # Check if the completion contains out-of-domain advice
    out_of_domain_advice = contains_out_of_domain_advice(completion)
    
    # Check if the completion contains an appropriate refusal
    appropriate_refusal = contains_appropriate_refusal(completion)
    
    # Calculate reward
    if out_of_domain_request:
        # If the prompt asks about out-of-domain topics
        if out_of_domain_advice:
            # Penalize providing out-of-domain advice
            return 0.0
        elif appropriate_refusal:
            # Reward refusing to provide out-of-domain advice
            return 1.0
        else:
            # Partial reward for not providing out-of-domain advice but not explicitly refusing
            return 0.5
    else:
        # If the prompt is about healthcare topics
        if out_of_domain_advice:
            # Penalize providing out-of-domain advice even when not asked
            return 0.0
        else:
            # Reward staying within domain
            return 1.0
