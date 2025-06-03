"""
Utility functions for reward calculation in ArGen GRPO fine-tuning.
"""

import re
from typing import Dict, List, Optional, Union, Tuple


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


def count_keywords(text: str, keywords: List[str], case_sensitive: bool = False) -> int:
    """
    Count occurrences of keywords in text.
    
    Args:
        text: The text to check
        keywords: List of keywords to count
        case_sensitive: Whether to perform case-sensitive matching
        
    Returns:
        Total count of all keywords found
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
    
    count = 0
    for keyword in keywords:
        count += text.count(keyword)
    
    return count


def has_disclaimer(
    text: str, 
    disclaimer_patterns: Optional[List[str]] = None
) -> bool:
    """
    Check if text contains a medical disclaimer.
    
    Args:
        text: The text to check
        disclaimer_patterns: List of regex patterns for disclaimers
        
    Returns:
        True if a disclaimer is found, False otherwise
    """
    if disclaimer_patterns is None:
        disclaimer_patterns = [
            r"I('m| am) not a (doctor|physician|healthcare professional)",
            r"This (is not|isn't) medical advice",
            r"consult (with )?(your|a) (doctor|physician|healthcare professional)",
            r"seek (professional|medical) (help|advice|care|attention)",
            r"This information (should not|shouldn't) replace professional medical advice",
        ]
    
    for pattern in disclaimer_patterns:
        if contains_pattern(text, pattern):
            return True
    
    return False


def has_citation(text: str) -> bool:
    """
    Check if text contains citations or references to medical sources.
    
    Args:
        text: The text to check
        
    Returns:
        True if citations are found, False otherwise
    """
    citation_patterns = [
        r"according to [^.]*",
        r"(studies|research) (show|suggest|indicate)",
        r"(published|reported) in [^.]*",
        r"(cited|referenced) (by|in) [^.]*",
        r"(source|reference):",
    ]
    
    for pattern in citation_patterns:
        if contains_pattern(text, pattern):
            return True
    
    return False


def expresses_uncertainty(text: str) -> bool:
    """
    Check if text appropriately expresses uncertainty when needed.
    
    Args:
        text: The text to check
        
    Returns:
        True if uncertainty expressions are found, False otherwise
    """
    uncertainty_patterns = [
        r"(may|might|could) (be|cause|help|indicate)",
        r"it('s| is) possible that",
        r"(some|many) people",
        r"(not|isn't|aren't) (always|necessarily)",
        r"(varies|depend|different) (from person to person|between individuals)",
        r"(limited|insufficient) (evidence|research|data|studies)",
    ]
    
    for pattern in uncertainty_patterns:
        if contains_pattern(text, pattern):
            return True
    
    return False


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to be within the specified range.
    
    Args:
        score: The score to normalize
        min_val: Minimum value of the normalized range
        max_val: Maximum value of the normalized range
        
    Returns:
        Normalized score
    """
    return min(max(score, min_val), max_val)


def detect_harmful_advice(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect potentially harmful medical advice.
    
    Args:
        text: The text to check
        
    Returns:
        Tuple of (is_harmful, reason)
    """
    harmful_patterns = [
        (r"stop taking (your|the) (medication|medicine|prescription)", 
         "Advising to stop prescribed medication"),
        (r"(instead of|alternative to|rather than) (seeing|consulting) (a|your) doctor", 
         "Discouraging professional medical care"),
        (r"(cure|treat|heal) (cancer|diabetes|heart disease|depression)", 
         "Claiming to cure serious conditions"),
        (r"(no need|don't need|unnecessary) to (see|consult|visit) (a|your) doctor", 
         "Suggesting professional care is unnecessary"),
        (r"(guaranteed|100%|completely) (cure|treatment|remedy)", 
         "Making absolute guarantees about treatments"),
    ]
    
    for pattern, reason in harmful_patterns:
        if contains_pattern(text, pattern):
            return True, reason
    
    return False, None


def detect_compassionate_language(text: str) -> bool:
    """
    Detect compassionate and empathetic language.
    
    Args:
        text: The text to check
        
    Returns:
        True if compassionate language is found, False otherwise
    """
    compassion_patterns = [
        r"I understand (that|how|your)",
        r"(it|that) must be (difficult|challenging|hard|tough)",
        r"I('m| am) sorry (to hear|that) you('re| are)",
        r"(your|this) (concern|worry|pain|discomfort|situation) is (valid|understandable|common)",
        r"(take care of|prioritize) (yourself|your health|your wellbeing)",
    ]
    
    compassion_count = 0
    for pattern in compassion_patterns:
        if contains_pattern(text, pattern):
            compassion_count += 1
    
    # Consider it compassionate if at least one pattern is found
    return compassion_count > 0
