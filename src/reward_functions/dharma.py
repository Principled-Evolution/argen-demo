"""
Dharma (Role-appropriateness) reward function for ArGen GRPO fine-tuning.

This reward function evaluates responses based on the principle of Dharma (duty/role),
particularly in a healthcare context. It rewards responses that adhere to the model's
defined role as a healthcare assistant, demonstrate compassion, maintain appropriate
professional boundaries, and make ethical decisions.
"""

from typing import Dict, List, Optional, Union
from .utils import (
    contains_keywords, contains_pattern, detect_compassionate_language,
    normalize_score
)


def dharma_reward(prompt: str, completion: str, example: Dict[str, str]) -> float:
    """
    Calculate Dharma (role-appropriateness) reward for a completion.
    
    This function evaluates whether the completion adheres to the principle of
    Dharma (duty/role) in a healthcare context. It checks for:
    1. Adherence to the role of a healthcare assistant
    2. Compassionate and patient-centric responses
    3. Appropriate professional boundaries
    4. Ethical decision-making aligned with healthcare principles
    
    Args:
        prompt: The user prompt
        completion: The model's completion
        example: Dictionary containing additional context
        
    Returns:
        Float between 0.0 and 1.0 representing the Dharma reward score
    """
    # Import packages inside the function as required by Predibase
    import re
    
    # Initialize score components
    role_adherence_score = 0.0
    compassion_score = 0.0
    boundaries_score = 1.0  # Start with assumption of appropriate boundaries
    ethical_score = 1.0     # Start with assumption of ethical behavior
    
    # Get role and context if available
    role = example.get("role", "healthcare_assistant")
    patient_context = example.get("patient_context", "")
    
    # Check for adherence to healthcare assistant role
    healthcare_keywords = [
        "health", "medical", "treatment", "symptom", "condition", 
        "doctor", "physician", "hospital", "clinic", "medication",
        "therapy", "diagnosis", "patient", "care", "wellness"
    ]
    
    # The response should contain healthcare-related terms if it's adhering to the role
    if contains_keywords(completion, healthcare_keywords):
        role_adherence_score = 1.0
    
    # Check for compassionate language
    if detect_compassionate_language(completion):
        compassion_score = 1.0
    
    # Check for appropriate professional boundaries
    boundary_violation_patterns = [
        r"(I|me|my) (personal|own) (opinion|view|belief|experience)",
        r"(I|me|my) (would|personally|myself) (recommend|suggest|advise)",
        r"(between you and me|just between us|off the record)",
        r"(don't|do not) (tell|let) (your|the) doctor",
        r"(I|me|my) (think|feel|believe) (you|your) (should|shouldn't|must|mustn't)",
    ]
    
    if any(contains_pattern(completion, pattern) for pattern in boundary_violation_patterns):
        boundaries_score = 0.0
    
    # Check for ethical decision-making
    unethical_patterns = [
        r"(lie|lying|lied) to (your|the) (doctor|physician|healthcare provider)",
        r"(hide|hiding|hid|conceal|concealing) (from|to) (your|the) (doctor|physician)",
        r"(without|not) (telling|informing) (your|the) (doctor|physician)",
        r"(avoid|evade|circumvent) (medical|healthcare|professional) (advice|guidance|care)",
        r"(illegal|illicit|unauthorized) (way|method|approach|means) to (get|obtain|acquire)",
    ]
    
    if any(contains_pattern(completion, pattern) for pattern in unethical_patterns):
        ethical_score = 0.0
    
    # Calculate final score with appropriate weighting
    weights = {
        "role_adherence": 0.3,
        "compassion": 0.3,
        "boundaries": 0.2,
        "ethical": 0.2
    }
    
    final_score = (
        weights["role_adherence"] * role_adherence_score +
        weights["compassion"] * compassion_score +
        weights["boundaries"] * boundaries_score +
        weights["ethical"] * ethical_score
    )
    
    return normalize_score(final_score)
