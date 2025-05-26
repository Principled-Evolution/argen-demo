"""
Centralized JSON extraction utility for Gemini API responses.

This module provides robust JSON extraction from Gemini API responses that may be
wrapped in markdown code blocks or contain various formatting issues.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Tuple
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Global counter for tracking default score usage
_default_score_lock = threading.Lock()
_default_score_count = 0


def increment_default_score_count() -> int:
    """
    Increment and return the global count of times default scores were used.
    
    Returns:
        int: The new total count of default score usage
    """
    global _default_score_count
    with _default_score_lock:
        _default_score_count += 1
        return _default_score_count


def get_default_score_count() -> int:
    """
    Get the current count of times default scores were used.
    
    Returns:
        int: The total count of default score usage
    """
    global _default_score_count
    with _default_score_lock:
        return _default_score_count


def reset_default_score_count() -> None:
    """Reset the global default score count to zero."""
    global _default_score_count
    with _default_score_lock:
        _default_score_count = 0


def preprocess_json_content(json_content: str) -> str:
    """
    Preprocess JSON content to fix common issues with Gemini responses.
    
    Args:
        json_content: The JSON content string to preprocess
        
    Returns:
        Preprocessed JSON content string
    """
    if not json_content:
        return json_content
    
    # Remove control characters
    json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)
    
    # Replace 'yes'/'no' with 'true'/'false'
    json_content = re.sub(r':\s*yes\s*([,}])', r': true\1', json_content, flags=re.IGNORECASE)
    json_content = re.sub(r':\s*no\s*([,}])', r': false\1', json_content, flags=re.IGNORECASE)
    
    # Fix common escape sequence issues
    json_content = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_content)
    
    # Remove trailing commas
    json_content = re.sub(r',\s*}', '}', json_content)
    json_content = re.sub(r',\s*]', ']', json_content)
    
    return json_content.strip()


def extract_json_from_response(response_text: str, context: str = "unknown") -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Extract and parse JSON from a Gemini API response with multiple fallback strategies.
    
    Args:
        response_text: The raw response text from Gemini API
        context: Context string for logging (e.g., "ahimsa", "dharma", "helpfulness")
        
    Returns:
        Tuple of (parsed_json_dict, success_flag)
        - parsed_json_dict: The parsed JSON as a dictionary, or None if all attempts failed
        - success_flag: True if parsing succeeded, False if default values should be used
    """
    if not response_text:
        logger.error(f"Empty response text provided for {context} JSON extraction")
        count = increment_default_score_count()
        logger.error(f"Using default scores due to empty response. Total default usage count: {count}")
        return None, False
    
    # Strategy 1: Extract from markdown code blocks (```json ... ```)
    json_content = _extract_from_markdown_fences(response_text)
    if json_content:
        result = _attempt_json_parse(json_content, context, "markdown fences")
        if result is not None:
            return result, True
    
    # Strategy 2: Extract from any code blocks (``` ... ```)
    json_content = _extract_from_any_code_blocks(response_text)
    if json_content:
        result = _attempt_json_parse(json_content, context, "any code blocks")
        if result is not None:
            return result, True
    
    # Strategy 3: Find JSON object boundaries ({ ... })
    json_content = _extract_json_object(response_text)
    if json_content:
        result = _attempt_json_parse(json_content, context, "JSON object boundaries")
        if result is not None:
            return result, True
    
    # Strategy 4: Find JSON array boundaries ([ ... ])
    json_content = _extract_json_array(response_text)
    if json_content:
        result = _attempt_json_parse(json_content, context, "JSON array boundaries")
        if result is not None:
            return result, True
    
    # Strategy 5: Try parsing the entire response as JSON
    result = _attempt_json_parse(response_text, context, "entire response")
    if result is not None:
        return result, True
    
    # All strategies failed
    logger.error(f"All JSON extraction strategies failed for {context}. Response preview: {response_text[:200]}...")
    count = increment_default_score_count()
    logger.error(f"Using default scores due to JSON parsing failure. Total default usage count: {count}")
    return None, False


def _extract_from_markdown_fences(text: str) -> Optional[str]:
    """Extract JSON from ```json ... ``` markdown fences."""
    # Pattern to match ```json ... ``` with optional whitespace
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_from_any_code_blocks(text: str) -> Optional[str]:
    """Extract content from any ``` ... ``` code blocks."""
    # Pattern to match ``` ... ``` (not specifically json)
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # Only return if it looks like JSON (starts with { or [)
        if content.startswith(('{', '[')):
            return content
    return None


def _extract_json_object(text: str) -> Optional[str]:
    """Extract JSON object using { ... } boundaries."""
    # Find the first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    return None


def _extract_json_array(text: str) -> Optional[str]:
    """Extract JSON array using [ ... ] boundaries."""
    # Find the first [ and last ]
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    return None


def _attempt_json_parse(json_content: str, context: str, strategy: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse JSON content with preprocessing.
    
    Args:
        json_content: The JSON content to parse
        context: Context for logging
        strategy: The extraction strategy being used
        
    Returns:
        Parsed JSON dictionary or None if parsing failed
    """
    try:
        # First attempt: direct parsing
        return json.loads(json_content)
    except json.JSONDecodeError:
        pass
    
    try:
        # Second attempt: with preprocessing
        preprocessed = preprocess_json_content(json_content)
        result = json.loads(preprocessed)
        logger.info(f"Successfully parsed {context} JSON using {strategy} with preprocessing")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse {context} JSON using {strategy}: {e}")
        logger.debug(f"Failed JSON content: {json_content[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {context} JSON using {strategy}: {e}")
        return None
