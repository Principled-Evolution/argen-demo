"""
Helper functions for processing chat responses in reward functions.
"""

import logging
import re
import json
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

def extract_content_from_chat_response(response):
    """
    Extract the content from a chat response.

    Args:
        response: The response to extract content from, which could be a string or a dictionary.

    Returns:
        The extracted content as a string.
    """
    # If the response is already a string, return it
    if isinstance(response, str):
        return response

    # If the response is a dictionary with a 'content' field, return the content
    if isinstance(response, dict) and 'content' in response:
        return response['content']

    # If the response is a dictionary with a 'role' and 'content' field, return the content
    if isinstance(response, dict) and 'role' in response and 'content' in response:
        return response['content']

    # Try to parse the response as JSON
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and 'content' in parsed:
                return parsed['content']
        except:
            pass

    # Try to extract content using regex
    if isinstance(response, str):
        match = re.search(r'"content":\s*"([^"]*)"', response)
        if match:
            return match.group(1)

    # If all else fails, return the response as is
    return str(response)

def process_completions(completions: List[Union[str, List[Dict], Dict]]) -> List[str]:
    """
    Process a list of completions to extract content from chat responses.

    Args:
        completions: List of completions, which could be strings, dictionaries, or lists of dictionaries.

    Returns:
        List of extracted content as strings.
    """
    processed_completions = []

    # Handle different completion formats
    for completion in completions:
        if isinstance(completion, list):
            # Handle list of messages (e.g., [{"role": "assistant", "content": "..."}])
            if len(completion) > 0:
                processed_completion = extract_content_from_chat_response(completion[0])
                processed_completions.append(processed_completion)
            else:
                # Empty list, append empty string
                processed_completions.append("")
        else:
            # Handle single message (string or dict)
            processed_completion = extract_content_from_chat_response(completion)
            processed_completions.append(processed_completion)

    return processed_completions
