"""
Centralized JSON extraction utility for Gemini API responses.

This module provides robust JSON extraction from Gemini API responses that may be
wrapped in markdown code blocks or contain various formatting issues.
"""

import json
import re
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
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
    Preprocess JSON content to fix common issues with API responses.

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

    # Handle truncated strings at the end - close incomplete string values
    lines = json_content.split('\n')
    if lines:
        last_line = lines[-1].strip()
        # Check if last line has an incomplete string (odd number of quotes after colon)
        if ':' in last_line and last_line.count('"') % 2 == 1:
            # Find the position after the last colon
            colon_pos = last_line.rfind(':')
            after_colon = last_line[colon_pos+1:].strip()
            if after_colon.startswith('"') and not after_colon.endswith('"'):
                # Close the incomplete string
                lines[-1] = last_line + '"'
                json_content = '\n'.join(lines)

    return json_content.strip()


def is_truncated_response(response_text: str) -> bool:
    """
    Detect if a response appears to be truncated based on common patterns.

    Args:
        response_text: The response text to check

    Returns:
        True if the response appears truncated, False otherwise
    """
    if not response_text:
        return True

    # Very short responses are likely truncated
    if len(response_text.strip()) < 50:
        return True

    # Check for truncation patterns
    truncation_patterns = [
        r':\s*$',  # Ends with colon and whitespace (field name but no value)
        r':\s*"[^"]*$',  # Ends with incomplete string value
        r'[,{]\s*$',  # Ends with comma or opening brace
        r'"[^"]*$',  # Ends with incomplete quoted string
    ]

    for pattern in truncation_patterns:
        if re.search(pattern, response_text.strip()):
            return True

    return False


def extract_json_from_response(response_text: str, context: str = "unknown") -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Extract and parse JSON from an API response with multiple fallback strategies.

    Args:
        response_text: The raw response text from API
        context: Context string for logging (e.g., "ahimsa", "dharma", "helpfulness")

    Returns:
        Tuple of (parsed_json_dict, success_flag, is_truncated)
        - parsed_json_dict: The parsed JSON as a dictionary, or None if all attempts failed
        - success_flag: True if parsing succeeded, False if default values should be used
    """
    if not response_text:
        logger.error(f"Empty response text provided for {context} JSON extraction")
        count = increment_default_score_count()
        logger.error(f"Using default scores due to empty response. Total default usage count: {count}")
        return None, False

    # Check if response appears truncated
    if is_truncated_response(response_text):
        logger.warning(f"Response appears truncated for {context} (length: {len(response_text)}): {response_text}")
        # Continue with extraction but flag as potentially problematic
    
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

    # Strategy 6: Try partial JSON extraction for truncated responses
    partial_result = _extract_partial_json(response_text, context)
    if partial_result is not None:
        logger.info(f"Successfully extracted partial JSON for {context}")
        return partial_result, True

    # All strategies failed
    logger.error(f"All JSON extraction strategies failed for {context}.")
    logger.error(f"FULL RESPONSE CONTENT FOR DEBUGGING:")
    logger.error(f"{'='*80}")
    logger.error(response_text)
    logger.error(f"{'='*80}")
    logger.error(f"Response length: {len(response_text)} characters")

    # Check if this looks like a truncation issue
    if is_truncated_response(response_text):
        logger.error(f"TRUNCATION DETECTED: Response appears to be cut off. Consider:")
        logger.error(f"  1. Increasing max_tokens in API call")
        logger.error(f"  2. Retrying with higher token limit")
        logger.error(f"  3. Checking for API timeout issues")

    count = increment_default_score_count()
    logger.error(f"Using default scores due to JSON parsing failure. Total default usage count: {count}")
    return None, False


def _extract_partial_json(text: str, context: str) -> Optional[Dict[str, Any]]:
    """
    Extract valid fields from partial/truncated JSON responses.

    This function attempts to parse individual key-value pairs even when
    the overall JSON is malformed or truncated.

    Args:
        text: The response text that may contain partial JSON
        context: Context for logging (e.g., "ahimsa", "dharma", "helpfulness")

    Returns:
        Dictionary with extracted fields or None if no valid fields found
    """
    logger.info(f"Attempting partial JSON extraction for {context}")

    # Find JSON-like content
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    # Extract everything from the opening brace
    json_like = text[start_idx:]

    # Try to extract individual key-value pairs
    extracted_fields = {}

    # Pattern to match key-value pairs
    # Matches: "key": value where value can be string, number, boolean, or null
    patterns = [
        r'"([^"]+)":\s*"([^"]*)"',  # String values
        r'"([^"]+)":\s*(\d+\.?\d*)',  # Numeric values
        r'"([^"]+)":\s*(true|false)',  # Boolean values
        r'"([^"]+)":\s*(null)',  # Null values
    ]

    for pattern in patterns:
        matches = re.findall(pattern, json_like, re.IGNORECASE)
        for key, value in matches:
            if key not in extracted_fields:  # Don't overwrite already extracted values
                # Convert value to appropriate type
                if pattern.endswith('true|false)'):
                    extracted_fields[key] = value.lower() == 'true'
                elif pattern.endswith('null)'):
                    extracted_fields[key] = None
                elif pattern.endswith(r'(\d+\.?\d*)'):
                    try:
                        extracted_fields[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        extracted_fields[key] = value
                else:
                    # String value - handle yes/no conversion
                    if value.lower() in ('yes', 'no'):
                        extracted_fields[key] = value.lower() == 'yes'
                    else:
                        extracted_fields[key] = value

    if extracted_fields:
        logger.info(f"Extracted {len(extracted_fields)} fields from partial JSON: {list(extracted_fields.keys())}")
        return extracted_fields

    return None


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

    # If no closing brace found, try to repair truncated JSON
    if start_idx != -1:
        return _attempt_json_repair(text[start_idx:])

    return None


def _attempt_json_repair(json_text: str) -> Optional[str]:
    """
    Attempt to repair truncated JSON by closing incomplete strings and objects.

    Args:
        json_text: Potentially truncated JSON text starting with '{'

    Returns:
        Repaired JSON string or None if repair is not possible
    """
    if not json_text.startswith('{'):
        return None

    # Try to find where the JSON was truncated
    lines = json_text.split('\n')
    repaired_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # If this is the last line and it looks incomplete
        if i == len(lines) - 1 and stripped and not stripped.endswith(('}', '"', ',')):
            # Check if it's an incomplete string value
            if ':' in stripped and '"' in stripped:
                # Find the last quote to see if string is incomplete
                last_quote_idx = stripped.rfind('"')
                colon_idx = stripped.rfind(':')

                if colon_idx > last_quote_idx:
                    # String value is incomplete, close it
                    repaired_lines.append(stripped + '"')
                else:
                    # String might be incomplete, try to close it
                    if stripped.count('"') % 2 == 1:  # Odd number of quotes = incomplete string
                        repaired_lines.append(stripped + '"')
                    else:
                        repaired_lines.append(stripped)
            else:
                repaired_lines.append(stripped)
        else:
            repaired_lines.append(line)

    # Join lines and ensure proper closing
    repaired_json = '\n'.join(repaired_lines)

    # Make sure JSON object is properly closed
    if not repaired_json.rstrip().endswith('}'):
        repaired_json = repaired_json.rstrip() + '\n}'

    return repaired_json


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
    logger.debug(f"Attempting to parse {context} JSON using {strategy}")
    logger.debug(f"Content to parse: {json_content[:200]}...")

    try:
        # First attempt: direct parsing
        result = json.loads(json_content)
        logger.info(f"Successfully parsed {context} JSON using {strategy} (direct)")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"Direct parsing failed for {strategy}: {e}")

    try:
        # Second attempt: with preprocessing
        preprocessed = preprocess_json_content(json_content)
        logger.debug(f"Preprocessed content: {preprocessed[:200]}...")
        result = json.loads(preprocessed)
        logger.info(f"Successfully parsed {context} JSON using {strategy} with preprocessing")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse {context} JSON using {strategy}: {e}")
        logger.debug(f"Failed JSON content (original): {json_content}")
        logger.debug(f"Failed JSON content (preprocessed): {preprocess_json_content(json_content)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {context} JSON using {strategy}: {e}")
        return None


# Provider-specific JSON recovery functions
async def fix_missing_keys_with_openai(
    evaluation_result: Dict[str, Any],
    required_keys: List[str],
    evaluation_type: str,
    openai_api_key: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Attempt to fix missing keys in an OpenAI evaluation result by making a second call
    that specifically asks OpenAI to fix the JSON.

    Args:
        evaluation_result: The original evaluation result with missing keys
        required_keys: List of required keys that should be in the result
        evaluation_type: Type of evaluation (ahimsa, dharma, helpfulness)
        openai_api_key: The OpenAI API key to use
        max_retries: Maximum number of retries for the fix attempt

    Returns:
        Fixed evaluation result with all required keys

    Raises:
        Exception: If unable to fix the missing keys after retries
    """
    from openai import AsyncOpenAI

    # Identify missing keys
    missing_keys = [key for key in required_keys if key not in evaluation_result]
    logger.warning(f"Attempting to fix missing keys in {evaluation_type} evaluation with OpenAI: {missing_keys}")

    # Create a prompt to fix the missing keys
    system_prompt = f"""
    You are a JSON repair specialist. I have a JSON object that is missing some required keys.

    The JSON object should have these required keys: {', '.join(required_keys)}

    But it's missing these keys: {', '.join(missing_keys)}

    Please provide a complete, valid JSON object with all required keys filled in with appropriate default values.
    """

    user_prompt = f"""
    Here is the incomplete JSON object:
    {json.dumps(evaluation_result, indent=2)}

    Please return a complete JSON object with all required keys: {', '.join(required_keys)}
    """

    retry_delay = 2
    client = AsyncOpenAI(api_key=openai_api_key)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Use a reliable model for JSON repair
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.1
            )

            content = response.choices[0].message.content
            if content:
                # Use our own extraction function to parse the fixed JSON
                fixed_result, extraction_success = extract_json_from_response(content, f"{evaluation_type}_fix")

                if extraction_success and fixed_result:
                    # Verify all required keys are now present
                    if all(key in fixed_result for key in required_keys):
                        logger.info(f"Successfully fixed missing keys in {evaluation_type} evaluation with OpenAI (attempt {attempt + 1}).")
                        return fixed_result
                    else:
                        still_missing = [key for key in required_keys if key not in fixed_result]
                        logger.warning(f"OpenAI fix attempt {attempt + 1} still missing keys: {still_missing}")
                else:
                    logger.error(f"OpenAI fix attempt {attempt + 1}: Failed to extract JSON from content")
            else:
                logger.warning(f"OpenAI fix attempt {attempt + 1}: Received empty content.")

        except Exception as e:
            logger.error(f"OpenAI fix attempt {attempt + 1}: Unexpected error: {e}", exc_info=True)

        if attempt < max_retries - 1:
            logger.info(f"OpenAI fix attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    # If we get here, all fix attempts failed
    raise Exception(f"Failed to fix missing keys in {evaluation_type} evaluation with OpenAI after {max_retries} attempts")


async def fix_missing_keys_with_anthropic(
    evaluation_result: Dict[str, Any],
    required_keys: List[str],
    evaluation_type: str,
    anthropic_api_key: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Attempt to fix missing keys in an Anthropic evaluation result by making a second call
    that specifically asks Anthropic to fix the JSON.

    Args:
        evaluation_result: The original evaluation result with missing keys
        required_keys: List of required keys that should be in the result
        evaluation_type: Type of evaluation (ahimsa, dharma, helpfulness)
        anthropic_api_key: The Anthropic API key to use
        max_retries: Maximum number of retries for the fix attempt

    Returns:
        Fixed evaluation result with all required keys

    Raises:
        Exception: If unable to fix the missing keys after retries
    """
    from anthropic import AsyncAnthropic

    # Identify missing keys
    missing_keys = [key for key in required_keys if key not in evaluation_result]
    logger.warning(f"Attempting to fix missing keys in {evaluation_type} evaluation with Anthropic: {missing_keys}")

    # Create a prompt to fix the missing keys
    system_prompt = f"""
    You are a JSON repair specialist. I have a JSON object that is missing some required keys.

    The JSON object should have these required keys: {', '.join(required_keys)}

    But it's missing these keys: {', '.join(missing_keys)}

    Please provide a complete, valid JSON object with all required keys filled in with appropriate default values.
    Return only the JSON object, no markdown formatting.
    """

    user_prompt = f"""
    Here is the incomplete JSON object:
    {json.dumps(evaluation_result, indent=2)}

    Please return a complete JSON object with all required keys: {', '.join(required_keys)}
    """

    retry_delay = 2
    client = AsyncAnthropic(api_key=anthropic_api_key)

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model="claude-3-5-haiku-20241022",  # Use a reliable model for JSON repair
                max_tokens=1000,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            content = response.content[0].text if response.content else None
            if content:
                # Use our own extraction function to parse the fixed JSON
                fixed_result, extraction_success = extract_json_from_response(content, f"{evaluation_type}_fix")

                if extraction_success and fixed_result:
                    # Verify all required keys are now present
                    if all(key in fixed_result for key in required_keys):
                        logger.info(f"Successfully fixed missing keys in {evaluation_type} evaluation with Anthropic (attempt {attempt + 1}).")
                        return fixed_result
                    else:
                        still_missing = [key for key in required_keys if key not in fixed_result]
                        logger.warning(f"Anthropic fix attempt {attempt + 1} still missing keys: {still_missing}")
                else:
                    logger.error(f"Anthropic fix attempt {attempt + 1}: Failed to extract JSON from content")
            else:
                logger.warning(f"Anthropic fix attempt {attempt + 1}: Received empty content.")

        except Exception as e:
            logger.error(f"Anthropic fix attempt {attempt + 1}: Unexpected error: {e}", exc_info=True)

        if attempt < max_retries - 1:
            logger.info(f"Anthropic fix attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    # If we get here, all fix attempts failed
    raise Exception(f"Failed to fix missing keys in {evaluation_type} evaluation with Anthropic after {max_retries} attempts")
