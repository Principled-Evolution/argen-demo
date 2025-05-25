"""
Reward functions for ArGen GRPO fine-tuning using Gemini evaluations.
"""

import json
import asyncio
import logging
import concurrent.futures
import functools
import re
import os
import sys
import uuid
import random
from typing import Dict, Optional, List, Any, Callable, Tuple, Set

import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException
from src.utils.env import get_gemini_api_key
from src.utils.data_integrity import verify_prompt_tier_hash, _DELIMITER
from src.utils.response_validator import ModelResponse, ModelResponseProcessor
from src.utils.gemini_api_wrapper import TrackedGenerativeModel
from src.utils.gemini_api_tracker import GeminiAPITracker

# Import OpenAI evaluators for fallback
# OPENAI_AVAILABLE will now primarily check for the base 'openai' library.
# Specific function imports from 'openai_rewards' are moved into 'fallback_to_openai'.
OPENAI_AVAILABLE = False
try:
    from openai import AsyncOpenAI # Check for the base OpenAI library
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False # Stays False if 'openai' itself is not found

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Error tracking (legacy - kept for backward compatibility)
GEMINI_ERROR_COUNT = 0
GEMINI_TOTAL_CALLS = 0
GEMINI_ERROR_THRESHOLD = 0.05  # 5% error threshold
VERBOSE_LOGGING = False

# Flag to control whether to include reasoning in evaluation responses
# Default to True for backward compatibility with existing code
INCLUDE_REASONING = False

def set_include_reasoning(include_reasoning: bool):
    """
    Set whether to include reasoning in evaluation responses.

    Args:
        include_reasoning: Whether to include reasoning in evaluation responses
    """
    global INCLUDE_REASONING
    INCLUDE_REASONING = include_reasoning
    logger.info(f"Set include_reasoning to {include_reasoning}")

def get_gemini_api_call_count() -> int:
    """Returns the total number of Gemini API calls made."""
    return GeminiAPITracker().get_total_calls()

def get_gemini_api_stats() -> Dict[str, Dict[str, int]]:
    """Returns detailed statistics for Gemini API calls."""
    return GeminiAPITracker().get_stats()

def print_gemini_api_summary():
    """Print a formatted summary of Gemini API usage."""
    GeminiAPITracker().print_summary()

# Temporary structure for backward compatibility - should never be returned
DEFAULT_EVAL_RESPONSE = {
    "error": "Evaluation failed",
    "reasoning": "Could not get response from Gemini API.",
    "ahimsa_score": 0.0,
    "harm_avoidance_score": 0.0,
    "safety_consideration_score": 0.0,
    "has_medical_disclaimer": False,
    "has_professional_referral": False,
    "ahimsa_violation": True,
    "dharma_score": 0.0,
    "domain_adherence_score": 0.0,
    "out_of_domain_advice": True,
    "response_scope": "S3", # Default to most restrictive scope on error
    "scope_risk": "high", # Default to highest risk on error
    "scope_penalty_factor": 0.0, # Default to maximum penalty on error
    "dharma_violation": True,
    "helpfulness_score": 0.0,
    "clarity_score": 0.0,
    "relevance_score": 0.0,
    "completeness_score": 0.0,
    "helpfulness_violation": True
}

class GeminiErrorRateExceeded(Exception):
    """Exception raised when Gemini error rate exceeds the threshold."""
    pass

class GeminiMissingKeysError(Exception):
    """Exception raised when Gemini response is missing required keys."""
    pass

class HashVerificationError(Exception):
    """Exception raised when hash verification fails."""
    pass

def track_gemini_error():
    """
    Track a Gemini API error and check if the error rate exceeds the threshold.
    Raises GeminiErrorRateExceeded if the threshold is exceeded.

    Note: This function is kept for backward compatibility but is now a no-op
    since error tracking is handled by the TrackedGenerativeModel wrapper.
    """
    # Legacy function - tracking is now handled by TrackedGenerativeModel
    pass

def track_gemini_success():
    """
    Track a successful Gemini API call.

    Note: This function is kept for backward compatibility but is now a no-op
    since success tracking is handled by the TrackedGenerativeModel wrapper.
    """
    # Legacy function - tracking is now handled by TrackedGenerativeModel
    pass

# Counter for missing keys occurrences
GEMINI_MISSING_KEYS_COUNT = 0
GEMINI_MISSING_KEYS_FIXED_COUNT = 0
GEMINI_OPENAI_FALLBACK_COUNT = 0

# Configuration for hash verification
VERIFY_HASH_SAMPLE_RATE = 0.1  # Verify 10% of items
VERIFY_HASH_SEED = 42  # Fixed seed for reproducible sampling

# Counters for hash verification
HASH_VERIFICATION_COUNT = 0
HASH_VERIFICATION_FAILED_COUNT = 0

def verify_hash_or_exit(prompt: str, compound_value: str, scenario_id: str, field_name: str, eval_type: str) -> str:
    """
    Verify that the hash in the compound value matches the hash calculated from the prompt.
    If verification fails, exit the program with a detailed error message.

    Args:
        prompt: The prompt string to verify
        compound_value: The compound value containing the hash (e.g., "A:hash" or "S0:hash")
        scenario_id: Identifier for the scenario being verified
        field_name: Name of the field being verified (e.g., "tier" or "scope")
        eval_type: Type of evaluation (e.g., "ahimsa" or "dharma")

    Returns:
        The base value (e.g., "A" or "S0") extracted from the compound value

    Raises:
        HashVerificationError: If hash verification fails
    """
    global HASH_VERIFICATION_COUNT, HASH_VERIFICATION_FAILED_COUNT

    # Check if compound_value contains a hash
    if _DELIMITER not in str(compound_value):
        logger.debug(f"{eval_type.capitalize()} evaluation: No hash found in {field_name} '{compound_value}', using as is.")
        return compound_value

    # Extract base value and hash
    base_value, embedded_hash = compound_value.split(_DELIMITER, 1)

    # Increment verification counter
    HASH_VERIFICATION_COUNT += 1

    # Verify hash
    if not verify_prompt_tier_hash(prompt, compound_value):
        # Calculate the expected hash for detailed error message
        from src.utils.data_integrity import calculate_prompt_hash
        expected_hash = calculate_prompt_hash(prompt)

        # Increment failure counter
        HASH_VERIFICATION_FAILED_COUNT += 1

        # Prepare detailed error message
        error_msg = (
            f"\n{'='*80}\n"
            f"CRITICAL ERROR: HASH VERIFICATION FAILED\n"
            f"Scenario ID: {scenario_id}\n"
            f"Evaluation Type: {eval_type}\n"
            f"Field: {field_name}\n"
            f"Compound Value: {compound_value}\n"
            f"Base Value: {base_value}\n"
            f"Embedded Hash: {embedded_hash}\n"
            f"Expected Hash: {expected_hash}\n"
            f"Prompt: {prompt[:100]}...\n"
            f"{'='*80}\n"
        )

        # Log the error
        logger.critical(error_msg)

        # Raise exception
        raise HashVerificationError(f"Hash verification failed for {field_name} in {eval_type} evaluation of scenario {scenario_id}")

    logger.debug(f"{eval_type.capitalize()} evaluation: Hash verification passed for {field_name}.")
    return base_value

async def fix_missing_keys_with_gemini(
    evaluation_result: Dict[str, Any],
    required_keys: List[str],
    evaluation_type: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Attempt to fix missing keys in a Gemini evaluation result by making a second call
    that specifically asks Gemini to fix the JSON.

    Args:
        evaluation_result: The original evaluation result with missing keys
        required_keys: List of required keys that should be in the result
        evaluation_type: Type of evaluation (ahimsa, dharma, helpfulness)
        max_retries: Maximum number of retries for the fix attempt

    Returns:
        Fixed evaluation result with all required keys

    Raises:
        GeminiMissingKeysError: If unable to fix the missing keys after retries
    """
    global GEMINI_MISSING_KEYS_COUNT, GEMINI_MISSING_KEYS_FIXED_COUNT

    # Track the occurrence
    GEMINI_MISSING_KEYS_COUNT += 1

    # Identify missing keys
    missing_keys = [key for key in required_keys if key not in evaluation_result]
    logger.warning(f"Attempting to fix missing keys in {evaluation_type} evaluation: {missing_keys}")

    # Create a prompt to fix the missing keys
    system_prompt = f"""
    You are a JSON repair specialist. I have a JSON object that is missing some required keys.

    The JSON object should have these required keys: {', '.join(required_keys)}

    But it's missing these keys: {', '.join(missing_keys)}

    Please fix the JSON by adding the missing keys with appropriate values based on the context.
    Return ONLY the complete, fixed JSON object with no explanation or markdown formatting.
    """

    user_prompt = f"""
    Here is the incomplete JSON:
    {json.dumps(evaluation_result, indent=2)}

    Please add the missing keys: {', '.join(missing_keys)}

    Return the complete, fixed JSON.
    """

    retry_delay = 2
    fixed_result = None

    for attempt in range(max_retries):
        try:
            # Define a synchronous function to make the Gemini API call
            def make_gemini_call():
                model = TrackedGenerativeModel(GEMINI_EVAL_MODEL)

                # Create the chat session
                chat = model.start_chat(history=[
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "model", "parts": ["I'll fix the JSON by adding the missing keys with appropriate values."]}
                ])

                # Send the user prompt
                response = chat.send_message(user_prompt)
                return response.text

            # Run the synchronous function in a thread
            try:
                content = await run_in_thread(make_gemini_call)
            except BlockedPromptException as e:
                logger.error(f"Fix attempt {attempt + 1}: Prompt blocked by Gemini API: {e}")
                # Skip to next attempt or fail
                content = None

            if content:
                try:
                    # Extract JSON from the response
                    if "```json" in content and "```" in content:
                        json_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_content = content.split("```")[1].split("```")[0].strip()
                    else:
                        json_content = content

                    # Preprocess JSON content
                    json_content = preprocess_json_content(json_content)

                    try:
                        fixed_result = json.loads(json_content)
                        if VERBOSE_LOGGING:
                            logger.info(f"Successfully fixed missing keys in {evaluation_type} evaluation (attempt {attempt + 1}).")

                        # Verify all required keys are now present
                        if all(key in fixed_result for key in required_keys):
                            GEMINI_MISSING_KEYS_FIXED_COUNT += 1
                            return fixed_result
                        else:
                            still_missing = [key for key in required_keys if key not in fixed_result]
                            logger.warning(f"Fix attempt {attempt + 1} still missing keys: {still_missing}")
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Fix attempt {attempt + 1}: Failed to decode JSON: {json_err}\nContent: {content}")
                except Exception as e:
                    logger.error(f"Fix attempt {attempt + 1}: Error processing content: {e}")
            else:
                logger.warning(f"Fix attempt {attempt + 1}: Received empty content.")

        except Exception as e:
            logger.error(f"Fix attempt {attempt + 1}: Unexpected error: {e}", exc_info=True)

        if attempt < max_retries - 1:
            logger.info(f"Fix attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    # If we get here, all fix attempts failed
    raise GeminiMissingKeysError(f"Failed to fix missing keys in {evaluation_type} evaluation after {max_retries} attempts")

async def fallback_to_openai(
    original_prompt: str,
    model_response: str,
    evaluation_type: str,
    original_prompt_meta: Optional[Dict[str, Any]] = None,
    original_gemini_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fallback to OpenAI evaluator when Gemini fails.

    Args:
        original_prompt: The user prompt
        model_response: The model response
        evaluation_type: Type of evaluation (ahimsa, dharma, helpfulness)
        original_prompt_meta: Metadata about the original prompt
        original_gemini_result: The original failed Gemini result

    Returns:
        Evaluation result from OpenAI

    Raises:
        GeminiErrorRateExceeded: If OpenAI fallback also fails
    """
    global GEMINI_OPENAI_FALLBACK_COUNT

    if not OPENAI_AVAILABLE:
        logger.error(f"OpenAI base library not available, cannot attempt fallback for {evaluation_type} evaluation")
        track_gemini_error()
        raise GeminiErrorRateExceeded(f"OpenAI base library not available for {evaluation_type} evaluation")

    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error(f"OpenAI API key not found for {evaluation_type} fallback")
        track_gemini_error()
        raise GeminiErrorRateExceeded(f"OpenAI API key not found for {evaluation_type} fallback")

    # Generate a scenario ID for logging
    scenario_id = str(uuid.uuid4())[:8]

    # Log the fallback event
    logger.warning(
        f"FALLBACK-{scenario_id}: Falling back to OpenAI for {evaluation_type} evaluation. "
        f"Original Gemini result: {original_gemini_result}"
    )

    GEMINI_OPENAI_FALLBACK_COUNT += 1

    try:
        # Import specific OpenAI evaluator functions here, inside the fallback logic
        from src.reward_functions.openai_rewards import (
            evaluate_ahimsa_with_openai,
            evaluate_dharma_with_openai,
            evaluate_helpfulness_with_openai
        )

        # Call the appropriate OpenAI evaluator
        if evaluation_type == "ahimsa":
            result = await evaluate_ahimsa_with_openai(
                original_prompt, model_response, api_key, original_prompt_meta
            )
        elif evaluation_type == "dharma":
            result = await evaluate_dharma_with_openai(
                original_prompt, model_response, api_key, original_prompt_meta
            )
        elif evaluation_type == "helpfulness":
            result = await evaluate_helpfulness_with_openai(
                original_prompt, model_response, api_key
            )
        else:
            logger.error(f"Unknown evaluation type for OpenAI fallback: {evaluation_type}")
            track_gemini_error()
            raise GeminiErrorRateExceeded(f"Unknown evaluation type for OpenAI fallback: {evaluation_type}")

        logger.info(f"FALLBACK-{scenario_id}: Successfully completed OpenAI fallback for {evaluation_type} evaluation")
        return result
    except ImportError as e:
        logger.error(f"FALLBACK-{scenario_id}: Failed to import OpenAI reward functions for {evaluation_type}: {e}", exc_info=True)
        track_gemini_error()
        raise GeminiErrorRateExceeded(f"OpenAI fallback failed for {evaluation_type} evaluation due to import error: {e}")
    except Exception as e:
        logger.error(f"FALLBACK-{scenario_id}: OpenAI fallback failed for {evaluation_type} evaluation: {e}", exc_info=True)
        track_gemini_error()

def preprocess_json_content(json_content: str) -> str:
    """
    Preprocess JSON content to fix common issues with Gemini responses.
    Enhanced to handle control characters and other JSON parsing issues.

    Args:
        json_content: The JSON content string to preprocess

    Returns:
        Preprocessed JSON content string
    """
    import unicodedata

    # Step 1: Remove or replace control characters
    # Remove ASCII control characters (0x00-0x1F) except tab, newline, carriage return
    # Also remove DEL character (0x7F) and Unicode control characters
    cleaned_chars = []
    for char in json_content:
        code = ord(char)
        # Keep printable ASCII, tab, newline, carriage return
        if (32 <= code <= 126) or char in '\t\n\r':
            cleaned_chars.append(char)
        # Replace other control characters with space
        elif code < 32 or code == 127 or unicodedata.category(char).startswith('C'):
            cleaned_chars.append(' ')
        else:
            # Keep other Unicode characters
            cleaned_chars.append(char)

    json_content = ''.join(cleaned_chars)

    # Step 2: Normalize whitespace
    # Replace multiple consecutive whitespace with single space
    json_content = re.sub(r'\s+', ' ', json_content)

    # Step 3: Fix common boolean value issues
    # Replace 'yes' with 'true' and 'no' with 'false'
    json_content = re.sub(r':\s*yes\s*,', ': true,', json_content, flags=re.IGNORECASE)
    json_content = re.sub(r':\s*no\s*,', ': false,', json_content, flags=re.IGNORECASE)
    # Handle the case where yes/no is the last value in an object
    json_content = re.sub(r':\s*yes\s*\n', ': true\n', json_content, flags=re.IGNORECASE)
    json_content = re.sub(r':\s*no\s*\n', ': false\n', json_content, flags=re.IGNORECASE)
    json_content = re.sub(r':\s*yes\s*\}', ': true}', json_content, flags=re.IGNORECASE)
    json_content = re.sub(r':\s*no\s*\}', ': false}', json_content, flags=re.IGNORECASE)

    # Step 4: Fix invalid escape sequences
    # Fix the invalid pipe escape
    json_content = json_content.replace(r'\|', '|')
    # Fix invalid underscore escape
    json_content = json_content.replace(r'\_', '_')
    # Fix other common invalid escapes
    json_content = json_content.replace(r'\-', '-')
    json_content = json_content.replace(r'\+', '+')
    json_content = json_content.replace(r'\=', '=')
    json_content = json_content.replace(r'\(', '(')
    json_content = json_content.replace(r'\)', ')')
    json_content = json_content.replace(r'\[', '[')
    json_content = json_content.replace(r'\]', ']')

    # Step 5: Remove trailing commas before closing curly braces or square brackets
    # This handles trailing commas in objects and arrays
    json_content = re.sub(r',\s*(\}|\])', r'\1', json_content)

    # Step 6: Fix common quote issues
    # Replace smart quotes with regular quotes
    json_content = json_content.replace('"', '"').replace('"', '"')
    json_content = json_content.replace(''', "'").replace(''', "'")

    # Step 7: Ensure proper JSON structure
    # Strip leading/trailing whitespace
    json_content = json_content.strip()

    return json_content


def detect_control_characters(text: str) -> bool:
    """
    Detect if text contains problematic control characters that could cause JSON parsing issues.

    Args:
        text: The text to check

    Returns:
        True if control characters are detected, False otherwise
    """
    import unicodedata

    for char in text:
        code = ord(char)
        # Check for problematic control characters
        if (code < 32 and char not in '\t\n\r') or code == 127:
            return True
        # Check for Unicode control characters
        if unicodedata.category(char).startswith('C') and char not in '\t\n\r':
            return True
    return False


def create_control_character_retry_prompt(original_prompt: str, attempt_number: int) -> str:
    """
    Create a modified prompt for retry attempts that explicitly asks to avoid control characters.

    Args:
        original_prompt: The original prompt text
        attempt_number: Which retry attempt this is (1-based)

    Returns:
        Modified prompt with control character avoidance instructions
    """
    control_char_instruction = (
        "\n\nIMPORTANT: Your response must contain only standard printable characters. "
        "Do not include any control characters, special formatting characters, or "
        "non-printable characters in your JSON response. Use only standard ASCII "
        "characters (32-126) plus newlines and spaces."
    )

    if attempt_number > 1:
        retry_instruction = f"\n\nThis is retry attempt {attempt_number}. "
        if attempt_number > 2:
            retry_instruction += "Previous attempts failed due to formatting issues. "
        retry_instruction += "Please ensure your JSON response is clean and parseable."
        control_char_instruction = retry_instruction + control_char_instruction

    return original_prompt + control_char_instruction


# Gemini model to use for evaluations
GEMINI_EVAL_MODEL = "gemini-2.0-flash"

# Thread pool for running synchronous Gemini API calls
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=200)

# Initialize the ModelResponseProcessor for sanitizing model responses
response_processor = ModelResponseProcessor()

def configure_gemini():
    """Configure the Gemini API with the API key."""
    try:
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        return False

async def run_in_thread(func: Callable, *args, **kwargs):
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_pool,
        functools.partial(func, *args, **kwargs)
    )

# evaluate_dharma_with_gemini is now in src.reward_functions.gemini.dharma
# from src.reward_functions.gemini.dharma import evaluate_dharma_with_gemini

# evaluate_helpfulness_with_gemini will be moved to src.reward_functions.gemini.helpfulness


# --- Batch Evaluation Function ---
async def batch_evaluate_with_gemini(
    prompts: List[str],
    responses: List[str],
    max_concurrency: int = 200,  # Increased to match thread pool size
    metadata_list: Optional[List[Dict]] = None,
    verify_hash_sample_rate: float = VERIFY_HASH_SAMPLE_RATE  # Default to global sample rate
) -> List[Dict]:
    """
    Batch evaluate multiple prompt-response pairs with Gemini API.
    Uses a semaphore to control concurrency and prevent rate limiting.
    Performs hash verification on a sample of prompts based on the sample rate.
    Sanitizes model responses before evaluation to prevent issues with malformed responses.

    Args:
        prompts: List of user prompts.
        responses: List of model responses.
        max_concurrency: Maximum number of concurrent API calls (default: 200).
        metadata_list: Optional list of metadata dicts containing tier information.
                      Tier information is used only for post-evaluation calculations,
                      not passed to the LLM evaluators.
        verify_hash_sample_rate: Fraction of prompts to verify (0.0-1.0).
                                Default is VERIFY_HASH_SAMPLE_RATE (0.1).

    Returns:
        List of dictionaries containing evaluation results for each prompt-response pair.
    """
    # Log response processor statistics
    stats = response_processor.get_stats()
    logger.info(f"Response processor stats: {stats}")
    if not configure_gemini():
        logger.error("Gemini API key not configured for batch evaluation.")
        track_gemini_error()
        raise GeminiErrorRateExceeded("Gemini API key not configured for batch evaluation.")

    if len(prompts) != len(responses):
        logger.error(f"Mismatch between prompts ({len(prompts)}) and responses ({len(responses)})")
        track_gemini_error()
        raise GeminiErrorRateExceeded(f"Mismatch between prompts ({len(prompts)}) and responses ({len(responses)})")

    # Create a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate_pair_with_semaphore(prompt, response, metadata=None):
        """Evaluate a single prompt-response pair with semaphore control."""
        async with semaphore:
            # Import moved here to break circular dependency
            from src.reward_functions.gemini.ahimsa import evaluate_ahimsa_with_gemini
            # Import the refactored evaluate_dharma_with_gemini
            from src.reward_functions.gemini.dharma import evaluate_dharma_with_gemini
            # Import the refactored evaluate_helpfulness_with_gemini
            from src.reward_functions.gemini.helpfulness import evaluate_helpfulness_with_gemini
            # Create tasks for all three evaluations
            ahimsa_task = evaluate_ahimsa_with_gemini(prompt, response, metadata)
            dharma_task = evaluate_dharma_with_gemini(prompt, response, metadata)
            helpfulness_task = evaluate_helpfulness_with_gemini(prompt, response) # No metadata for helpfulness

            # Run all three evaluations concurrently
            results = await asyncio.gather(
                ahimsa_task, dharma_task, helpfulness_task,
                return_exceptions=True
            )

            # Process results
            eval_results = {}
            keys = ["ahimsa", "dharma", "helpfulness"]
            all_successful = True

            for i, res in enumerate(results):
                key_base = keys[i]
                if isinstance(res, Exception):
                    logger.error(f"Evaluation failed for {key_base}: {res}")
                    eval_results[f"{key_base}_score"] = 0.0
                    eval_results[f"{key_base}_violation"] = True
                    eval_results[f"{key_base}_error"] = str(res)
                    all_successful = False
                elif isinstance(res, dict):
                    # For Ahimsa, capture raw scores before penalties
                    if key_base == "ahimsa":
                        # Get raw scores without penalty for disclaimers/referrals
                        harm_avoidance = res.get("harm_avoidance_score", 0.0)
                        safety_context = res.get("safety_context_score", 0.0)
                        raw_ahimsa_score = (harm_avoidance + safety_context) / 2.0
                        eval_results[f"{key_base}_raw_score"] = raw_ahimsa_score

                        # Store whether medical disclaimer and referral were present
                        eval_results["has_medical_disclaimer"] = res.get("has_medical_disclaimer", False)
                        eval_results["has_professional_referral"] = res.get("has_professional_referral", False)

                        # Still store the penalized score for violation check
                        eval_results[f"{key_base}_score"] = res.get(f"{key_base}_score", 0.0)
                    else:
                        # For other metrics, just use the score as provided
                        eval_results[f"{key_base}_score"] = res.get(f"{key_base}_score", 0.0)

                    eval_results[f"{key_base}_violation"] = res.get(f"{key_base}_violation", True)
                else:
                    logger.error(f"Unexpected result type for {key_base}: {type(res)}")
                    eval_results[f"{key_base}_score"] = 0.0
                    eval_results[f"{key_base}_violation"] = True
                    all_successful = False

            if all_successful:
                # Get scope penalty factor if available
                scope_penalty_factor = eval_results.get("scope_penalty_factor", 1.0)

                # Calculate overall_risk based on summing individual risks (1-score)
                # Use ahimsa_raw_score (before penalties) if available, otherwise use regular score
                ahimsa_score = eval_results.get("ahimsa_raw_score", eval_results.get("ahimsa_score", 0.0))

                # Apply scope penalty to all component scores
                if scope_penalty_factor < 1.0:
                    # Apply scope penalty to ahimsa score
                    ahimsa_score = ahimsa_score * scope_penalty_factor
                    eval_results["ahimsa_score"] = ahimsa_score

                    # Apply scope penalty to helpfulness score
                    helpfulness_score = eval_results.get("helpfulness_score", 0.0) * scope_penalty_factor
                    eval_results["helpfulness_score"] = helpfulness_score

                    # Set overall violation if penalty is 0
                    if scope_penalty_factor == 0.0:
                        eval_results["overall_violation"] = True
                        eval_results["ahimsa_violation"] = True
                        eval_results["helpfulness_violation"] = True

                # Calculate risks
                ahimsa_risk = 1.0 - ahimsa_score
                dharma_score = eval_results.get("dharma_score", 0.0)
                dharma_risk = 1.0 - dharma_score
                helpfulness_score = eval_results.get("helpfulness_score", 0.0)
                helpfulness_risk = 1.0 - helpfulness_score

                # Sum the individual risks
                total_risk = ahimsa_risk + dharma_risk + helpfulness_risk

                # Store the individual risks and total risk
                eval_results["ahimsa_risk"] = ahimsa_risk
                eval_results["dharma_risk"] = dharma_risk
                eval_results["helpfulness_risk"] = helpfulness_risk
                eval_results["overall_risk"] = total_risk

                # Add a single violation flag for easier downstream checks
                eval_results["violation"] = (
                    eval_results.get("ahimsa_violation", False) or
                    eval_results.get("dharma_violation", False) or
                    eval_results.get("helpfulness_violation", False) or
                    scope_penalty_factor == 0.0
                )

            return eval_results

    # Determine which prompts to verify based on sample rate
    num_items = len(prompts)
    if verify_hash_sample_rate > 0:
        # Use fixed seed for reproducible sampling
        rng = random.Random(VERIFY_HASH_SEED)
        num_to_sample = max(1, min(num_items, int(num_items * verify_hash_sample_rate)))
        sample_indices = set(rng.sample(range(num_items), num_to_sample))
        logger.info(f"Selected {num_to_sample} out of {num_items} prompts for hash verification (sample rate: {verify_hash_sample_rate:.2f})")
        logger.debug(f"Hash verification indices: {sorted(sample_indices)}")
    else:
        sample_indices = set()
        logger.info("Hash verification disabled (sample rate is 0)")

    # Create tasks for all prompt-response pairs
    tasks = []
    for i, (p, r) in enumerate(zip(prompts, responses)):
        # Get metadata for this prompt
        meta = metadata_list[i] if metadata_list and i < len(metadata_list) else None

        # If this prompt is selected for verification, add a flag to metadata
        if meta is not None and i in sample_indices:
            # Make a copy to avoid modifying the original
            meta = meta.copy()
            meta["verify_hash"] = True

        # Add task
        tasks.append(evaluate_pair_with_semaphore(p, r, meta))

    # Run all tasks concurrently with semaphore control
    logger.info(f"Batch evaluating {len(prompts)} prompt-response pairs with concurrency {max_concurrency}...")
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()

    logger.info(f"Completed batch evaluation of {len(prompts)} pairs in {end_time - start_time:.2f} seconds "
                f"(avg: {(end_time - start_time) / max(1, len(prompts)):.2f}s per pair)")

    # Log error handling statistics
    log_error_handling_stats()

    return results


def ensure_reasoning_field(evaluation_result: Dict) -> Dict:
    """
    Ensure that the evaluation result has a reasoning field, even if reasoning was disabled.
    This maintains backward compatibility with code that expects the reasoning field.

    Args:
        evaluation_result: The evaluation result dictionary

    Returns:
        The evaluation result with a reasoning field added if it was missing
    """
    if "reasoning" not in evaluation_result:
        if INCLUDE_REASONING:
            # Reasoning was enabled but missing from response - add a fallback message
            evaluation_result["reasoning"] = "Reasoning field was requested but not provided by the evaluator."
        else:
            # Reasoning was disabled - add the standard placeholder
            evaluation_result["reasoning"] = "Reasoning field disabled during evaluation."
    return evaluation_result

def log_error_handling_stats():
    """
    Log statistics about error handling in Gemini evaluations.
    This includes counts of missing keys, successful fixes, OpenAI fallbacks,
    and hash verification statistics.
    """
    global GEMINI_ERROR_COUNT, GEMINI_TOTAL_CALLS
    global GEMINI_MISSING_KEYS_COUNT, GEMINI_MISSING_KEYS_FIXED_COUNT, GEMINI_OPENAI_FALLBACK_COUNT
    global HASH_VERIFICATION_COUNT, HASH_VERIFICATION_FAILED_COUNT

    # Calculate percentages
    total_calls = max(1, GEMINI_TOTAL_CALLS)  # Avoid division by zero
    error_rate = GEMINI_ERROR_COUNT / total_calls

    missing_keys_rate = GEMINI_MISSING_KEYS_COUNT / total_calls if total_calls > 0 else 0
    fix_success_rate = GEMINI_MISSING_KEYS_FIXED_COUNT / max(1, GEMINI_MISSING_KEYS_COUNT)
    fallback_rate = GEMINI_OPENAI_FALLBACK_COUNT / total_calls if total_calls > 0 else 0

    # Hash verification statistics
    hash_verification_rate = HASH_VERIFICATION_COUNT / total_calls if total_calls > 0 else 0
    hash_failure_rate = HASH_VERIFICATION_FAILED_COUNT / max(1, HASH_VERIFICATION_COUNT)

    # Log the statistics
    logger.info("=== Gemini Evaluation Error Handling Statistics ===")
    logger.info(f"Total API calls: {GEMINI_TOTAL_CALLS}")
    logger.info(f"Total errors: {GEMINI_ERROR_COUNT} ({error_rate:.2%})")
    logger.info(f"Missing keys occurrences: {GEMINI_MISSING_KEYS_COUNT} ({missing_keys_rate:.2%} of total calls)")
    logger.info(f"Successfully fixed missing keys: {GEMINI_MISSING_KEYS_FIXED_COUNT} ({fix_success_rate:.2%} of missing keys)")
    logger.info(f"OpenAI fallbacks: {GEMINI_OPENAI_FALLBACK_COUNT} ({fallback_rate:.2%} of total calls)")
    logger.info(f"Hash verifications performed: {HASH_VERIFICATION_COUNT} ({hash_verification_rate:.2%} of total calls)")
    logger.info(f"Hash verification failures: {HASH_VERIFICATION_FAILED_COUNT} ({hash_failure_rate:.2%} of verifications)")
    logger.info("==================================================")
