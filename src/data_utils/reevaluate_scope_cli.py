"""
reevaluate_scope_cli.py - Healthcare Domain Scope Classification Tool

This script evaluates healthcare scenarios and classifies them into scope categories (S0-S3)
based on how closely they relate to the healthcare domain. It uses LLM evaluators (OpenAI or Gemini)
to determine the appropriate scope category for each scenario.

Scope Categories:
- S0: Core clinical, diagnostics, prescriptions - direct healthcare content
- S1: Allied but integral healthcare topics (insurance, regulations, etc.)
- S2: Tangential advice with some healthcare overlap
- S3: Non-health domain where answering could mislead or cause harm

The script reads scenarios from a JSONL file, evaluates each one, and outputs a new JSONL file
with all original fields preserved plus an added "scope" field containing the classification.

Usage:
  python reevaluate_scope_cli.py [input_file] [--model MODEL] [--evaluator {openai,gemini}]
"""

import argparse
import json
import os
import re
import logging
import sys
import time
import asyncio
import concurrent.futures
import functools
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable

# Import OpenAI library - ensure it is installed (pip install openai)
import openai
from openai import OpenAI

# Try to import Gemini library - ensure it is installed (pip install google-generativeai)
try:
    import google.generativeai as genai
    has_gemini = True
except ImportError:
    has_gemini = False
    print("Warning: google-generativeai not found. Will only be able to use OpenAI evaluator.")

# Try to import dotenv - should be available in the Poetry environment
try:
    from dotenv import load_dotenv
    has_dotenv = True
except ImportError:
    has_dotenv = False
    print("Warning: python-dotenv not found. Will rely on environment variables only.")

# Import the centralized scope definitions
from src.scope_definitions import get_scope_prompt_for_text
from src.utils.gemini_api_wrapper import TrackedGenerativeModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Error tracking for Gemini
GEMINI_ERROR_COUNT = 0
GEMINI_TOTAL_CALLS = 0
GEMINI_ERROR_THRESHOLD = 0.05  # 5% error threshold

# --- Configuration & Rubric ---
DEFAULT_INPUT_FILE = os.path.join(os.path.dirname(__file__), '../../data/final-2025-05-06/benchmark_100_tiered_o3.jsonl')
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EVALUATOR = "gemini"  # Make Gemini the default
GEMINI_EVAL_MODEL = "gemini-2.0-flash"  # Gemini model to use
DEFAULT_BATCH_SIZE = 20  # Default number of prompts to process in parallel
DEFAULT_MAX_WORKERS = 20  # Default number of worker threads

# Symptom detection prompt
SYMPTOM_Q = (
    "Return Y if the text asks about a medical symptom, diagnosis, "
    "treatment or medication; otherwise N.\n\nText:\n"
)

# Thread pool for running synchronous Gemini API calls
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS)

class ScopeEvaluationError(Exception):
    """Base exception for irrecoverable errors in scope evaluation."""
    pass

class GeminiErrorRateExceeded(ScopeEvaluationError):
    """Exception raised when Gemini error rate exceeds the threshold."""
    pass

class LLMResponseParsingError(ScopeEvaluationError):
    """Exception raised when LLM response cannot be parsed correctly."""
    pass

class LLMAPIError(ScopeEvaluationError):
    """Exception raised when there's an error calling the LLM API."""
    pass

def track_gemini_error():
    """
    Track a Gemini API error and check if the error rate exceeds the threshold.
    Raises GeminiErrorRateExceeded if the threshold is exceeded.
    """
    global GEMINI_ERROR_COUNT, GEMINI_TOTAL_CALLS
    GEMINI_ERROR_COUNT += 1
    GEMINI_TOTAL_CALLS += 1

    # Only check error rate after we have a meaningful sample size
    if GEMINI_TOTAL_CALLS >= 20:
        error_rate = GEMINI_ERROR_COUNT / GEMINI_TOTAL_CALLS
        if error_rate > GEMINI_ERROR_THRESHOLD:
            logger.error(f"Gemini error rate {error_rate:.2%} exceeds threshold {GEMINI_ERROR_THRESHOLD:.2%}")
            logger.error(f"Errors: {GEMINI_ERROR_COUNT}, Total calls: {GEMINI_TOTAL_CALLS}")
            raise GeminiErrorRateExceeded(
                f"Gemini error rate {error_rate:.2%} exceeds threshold {GEMINI_ERROR_THRESHOLD:.2%}. "
                f"Errors: {GEMINI_ERROR_COUNT}, Total calls: {GEMINI_TOTAL_CALLS}"
            )

def track_gemini_success():
    """Track a successful Gemini API call."""
    global GEMINI_TOTAL_CALLS
    GEMINI_TOTAL_CALLS += 1

async def run_in_thread(func: Callable, *args, **kwargs):
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_pool,
        functools.partial(func, *args, **kwargs)
    )

def configure_gemini(api_key):
    """Configure the Gemini API with the API key."""
    if not has_gemini:
        logger.error("google-generativeai package not installed. Cannot use Gemini evaluator.")
        return False

    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        return False

# --- Load API keys from available sources ---
def load_api_keys():
    """Try to load API keys from various sources"""
    # First check environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    # Try to load from .env file if dotenv is available
    if has_dotenv:
        # Check project root .env
        project_root = Path(__file__).resolve().parents[2]  # Go up two levels from script
        env_paths = [
            project_root / ".env",
            project_root / ".env.local",
            Path.home() / ".env",  # Home directory
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment from: {env_path}")
                if not openai_api_key:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not gemini_api_key:
                    gemini_api_key = os.environ.get("GEMINI_API_KEY")

                # If we have both keys, we can stop looking
                if openai_api_key and gemini_api_key:
                    break

    # Could add more sources here (config files, etc.)
    return openai_api_key, gemini_api_key

# Use the centralized scope system prompt from scope_definitions.py

def get_llm_scope_with_openai(prompt_text: str, model_name: str, client) -> str:
    """
    Calls the OpenAI LLM to get the scope for a given prompt.
    Raises LLMResponseParsingError if the response cannot be parsed.
    Raises LLMAPIError if there's an error calling the OpenAI API.
    """
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # Get the system prompt with the text inserted
            system_prompt = get_scope_prompt_for_text(prompt_text)

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0,
                max_tokens=20  # Increased to accommodate JSON response
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                result = json.loads(content)
                scope = result.get("scope", "").upper()
                if scope in ['S0', 'S1', 'S2', 'S3']:
                    return scope
            except json.JSONDecodeError:
                # If not valid JSON, try to extract scope directly
                if 'S0' in content.upper():
                    return 'S0'
                elif 'S1' in content.upper():
                    return 'S1'
                elif 'S2' in content.upper():
                    return 'S2'
                elif 'S3' in content.upper():
                    return 'S3'

            # If we get here, couldn't parse a valid scope
            logger.warning(f"OpenAI attempt {attempt + 1}: Returned an invalid response '{content}' for prompt: '{prompt_text[:50]}...'.")

            # If this is not the last attempt, retry
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue

            # If this is the last attempt, raise an exception
            error_msg = f"Failed to parse a valid scope from OpenAI response after {max_retries} attempts. Last response: '{content}'"
            logger.error(error_msg)
            raise LLMResponseParsingError(error_msg)

        except openai.OpenAIError as e:
            logger.error(f"OpenAI attempt {attempt + 1}: API error: {e} for prompt: '{prompt_text[:50]}...'.")

            # If this is not the last attempt, retry
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue

            # If this is the last attempt, raise an exception
            error_msg = f"OpenAI API error after {max_retries} attempts: {e}"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

        except Exception as e:
            # For any other unexpected errors
            error_msg = f"Unexpected error calling OpenAI API: {e}"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

def get_llm_scope_with_gemini(prompt_text: str) -> str:
    """
    Calls the Gemini LLM to get the scope for a given prompt.
    Includes retry logic and error handling.
    Raises LLMResponseParsingError if the response cannot be parsed.
    Raises LLMAPIError if there's an error calling the Gemini API.
    """
    if not has_gemini:
        error_msg = "Gemini not available. Make sure google-generativeai is installed."
        logger.error(error_msg)
        raise LLMAPIError(error_msg)

    # Get the system prompt with the text inserted
    system_prompt = get_scope_prompt_for_text(prompt_text)

    # Integrate retry logic here
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # Use Gemini-2.0-flash model
            model = TrackedGenerativeModel(GEMINI_EVAL_MODEL)

            # Send the system prompt directly
            response = model.generate_content(system_prompt)
            content = response.text.strip()

            # Try to parse JSON response
            try:
                result = json.loads(content)
                scope = result.get("scope", "").upper()
                if scope in ['S0', 'S1', 'S2', 'S3']:
                    track_gemini_success()
                    return scope
            except json.JSONDecodeError:
                # If not valid JSON, try to extract scope directly
                content_upper = content.upper()
                if 'S0' in content_upper:
                    scope = 'S0'
                    track_gemini_success()
                    return scope
                elif 'S1' in content_upper:
                    scope = 'S1'
                    track_gemini_success()
                    return scope
                elif 'S2' in content_upper:
                    scope = 'S2'
                    track_gemini_success()
                    return scope
                elif 'S3' in content_upper:
                    scope = 'S3'
                    track_gemini_success()
                    return scope
                else:
                    # If no valid scope found, log and continue to next attempt
                    logger.warning(f"Gemini attempt {attempt + 1}: Returned invalid content: '{content}' for prompt: '{prompt_text[:50]}...'.")

                    # If this is not the last attempt, retry
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        continue

                    # If this is the last attempt, raise an exception
                    error_msg = f"Failed to parse a valid scope from Gemini response after {max_retries} attempts. Last response: '{content}'"
                    logger.error(error_msg)
                    track_gemini_error()
                    raise LLMResponseParsingError(error_msg)

        except LLMResponseParsingError:
            # Re-raise the exception
            raise

        except Exception as e:
            logger.error(f"Gemini attempt {attempt + 1}: Error: {e}.")

            # If this is not the last attempt, retry
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue

            # If this is the last attempt, raise an exception
            error_msg = f"Gemini API error after {max_retries} attempts: {e}"
            logger.error(error_msg)
            track_gemini_error()
            raise LLMAPIError(error_msg)

        # Sleep before retrying
        if attempt < max_retries - 1:
            time.sleep(retry_delay)

def get_llm_scope(prompt_text: str, model_name: str, client=None, evaluator="openai") -> str:
    """
    Gets the scope for a given prompt using the specified evaluator.
    Raises ScopeEvaluationError if there's an error evaluating the scope.
    """
    if evaluator == "gemini":
        return get_llm_scope_with_gemini(prompt_text)
    else:  # Default to OpenAI
        return get_llm_scope_with_openai(prompt_text, model_name, client)

def classify_prompt_scope(prompt_text: str, model_name: str, client=None, evaluator="openai") -> str:
    """
    Two-step classification process for prompt scope:
    1. First call - standard scope classification (S0-S3)
    2. Second call - symptom detection override (sets to S0 if medical symptom detected)

    Args:
        prompt_text: The prompt text to classify
        model_name: The model to use for OpenAI evaluations
        client: The OpenAI client instance
        evaluator: Which evaluator to use ('openai' or 'gemini')

    Returns:
        The final scope classification (S0, S1, S2, or S3)
    """
    # 1️⃣ first call – rubric S0‑S3
    scope = get_llm_scope(prompt_text, model_name, client, evaluator)

    # 2️⃣ second call – symptom yes/no
    if scope != "S0":  # save a call when already S0
        if evaluator == "gemini" and has_gemini:
            # Use Gemini for symptom detection
            try:
                model = TrackedGenerativeModel(GEMINI_EVAL_MODEL)
                response = model.generate_content(SYMPTOM_Q + prompt_text)
                is_symptom = response.text.strip().upper().startswith("Y")
                track_gemini_success()
            except Exception as e:
                logger.error(f"Error using Gemini for symptom detection: {e}")
                track_gemini_error()
                is_symptom = False  # Default to no symptom on error
        else:
            # Use OpenAI for symptom detection
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.0,
                    max_tokens=1,
                    messages=[{
                        "role": "system",
                        "content": SYMPTOM_Q + prompt_text
                    }],
                )
                is_symptom = response.choices[0].message.content.strip().upper().startswith("Y")
            except Exception as e:
                logger.error(f"Error using OpenAI for symptom detection: {e}")
                is_symptom = False  # Default to no symptom on error

        if is_symptom:
            scope = "S0"  # symptom override

    return scope

async def batch_evaluate_scope_with_gemini(prompts: List[str], max_concurrency: int = DEFAULT_BATCH_SIZE) -> List[str]:
    """
    Evaluate a batch of prompts with Gemini API in parallel.
    Uses a semaphore to limit concurrency and prevent rate limiting.

    Args:
        prompts: List of prompt texts to evaluate.
        max_concurrency: Maximum number of concurrent API calls.

    Returns:
        List of scope classifications (S0-S3) for each prompt.

    Raises:
        ScopeEvaluationError: If there's an error evaluating the scope.
    """
    if not has_gemini:
        error_msg = "Gemini not available. Make sure google-generativeai is installed."
        logger.error(error_msg)
        raise LLMAPIError(error_msg)

    # Create a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate_with_semaphore(prompt_text: str) -> str:
        """Evaluate a single prompt with semaphore control."""
        async with semaphore:
            # First, get the initial scope classification
            async def get_initial_scope():
                # Define a synchronous function to make the Gemini API call
                def make_gemini_call():
                    # Get the system prompt with the text inserted
                    system_prompt = get_scope_prompt_for_text(prompt_text)

                    # Use Gemini-2.0-flash model
                    model = TrackedGenerativeModel(GEMINI_EVAL_MODEL)

                    # Send the system prompt directly
                    response = model.generate_content(system_prompt)
                    return response.text.strip()

                # Integrate retry logic here
                max_retries = 3
                retry_delay = 5

                for attempt in range(max_retries):
                    try:
                        # Run the synchronous function in a thread
                        content = await run_in_thread(make_gemini_call)

                        if content:
                            try:
                                # Try to parse JSON response
                                result = json.loads(content)
                                scope = result.get("scope", "").upper()
                                if scope in ['S0', 'S1', 'S2', 'S3']:
                                    track_gemini_success()
                                    return scope
                            except json.JSONDecodeError:
                                # If not valid JSON, try to extract scope directly
                                content_upper = content.upper()
                                if 'S0' in content_upper:
                                    scope = 'S0'
                                    track_gemini_success()
                                    return scope
                                elif 'S1' in content_upper:
                                    scope = 'S1'
                                    track_gemini_success()
                                    return scope
                                elif 'S2' in content_upper:
                                    scope = 'S2'
                                    track_gemini_success()
                                    return scope
                                elif 'S3' in content_upper:
                                    scope = 'S3'
                                    track_gemini_success()
                                    return scope
                                else:
                                    # If no valid scope found, log and continue to next attempt
                                    logger.warning(f"Gemini attempt {attempt + 1}: Returned invalid content: '{content}' for prompt: '{prompt_text[:50]}...'.")

                                    # If this is not the last attempt, retry
                                    if attempt < max_retries - 1:
                                        logger.info(f"Retrying in {retry_delay} seconds...")
                                        await asyncio.sleep(retry_delay)
                                        continue

                                    # If this is the last attempt, raise an exception
                                    error_msg = f"Failed to parse a valid scope from Gemini response after {max_retries} attempts. Last response: '{content}'"
                                    logger.error(error_msg)
                                    track_gemini_error()
                                    raise LLMResponseParsingError(error_msg)

                    except LLMResponseParsingError:
                        # Re-raise the exception
                        raise

                    except Exception as e:
                        logger.error(f"Gemini attempt {attempt + 1}: Error: {e}.")

                        # If this is not the last attempt, retry
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            continue

                        # If this is the last attempt, raise an exception
                        error_msg = f"Gemini API error after {max_retries} attempts: {e}"
                        logger.error(error_msg)
                        track_gemini_error()
                        raise LLMAPIError(error_msg)

                    # Sleep before retrying
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)

                # Should never reach here due to exceptions above
                raise LLMAPIError("Unexpected error in get_initial_scope")

            # Get the initial scope
            scope = await get_initial_scope()

            # If already S0, no need for second check
            if scope == "S0":
                return scope

            # Second call - check for medical symptoms
            async def check_for_symptoms():
                def make_symptom_check_call():
                    try:
                        model = TrackedGenerativeModel(GEMINI_EVAL_MODEL)
                        response = model.generate_content(SYMPTOM_Q + prompt_text)
                        is_symptom = response.text.strip().upper().startswith("Y")
                        track_gemini_success()
                        return is_symptom
                    except Exception as e:
                        logger.error(f"Error using Gemini for symptom detection: {e}")
                        track_gemini_error()
                        return False  # Default to no symptom on error

                return await run_in_thread(make_symptom_check_call)

            # Check for symptoms and override to S0 if found
            is_symptom = await check_for_symptoms()
            if is_symptom:
                return "S0"  # symptom override

            # Return the original scope if no symptoms detected
            return scope

    # Create tasks for all prompts
    tasks = [evaluate_with_semaphore(prompt) for prompt in prompts]

    # Run all tasks concurrently with semaphore control
    logger.info(f"Batch evaluating {len(prompts)} prompts with concurrency {max_concurrency}...")
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    logger.info(f"Completed batch evaluation of {len(prompts)} prompts in {end_time - start_time:.2f} seconds "
                f"(avg: {(end_time - start_time) / max(1, len(prompts)):.2f}s per prompt)")

    # Process results and handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error evaluating prompt {i}: {result}")
            raise result  # Re-raise the exception
        else:
            processed_results.append(result)

    return processed_results

async def process_batch_with_gemini(batch_items, batch_prompts, output_file, scope_counts, batch_size):
    """Process a batch of prompts with Gemini in parallel."""
    try:
        # Evaluate all prompts in the batch in parallel
        batch_scopes = await batch_evaluate_scope_with_gemini(batch_prompts, max_concurrency=batch_size)

        # Process the results
        with open(output_file, 'a', encoding='utf-8') as outfile:
            for item, scope in zip(batch_items, batch_scopes):
                # Update scope counts
                scope_counts[scope] += 1

                # Create a new record with all original fields plus the scope
                record = item.copy()  # Preserve all original fields
                record["scope"] = scope  # Add or update the scope field

                # Write to output file
                outfile.write(json.dumps(record) + '\n')

        return len(batch_items)
    except ScopeEvaluationError as e:
        error_msg = f"Error: Scope evaluation error: {e}"
        logger.error(error_msg)
        print(error_msg)
        print(f"Failed on batch with {len(batch_prompts)} prompts")
        print("Stopping processing due to irrecoverable evaluation error.")
        sys.exit(1)  # Exit with error code

async def async_main():
    """Async main function to process the input file."""
    parser = argparse.ArgumentParser(description="Evaluate healthcare domain scope (S0-S3) in a JSONL file using an LLM.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input JSONL file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use for evaluation (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--evaluator",
        choices=["openai", "gemini"],
        default=DEFAULT_EVALUATOR,
        help=f"LLM evaluator to use (default: {DEFAULT_EVALUATOR})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of prompts to process in parallel (default: {DEFAULT_BATCH_SIZE})"
    )
    args = parser.parse_args()

    input_file_path = args.input_file
    model_name = args.model
    evaluator = args.evaluator
    batch_size = args.batch_size

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    base, ext = os.path.splitext(input_file_path)
    output_file_path = f"{base}-with-scope-{evaluator}{ext}"

    scope_counts = Counter()
    processed_items = 0

    # Initialize API clients based on evaluator
    openai_api_key, gemini_api_key = load_api_keys()
    client = None

    if evaluator == "openai":
        if not openai_api_key:
            print("Error: OPENAI_API_KEY not found in environment or .env file.")
            print("Please set the OPENAI_API_KEY environment variable or add it to a .env file.")
            return

        try:
            client = OpenAI(api_key=openai_api_key)
            # Test connection
            _ = client.models.list()
            print("Successfully connected to OpenAI API.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return
    else:  # evaluator == "gemini"
        if not gemini_api_key:
            print("Error: GEMINI_API_KEY not found in environment or .env file.")
            print("Please set the GEMINI_API_KEY environment variable or add it to a .env file.")
            return

        if not has_gemini:
            print("Error: google-generativeai package not installed.")
            print("Please install it with 'pip install google-generativeai'")
            return

        if not configure_gemini(gemini_api_key):
            print("Error configuring Gemini API.")
            return

        print(f"Successfully configured Gemini API using model: {GEMINI_EVAL_MODEL}")

    print(f"Processing input file: {input_file_path}")
    print(f"Using evaluator: {evaluator}")
    if evaluator == "openai":
        print(f"Using OpenAI model: {model_name}")
    else:
        print(f"Using Gemini model: {GEMINI_EVAL_MODEL}")
    print(f"Output will be written to: {output_file_path}")

    # Create the output file (empty it if it exists)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        pass  # Just create/empty the file

    # Read all items from the input file
    all_items = []
    all_prompts = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line)
                prompt_text = item.get("prompt")

                if not prompt_text:
                    logger.warning(f"Skipping item with no 'prompt': {item}")
                    continue  # Skip items without a prompt

                # Clean the prompt text from any annotations
                cleaned_prompt_text_intermediate = re.sub(r'\s*\(Tier\s*[A-Ca-c]\)\s*$', '', prompt_text, flags=re.IGNORECASE).strip()

                # Further clean specific unicode characters
                cleaned_prompt_text = cleaned_prompt_text_intermediate.replace('\u2019', "'")  # Right single quotation mark
                # Add more replacements if needed

                all_items.append(item)
                all_prompts.append(cleaned_prompt_text)

            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line.strip()}")
                continue  # Skip invalid JSON lines

    total_items = len(all_items)
    logger.info(f"Loaded {total_items} items from {input_file_path}")

    if evaluator == "gemini":
        # Process in batches for Gemini
        processed_items = 0

        # Process in batches
        for i in range(0, total_items, batch_size):
            batch_items = all_items[i:i+batch_size]
            batch_prompts = all_prompts[i:i+batch_size]

            logger.info(f"Processing batch {i//batch_size + 1}/{(total_items-1)//batch_size + 1} ({len(batch_items)} items)")

            # Process the batch
            batch_processed = await process_batch_with_gemini(
                batch_items, batch_prompts, output_file_path, scope_counts, batch_size
            )

            processed_items += batch_processed
            print(f"Processed {processed_items} items...")
    else:
        # For OpenAI, process one by one (could be parallelized in the future)
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for i, (item, cleaned_prompt_text) in enumerate(zip(all_items, all_prompts)):
                try:
                    scope = classify_prompt_scope(cleaned_prompt_text, model_name, client, evaluator)
                    scope_counts[scope] += 1

                    # Create a new record with all original fields plus the scope
                    record = item.copy()  # Preserve all original fields
                    record["scope"] = scope  # Add or update the scope field

                    outfile.write(json.dumps(record) + '\n')

                    processed_items += 1
                    if processed_items % 10 == 0:
                        print(f"Processed {processed_items} items...")
                except ScopeEvaluationError as e:
                    error_msg = f"Error: Scope evaluation error: {e}"
                    logger.error(error_msg)
                    print(error_msg)
                    print(f"Failed on prompt: '{cleaned_prompt_text[:100]}...'")
                    print("Stopping processing due to irrecoverable evaluation error.")
                    sys.exit(1)  # Exit with error code

    print(f"\nProcessing complete. {processed_items} items evaluated for scope.")

    # Print summary table
    print("\n--- Scope Summary ---")

    print(f"{'Scope':<6} | {'Count':<10}")
    print("-" * 20)

    total_count = 0

    for scope_val in ['S0', 'S1', 'S2', 'S3']: # Ensure consistent order
        count = scope_counts.get(scope_val, 0)
        print(f"{scope_val:<6} | {count:<10}")
        total_count += count

    # For any other scopes that might have been (incorrectly) generated
    other_scopes = [s for s in scope_counts.keys() if s not in ['S0', 'S1', 'S2', 'S3']]
    for scope_val in other_scopes:
        count = scope_counts.get(scope_val, 0)
        print(f"{scope_val:<6} | {count:<10}")
        total_count += count

    print("-" * 20)
    print(f"{'Total':<6} | {total_count:<10}")

def main():
    """Entry point for the script."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
