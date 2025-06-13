#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
openai_utils.py - OpenAI API interactions for ArGen dataset generator
====================================================================
Contains utilities for OpenAI API calls, token counting, and backoff strategies
"""

import time
import json
import re
import tiktoken
import openai
from openai import RateLimitError, OpenAIError, OpenAI
from typing import Dict, List, Optional, Callable, Any

# Determine if we're running as a standalone package or as part of the parent project
try:
    import argen.data.utils
    STANDALONE_MODE = False
except ImportError:
    STANDALONE_MODE = True

# Import with the appropriate style based on mode
if STANDALONE_MODE:
    # Standalone mode - use direct imports
    from config import log
else:
    # Integrated mode - use relative imports
    from .config import log

# OpenAI client initialization
def init_openai_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Initialize the OpenAI client with the given API key."""
    if not api_key:
        api_key = openai.api_key
        if not api_key:
            log.error("OpenAI API key not provided and not found in environment.")
            return None

    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        log.error(f"Failed to initialize OpenAI client: {e}")
        return None

# Token counting utilities
def count_tokens(text: str, model_name: str) -> int:
    """Count the number of tokens in a text string for a specific model."""
    try:
        # Import MODEL_ALIASES from config
        if STANDALONE_MODE:
            from config import MODEL_ALIASES
        else:
            from .config import MODEL_ALIASES

        # Check if model has an alias for tokenization
        if model_name in MODEL_ALIASES:
            tokenizer_model = MODEL_ALIASES[model_name]
            log.debug(f"Using tokenizer for {tokenizer_model} instead of {model_name}")
            model_name = tokenizer_model

        # Use the model name to get the right encoding
        enc = tiktoken.encoding_for_model(model_name)
        token_count = len(enc.encode(text))
        return token_count
    except KeyError:
        # Fallback for models not recognized by tiktoken
        log.warning(f"Tiktoken encoding not found for {model_name}. Using approximate count (chars/4).")
        return len(text) // 4

def compute_max_tokens(system_prompt: str, user_prompt: str, model_limit: Dict[str, int]) -> int:
    """Compute the maximum tokens available for generation given the input."""
    model_context_window = model_limit.get('context_window', 4096)
    model_max_output = model_limit.get('max_output_tokens', 2048)

    # Count tokens used by prompts
    used = count_tokens(system_prompt + user_prompt, model_limit.get('model_name', 'gpt-3.5-turbo'))

    # Ensure at least 64 tokens, reserve 100 for safety margin
    available_for_output = max(64, model_context_window - used - 100)  # Reserve 100 for safety

    # Cap the available tokens by the model's maximum output tokens
    max_tok = min(available_for_output, model_max_output)

    log.debug(f"compute_max_tokens: Context={model_context_window}, MaxOutput={model_max_output}, "
             f"Used={used}, Available={available_for_output}, Capped MaxTokens={max_tok}")
    return max_tok

# Backoff strategies for API calls
def call_with_backoff(fn: Callable, *args: Any, max_retries: int = 5,
                     initial_delay: float = 1.0, **kwargs: Any) -> Any:
    """Call a function with exponential backoff for handling rate limits and errors."""
    delay = initial_delay
    for i in range(max_retries):
        try:
            log.debug(f"call_with_backoff: attempt {i+1}/{max_retries} for function {fn.__name__}")
            return fn(*args, **kwargs)
        except (RateLimitError, OpenAIError) as e:
            log.warning(f"{fn.__name__} error: {e}. Retrying after {delay:.2f} seconds.")
            time.sleep(delay)
            delay *= 2
        except Exception as e:  # Catch other potential errors
            log.error(f"Unexpected error in {fn.__name__} on attempt {i+1}: {e}")
            # Decide if retry makes sense for other errors, here we retry
            time.sleep(delay)
            delay *= 2

    log.error(f"{fn.__name__} failed after {max_retries} attempts. Raising last exception.")
    # Reraise the exception after max retries
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.error(f"Final attempt failed for {fn.__name__}: {e}")
        raise e

# Generation function with batch handling
def generate_chat_completion(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    model_limit: Optional[Dict[str, int]] = None,
    max_retries: int = 5,
    initial_delay: float = 1.0
) -> Optional[str]:
    """Generate a chat completion with proper error handling and token management."""
    # Import MODEL_ALIASES from config
    if STANDALONE_MODE:
        from config import get_model_limits, MODEL_ALIASES
    else:
        from .config import get_model_limits, MODEL_ALIASES

    # Check if model has an alias for API calls - only for gpt-4.1 models
    # gpt-4o-mini now exists natively in the OpenAI API, so we don't need to alias it
    original_model_name = model_name
    if model_name in MODEL_ALIASES and model_name.startswith("gpt-4.1"):
        model_name = MODEL_ALIASES[model_name]
        log.info(f"Using model {model_name} as an alias for {original_model_name}")

    if model_limit is None:
        model_limit = get_model_limits(model_name)

    if max_tokens is None:
        max_tokens = compute_max_tokens(system_prompt, user_prompt, model_limit)

    for attempt in range(1, max_retries + 1):
        log.debug(f"generate_chat_completion: attempt {attempt}/{max_retries}")
        try:
            # Use max_completion_tokens for o3 models, max_tokens for others
            # Also handle temperature parameter for o3 models
            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_retries": max_retries,
                "initial_delay": initial_delay
            }

            if model_name.startswith("o3"):
                api_params["max_completion_tokens"] = max_tokens
                # o3 models don't support temperature parameter
            else:
                api_params["max_tokens"] = max_tokens
                api_params["temperature"] = temperature

            resp = call_with_backoff(
                client.chat.completions.create,
                **api_params
            )

            if not resp or not resp.choices or not resp.choices[0].message:
                log.warning(f"Empty response received on attempt {attempt}")
                time.sleep(initial_delay * (2**(attempt-1)))
                continue

            content = resp.choices[0].message.content.strip()

            # Use regex to extract JSON array from the response
            # This will find the first occurrence of a JSON array (starting with '[' and ending with ']')
            array_match = re.search(r"^[\s\S]*?(\[[\s\S]*\])[\s\S]*$", content)
            if array_match:
                content = array_match.group(1).strip()

            return content

        except json.JSONDecodeError as e:
            log.error(f"JSONDecodeError on attempt {attempt}: {e}")
            log.debug(f"Raw content: {resp.choices[0].message.content[:100]}...")
            if attempt == max_retries:
                raise ValueError(f"Failed to parse JSON after {max_retries} attempts") from e
            time.sleep(initial_delay * (2**(attempt-1)))

        except (RateLimitError, OpenAIError) as e:
            log.error(f"OpenAI API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise ValueError(f"OpenAI API error persisted after {max_retries} attempts.") from e
            # Let call_with_backoff handle the retry delay for these errors

        except Exception as e:
            log.error(f"Unexpected error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise ValueError(f"Unexpected error persisted after {max_retries} attempts.") from e
            time.sleep(initial_delay * (2**(attempt-1)))

    log.error(f"Failed to generate completion after {max_retries} attempts.")
    return None