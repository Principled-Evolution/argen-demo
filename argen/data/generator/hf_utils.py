#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hf_utils.py - HuggingFace model utilities for ArGen dataset generator
====================================================================
Contains utilities for local HuggingFace model initialization and generation
"""

import torch
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the import helper
try:
    # Try relative import first
    from .import_helper import STANDALONE_MODE, get_import
except ImportError:
    # Fall back to direct import
    try:
        from import_helper import STANDALONE_MODE, get_import
    except ImportError:
        # Last resort: define our own standalone mode detection
        try:
            import argen.data.utils
            STANDALONE_MODE = False
        except ImportError:
            STANDALONE_MODE = True

        # Define a simple get_import function
        import importlib

# Export these functions for use in other modules
__all__ = ['init_hf_model', 'generate_hf_completion', 'get_hf_model_name']

def get_import(module_name):
    if STANDALONE_MODE:
        return importlib.import_module(module_name)
    else:
        return importlib.import_module(f".{module_name}", package="argen.data.generator")

# Import config module using the helper
try:
    config_module = get_import('config')
    log = config_module.log
except Exception as e:
    # Fallback to direct import if helper fails
    if STANDALONE_MODE:
        from config import log
    else:
        from .config import log

# Global model variables
_hf_model = None
_hf_tokenizer = None
_hf_device = None
_hf_model_name = None

def get_hf_model_name():
    """
    Get the name of the currently initialized HuggingFace model.

    Returns:
        str: The name of the initialized model, or None if no model is initialized
    """
    global _hf_model_name
    return _hf_model_name

def init_hf_model(model_name: str) -> bool:
    """
    Initialize a HuggingFace model for generation.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global _hf_model, _hf_tokenizer, _hf_device, _hf_model_name

    log.info(f"Initializing HuggingFace model: {model_name}")
    try:
        # Set device
        _hf_device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {_hf_device}")

        # Load tokenizer with special handling for medalpaca
        log.info(f"Loading tokenizer for {model_name}...")

        # Special handling for medalpaca models
        if "medalpaca" in model_name.lower():
            log.info("Using special tokenizer configuration for medalpaca model")
            try:
                # Try with legacy=True for medalpaca
                _hf_tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
                log.info("Successfully loaded medalpaca tokenizer with legacy=True")
            except Exception as e:
                log.warning(f"Failed to load medalpaca tokenizer with legacy=True: {e}")
                # Fallback to LlamaTokenizer which is known to work with medalpaca
                try:
                    from transformers import LlamaTokenizer
                    _hf_tokenizer = LlamaTokenizer.from_pretrained(model_name)
                    log.info("Successfully loaded medalpaca tokenizer using LlamaTokenizer")
                except Exception as e2:
                    log.warning(f"Failed to load medalpaca tokenizer with LlamaTokenizer: {e2}")
                    # Last resort: try AutoTokenizer with use_fast=False
                    _hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    log.info("Loaded medalpaca tokenizer with use_fast=False as fallback")
        else:
            # Standard tokenizer loading for other models
            _hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if needed
        if _hf_tokenizer.pad_token is None:
            _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
            log.info(f"Set pad_token to eos_token: {_hf_tokenizer.pad_token}")

        # Load model
        log.info(f"Loading model {model_name} on {_hf_device}...")
        model_dtype = torch.bfloat16 if _hf_device == "cuda" else torch.float32
        log.info(f"Using dtype: {model_dtype}")

        _hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        _hf_model_name = model_name
        log.info(f"Model loaded successfully. Running quick test...")

        # Test model with simple prompt
        test_result = generate_hf_completion(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello, how are you?",
            temperature=0.7
        )

        if test_result:
            log.info(f"HuggingFace model initialized successfully: {model_name}")
            log.info(f"Test response: {test_result[:100]}...")
            return True
        else:
            log.error(f"HuggingFace model initialization test failed")
            return False

    except Exception as e:
        log.error(f"Failed to initialize HuggingFace model: {e}")
        return False

def generate_hf_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 512,
    model_limit: Optional[Dict[str, int]] = None
) -> Optional[str]:
    """
    Generate a completion using a local HuggingFace model.

    Args:
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        temperature: Temperature for generation
        max_tokens: Maximum number of tokens to generate
        model_limit: Token limits for the model (not used for HF models)

    Returns:
        str: Generated text or None if generation failed
    """
    global _hf_model, _hf_tokenizer, _hf_device, _hf_model_name

    if _hf_model is None or _hf_tokenizer is None:
        log.error("HuggingFace model not initialized.")
        return None

    try:
        # Format prompt based on model type
        # Try to detect the model type and use the appropriate format
        model_name_lower = _hf_model_name.lower()

        if "medalpaca" in model_name_lower:
            # Medalpaca format (based on Alpaca)
            # For MedAlpaca, we're now using a simplified format where the system_prompt already contains
            # the instruction format, so we just pass it through
            formatted_prompt = system_prompt
            log.info("Using simplified Medalpaca instruction format")
        elif "meditron-7b-chat" in model_name_lower:
            # Special format for meditron-7b-chat
            formatted_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
            log.info("Using Meditron-7b-chat format")
        elif "llama" in model_name_lower:
            # Llama 2/3 format
            formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            log.info("Using Llama-style chat format")
        elif "mistral" in model_name_lower:
            # Mistral format
            formatted_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
            log.info("Using Mistral-style chat format")
        elif "meditron" in model_name_lower:
            # Meditron format (similar to Llama)
            formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            log.info("Using Meditron-style chat format (Llama-based)")
        elif "phi" in model_name_lower:
            # Phi format
            formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
            log.info("Using Phi-style chat format")
        else:
            # Generic format as fallback
            formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
            log.info("Using generic chat format (fallback)")

        log.info(f"Prompt format: {formatted_prompt[:100]}...")

        log.info(f"Tokenizing prompt for {_hf_model_name}...")

        # Tokenize input
        inputs = _hf_tokenizer(formatted_prompt, return_tensors="pt").to(_hf_device)
        input_token_length = inputs.input_ids.shape[1]

        # Check if input is too long and log a warning
        if input_token_length > 1024:
            log.warning(f"Input prompt is {input_token_length} tokens, which is quite long. This may affect generation quality.")

        # Set model context length to 2048 tokens by default for LLaMA/Alpaca 7-B models
        MODEL_CTX = 2048

        # Adjust max_tokens if needed to avoid warnings
        model_max_length = getattr(_hf_tokenizer, "model_max_length", MODEL_CTX)

        # Calculate safe max tokens with proper budgeting
        prompt_len = input_token_length
        safe_max = model_max_length - prompt_len

        # Use at least 128 tokens, at most 512, but stay within context window
        safe_max_tokens = max(128, min(512, safe_max))

        log.info(f"Model context length: {model_max_length}, Prompt length: {prompt_len}")
        log.info(f"Available token budget: {safe_max}, Using max_new_tokens: {safe_max_tokens}")

        # Special handling for Meditron-7b-chat (still be a bit more conservative)
        if "meditron-7b-chat" in _hf_model_name.lower():
            # Be more conservative with Meditron-7b-chat
            safe_max_tokens = min(safe_max_tokens, safe_max - 50)  # 50 token buffer
            log.info(f"Using conservative token buffer for Meditron-7b-chat: {safe_max_tokens}")

        if safe_max_tokens < max_tokens:
            log.info(f"Adjusted max_tokens from {max_tokens} to {safe_max_tokens} to fit within model context window")

        # Ensure we have a positive number of tokens
        if safe_max_tokens <= 0:
            log.warning(f"Calculated safe_max_tokens is {safe_max_tokens}, setting to minimum of 128")
            safe_max_tokens = 128

        log.info(f"Starting generation with temperature={temperature}, max_new_tokens={safe_max_tokens}...")

        # Generate with progress updates
        generation_start_time = time.time()

        # Generate with enhanced parameters to fight repetition
        generate_kwargs = dict(
            max_new_tokens=safe_max_tokens,
            do_sample=True,
            temperature=1.1,          # Higher temperature for more diversity
            top_p=0.92,               # Slightly higher top_p
            repetition_penalty=1.15,  # Penalize repetition
            no_repeat_ngram_size=4,   # Ban 4-gram repeats
            pad_token_id=_hf_tokenizer.pad_token_id
        )

        # For medalpaca, use even stronger anti-repetition settings
        if "medalpaca" in _hf_model_name.lower():
            generate_kwargs["temperature"] = 1.2
            generate_kwargs["repetition_penalty"] = 1.2
            log.info(f"Using enhanced anti-repetition settings for Medalpaca: temp={generate_kwargs['temperature']}, rep_penalty={generate_kwargs['repetition_penalty']}")

        log.info(f"Generation parameters: temperature={generate_kwargs['temperature']}, top_p={generate_kwargs['top_p']}, " +
                 f"repetition_penalty={generate_kwargs['repetition_penalty']}, no_repeat_ngram_size={generate_kwargs['no_repeat_ngram_size']}")

        # Use lower-level model.generate call for better control and performance
        tokenized = inputs.to(_hf_device)

        with torch.inference_mode():
            outputs = _hf_model.generate(**tokenized, **generate_kwargs)

        generation_time = time.time() - generation_start_time
        log.info(f"Generation completed in {generation_time:.2f} seconds")

        # Decode output
        log.info("Decoding output...")
        full_output = _hf_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response based on model type
        model_name_lower = _hf_model_name.lower()

        if "medalpaca" in model_name_lower:
            # For Medalpaca with our new format, the response is everything after "### Response"
            if "### Response" in full_output:
                assistant_response = full_output.split("### Response")[-1].strip()
                log.info("Extracted response using Medalpaca format")
            else:
                # Fallback if the expected format isn't found
                assistant_response = full_output.replace(formatted_prompt, "").strip()
                log.info("Used fallback response extraction for Medalpaca (prompt removal)")
        elif "llama" in model_name_lower or "mistral" in model_name_lower or "meditron" in model_name_lower:
            # For Llama/Mistral/Meditron, the response is everything after [/INST]
            if "[/INST]" in full_output:
                assistant_response = full_output.split("[/INST]")[-1].strip()
                log.info("Extracted response using Llama/Mistral format")
            else:
                # Fallback if the expected format isn't found
                assistant_response = full_output.replace(formatted_prompt, "").strip()
                log.info("Used fallback response extraction (prompt removal)")
        elif "phi" in model_name_lower:
            # For Phi, the response is after <|assistant|>
            if "<|assistant|>" in full_output:
                assistant_response = full_output.split("<|assistant|>")[-1].strip()
                log.info("Extracted response using Phi format")
            else:
                # Fallback
                assistant_response = full_output.replace(formatted_prompt, "").strip()
                log.info("Used fallback response extraction (prompt removal)")
        else:
            # Generic fallback - try different separators
            if "<|assistant|>" in full_output:
                assistant_response = full_output.split("<|assistant|>")[-1].strip()
            elif "[/INST]" in full_output:
                assistant_response = full_output.split("[/INST]")[-1].strip()
            elif "### Response:" in full_output:
                assistant_response = full_output.split("### Response:")[-1].strip()
            else:
                # Last resort fallback
                assistant_response = full_output.replace(formatted_prompt, "").strip()
                log.info("Used last resort fallback response extraction (prompt removal)")

        log.info(f"Extracted response: {assistant_response[:100]}...")

        # Try to extract JSON array if that's what we're expecting
        log.info("Extracting and validating JSON response...")

        # Special handling for medical models
        if "meditron-7b-chat" in _hf_model_name.lower() or "medalpaca" in _hf_model_name.lower():
            # First, try to clean up the response by removing common prefixes and suffixes
            cleaned_response = assistant_response.strip()

            # Remove common prefixes
            prefixes = [
                "Here's a JSON array of",
                "Here is a JSON array of",
                "Here are",
                "Sure, here's",
                "Sure, here is",
                "I'll generate",
                "Here's the JSON array",
                "Here is the JSON array",
                "I've generated",
                "Below is a JSON array",
                "Here are the healthcare questions:",
                "Here's a list of",
                "Here is a list of"
            ]
            for prefix in prefixes:
                if cleaned_response.lower().startswith(prefix.lower()):
                    cleaned_response = cleaned_response[len(prefix):].strip()

            # Remove common suffixes
            suffixes = [
                "Let me know if you need any adjustments.",
                "Let me know if you need anything else.",
                "I hope this helps!",
                "Is there anything else you'd like me to help with?",
                "Let me know if you need more scenarios.",
                "These questions cover various healthcare topics.",
                "I hope these questions are helpful for your healthcare scenario generation."
            ]
            for suffix in suffixes:
                if cleaned_response.lower().endswith(suffix.lower()):
                    cleaned_response = cleaned_response[:-len(suffix)].strip()

            model_name = "Meditron-7b-chat" if "meditron-7b-chat" in _hf_model_name.lower() else "Medalpaca"
            log.info(f"Cleaned response for {model_name}: {cleaned_response[:100]}...")

            # More aggressive cleaning for Medalpaca specifically
            if "medalpaca" in _hf_model_name.lower():
                # Remove any text before the first '[' character
                if '[' in cleaned_response:
                    cleaned_response = re.sub(r'^[\s\S]*?(\[)', r'\1', cleaned_response)
                    log.info(f"Removed text before first '[': {cleaned_response[:100]}...")

                # Remove any text after the last ']' character
                if ']' in cleaned_response:
                    cleaned_response = re.sub(r'(\])[\s\S]*$', r'\1', cleaned_response)
                    log.info(f"Removed text after last ']': {cleaned_response[:100]}...")

                # Fix common JSON formatting issues
                # Replace single quotes with double quotes
                cleaned_response = cleaned_response.replace("'", '"')
                # Fix missing commas between array items
                cleaned_response = re.sub(r'"\s*"', '", "', cleaned_response)
                # Fix trailing commas
                cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)

                log.info(f"Fixed JSON formatting issues: {cleaned_response[:100]}...")

            # Try to find a JSON array in the cleaned response
            # Use a more precise regex that ensures we get a complete array
            array_match = re.search(r"(\[\s*\".*?\"\s*(,\s*\".*?\"\s*)*\])", cleaned_response, re.DOTALL)
            if array_match:
                content = array_match.group(1).strip()
                # Validate JSON
                try:
                    json_data = json.loads(content)
                    if isinstance(json_data, list):
                        log.info(f"Successfully parsed JSON array with {len(json_data)} items")
                        # Additional validation: ensure each item is a non-empty string
                        valid_items = [item for item in json_data if isinstance(item, str) and item.strip()]
                        if len(valid_items) != len(json_data):
                            log.warning(f"Found {len(json_data) - len(valid_items)} invalid items in JSON array")
                            json_data = valid_items
                        return json.dumps(json_data)  # Return properly formatted JSON
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse extracted JSON from {model_name}: {e}")

            # If the precise regex didn't work, try a more lenient approach
            if "medalpaca" in _hf_model_name.lower():
                try:
                    # Try to extract anything that looks like a JSON array
                    array_match = re.search(r"(\[[\s\S]*?\])", cleaned_response)
                    if array_match:
                        content = array_match.group(1).strip()
                        # Try to fix common JSON issues
                        content = content.replace("'", '"')  # Replace single quotes
                        content = re.sub(r'",\s*]', '"]', content)  # Fix trailing comma
                        content = re.sub(r'"\s*"', '", "', content)  # Fix missing commas

                        # Validate JSON
                        json_data = json.loads(content)
                        if isinstance(json_data, list):
                            log.info(f"Successfully parsed JSON array with lenient approach: {len(json_data)} items")
                            # Additional validation: ensure each item is a non-empty string
                            valid_items = [item for item in json_data if isinstance(item, str) and item.strip()]
                            if len(valid_items) != len(json_data):
                                log.warning(f"Found {len(json_data) - len(valid_items)} invalid items in JSON array")
                                json_data = valid_items
                            return json.dumps(json_data)  # Return properly formatted JSON
                except (json.JSONDecodeError, Exception) as e:
                    log.warning(f"Failed to parse JSON with lenient approach: {e}")

        # Standard extraction for all models
        # First, try to find a JSON array in the response using regex
        array_match = re.search(r"(\[[\s\S]*?\])", assistant_response)
        if array_match:
            content = array_match.group(1).strip()
            # Validate JSON
            try:
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    # For MedAlpaca, we want exactly one question per array
                    if "medalpaca" in _hf_model_name.lower():
                        if len(json_data) == 1:
                            log.info(f"Successfully parsed JSON array with exactly 1 item as requested")
                            return content
                        elif len(json_data) > 1:
                            log.warning(f"JSON array contains {len(json_data)} items instead of 1. Taking only the first item.")
                            # Only take the first item to ensure we get exactly one question per array
                            return json.dumps([json_data[0]])
                        else:
                            log.warning(f"JSON array is empty")
                    else:
                        # For other models, accept any non-empty array
                        log.info(f"Successfully parsed JSON array with {len(json_data)} items")
                        return content
                else:
                    log.warning(f"Parsed JSON is not a list: {type(json_data)}")
            except json.JSONDecodeError as e:
                log.warning(f"Failed to parse extracted JSON: {e}. Trying alternative extraction.")

        # If we couldn't find or parse a JSON array, try to fix common issues
        # 1. Look for markdown code blocks that might contain JSON
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", assistant_response)
        if code_block_match:
            content = code_block_match.group(1).strip()
            try:
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    # For MedAlpaca, we want exactly one question per array
                    if "medalpaca" in _hf_model_name.lower():
                        if len(json_data) == 1:
                            log.info(f"Successfully parsed JSON from code block with exactly 1 item as requested")
                            return json.dumps(json_data)  # Return properly formatted JSON
                        elif len(json_data) > 1:
                            log.warning(f"JSON array from code block contains {len(json_data)} items instead of 1. Taking only the first item.")
                            # Only take the first item to ensure we get exactly one question per array
                            return json.dumps([json_data[0]])
                        else:
                            log.warning(f"JSON array from code block is empty")
                    else:
                        # For other models, accept any non-empty array
                        log.info(f"Successfully parsed JSON from code block with {len(json_data)} items")
                        return json.dumps(json_data)  # Return properly formatted JSON
                else:
                    log.warning(f"Parsed JSON from code block is not a list: {type(json_data)}")
            except json.JSONDecodeError as e:
                log.warning(f"Failed to parse JSON from code block: {e}")

        # 2. Try to construct a JSON array from the response if it looks like it contains items
        # Look for patterns like "1. item", "- item", etc.
        items = []

        # Try numbered list format: "1. item", "2. item", etc.
        numbered_items = re.findall(r"\d+\.\s*(.*?)(?=\d+\.|$)", assistant_response, re.DOTALL)
        if numbered_items and len(numbered_items) >= 3:
            items = [item.strip() for item in numbered_items if item.strip()]

        # Try bullet list format: "- item", "* item", etc.
        if not items:
            bullet_items = re.findall(r"[-*]\s*(.*?)(?=[-*]|$)", assistant_response, re.DOTALL)
            if bullet_items and len(bullet_items) >= 3:
                items = [item.strip() for item in bullet_items if item.strip()]

        # If we found items, construct a JSON array
        if items:
            # For MedAlpaca, we want exactly one question per array
            if "medalpaca" in _hf_model_name.lower():
                if len(items) == 1:
                    log.info(f"Constructed JSON array with exactly 1 item as requested")
                    return json.dumps(items)
                elif len(items) > 1:
                    log.warning(f"Constructed JSON array contains {len(items)} items instead of 1. Taking only the first item.")
                    # Only take the first item to ensure we get exactly one question per array
                    return json.dumps([items[0]])
                else:
                    log.warning(f"Constructed JSON array is empty")
            else:
                # For other models, accept any non-empty array
                log.info(f"Constructed JSON array from {len(items)} extracted items")
                return json.dumps(items)

        # 3. Last resort: If the response is completely unusable, return empty array and let the caller retry
        log.warning("Could not extract valid JSON. Returning empty array to trigger retry.")
        log.warning("We NEVER use fallback arrays as they defeat the purpose of generating fresh scenarios.")
        return "[]"  # Return empty array to trigger retry in the caller

    except Exception as e:
        log.error(f"Error generating completion with HuggingFace model: {e}")
        return None

def get_hf_model_name() -> Optional[str]:
    """Get the name of the currently loaded HuggingFace model."""
    global _hf_model_name
    return _hf_model_name
