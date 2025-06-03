#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_model.py - Baseline model utilities for ArGen dataset generator
=======================================================================
Contains functions for initializing and generating responses from baseline models
"""

import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import log, DEFAULT_BASELINE

# Global model variables
_baseline_model = None
_baseline_tokenizer = None
_baseline_device = None

def init_baseline_model(model_name: str = DEFAULT_BASELINE) -> bool:
    """Initialize baseline model for response generation."""
    global _baseline_model, _baseline_tokenizer, _baseline_device

    log.info(f"Initializing baseline model: {model_name}")
    try:
        # Set device
        _baseline_device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {_baseline_device}")

        # Load tokenizer
        _baseline_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _baseline_tokenizer.pad_token is None:
            _baseline_tokenizer.pad_token = _baseline_tokenizer.eos_token
        _baseline_tokenizer.padding_side = "left"  # Required for batch generation

        # Load model
        _baseline_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if _baseline_device=="cuda" else torch.float32,
            device_map="auto"
        )

        # Test model with simple prompt
        _ = generate_baseline_responses(["Test prompt"], batch_size=1)
        log.info(f"Baseline model initialized successfully: {model_name}")
        return True
    except Exception as e:
        log.error(f"Failed to initialize baseline model: {e}")
        return False

def generate_baseline_responses(prompts: List[str], system_prompt: str = "", batch_size: int = 8, output_scores: bool = False) -> List[str]:
    """
    Generate responses from baseline model for a list of prompts.

    Args:
        prompts: List of prompt strings
        system_prompt: System prompt to prepend
        batch_size: Batch size for generation
        output_scores: Whether to return logits for NLL calculation

    Returns:
        If output_scores=False: List of response strings
        If output_scores=True: Tuple of (responses, outputs, input_ids)
    """
    global _baseline_model, _baseline_tokenizer, _baseline_device

    if _baseline_model is None or _baseline_tokenizer is None:
        log.error("Baseline model not initialized.")
        return ["Error: Model not initialized."] * len(prompts)

    log.info(f"Generating responses for {len(prompts)} prompts with batch size {batch_size}")
    responses = []
    all_outputs = []
    all_input_ids = []

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        log.debug(f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")

        # Format prompts
        formatted_prompts = []
        for prompt in batch_prompts:
            # Use the format appropriate for your model (Llama, Mistral, etc.)
            formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
            formatted_prompts.append(formatted_prompt)

        # Tokenize batch with padding
        batch_encoding = _baseline_tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(_baseline_device)

        # Generate
        with torch.inference_mode():
            if output_scores:
                # Generate with output_scores=True for NLL calculation
                outputs = _baseline_model.generate(
                    **batch_encoding,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=_baseline_tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                # Store outputs and input_ids for NLL calculation
                all_outputs.append(outputs)
                all_input_ids.append(batch_encoding['input_ids'])
            else:
                # Standard generation without scores
                outputs = _baseline_model.generate(
                    **batch_encoding,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=_baseline_tokenizer.pad_token_id
                )

        # Decode and extract assistant response
        batch_responses = []
        if output_scores:
            # For output_scores=True, outputs is a dict with 'sequences'
            sequences = outputs.sequences
            for output, prompt in zip(sequences, formatted_prompts):
                text = _baseline_tokenizer.decode(output, skip_special_tokens=True)
                # Extract just the assistant part
                assistant_response = text.split("<|assistant|>")[-1].strip()
                batch_responses.append(assistant_response)
        else:
            # For standard generation, outputs is just the sequences
            for output, prompt in zip(outputs, formatted_prompts):
                text = _baseline_tokenizer.decode(output, skip_special_tokens=True)
                # Extract just the assistant part
                assistant_response = text.split("<|assistant|>")[-1].strip()
                batch_responses.append(assistant_response)

        responses.extend(batch_responses)

    if output_scores:
        # Return tuple of (responses, outputs, input_ids) for NLL calculation
        return responses, all_outputs, all_input_ids
    else:
        # Return just the responses
        return responses

def compute_sentence_nll(outputs, input_ids) -> float:
    """Mean NLL per token; conservative high value if scores missing."""
    global _baseline_tokenizer, _baseline_model
    from .config import EPSILON, log

    # Return a large finite sentinel value (1e6) when logits are missing
    if not outputs or not getattr(outputs, "scores", None):
        log.warning("Missing scores in outputs, returning sentinel NLL value")
        return 1e6  # Large finite sentinel value instead of infinity

    # Check if input_ids exceed model's position embedding limit
    if hasattr(_baseline_model, 'config') and hasattr(_baseline_model.config, 'max_position_embeddings'):
        max_len = _baseline_model.config.max_position_embeddings
        if input_ids.shape[1] > max_len:
            log.warning(f"Input sequence length {input_ids.shape[1]} exceeds model's max position embeddings {max_len}")
            # Truncate from the beginning to fit within model's context window
            input_ids = input_ids[:, -max_len:]
            log.info(f"Truncated input to length {input_ids.shape[1]} for NLL calculation")

    total, count = 0.0, 0
    try:
        for step, logits in enumerate(outputs.scores):
            if step+1 >= input_ids.shape[1]:
                break
            tok = input_ids[0, step+1].item()
            if tok == _baseline_tokenizer.pad_token_id:
                continue
            lp = torch.log_softmax(logits[0], dim=-1)[tok]
            total -= lp.item()
            count += 1
    except Exception as e:
        # If any error occurs during computation, return the sentinel value
        log.warning(f"Error computing NLL: {e}. Using sentinel value.")
        return 1e6

    # If no valid tokens were found, return the sentinel value
    if count == 0:
        return 1e6

    # Ensure we don't divide by zero
    return max(total / max(count, EPSILON), EPSILON)

# Difficulty banding helpers
_current_ema = None  # For difficulty banding

def update_ema(new: float, alpha: float = 0.1) -> None:
    """Update exponential moving average with new value."""
    global _current_ema
    _current_ema = new if _current_ema is None else alpha * new + (1 - alpha) * _current_ema

def get_current_ema() -> Optional[float]:
    """Get the current exponential moving average value."""
    global _current_ema
    return _current_ema