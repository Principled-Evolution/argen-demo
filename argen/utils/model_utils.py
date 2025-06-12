"""
Model loading and generation utilities for ArGen evaluation.

This module provides functions for loading models, tokenizers, and generating responses
that are used across different evaluation modules.
"""

import os
import logging
import torch
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from HuggingFace or local path.
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        logger.info(f"Successfully loaded model: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def apply_chat_template_if_needed(
    tokenizer: AutoTokenizer, 
    prompt: str, 
    system_prompt: str
) -> str:
    """
    Apply chat template if the tokenizer supports it, otherwise use fallback formatting.
    
    Args:
        tokenizer: The tokenizer to use
        prompt: The user prompt
        system_prompt: The system prompt
        
    Returns:
        Formatted prompt string
    """
    try:
        # Try to use the tokenizer's chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return formatted_prompt
    except Exception as e:
        logger.warning(f"Could not apply chat template: {e}. Using fallback formatting.")
    
    # Fallback formatting
    return f"{system_prompt}\nUser: {prompt}\nAssistant:"


def generate_responses_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int = 512
) -> List[str]:
    """
    Generate responses for a batch of prompts.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompts: List of formatted prompts
        temperature: Temperature for generation
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        List of generated responses
    """
    responses = []
    device = next(model.parameters()).device
    
    for prompt in prompts:
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response (remove input tokens)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            responses.append(response.strip())
            
        except Exception as e:
            logger.error(f"Failed to generate response for prompt: {e}")
            responses.append(f"Error generating response: {str(e)}")
    
    return responses
