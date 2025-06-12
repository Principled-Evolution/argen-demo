"""
LLM-based evaluator for ArGen fine-tuning.

This module provides functions to evaluate a model using LLM APIs (OpenAI, Gemini, etc.)
for Ahimsa, Dharma, and Helpfulness principles.

This is the main evaluation module that supports multiple LLM providers.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LlamaTokenizer
import re
import datetime
import logging
import random # Import random for sampling

from argen.reward_functions.openai_rewards import (
    evaluate_dharma_with_openai,
    evaluate_ahimsa_with_openai,
    evaluate_helpfulness_with_openai,
    batch_evaluate_with_openai,
    DEFAULT_EVAL_RESPONSE
)

# Import Gemini reward functions.
# If these imports fail, an ImportError will be raised, clearly indicating the problem.
from argen.reward_functions.gemini.ahimsa import evaluate_ahimsa_with_gemini
from argen.reward_functions.gemini.ahimsa import evaluate_ahimsa_multi_with_gemini
from argen.reward_functions.gemini.ahimsa import batch_process_ahimsa_evaluations_concurrently
from argen.reward_functions.gemini.dharma import evaluate_dharma_with_gemini
from argen.reward_functions.gemini.dharma import batch_process_dharma_evaluations_concurrently
from argen.reward_functions.gemini.helpfulness import evaluate_helpfulness_with_gemini
from argen.reward_functions.gemini.helpfulness import batch_process_helpfulness_evaluations_concurrently
from argen.reward_functions.gemini_rewards import (
    GEMINI_EVAL_MODEL
)

from argen.config import (
    DEFAULT_MAX_NEW_TOKENS,
    ENHANCED_SYSTEM_PROMPT,
    BASIC_SYSTEM_PROMPT,
    OPENAI_EVAL_MODEL,
    OPENAI_EVAL_TEMPERATURE,
    OPENAI_EVAL_MAX_TOKENS,
    REWARD_WEIGHTS,
    get_system_prompt,
    DEFAULT_GENERATION_BATCH_SIZE,
    GRPO_CONFIG
)
from argen.penalty_config import PENALTY_CONFIG
from tqdm import tqdm

# Import data integrity utilities
from argen.utils.data_integrity import (
    verify_prompt_tier_hash,
    extract_tier_from_compound
)

# Configuration for the check (can be moved to a config file)
VERIFY_HASH_SAMPLE_RATE = 0.1 # Check 15% of items per batch

# Batch size overrides for known large models
LARGE_MODEL_BATCH_OVERRIDES = {
    "medalpaca/medalpaca-7b": 8,  # Reduce batch size for this 7B model
    # Add other model_name: batch_size pairs here if needed
    # e.g., "meta-llama/Llama-3.2-8B-Instruct": 2
    "meta-llama/Llama-3.2-3B-Instruct": 20
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Add this line to define the module logger

# Set the logging level for httpx to WARNING or higher
logging.getLogger("httpx").setLevel(logging.WARNING)

# We rely on the model's native chat template
# Each model (Llama 3, MedAlpaca, etc.) has its own template
# that's loaded with the tokenizer

def _create_fallback_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Creates a fallback prompt format when chat template is not available.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        A formatted string prompt
    """
    prompt_parts = []
    for msg in messages:
        role = msg.get('role', '').capitalize()
        content = msg.get('content', '')
        prompt_parts.append(f"{role}: {content}")

    # Format looks like: "System: ... \nUser: ... \nAssistant:"
    formatted_prompt = "\n".join(prompt_parts)
    # Add the assistant prefix to indicate where model should continue
    if not formatted_prompt.endswith("Assistant:"):
        formatted_prompt += "\nAssistant:"

    return formatted_prompt

def _format_alpaca_instruct_prompt(system_prompt_content: str, user_prompt_content: str, history: Optional[List[Dict]] = None) -> str:
    """
    Formats a prompt for Alpaca-style instruction-following models.
    """
    prompt_parts = []

    if system_prompt_content:
        prompt_parts.append(system_prompt_content)

    if history:
        for msg in history:
            role = "Instruction" if msg['role'] == 'user' else "Response"
            prompt_parts.append(f"### Previous {role}:\n{msg['content']}")

    prompt_parts.append(f"### Instruction:\n{user_prompt_content}")
    prompt_parts.append("### Response:\n")

    return "\n\n".join(prompt_parts)

def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculates summary metrics from individual evaluation results.

    Args:
        results: A list of dictionaries, where each dictionary contains
                 evaluation results for a single scenario.

    Returns:
        A dictionary containing summary metrics.
    """
    num_scenarios = len(results)
    if num_scenarios == 0:
        return {
            "ahimsa_violations": 0, "ahimsa_violation_rate": 0.0, "average_ahimsa_score": 0.0,
            "dharma_violations": 0, "dharma_violation_rate": 0.0, "average_dharma_score": 0.0,
            "average_combined_score": 0.0,
            "helpfulness_violations": 0, "helpfulness_violation_rate": 0.0, "average_helpfulness_score": 0.0,
            "average_clarity_score": 0.0, "average_relevance_score": 0.0, "average_completeness_score": 0.0,
            "num_clipped": 0,
            "clipped_ratio": 0.0
        }

    helpfulness_violations = sum(1 for r in results if r.get("helpfulness_violation", False))

    ahimsa_violations = sum(1 for r in results if r.get("ahimsa_violation", False))
    dharma_violations = sum(1 for r in results if r.get("dharma_violation", False))

    total_ahimsa_score = sum(r.get("ahimsa_score", 0.0) for r in results)
    total_dharma_score = sum(r.get("dharma_score", 0.0) for r in results)
    total_helpfulness_score = sum(r.get("helpfulness_score", 0.0) for r in results)
    total_clarity_score = sum(r.get("clarity_score", 0.0) for r in results)
    total_relevance_score = sum(r.get("relevance_score", 0.0) for r in results)
    total_completeness_score = sum(r.get("completeness_score", 0.0) for r in results)
    total_combined_score = sum(r.get("combined_score", 0.0) for r in results)

    num_clipped = sum(1 for r in results if r.get("was_clipped", False))

    # Calculate scope penalty metrics
    scope_penalties = [r.get("scope_penalty_factor", 1.0) for r in results]
    total_scope_penalty = sum(scope_penalties)
    avg_scope_penalty = total_scope_penalty / num_scenarios if num_scenarios > 0 else 1.0

    # Count responses by scope category
    scope_counts = {
        "S0": sum(1 for r in results if r.get("response_scope", "S0") == "S0"),
        "S1": sum(1 for r in results if r.get("response_scope", "S0") == "S1"),
        "S2": sum(1 for r in results if r.get("response_scope", "S0") == "S2"),
        "S3": sum(1 for r in results if r.get("response_scope", "S0") == "S3")
    }

    # Count severe scope penalties (0.0)
    severe_scope_penalties = sum(1 for p in scope_penalties if p == 0.0)

    return {
        "ahimsa_violations": ahimsa_violations,
        "ahimsa_violation_rate": ahimsa_violations / num_scenarios,
        "average_ahimsa_score": total_ahimsa_score / num_scenarios,
        "dharma_violations": dharma_violations,
        "dharma_violation_rate": dharma_violations / num_scenarios,
        "average_dharma_score": total_dharma_score / num_scenarios,
        "average_combined_score": total_combined_score / num_scenarios,
        "helpfulness_violations": helpfulness_violations,
        "helpfulness_violation_rate": helpfulness_violations / num_scenarios,
        "average_helpfulness_score": total_helpfulness_score / num_scenarios,
        "average_clarity_score": total_clarity_score / num_scenarios,
        "average_relevance_score": total_relevance_score / num_scenarios,
        "average_completeness_score": total_completeness_score / num_scenarios,
        "num_clipped": num_clipped,
        "clipped_ratio": num_clipped / num_scenarios if num_scenarios > 0 else 0.0,
        # Scope penalty metrics
        "average_scope_penalty_factor": avg_scope_penalty,
        "severe_scope_penalties": severe_scope_penalties,
        "severe_scope_penalty_rate": severe_scope_penalties / num_scenarios if num_scenarios > 0 else 0.0,
        "scope_response_counts": scope_counts
    }


def get_mock_response(prompt: str) -> str:
    """Returns a generic mock response for testing."""
    return (
        "Mock Response: I understand your concern. As an AI, I cannot provide medical advice. "
        "It is best to consult with a qualified healthcare professional for any health questions or concerns."
    )


def _generate_batch_response(
    model_name: str,
    prompts: List[str],
    temperature: float,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: str = "cuda"
) -> Tuple[List[Optional[str]], List[bool], float]:
    """
    Generates responses for a batch of prompts using a local Hugging Face model.

    Args:
        model_name: Identifier for the model (HF name/path).
        prompts: A list of fully formatted prompts to send to the model.
        temperature: Temperature for generation.
        model: Pre-loaded Hugging Face model.
        tokenizer: Pre-loaded Hugging Face tokenizer.
        max_new_tokens: Max new tokens for generation.
        device: The device to run generation on ('cuda' or 'cpu').

    Returns:
        A tuple containing a list of generated response strings (or None for errors),
        a list of clip flags (based on refined logic ignoring end-padding),
        and the total batch generation time.
    """
    start_time = time.time()
    responses = [None] * len(prompts)
    clip_flags = [False] * len(prompts)

    if not prompts:
        return [], [], 0.0

    logging.info(f"Generating batch of {len(prompts)} responses locally for model {model_name}...")

    try:
        # Ensure tokenizer has a padding token; use EOS if not set
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            if eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                pad_token_id = eos_token_id # Update local variable too
                logging.info(f"Tokenizer pad_token was None. Setting pad_token to eos_token: {tokenizer.eos_token}")
            else:
                # This is problematic, generation might fail or behave unexpectedly.
                logging.error("Tokenizer has no pad_token_id and no eos_token_id! Clipping detection and padding may fail.")
                # We'll have to proceed without reliable padding checks.

        # Tokenize the batch with left padding
        logging.info(f"Tokenizing batch with padding_side='left'")
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            padding_side='left'
        ).to(device)

        # Generation arguments
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": pad_token_id, # Use the potentially updated pad_token_id
            "eos_token_id": eos_token_id
        }

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode and check for clipping
        decoded_responses = []
        for i, output_sequence in enumerate(outputs):
            input_length = inputs.input_ids[i].shape[0]
            # Slice the output sequence to get only the generated part
            generated_tokens = output_sequence[input_length:]
            generated_token_count = len(generated_tokens)

            # +++ Add Debug Logging Here +++
            logging.debug(f"Sequence {i}: Generated token count = {generated_token_count}, max_new_tokens = {max_new_tokens}")
            logging.debug(f"Sequence {i}: EOS ID = {eos_token_id}, PAD ID = {pad_token_id}")
            # Log last few token IDs to see what's happening at the end
            if generated_token_count > 0:
                last_few_tokens = generated_tokens[-5:].tolist() # Get last 5 token IDs as a list
                logging.debug(f"Sequence {i}: Last {len(last_few_tokens)} generated token IDs = {last_few_tokens}")
            # +++ End Debug Logging +++

            # --- Padding-Aware Check for Clipping ---
            is_clipped = False
            if generated_token_count == max_new_tokens:
                # Find the ID of the actual last generated token, ignoring potential padding
                actual_last_token_id = None
                # Iterate backwards through the generated tokens
                for token_id_tensor in reversed(generated_tokens):
                    token_id = token_id_tensor.item()
                    # Check against pad_token_id *only if* it's defined
                    if pad_token_id is not None and token_id == pad_token_id:
                        continue # Skip padding tokens
                    else:
                        # Found the last non-padding token
                        actual_last_token_id = token_id
                        break # Stop searching

                # If we found a last non-padding token (should always happen unless max_new_tokens=0)
                # AND that token is not the EOS token, then it was likely cut off.
                if actual_last_token_id is not None and actual_last_token_id != eos_token_id:
                    is_clipped = True
                    logging.debug(
                        f"Clipping detected for sequence {i}: "
                        f"Reached max_new_tokens ({max_new_tokens}). "
                        f"Last non-pad token ID ({actual_last_token_id}) was not EOS ID ({eos_token_id})."
                    )
                # else: Either didn't reach max_new_tokens, or did and the last non-pad token was EOS. Not clipped.

            clip_flags[i] = is_clipped
            # ---

            decoded_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            decoded_responses.append(decoded_response.strip())

        responses = decoded_responses

    except Exception as e:
        logging.exception(f"Error during batch generation for model {model_name}: {e}")
        # Return list of Nones on error
        responses = [None] * len(prompts)
        # Ensure clip_flags has the correct length even on error
        clip_flags = [False] * len(prompts)


    end_time = time.time()
    generation_time = end_time - start_time
    logging.info(f"Batch generation completed in {generation_time:.2f}s")
    return responses, clip_flags, generation_time


async def evaluate_model_with_llm(
    model_name: str,
    scenarios: List[Dict],
    output_file: str,
    temperature: float,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    evaluator: str = "openai",
    test_mode: bool = False,
    system_prompt_type: str = 'ENHANCED',
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    apply_medical_disclaimer_penalty: Optional[bool] = None,
    apply_professional_referral_penalty: Optional[bool] = None,
    gemini_eval_mode: str = "batch",
    skip_evaluation: bool = False
) -> None:
    """
    Evaluates a model using an LLM (OpenAI or Gemini) for Ahimsa, Dharma, and Helpfulness.
    Generates responses locally if model_name is a path, otherwise uses Predibase.
    Allows for batching of generations and evaluations.
    """
    logger.info(f"Starting evaluation for model: {model_name} with evaluator: {evaluator.upper()}")
    logger.info(f"Processing {len(scenarios)} scenarios with generation batch size: {generation_batch_size}")

    # Determine if it's a local model path
    is_filesystem_path = os.path.exists(model_name)
    # Heuristic: if it contains '/' and is not a filesystem path, it's likely an HF ID for local loading.
    attempt_hf_hub_load = not is_filesystem_path and ('/' in model_name)

    # Initialize model and tokenizer for local generation if applicable
    local_model = None
    local_tokenizer = None
    device = "cpu"

    if is_filesystem_path or attempt_hf_hub_load:
        logger.info(f"Attempting to load model locally: {model_name}")
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("CUDA available, using GPU for local model.")
        else:
            logger.warning("CUDA not available, using CPU for local model. This might be slow.")

        try:
            # Use LlamaTokenizer for Llama models for correct chat template handling
            # if "llama" in model_name.lower():
            #     local_tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # else:
            #     local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Always use AutoTokenizer to correctly infer the tokenizer type from model config
            local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # bf16 for GPU, fp32 for CPU
                device_map="auto" if device == "cuda" else None, # Use "auto" for multi-GPU if available
                trust_remote_code=True
            )
            if device == "cpu": # Explicitly move to CPU if that's the target
                local_model.to(device)
            local_model.eval() # Set model to evaluation mode
            logger.info(f"Local model {model_name} loaded successfully on {device}.")

            # Set pad_token_id if not set (common for Llama models)
            if local_tokenizer.pad_token_id is None:
                local_tokenizer.pad_token_id = local_tokenizer.eos_token_id
                logger.info(f"Set tokenizer.pad_token_id to eos_token_id: {local_tokenizer.eos_token_id}")

        except Exception as e:
            logger.error(f"Error loading local model {model_name}: {e}", exc_info=True)
            # Fallback or error handling if local model fails to load
            return # Exit if model loading fails

    evaluation_results = []
    total_scenarios = len(scenarios)
    processed_scenarios = 0

    # Determine effective generation batch_size, considering model-specific overrides
    effective_batch_size = generation_batch_size
    if model_name in LARGE_MODEL_BATCH_OVERRIDES:
        effective_batch_size = LARGE_MODEL_BATCH_OVERRIDES[model_name]
        logger.info(f"Overriding generation batch size to {effective_batch_size} for model {model_name}")
    if test_mode:
        effective_batch_size = 1 # Ensure single processing for test mode simplicity
        logger.info(f"Test mode: Setting generation batch size to 1.")

    # Main loop for processing scenarios in batches
    for i in tqdm(range(0, total_scenarios, effective_batch_size), desc="Evaluating scenarios"):
        batch_scenarios_data = scenarios[i : i + effective_batch_size]
        batch_prompts_for_generation = []
        batch_original_prompts = [] # To store original prompts for logging/verification
        batch_metadata = [] # To store metadata like tier and scope for each prompt

        # Prepare prompts for the current batch
        for scenario_data in batch_scenarios_data:
            user_prompt = scenario_data["prompt"]
            history = scenario_data.get("history") # Optional history
            system_prompt_content = get_system_prompt(system_prompt_type)

            # Format the prompt - check for MedAlpaca first before trying chat template
            if "medalpaca" in model_name.lower() and (is_filesystem_path or attempt_hf_hub_load):
                # Specific handling for MedAlpaca models using Alpaca format
                formatted_prompt_for_gen = _format_alpaca_instruct_prompt(system_prompt_content, user_prompt, history)
            elif is_filesystem_path or attempt_hf_hub_load:
                # Try chat template for other local models
                messages = []
                if system_prompt_content:
                    messages.append({"role": "system", "content": system_prompt_content})
                if history: # Add history if present
                    messages.extend(history)
                messages.append({"role": "user", "content": user_prompt})

                try:
                    formatted_prompt_for_gen = local_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(f"Could not apply chat template for local model: {e}. Using fallback formatting.")
                    # Fallback: simple concatenation or a custom basic formatter
                    formatted_prompt_for_gen = f"{system_prompt_content}\nUser: {user_prompt}\nAssistant:"
            else: # For Predibase or local models without advanced chat templates
                formatted_prompt_for_gen = f"{system_prompt_content}\nUser: {user_prompt}\nAssistant:"

            batch_prompts_for_generation.append(formatted_prompt_for_gen)
            batch_original_prompts.append(user_prompt) # Store original user prompt
            batch_metadata.append({
                "tier": scenario_data.get("tier", "C"), # Default to Tier C if not specified
                "scope": scenario_data.get("scope") # Optional scope
            })

        # Generate responses for the batch
        batch_generated_responses = [None] * len(batch_prompts_for_generation) # Initialize with None
        batch_was_clipped = [False] * len(batch_prompts_for_generation)
        generation_time = 0.0

        if test_mode:
            logger.info("Test mode: Using mock responses.")
            batch_generated_responses = [get_mock_response(p) for p in batch_original_prompts]
        elif local_model and local_tokenizer: # Successfully loaded a local/HF model
            logger.debug(f"Generating responses for batch of {len(batch_prompts_for_generation)} locally with {model_name}.")
            responses, clipped_flags, gen_time = _generate_batch_response(
                model_name, batch_prompts_for_generation, temperature, local_model, local_tokenizer, device=device
            )
            batch_generated_responses = responses
            batch_was_clipped = clipped_flags
            generation_time = gen_time
            logger.debug(f"Local generation for batch took {generation_time:.2f}s.")
        else: # Not test mode, and local_model (or tokenizer) is None
            if is_filesystem_path or attempt_hf_hub_load:
                # This means an attempt to load a local/HF model was made but failed (local_model is None)
                logger.error(f"Failed to load the intended local/HuggingFace model '{model_name}'. Generation cannot proceed locally.")
                # Fill with error placeholders
                batch_generated_responses = ["MODEL_LOAD_FAILURE"] * len(batch_prompts_for_generation)
            else:
                # Model name was not a path and didn't look like an HF ID.
                # No Predibase fallback, so this is an unsupported model type for generation.
                logger.error(f"Model '{model_name}' is not a local path, not recognized as a HuggingFace Hub ID, and Predibase is not used. Cannot generate responses.")
                batch_generated_responses = ["UNSUPPORTED_MODEL_FOR_GENERATION"] * len(batch_prompts_for_generation)


        # Evaluate responses for the batch (skip if skip_evaluation=True)
        evaluation_tasks_to_gather = []

        if skip_evaluation:
            # Skip evaluation, just create results with responses
            current_batch_eval_results = []
            for i_item in range(len(batch_scenarios_data)):
                result = {
                    "prompt": batch_original_prompts[i_item],
                    "response": batch_generated_responses[i_item],
                    "scenario_id": batch_scenarios_data[i_item].get("scenario_id", f"batch_{i}_item_{i_item}")
                }
                current_batch_eval_results.append(result)

            evaluation_results.extend(current_batch_eval_results)
            processed_scenarios += len(batch_scenarios_data)
            continue

        if evaluator == "openai":
            openai_eval_items = []
            for idx, scenario_data in enumerate(batch_scenarios_data):
                original_prompt = batch_original_prompts[idx]
                generated_response = batch_generated_responses[idx]
                actual_tier = extract_tier_from_compound(scenario_data.get("tier", "C"))
                current_prompt_meta = {"tier": actual_tier, "scope": scenario_data.get("scope")}
                openai_eval_items.append({
                    "prompt": original_prompt, "response": generated_response, "metadata": current_prompt_meta
                })
            if openai_eval_items:
                evaluation_tasks_to_gather.append(batch_evaluate_with_openai(openai_eval_items, openai_api_key))

        elif evaluator == "gemini":
            if gemini_eval_mode == "batch":
                # Batch evaluation mode (current implementation)
                ahimsa_payload_for_concurrent_eval = []
                dharma_payload_for_concurrent_eval = []
                helpfulness_payload_for_concurrent_eval = []

                for idx, scenario_data in enumerate(batch_scenarios_data):
                    original_prompt = batch_original_prompts[idx]
                    generated_response = batch_generated_responses[idx]

                    if generated_response is None:
                        logger.warning(f"Scenario {scenario_data.get('scenario_id', 'N/A')} has None generated_response. Using default error for Gemini evals.")
                        # For concurrent_eval, we add an item that its underlying multi_with_gemini will handle as error.
                        ahimsa_payload_for_concurrent_eval.append({
                            "prompt": original_prompt,
                            "model_response": "GENERATION_FAILED_PLACEHOLDER",
                            "original_prompt_meta": {"tier": "N/A", "scope": "N/A"} # Ensure meta is always a dict
                        })
                        dharma_payload_for_concurrent_eval.append({ # Add placeholder for Dharma batch too
                            "prompt": original_prompt,
                            "model_response": "GENERATION_FAILED_PLACEHOLDER",
                            "original_prompt_meta": {"tier": "N/A", "scope": "N/A"}
                        })
                        helpfulness_payload_for_concurrent_eval.append({ # Add placeholder for Helpfulness batch too
                            "prompt": original_prompt,
                            "model_response": "GENERATION_FAILED_PLACEHOLDER",
                            "original_prompt_meta": {}
                        })
                        continue

                    actual_tier = extract_tier_from_compound(scenario_data.get("tier", "C"))
                    current_prompt_meta = {"tier": actual_tier, "scope": scenario_data.get("scope")}

                    ahimsa_payload_for_concurrent_eval.append({
                        "prompt": original_prompt,
                        "model_response": generated_response,
                        "original_prompt_meta": current_prompt_meta
                    })
                    # Populate Dharma payload for batch processing
                    dharma_payload_for_concurrent_eval.append({
                        "prompt": original_prompt,
                        "model_response": generated_response,
                        "original_prompt_meta": current_prompt_meta
                    })
                    # Populate Helpfulness payload for batch processing
                    helpfulness_payload_for_concurrent_eval.append({
                        "prompt": original_prompt,
                        "model_response": generated_response,
                        "original_prompt_meta": {} # Helpfulness multi eval does not use meta currently
                    })

                if ahimsa_payload_for_concurrent_eval:
                    items_per_ahimsa_call_cfg = GRPO_CONFIG.get("gemini_ahimsa_items_per_call_eval", 10)
                    max_concurrent_ahimsa_cfg = GRPO_CONFIG.get("gemini_ahimsa_max_concurrent_eval", 5)
                    evaluation_tasks_to_gather.append(
                        batch_process_ahimsa_evaluations_concurrently(
                            ahimsa_payload_for_concurrent_eval,
                            items_per_gemini_call=items_per_ahimsa_call_cfg,
                            max_concurrent_calls=max_concurrent_ahimsa_cfg,
                            temperature=temperature
                        )
                    )

                if dharma_payload_for_concurrent_eval: # Check Dharma payload
                    items_per_dharma_call_cfg = GRPO_CONFIG.get("gemini_dharma_items_per_call_eval", 10)
                    max_concurrent_dharma_cfg = GRPO_CONFIG.get("gemini_dharma_max_concurrent_eval", 5)
                    evaluation_tasks_to_gather.append(
                        batch_process_dharma_evaluations_concurrently( # Use Dharma batch processor
                            dharma_payload_for_concurrent_eval,
                            items_per_gemini_call=items_per_dharma_call_cfg,
                            max_concurrent_calls=max_concurrent_dharma_cfg,
                            temperature=temperature
                        )
                    )

                # Add concurrent helpfulness task if payload exists
                if helpfulness_payload_for_concurrent_eval:
                    items_per_helpfulness_call_cfg = GRPO_CONFIG.get("gemini_helpfulness_items_per_call_eval", 10)
                    max_concurrent_helpfulness_cfg = GRPO_CONFIG.get("gemini_helpfulness_max_concurrent_eval", 5)
                    evaluation_tasks_to_gather.append(
                        batch_process_helpfulness_evaluations_concurrently(
                            helpfulness_payload_for_concurrent_eval,
                            items_per_gemini_call=items_per_helpfulness_call_cfg,
                            max_concurrent_calls=max_concurrent_helpfulness_cfg,
                            temperature=temperature
                        )
                    )

            elif gemini_eval_mode == "individual":
                # Individual evaluation mode (one API call per evaluation)
                individual_ahimsa_tasks = []
                individual_dharma_tasks = []
                individual_helpfulness_tasks = []

                for idx, scenario_data in enumerate(batch_scenarios_data):
                    original_prompt = batch_original_prompts[idx]
                    generated_response = batch_generated_responses[idx]

                    if generated_response is None:
                        logger.warning(f"Scenario {scenario_data.get('scenario_id', 'N/A')} has None generated_response. Using default error for individual Gemini evals.")
                        # Add error tasks that will return default error responses
                        error_result = {"error": "Model generation failed", **DEFAULT_EVAL_RESPONSE}
                        individual_ahimsa_tasks.append(asyncio.sleep(0, result=error_result))
                        individual_dharma_tasks.append(asyncio.sleep(0, result=error_result))
                        individual_helpfulness_tasks.append(asyncio.sleep(0, result=error_result))
                        continue

                    actual_tier = extract_tier_from_compound(scenario_data.get("tier", "C"))
                    current_prompt_meta = {"tier": actual_tier, "scope": scenario_data.get("scope")}

                    # Create individual evaluation tasks
                    individual_ahimsa_tasks.append(evaluate_ahimsa_with_gemini(
                        original_prompt, generated_response, current_prompt_meta, temperature=temperature
                    ))
                    individual_dharma_tasks.append(evaluate_dharma_with_gemini(
                        original_prompt, generated_response, current_prompt_meta, temperature=temperature
                    ))
                    individual_helpfulness_tasks.append(evaluate_helpfulness_with_gemini(
                        original_prompt, generated_response, temperature=temperature
                    ))

                # Add individual evaluation tasks with concurrency control
                if individual_ahimsa_tasks:
                    evaluation_tasks_to_gather.append(asyncio.gather(*individual_ahimsa_tasks, return_exceptions=True))
                if individual_dharma_tasks:
                    evaluation_tasks_to_gather.append(asyncio.gather(*individual_dharma_tasks, return_exceptions=True))
                if individual_helpfulness_tasks:
                    evaluation_tasks_to_gather.append(asyncio.gather(*individual_helpfulness_tasks, return_exceptions=True))

            else:
                logger.error(f"Unknown gemini_eval_mode: {gemini_eval_mode}. Must be 'batch' or 'individual'.")
                # Fall back to batch mode
                gemini_eval_mode = "batch"
                logger.warning("Falling back to batch mode.")
                # Recursive call with corrected mode would be complex, so just error out
                raise ValueError(f"Invalid gemini_eval_mode: {gemini_eval_mode}")
        else:
            logger.error(f"Unknown evaluator: {evaluator}. Skipping evaluation for this batch.")
            batch_eval_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in batch_scenarios_data]
            for idx, res_dict in enumerate(batch_eval_results):
                res_dict["prompt"] = batch_original_prompts[idx]
                res_dict["response"] = batch_generated_responses[idx]
                res_dict["scenario_id"] = batch_scenarios_data[idx].get("scenario_id", f"batch_{i}_item_{idx}")
                res_dict["error"] = f"Unknown evaluator: {evaluator}"
            evaluation_results.extend(batch_eval_results)
            processed_scenarios += len(batch_scenarios_data)
            continue

        # Run evaluation tasks
        gathered_eval_task_results = []
        if evaluation_tasks_to_gather:
            try:
                gathered_eval_task_results = await asyncio.gather(*evaluation_tasks_to_gather, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during asyncio.gather for evaluations: {e}", exc_info=True)
                # If gather itself fails, fill with errors based on expected structure
                if evaluator == "openai":
                    gathered_eval_task_results = [[DEFAULT_EVAL_RESPONSE.copy() for _ in batch_scenarios_data]]
                elif evaluator == "gemini":
                    # Expected: [ahimsa_list, dharma_list, helpfulness_list]
                    gathered_eval_task_results = []
                    if ahimsa_payload_for_concurrent_eval: gathered_eval_task_results.append([DEFAULT_EVAL_RESPONSE.copy() for _ in ahimsa_payload_for_concurrent_eval])
                    if dharma_payload_for_concurrent_eval: gathered_eval_task_results.append([DEFAULT_EVAL_RESPONSE.copy() for _ in dharma_payload_for_concurrent_eval])
                    if helpfulness_payload_for_concurrent_eval: gathered_eval_task_results.append([DEFAULT_EVAL_RESPONSE.copy() for _ in helpfulness_payload_for_concurrent_eval])

        # Process and combine results for the batch
        current_batch_eval_results = []
        if evaluator == "openai":
            if gathered_eval_task_results and not isinstance(gathered_eval_task_results[0], Exception) and isinstance(gathered_eval_task_results[0], list):
                current_batch_eval_results = gathered_eval_task_results[0]
            elif gathered_eval_task_results and isinstance(gathered_eval_task_results[0], Exception):
                logger.error(f"OpenAI batch evaluation failed: {gathered_eval_task_results[0]}")
                current_batch_eval_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in batch_scenarios_data]
            else:
                logger.error(f"Unexpected result structure from batch_evaluate_with_openai: {gathered_eval_task_results}")
                current_batch_eval_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in batch_scenarios_data]
        elif evaluator == "gemini":
            gemini_ahimsa_batch_results = []
            gemini_dharma_batch_results = []
            gemini_helpfulness_batch_results = []

            current_gathered_idx = 0

            if gemini_eval_mode == "batch":
                # Process batch mode results
                # Ahimsa results (from batch_process_ahimsa_evaluations_concurrently)
                if ahimsa_payload_for_concurrent_eval: # Check if the task was added
                    if gathered_eval_task_results and len(gathered_eval_task_results) > current_gathered_idx:
                        ahimsa_task_result = gathered_eval_task_results[current_gathered_idx]
                        if isinstance(ahimsa_task_result, Exception):
                            logger.error(f"Concurrent Ahimsa processing for Gemini failed: {ahimsa_task_result}")
                            gemini_ahimsa_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in ahimsa_payload_for_concurrent_eval]
                        elif isinstance(ahimsa_task_result, list):
                            gemini_ahimsa_batch_results = ahimsa_task_result # This is already the flat list of results
                        else:
                            logger.warning(f"Concurrent Ahimsa for Gemini returned unexpected type: {type(ahimsa_task_result)}. Results: {ahimsa_task_result}")
                            gemini_ahimsa_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in ahimsa_payload_for_concurrent_eval]
                    else:
                        logger.warning("Ahimsa results missing from gathered Gemini tasks (concurrent processing path).")
                        gemini_ahimsa_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in ahimsa_payload_for_concurrent_eval]
                    current_gathered_idx += 1

                # Dharma results (from batch_process_dharma_evaluations_concurrently)
                if dharma_payload_for_concurrent_eval: # Check if the Dharma batch task was added
                    if gathered_eval_task_results and len(gathered_eval_task_results) > current_gathered_idx:
                        dharma_task_result = gathered_eval_task_results[current_gathered_idx]
                        if isinstance(dharma_task_result, Exception):
                            logger.error(f"Concurrent Dharma processing for Gemini failed: {dharma_task_result}")
                            gemini_dharma_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in dharma_payload_for_concurrent_eval]
                        elif isinstance(dharma_task_result, list):
                            gemini_dharma_batch_results = dharma_task_result
                        else:
                            logger.warning(f"Concurrent Dharma for Gemini returned unexpected type: {type(dharma_task_result)}. Results: {dharma_task_result}")
                            gemini_dharma_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in dharma_payload_for_concurrent_eval]
                    else:
                        logger.warning("Dharma results missing from gathered Gemini tasks (concurrent processing path).")
                        gemini_dharma_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in dharma_payload_for_concurrent_eval]
                    current_gathered_idx += 1

                # Process helpfulness batch results
                if helpfulness_payload_for_concurrent_eval: # Check if the helpfulness batch task was added
                    if gathered_eval_task_results and len(gathered_eval_task_results) > current_gathered_idx:
                        helpfulness_batch_task_result = gathered_eval_task_results[current_gathered_idx]
                        if isinstance(helpfulness_batch_task_result, Exception):
                            logger.error(f"Concurrent Helpfulness processing for Gemini failed: {helpfulness_batch_task_result}")
                            gemini_helpfulness_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in helpfulness_payload_for_concurrent_eval]
                        elif isinstance(helpfulness_batch_task_result, list):
                            gemini_helpfulness_batch_results = helpfulness_batch_task_result
                        else:
                            logger.warning(f"Concurrent Helpfulness for Gemini returned unexpected type: {type(helpfulness_batch_task_result)}. Results: {helpfulness_batch_task_result}")
                            gemini_helpfulness_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in helpfulness_payload_for_concurrent_eval]
                    else:
                        logger.warning("Helpfulness batch results missing from gathered Gemini tasks (concurrent processing path).")
                        gemini_helpfulness_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in helpfulness_payload_for_concurrent_eval]
                    current_gathered_idx += 1

            elif gemini_eval_mode == "individual":
                # Process individual mode results
                # Ahimsa individual results
                if individual_ahimsa_tasks:
                    if gathered_eval_task_results and len(gathered_eval_task_results) > current_gathered_idx:
                        ahimsa_individual_result = gathered_eval_task_results[current_gathered_idx]
                        if isinstance(ahimsa_individual_result, Exception):
                            logger.error(f"Individual Ahimsa processing for Gemini failed: {ahimsa_individual_result}")
                            gemini_ahimsa_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_ahimsa_tasks]
                        elif isinstance(ahimsa_individual_result, list):
                            gemini_ahimsa_batch_results = ahimsa_individual_result
                        else:
                            logger.warning(f"Individual Ahimsa for Gemini returned unexpected type: {type(ahimsa_individual_result)}. Results: {ahimsa_individual_result}")
                            gemini_ahimsa_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_ahimsa_tasks]
                    else:
                        logger.warning("Individual Ahimsa results missing from gathered Gemini tasks.")
                        gemini_ahimsa_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_ahimsa_tasks]
                    current_gathered_idx += 1

                # Dharma individual results
                if individual_dharma_tasks:
                    if gathered_eval_task_results and len(gathered_eval_task_results) > current_gathered_idx:
                        dharma_individual_result = gathered_eval_task_results[current_gathered_idx]
                        if isinstance(dharma_individual_result, Exception):
                            logger.error(f"Individual Dharma processing for Gemini failed: {dharma_individual_result}")
                            gemini_dharma_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_dharma_tasks]
                        elif isinstance(dharma_individual_result, list):
                            gemini_dharma_batch_results = dharma_individual_result
                        else:
                            logger.warning(f"Individual Dharma for Gemini returned unexpected type: {type(dharma_individual_result)}. Results: {dharma_individual_result}")
                            gemini_dharma_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_dharma_tasks]
                    else:
                        logger.warning("Individual Dharma results missing from gathered Gemini tasks.")
                        gemini_dharma_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_dharma_tasks]
                    current_gathered_idx += 1

                # Helpfulness individual results
                if individual_helpfulness_tasks:
                    if gathered_eval_task_results and len(gathered_eval_task_results) > current_gathered_idx:
                        helpfulness_individual_result = gathered_eval_task_results[current_gathered_idx]
                        if isinstance(helpfulness_individual_result, Exception):
                            logger.error(f"Individual Helpfulness processing for Gemini failed: {helpfulness_individual_result}")
                            gemini_helpfulness_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_helpfulness_tasks]
                        elif isinstance(helpfulness_individual_result, list):
                            gemini_helpfulness_batch_results = helpfulness_individual_result
                        else:
                            logger.warning(f"Individual Helpfulness for Gemini returned unexpected type: {type(helpfulness_individual_result)}. Results: {helpfulness_individual_result}")
                            gemini_helpfulness_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_helpfulness_tasks]
                    else:
                        logger.warning("Individual Helpfulness results missing from gathered Gemini tasks.")
                        gemini_helpfulness_batch_results = [DEFAULT_EVAL_RESPONSE.copy() for _ in individual_helpfulness_tasks]
                    current_gathered_idx += 1

            # Combine results for each item in the original batch_scenarios_data
            for i_item in range(len(batch_scenarios_data)):
                # Safely get results for each item, defaulting if lists are shorter (e.g., due to earlier generation failures)
                ahimsa_res = gemini_ahimsa_batch_results[i_item] if i_item < len(gemini_ahimsa_batch_results) else DEFAULT_EVAL_RESPONSE.copy()
                dharma_res = gemini_dharma_batch_results[i_item] if i_item < len(gemini_dharma_batch_results) else DEFAULT_EVAL_RESPONSE.copy()
                # Use helpfulness_batch_results now
                help_res = gemini_helpfulness_batch_results[i_item] if i_item < len(gemini_helpfulness_batch_results) else DEFAULT_EVAL_RESPONSE.copy()

                # If any of the results are exceptions (from return_exceptions=True in inner gathers), convert to error dict
                if isinstance(ahimsa_res, Exception): ahimsa_res = {"error": f"Ahimsa Eval Exception: {str(ahimsa_res)}", **DEFAULT_EVAL_RESPONSE}
                if isinstance(dharma_res, Exception): dharma_res = {"error": f"Dharma Eval Exception: {str(dharma_res)}", **DEFAULT_EVAL_RESPONSE}
                if isinstance(help_res, Exception): help_res = {"error": f"Helpfulness Eval Exception: {str(help_res)}", **DEFAULT_EVAL_RESPONSE}

                # Special handling for GENERATION_FAILED_PLACEHOLDER from multi-evaluators
                if isinstance(ahimsa_res, dict) and ahimsa_res.get("error", "") == "Original generation failed":
                    logger.info(f"Detected error from multi_ahimsa due to GENERATION_FAILED_PLACEHOLDER for item {i_item}. Using default error.")
                    ahimsa_res = {"error": "Model generation failed (placeholder detected in multi_eval)", **DEFAULT_EVAL_RESPONSE}
                if isinstance(dharma_res, dict) and dharma_res.get("error", "") == "Original generation failed": # Check for Dharma too
                    logger.info(f"Detected error from multi_dharma due to GENERATION_FAILED_PLACEHOLDER for item {i_item}. Using default error.")
                    dharma_res = {"error": "Model generation failed (placeholder detected in multi_eval)", **DEFAULT_EVAL_RESPONSE}

                combined_item_result = DEFAULT_EVAL_RESPONSE.copy() # Start fresh

                # Ahimsa fields (prioritize from ahimsa_res)
                combined_item_result.update(ahimsa_res) # Base update, ensure this is present
                # Ensure specific Ahimsa fields are from ahimsa_res if they exist, otherwise default
                for key in ["ahimsa_score", "ahimsa_violation", "harm_avoidance_score", "safety_context_score", "has_medical_disclaimer", "has_professional_referral", "referral_strength", "severity"]:
                    ultimate_fallback_value = None
                    if key == "ahimsa_score": ultimate_fallback_value = 0.0
                    elif key == "ahimsa_violation": ultimate_fallback_value = True
                    elif key == "harm_avoidance_score": ultimate_fallback_value = 0.0
                    elif key == "safety_context_score": ultimate_fallback_value = 0.0
                    elif key == "has_medical_disclaimer": ultimate_fallback_value = False
                    elif key == "has_professional_referral": ultimate_fallback_value = False
                    elif key == "referral_strength": ultimate_fallback_value = "none"
                    elif key == "severity": ultimate_fallback_value = "none"

                    combined_item_result[key] = ahimsa_res.get(key, DEFAULT_EVAL_RESPONSE.get(key, ultimate_fallback_value))

                # Dharma fields (update, but don't overwrite ahimsa_score/violation)
                for key, value in dharma_res.items():
                    if key not in ["ahimsa_score", "ahimsa_violation"]:
                        combined_item_result[key] = value
                # Ensure specific Dharma fields are from dharma_res or default
                for key in ["dharma_score", "dharma_violation", "domain_adherence_score", "out_of_domain_advice", "response_scope", "scope_risk", "scope_penalty_factor"]:
                     combined_item_result[key] = dharma_res.get(key, DEFAULT_EVAL_RESPONSE.get(key, None)) # Use .get for DEFAULT_EVAL_RESPONSE too

                # Helpfulness fields (update, don't overwrite ahimsa/dharma scores/violations)
                for key, value in help_res.items():
                    if key not in ["ahimsa_score", "ahimsa_violation", "dharma_score", "dharma_violation"]:
                        combined_item_result[key] = value
                # Ensure specific Helpfulness fields are from help_res or default
                for key in ["helpfulness_score", "helpfulness_violation", "clarity_score", "relevance_score", "completeness_score"]:
                    combined_item_result[key] = help_res.get(key, DEFAULT_EVAL_RESPONSE.get(key, None))

                # Calculate combined score
                ahimsa_s = combined_item_result.get("ahimsa_score", 0.0)
                dharma_s = combined_item_result.get("dharma_score", 0.0)
                helpfulness_s = combined_item_result.get("helpfulness_score", 0.0)

                combined_item_result["combined_score"] = (\
                    (ahimsa_s * REWARD_WEIGHTS.get("ahimsa", 0.0)) +\
                    (dharma_s * REWARD_WEIGHTS.get("dharma", 0.0)) +\
                    (helpfulness_s * REWARD_WEIGHTS.get("helpfulness", 0.0))\
                )
                # Ensure the combined score is capped between 0 and 1 if scores are 0-1
                # This depends on how scores are scaled. Assuming they are effectively 0-1 for combined purposes.
                # combined_item_result["combined_score"] = max(0.0, min(1.0, combined_item_result["combined_score"]))

                # Consolidate reasoning and errors
                all_reasonings = []
                if ahimsa_res.get("reasoning") and ahimsa_res["reasoning"] != DEFAULT_EVAL_RESPONSE["reasoning"]: all_reasonings.append(f"A: {ahimsa_res['reasoning']}")
                if dharma_res.get("reasoning") and dharma_res["reasoning"] != DEFAULT_EVAL_RESPONSE["reasoning"]: all_reasonings.append(f"D: {dharma_res['reasoning']}")
                if help_res.get("reasoning") and help_res["reasoning"] != DEFAULT_EVAL_RESPONSE["reasoning"]: all_reasonings.append(f"H: {help_res['reasoning']}")
                combined_item_result["reasoning"] = " | ".join(all_reasonings) if all_reasonings else DEFAULT_EVAL_RESPONSE["reasoning"]

                all_errors = []
                if ahimsa_res.get("error") and ahimsa_res["error"] != DEFAULT_EVAL_RESPONSE["error"]: all_errors.append(f"A_Err: {ahimsa_res['error']}")
                if dharma_res.get("error") and dharma_res["error"] != DEFAULT_EVAL_RESPONSE["error"]: all_errors.append(f"D_Err: {dharma_res['error']}")
                if help_res.get("error") and help_res["error"] != DEFAULT_EVAL_RESPONSE["error"]: all_errors.append(f"H_Err: {help_res['error']}")

                if all_errors:
                    combined_item_result["error"] = " | ".join(all_errors)
                elif "error" in combined_item_result and combined_item_result["error"] == DEFAULT_EVAL_RESPONSE["error"]:
                     # If it only has the default error and no specific ones, remove it
                     # Make sure an actual error didn't occur that matched default message.
                     if not (isinstance(ahimsa_res, dict) and ahimsa_res.get("error") and ahimsa_res.get("error") != DEFAULT_EVAL_RESPONSE["error"] or
                             isinstance(dharma_res, dict) and dharma_res.get("error") and dharma_res.get("error") != DEFAULT_EVAL_RESPONSE["error"] or
                             isinstance(help_res, dict) and help_res.get("error") and help_res.get("error") != DEFAULT_EVAL_RESPONSE["error"]):
                        del combined_item_result["error"]
                elif "error" not in combined_item_result and not all_errors:
                     pass

                current_batch_eval_results.append(combined_item_result)

        # Add prompt, response, and scenario_id to each result in the batch
        for i_item in range(len(batch_scenarios_data)):
            current_batch_eval_results[i_item]["prompt"] = batch_original_prompts[i_item]
            current_batch_eval_results[i_item]["response"] = batch_generated_responses[i_item]
            current_batch_eval_results[i_item]["scenario_id"] = batch_scenarios_data[i_item].get("scenario_id", f"batch_{i}_item_{i_item}")

        evaluation_results.extend(current_batch_eval_results)
        processed_scenarios += len(batch_scenarios_data)

    # Calculate final metrics
    final_metrics = calculate_metrics(evaluation_results)

    # Set evaluator info based on which evaluator was used
    if evaluator == "openai":
        evaluator_info = f"openai ({OPENAI_EVAL_MODEL})"
    else:  # evaluator == "gemini"
        evaluator_info = f"gemini ({GEMINI_EVAL_MODEL})"

    # Get scope penalty configuration from the reward functions
    from argen.reward_functions.openai_rewards import SCOPE_PENALTY_TABLE as openai_scope_penalty

    # Create a copy of the scope penalty table for the output
    scope_penalty_config = {f"{k[0]}-{k[1]}": v for k, v in openai_scope_penalty.items()}

    output_data = {
        "evaluation_config": {
            "model_name": model_name,
            "evaluator": evaluator_info,
            "num_scenarios": len(scenarios),
            "temperature": temperature,
            "test_mode": test_mode,
            "system_prompt_type": system_prompt_type,
            "generation_batch_size": generation_batch_size,
            "ahimsa_weight": REWARD_WEIGHTS["ahimsa"],
            "dharma_weight": REWARD_WEIGHTS["dharma"],
            "helpfulness_weight": REWARD_WEIGHTS["helpfulness"],
            "apply_medical_disclaimer_penalty": PENALTY_CONFIG["apply_medical_disclaimer_penalty"],
            "referral_policy": PENALTY_CONFIG["referral_policy"],
            "scope_penalty_config": scope_penalty_config,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "summary_metrics": final_metrics,
        "individual_results": evaluation_results
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Evaluation results saved to {output_file}")
    except IOError as e:
        logging.error(f"Error saving results to {output_file}: {e}")


# Backward compatibility function
async def evaluate_model_with_openai(
    model_name: str,
    scenarios: List[Dict],
    output_file: str,
    temperature: float,
    openai_api_key: str,
    test_mode: bool = False,
    system_prompt_type: str = 'ENHANCED',
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    apply_medical_disclaimer_penalty: Optional[bool] = None,
    apply_professional_referral_penalty: Optional[bool] = None
) -> None:
    """
    Backward compatibility function that calls evaluate_model_with_llm with OpenAI evaluator.

    Args:
        model_name: Identifier of the model to evaluate (HF path/name or Predibase name).
        scenarios: A list of scenario dictionaries.
        output_file: Path to save the evaluation results (JSON).
        temperature: Temperature for model generation.
        openai_api_key: The OpenAI API key to use for evaluation.
        test_mode: If True, uses mock generation and potentially mock evaluation.
        system_prompt_type: Type of system prompt to use ('BASIC' or 'ENHANCED').
        generation_batch_size: Number of prompts to generate in parallel locally.
        apply_medical_disclaimer_penalty: Whether to apply penalty for missing medical disclaimer.
            If None, uses the default from config.
        apply_professional_referral_penalty: Whether to apply penalty for missing professional referral.
            If None, uses the default from config.
    """
    return await evaluate_model_with_llm(
        model_name=model_name,
        scenarios=scenarios,
        output_file=output_file,
        temperature=temperature,
        openai_api_key=openai_api_key,
        evaluator="openai",
        test_mode=test_mode,
        system_prompt_type=system_prompt_type,
        generation_batch_size=generation_batch_size,
        apply_medical_disclaimer_penalty=apply_medical_disclaimer_penalty,
        apply_professional_referral_penalty=apply_professional_referral_penalty
    )
