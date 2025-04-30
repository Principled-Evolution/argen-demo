"""
OpenAI-based evaluator for ArGen fine-tuning.

This module provides functions to evaluate a model using the OpenAI API (e.g., GPT-4o-mini) 
for both Ahimsa and Dharma principles.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import datetime
import logging

from src.reward_functions.openai_rewards import (
    evaluate_ahimsa_with_openai,
    evaluate_dharma_with_openai,
    DEFAULT_EVAL_RESPONSE
)
from src.config import (
    DEFAULT_MAX_NEW_TOKENS,
    ENHANCED_SYSTEM_PROMPT,
    BASIC_SYSTEM_PROMPT,
    OPENAI_EVAL_MODEL,
    OPENAI_EVAL_TEMPERATURE,
    OPENAI_EVAL_MAX_TOKENS,
    REWARD_WEIGHTS
)
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            "average_combined_score": 0.0
        }

    ahimsa_violations = sum(1 for r in results if r.get("ahimsa_violation", False))
    dharma_violations = sum(1 for r in results if r.get("dharma_violation", False))
    
    total_ahimsa_score = sum(r.get("ahimsa_score", 0.0) for r in results)
    total_dharma_score = sum(r.get("dharma_score", 0.0) for r in results)
    total_combined_score = sum(r.get("combined_score", 0.0) for r in results)

    return {
        "ahimsa_violations": ahimsa_violations,
        "ahimsa_violation_rate": ahimsa_violations / num_scenarios,
        "average_ahimsa_score": total_ahimsa_score / num_scenarios,
        "dharma_violations": dharma_violations,
        "dharma_violation_rate": dharma_violations / num_scenarios,
        "average_dharma_score": total_dharma_score / num_scenarios,
        "average_combined_score": total_combined_score / num_scenarios,
    }


def generate_model_response(
    model_name: str,
    prompt: str,
    temperature: float = 0.9,
    api_token: Optional[str] = None,
    tenant_id: str = "default",
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """
    Generate a response from a model using Predibase with retry logic.
    
    Args:
        model_name: Name of the model to use
        prompt: Prompt to send to the model (formatted by caller)
        temperature: Temperature for generation
        api_token: Predibase API token
        tenant_id: Predibase tenant ID
        max_retries: Maximum number of times to retry the API call
        retry_delay: Delay in seconds between retries
        
    Returns:
        The model's response, or an error string if all retries fail.
    """
    
    last_error = None

    for attempt in range(max_retries):
        current_api_token = api_token
        if current_api_token is None:
            try:
                config_path = os.path.expanduser("~/.predibase/config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    current_api_token = config.get("api_key")
                if not current_api_token:
                    current_api_token = os.getenv("PREDIBASE_API_TOKEN")
                if not current_api_token:
                     raise ValueError("Predibase API token not found in config file or PREDIBASE_API_TOKEN environment variable.")
            except Exception as e:
                last_error = f"Error getting API token: {str(e)}"
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed to get API token: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return last_error

        url = f"https://serving.app.predibase.com/{tenant_id}/deployments/v2/llms/{model_name}/generate"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_api_token}"
        }
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                "temperature": temperature,
                "do_sample": True
            }
        }
        
        try:
            import requests
            response_obj = requests.post(url, headers=headers, json=data, timeout=60) 
            response_obj.raise_for_status()
            response_data = response_obj.json()
            response = response_data.get("generated_text", "")

            if not response:
                 raise ValueError("Received empty response from API")

            if response.startswith(prompt):
                 response = response[len(prompt):]
            
            return response.strip()

        except requests.exceptions.RequestException as e:
            last_error = f"Error calling Predibase API: {str(e)}"
            print(f"Attempt {attempt + 1} failed: {last_error}")
        except Exception as e:
            last_error = f"Error processing Predibase response: {str(e)}"
            print(f"Attempt {attempt + 1} failed: {last_error}")

        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    print(f"All {max_retries} attempts failed. Returning last error.")
    return last_error


def get_mock_response(prompt: str) -> str:
    """Returns a generic mock response for testing."""
    return (
        "Mock Response: I understand your concern. As an AI, I cannot provide medical advice. "
        "It is best to consult with a qualified healthcare professional for any health questions or concerns."
    )


def _generate_response(
    model_name: str,
    prompt: str,
    temperature: float,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    test_mode: bool = False,
    use_basic_prompt: bool = False
) -> Tuple[Optional[str], float]:
    """
    Generates a response from the specified model (always uses local inference).

    Args:
        model_name: Identifier for the model (HF name/path).
        prompt: The fully formatted prompt to send to the model.
        temperature: Temperature for generation.
        model: Pre-loaded Hugging Face model (for local generation).
        tokenizer: Pre-loaded Hugging Face tokenizer (for local generation).
        max_new_tokens: Max tokens for local generation.
        max_retries: Max retries for generation (kept for API compatibility).
        retry_delay: Delay for retry (kept for API compatibility).
        test_mode: If True, returns a mock response instead of calling the model.
        use_basic_prompt: Flag potentially influencing prompt logic (kept for now).

    Returns:
        A tuple containing the model response (or None on error) and generation time.
    """
    if test_mode:
        return get_mock_response(prompt), 0.0

    start_time = time.time()
    response = None
    last_error = "Generation failed after retries."

    # Check if model and tokenizer are loaded
    if not model or not tokenizer:
        logging.error("Local generation called but model/tokenizer not provided.")
        return None, time.time() - start_time

    logging.info(f"Generating response locally for model {model_name} (first 50 chars): {prompt[:50]}...")
    
    try:
        # Ensure the model is on the correct device (it should be from device_map="auto")
        device = model.device
        logging.info(f"Using device: {device} from pre-loaded model.")

        # Create a text generation pipeline ON THE FLY with the loaded components
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            # Device is handled by accelerate (device_map="auto") when loading model
        )

        # Handle chat formatted prompts vs raw text (similar logic as before)
        is_chat_formatted = "<|start_header_id|>" in prompt and "<|end_header_id|>" in prompt
        formatted_inputs = prompt # Default to using the prompt as is

        if is_chat_formatted and hasattr(tokenizer, 'apply_chat_template'):
            # Attempt to reconstruct messages if possible, otherwise use raw prompt
            user_message_match = re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", prompt, re.DOTALL)
            system_message_match = re.search(r"<\|start_header_id\|>system<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", prompt, re.DOTALL)

            if user_message_match:
                user_content = user_message_match.group(1).strip()
                messages = []
                if system_message_match:
                    messages.append({"role": "system", "content": system_message_match.group(1).strip()})
                messages.append({"role": "user", "content": user_content})
                
                # Let the tokenizer handle the template application for the pipeline
                formatted_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                logging.debug(f"Applied chat template. Input to pipeline (start): {formatted_inputs[:100]}...")
        
        # Generate response using the pipeline
        outputs = pipe(formatted_inputs)
        response = outputs[0]['generated_text']
        logging.debug(f"Pipeline output (start): {response[:100]}...")

        # Remove the input prompt part from the pipeline output
        # This is crucial as pipeline often includes the input
        if response.startswith(formatted_inputs):
            response = response[len(formatted_inputs):]
        # Fallback: if original pre-formatted prompt was used, try removing that too
        elif formatted_inputs == prompt and response.startswith(prompt):
             response = response[len(prompt):]

        if not response:
            raise ValueError("Received empty response from local model pipeline")

        response = response.strip() # Ensure clean output

    except Exception as e:
        logging.exception(f"Error during local generation pipeline: {e}") # Log stack trace
        response = f"Error: Local generation failed - {str(e)}" # Set error response

    end_time = time.time()
    generation_time = end_time - start_time

    # Check for error strings or empty/None responses
    if isinstance(response, str) and response.startswith("Error:"):
        logging.warning(f"Model generation failed or returned error: {response}")
        return None, generation_time
    elif not isinstance(response, str) or not response.strip():
        logging.warning(f"Model generation returned unexpected or empty response: {response}")
        return None, generation_time

    logging.info(f"Generated response received (length {len(response)}). Time: {generation_time:.2f}s")
    return response, generation_time


async def evaluate_model_with_openai(
    model_name: str,
    scenarios: List[Dict],
    output_file: str,
    temperature: float,
    openai_api_key: str,
    test_mode: bool = False,
    use_basic_prompt: bool = False
) -> None:
    """
    Evaluates a model's responses to scenarios using OpenAI for scoring.

    Args:
        model_name: Identifier of the model to evaluate (HF path/name or Predibase name).
        scenarios: A list of scenario dictionaries.
        output_file: Path to save the evaluation results (JSON).
        temperature: Temperature for model generation.
        openai_api_key: The OpenAI API key to use for evaluation.
        test_mode: If True, uses mock generation and potentially mock evaluation.
        use_basic_prompt: If True, uses a basic system prompt for generation.
    """
    results = []
    failed_generations = 0
    local_model = None
    local_tokenizer = None

    # --- Always load model locally --- 
    if not test_mode:
        logging.info(f"Loading model locally: {model_name}")
        try:
            # Device selection (prefer GPU if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Setting up device: {device}")

            local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, # Use bfloat16 for L4 compatibility
                device_map="auto" # Automatically use available GPU(s)
            )
            logging.info(f"Successfully loaded model {model_name} onto device(s): {local_model.hf_device_map}")
        except Exception as e:
            logging.exception(f"Failed to load model {model_name}: {e}")
            print(f"ERROR: Could not load the model '{model_name}'. Please check the path/name and dependencies.")
            # Exit or handle error appropriately - Cannot proceed without the model
            return # Stop evaluation if model loading fails

    # Define system prompts based on the flag
    system_prompt = BASIC_SYSTEM_PROMPT if use_basic_prompt else ENHANCED_SYSTEM_PROMPT

    logging.info(f"Starting OpenAI evaluation for {len(scenarios)} scenarios...")

    for i, scenario in enumerate(tqdm(scenarios, desc="Evaluating scenarios")):
        original_prompt = scenario['prompt']
        conversation_history = scenario.get('conversation_history', [])
        
        # --- Prepare the prompt for the model being evaluated --- 
        # Using Unsloth's recommended chat template format for Llama-3.2
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        # Add previous turns if they exist
        if conversation_history:
            messages.extend(conversation_history)
        # Add the current user prompt
        messages.append({"role": "user", "content": original_prompt})
        
        # Format the prompt using tokenizer if available, otherwise use a fallback format
        if local_tokenizer and not test_mode:
            try:
                formatted_prompt = local_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                ) 
            except Exception as e:
                logging.warning(f"Error applying chat template with tokenizer for scenario {i}: {e}. Using fallback format.")
                formatted_prompt = _create_fallback_prompt(messages)
        else:
            # Fallback format for test mode or if tokenizer isn't available
            formatted_prompt = _create_fallback_prompt(messages)

        # --- Generate model response --- 
        model_response, generation_time = _generate_response(
            model_name=model_name, # Pass name for identification/logging
            prompt=formatted_prompt,
            temperature=temperature,
            model=local_model,      # Pass loaded model (or None)
            tokenizer=local_tokenizer, # Pass loaded tokenizer (or None)
            test_mode=test_mode,
            use_basic_prompt=use_basic_prompt # Keep passing this
        )

        # --- Evaluate response --- 
        ahimsa_eval = {}
        dharma_eval = {}
        # Remove individual timers, time the gather operation
        # ahimsa_eval_time = 0.0
        # dharma_eval_time = 0.0
        concurrent_eval_time = 0.0
        
        if model_response is None:
            failed_generations += 1
            # Keep default error responses
            ahimsa_eval = {"ahimsa_score": 0.0, "ahimsa_reasoning": "Model generation failed", "ahimsa_violation": True}
            dharma_eval = {"dharma_score": 0.0, "dharma_reasoning": "Model generation failed", "dharma_violation": True}
        else:
            # Use asyncio.gather to run evaluations concurrently
            eval_start_time = time.time()
            try:
                # Gather results from async functions
                results_list = await asyncio.gather(
                    evaluate_ahimsa_with_openai(original_prompt, model_response, openai_api_key=openai_api_key),
                    evaluate_dharma_with_openai(original_prompt, model_response, openai_api_key=openai_api_key)
                )
                concurrent_eval_time = time.time() - eval_start_time
                ahimsa_eval = results_list[0] # Result from first function call
                dharma_eval = results_list[1] # Result from second function call
                logging.info(f"Concurrent OpenAI evaluations completed in {concurrent_eval_time:.2f}s")
            except Exception as gather_err:
                # Handle potential errors during gather (e.g., network issues in one call)
                logging.error(f"Error during concurrent evaluation: {gather_err}")
                concurrent_eval_time = time.time() - eval_start_time # Still record time
                # Use default error responses if gather fails
                ahimsa_eval = {**DEFAULT_EVAL_RESPONSE, "error": f"Gather Error: {gather_err}"}
                dharma_eval = {**DEFAULT_EVAL_RESPONSE, "error": f"Gather Error: {gather_err}"}

        # --- Combine and store results --- 
        ahimsa_weight = REWARD_WEIGHTS["ahimsa"]
        dharma_weight = REWARD_WEIGHTS["dharma"]
        combined_score = (
            ahimsa_eval.get('ahimsa_score', 0.0) * ahimsa_weight + 
            dharma_eval.get('dharma_score', 0.0) * dharma_weight
        ) / (ahimsa_weight + dharma_weight)
        
        results.append({
            "scenario_index": i,
            "prompt": original_prompt,
            "conversation_history": conversation_history,
            "generated_response": model_response,
            "generation_time_seconds": generation_time,
            "concurrent_eval_time_seconds": concurrent_eval_time, # Store total concurrent time
            **ahimsa_eval,
            **dharma_eval,
            "combined_score": combined_score
        })

    # --- Cleanup after loop --- 
    if local_model is not None:
        logging.info("Unloading local model...")
        del local_model
    if local_tokenizer is not None:
        del local_tokenizer
    if torch.cuda.is_available():
        logging.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    logging.info(f"Evaluation loop completed. Processed: {len(scenarios)}, Failed Generation: {failed_generations}")

    # Calculate final metrics
    final_metrics = calculate_metrics(results)
    
    # Save results
    output_data = {
        "evaluation_config": {
            "model_name": model_name,
            "evaluator": f"openai ({OPENAI_EVAL_MODEL})",
            "num_scenarios": len(scenarios),
            "temperature": temperature,
            "test_mode": test_mode,
            "use_basic_prompt": use_basic_prompt,
            "ahimsa_weight": REWARD_WEIGHTS["ahimsa"],
            "dharma_weight": REWARD_WEIGHTS["dharma"],
            "timestamp": datetime.datetime.now().isoformat()
        },
        "summary_metrics": final_metrics,
        "individual_results": results
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Evaluation results saved to {output_file}")
    except IOError as e:
        logging.error(f"Error saving results to {output_file}: {e}")
