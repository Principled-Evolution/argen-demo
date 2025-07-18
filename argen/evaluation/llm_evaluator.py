"""
LLM-based evaluator for ArGen fine-tuning.

This module provides functions to evaluate a model using multiple LLM providers
(OpenAI, Gemini, Anthropic) for Ahimsa, Dharma, and Helpfulness principles.
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

from argen.reward_functions.gemini.ahimsa import (
    batch_process_ahimsa_evaluations_concurrently
)

from argen.reward_functions.gemini.dharma import (
    batch_process_dharma_evaluations_concurrently
)

from argen.reward_functions.gemini.helpfulness import (
    batch_process_helpfulness_evaluations_concurrently
)

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

# Import data integrity utilities
from argen.utils.data_integrity import (
    extract_tier_from_compound
)

# Import model loading utilities
from argen.utils.model_utils import (
    load_model_and_tokenizer,
    generate_responses_batch,
    apply_chat_template_if_needed
)

# Import evaluation utilities
from argen.utils.evaluation_utils import (
    calculate_combined_score,
    save_evaluation_results,
    create_evaluation_summary,
    load_scenarios_from_file
)

# Import environment utilities
from argen.utils.env import (
    get_openai_api_key,
    get_gemini_api_key
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompts_responses_from_json(json_file_path: str) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """
    Load prompt-response pairs from a JSON file.

    Args:
        json_file_path: Path to the JSON file containing evaluation results

    Returns:
        Tuple of (prompt_response_pairs, scenario_metadata)
        - prompt_response_pairs: List of (prompt, response) tuples
        - scenario_metadata: List of scenario metadata dictionaries
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract individual results
        individual_results = data.get('individual_results', [])
        if not individual_results:
            raise ValueError(f"No 'individual_results' found in JSON file: {json_file_path}")

        prompt_response_pairs = []
        scenario_metadata = []

        for result in individual_results:
            prompt = result.get('prompt', '')
            response = result.get('response', '')
            scenario_id = result.get('scenario_id', '')

            if not prompt or not response:
                logger.warning(f"Skipping result with missing prompt or response: {scenario_id}")
                continue

            prompt_response_pairs.append((prompt, response))

            # Create scenario metadata (extract what we can from the result)
            scenario_meta = {
                'scenario_id': scenario_id,
                'prompt': prompt,
                # Try to extract tier and scope if available in the original data
                'tier': 'C',  # Default tier
                'scope': 'S0'  # Default scope
            }
            scenario_metadata.append(scenario_meta)

        logger.info(f"Loaded {len(prompt_response_pairs)} prompt-response pairs from {json_file_path}")
        return prompt_response_pairs, scenario_metadata

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {json_file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading JSON file {json_file_path}: {e}")



def generate_responses_locally(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 1,
    system_prompt_type: str = 'ENHANCED'
) -> List[str]:
    """
    Generate responses using a locally loaded model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompts: List of prompts to generate responses for
        temperature: Temperature for generation
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for generation
        system_prompt_type: Type of system prompt to use
    
    Returns:
        List of generated responses
    """
    responses = []
    system_prompt = get_system_prompt(system_prompt_type)
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Apply chat template if needed
        formatted_prompts = []
        for prompt in batch_prompts:
            formatted_prompt = apply_chat_template_if_needed(
                tokenizer, prompt, system_prompt
            )
            formatted_prompts.append(formatted_prompt)
        
        # Generate responses for this batch
        batch_responses = generate_responses_batch(
            model, tokenizer, formatted_prompts, 
            temperature, max_new_tokens
        )
        responses.extend(batch_responses)
    
    return responses

def is_local_model_path(model_name: str) -> bool:
    """
    Check if the model name refers to a local model path.
    
    Args:
        model_name: The model name or path
    
    Returns:
        True if it's a local path, False otherwise
    """
    return os.path.exists(model_name) or "/" in model_name

async def generate_responses_for_scenarios(
    model_name: str,
    scenarios: List[Dict],
    temperature: float,
    system_prompt_type: str = 'ENHANCED',
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    test_mode: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Generate responses for a list of scenarios using local model.

    Args:
        model_name: Name or path of the model to use
        scenarios: List of scenario dictionaries
        temperature: Temperature for generation
        system_prompt_type: Type of system prompt to use
        generation_batch_size: Batch size for generation
        test_mode: Whether to use test mode (shorter responses)

    Returns:
        Tuple of (original_prompts, generated_responses)
    """
    original_prompts = [scenario['prompt'] for scenario in scenarios]
    
    if test_mode:
        # In test mode, return mock responses
        generated_responses = [f"Test response for: {prompt[:50]}..." for prompt in original_prompts]
        return original_prompts, generated_responses
    
    if is_local_model_path(model_name):
        # Load model locally
        logger.info(f"Loading local model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name)
        
        # Generate responses locally
        generated_responses = generate_responses_locally(
            model, tokenizer, original_prompts, temperature,
            batch_size=generation_batch_size,
            system_prompt_type=system_prompt_type
        )
        
        # Clean up model from memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    else:
        raise ValueError(f"Unknown model type: {model_name}. Only local model paths are supported.")
    
    return original_prompts, generated_responses


async def evaluate_model_with_llm(
    model_name: str,
    scenarios: List[Dict],
    output_file: str,
    temperature: float,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,  # Add Anthropic support
    evaluator: str = "openai",
    test_mode: bool = False,
    system_prompt_type: str = 'ENHANCED',
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    apply_medical_disclaimer_penalty: Optional[bool] = None,
    apply_professional_referral_penalty: Optional[bool] = None,
    gemini_eval_mode: str = "batch",
    skip_evaluation: bool = False,
    openai_model: Optional[str] = None,      # Add model selection
    anthropic_model: Optional[str] = None,   # Add model selection
    gemini_model: Optional[str] = None,      # Add model selection
    pregenerated_responses: Optional[List[Tuple[str, str]]] = None  # Add support for pre-generated (prompt, response) pairs
) -> None:
    """
    Evaluates a model using an LLM (OpenAI, Gemini, or Anthropic) for Ahimsa, Dharma, and Helpfulness.
    Can either generate responses locally using the specified model path, or evaluate pre-generated responses.
    Allows for batching of generations and evaluations.

    Args:
        model_name: Name or path of the model to evaluate (ignored if pregenerated_responses provided)
        scenarios: List of scenario dictionaries
        output_file: Path to save evaluation results
        temperature: Temperature for generation
        openai_api_key: OpenAI API key (if using OpenAI evaluator)
        gemini_api_key: Gemini API key (if using Gemini evaluator)
        anthropic_api_key: Anthropic API key (if using Anthropic evaluator)
        evaluator: Which LLM provider to use ("openai", "gemini", "anthropic")
        test_mode: Whether to use test mode (mock responses)
        system_prompt_type: Type of system prompt to use
        generation_batch_size: Batch size for generation
        apply_medical_disclaimer_penalty: Whether to apply medical disclaimer penalty
        apply_professional_referral_penalty: Whether to apply professional referral penalty
        gemini_eval_mode: Evaluation mode for Gemini ("batch" or "individual")
        skip_evaluation: Whether to skip evaluation and only generate responses
        openai_model: Specific OpenAI model to use
        anthropic_model: Specific Anthropic model to use
        gemini_model: Specific Gemini model to use
        pregenerated_responses: Optional list of (prompt, response) tuples to evaluate instead of generating new responses
    """
    logger.info(f"Starting evaluation for model: {model_name} with evaluator: {evaluator.upper()}")

    # Handle pre-generated responses or generate new ones
    if pregenerated_responses:
        logger.info(f"Using {len(pregenerated_responses)} pre-generated prompt-response pairs")
        original_prompts = [pair[0] for pair in pregenerated_responses]
        generated_responses = [pair[1] for pair in pregenerated_responses]

        # Validate that we have the right number of responses for scenarios
        if len(pregenerated_responses) != len(scenarios):
            logger.warning(f"Number of pre-generated responses ({len(pregenerated_responses)}) doesn't match scenarios ({len(scenarios)})")
            # Truncate or pad as needed
            min_len = min(len(pregenerated_responses), len(scenarios))
            original_prompts = original_prompts[:min_len]
            generated_responses = generated_responses[:min_len]
            scenarios = scenarios[:min_len]
    else:
        logger.info(f"Processing {len(scenarios)} scenarios with generation batch size: {generation_batch_size}")
        # Generate responses for all scenarios
        original_prompts, generated_responses = await generate_responses_for_scenarios(
            model_name, scenarios, temperature, system_prompt_type, generation_batch_size, test_mode
        )

    # Skip evaluation if requested
    if skip_evaluation:
        evaluation_results = []
        for i, scenario in enumerate(scenarios):
            result = {
                "prompt": original_prompts[i],
                "response": generated_responses[i],
                "scenario_id": scenario.get("scenario_id", f"scenario_{i}")
            }
            evaluation_results.append(result)

        # Save results and return
        save_evaluation_results(evaluation_results, output_file)
        return

    # Prepare evaluation based on evaluator type
    evaluation_results = []

    if evaluator == "openai":
        # Prepare OpenAI evaluation items
        openai_eval_items = []
        for i, scenario in enumerate(scenarios):
            actual_tier = extract_tier_from_compound(scenario.get("tier", "C"))
            current_prompt_meta = {"tier": actual_tier, "scope": scenario.get("scope")}
            openai_eval_items.append({
                "prompt": original_prompts[i],
                "response": generated_responses[i],
                "metadata": current_prompt_meta
            })

        if openai_eval_items:
            batch_results = await batch_evaluate_with_openai(
                [item["prompt"] for item in openai_eval_items],
                [item["response"] for item in openai_eval_items],
                openai_api_key,
                metadata_list=[item["metadata"] for item in openai_eval_items],
                model_name=openai_model
            )

            # Process batch results to add prompt, response, scenario_id and combined score
            for i, result in enumerate(batch_results):
                # Calculate combined score
                combined_score = calculate_combined_score(
                    result.get("ahimsa_score", 0.0),
                    result.get("dharma_score", 0.0),
                    result.get("helpfulness_score", 0.0)
                )
                result["combined_score"] = combined_score

                # Add prompt, response, and scenario_id to match input format
                result["prompt"] = original_prompts[i]
                result["response"] = generated_responses[i]
                result["scenario_id"] = scenarios[i].get("scenario_id", f"scenario_{i}")

                evaluation_results.append(result)

    elif evaluator == "anthropic":
        # Import Anthropic functions
        from argen.reward_functions.anthropic_rewards import (
            evaluate_ahimsa_with_anthropic,
            evaluate_dharma_with_anthropic,
            evaluate_helpfulness_with_anthropic
        )
        from argen.utils.env import get_anthropic_api_key

        if not anthropic_api_key:
            anthropic_api_key = get_anthropic_api_key()

        # Get concurrency settings for Anthropic (use updated conservative limits)
        max_concurrent = GRPO_CONFIG.get("anthropic_max_concurrent_eval", 5)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_scenario_with_semaphore(i, scenario):
            """Evaluate a single scenario with semaphore control."""
            async with semaphore:
                actual_tier = extract_tier_from_compound(scenario.get("tier", "C"))
                current_prompt_meta = {"tier": actual_tier, "scope": scenario.get("scope")}

                # Run all three evaluations concurrently for this scenario
                ahimsa_task = evaluate_ahimsa_with_anthropic(
                    original_prompts[i], generated_responses[i], anthropic_api_key,
                    current_prompt_meta, anthropic_model
                )
                dharma_task = evaluate_dharma_with_anthropic(
                    original_prompts[i], generated_responses[i], anthropic_api_key,
                    current_prompt_meta, anthropic_model
                )
                helpfulness_task = evaluate_helpfulness_with_anthropic(
                    original_prompts[i], generated_responses[i], anthropic_api_key,
                    anthropic_model
                )

                # Wait for all evaluations to complete
                ahimsa_result, dharma_result, helpfulness_result = await asyncio.gather(
                    ahimsa_task, dharma_task, helpfulness_task, return_exceptions=True
                )

                # Handle exceptions
                if isinstance(ahimsa_result, Exception):
                    ahimsa_result = {**DEFAULT_EVAL_RESPONSE, "error": f"Ahimsa eval failed: {str(ahimsa_result)}"}
                if isinstance(dharma_result, Exception):
                    dharma_result = {**DEFAULT_EVAL_RESPONSE, "error": f"Dharma eval failed: {str(dharma_result)}"}
                if isinstance(helpfulness_result, Exception):
                    helpfulness_result = {**DEFAULT_EVAL_RESPONSE, "error": f"Helpfulness eval failed: {str(helpfulness_result)}"}

                # Combine results
                combined_result = {
                    "prompt": original_prompts[i],
                    "response": generated_responses[i],
                    "scenario_id": scenario.get("scenario_id", f"scenario_{i}"),
                    **ahimsa_result,
                    **dharma_result,
                    **helpfulness_result
                }

                # Calculate combined score
                combined_score = calculate_combined_score(
                    ahimsa_result.get("ahimsa_score", 0.0),
                    dharma_result.get("dharma_score", 0.0),
                    helpfulness_result.get("helpfulness_score", 0.0)
                )
                combined_result["combined_score"] = combined_score

                return combined_result

        # Create tasks for all scenarios and run them concurrently
        logger.info(f"Anthropic evaluator: Processing {len(scenarios)} scenarios with max concurrency: {max_concurrent}")
        scenario_tasks = [
            evaluate_scenario_with_semaphore(i, scenario)
            for i, scenario in enumerate(scenarios)
        ]

        # Execute all scenario evaluations concurrently
        evaluation_results = await asyncio.gather(*scenario_tasks, return_exceptions=True)

        # Handle any exceptions at the scenario level
        final_results = []
        for i, result in enumerate(evaluation_results):
            if isinstance(result, Exception):
                logger.error(f"Exception in Anthropic evaluation for scenario {i}: {result}")
                # Create a default result for failed scenarios
                default_result = {
                    "prompt": original_prompts[i],
                    "response": generated_responses[i],
                    "scenario_id": scenarios[i].get("scenario_id", f"scenario_{i}"),
                    **DEFAULT_EVAL_RESPONSE,
                    "error": f"Scenario evaluation failed: {str(result)}",
                    "combined_score": 0.0
                }
                final_results.append(default_result)
            else:
                final_results.append(result)

        evaluation_results = final_results

    elif evaluator == "gemini":
        # Import Gemini functions
        from argen.reward_functions.gemini.ahimsa import evaluate_ahimsa_with_gemini
        from argen.reward_functions.gemini.dharma import evaluate_dharma_with_gemini
        from argen.reward_functions.gemini.helpfulness import evaluate_helpfulness_with_gemini

        if gemini_eval_mode == "individual":
            # Individual evaluation mode for Gemini
            for i, scenario in enumerate(scenarios):
                actual_tier = extract_tier_from_compound(scenario.get("tier", "C"))
                current_prompt_meta = {"tier": actual_tier, "scope": scenario.get("scope")}

                # Run all three evaluations concurrently for this scenario
                ahimsa_task = evaluate_ahimsa_with_gemini(
                    original_prompts[i], generated_responses[i], current_prompt_meta, temperature
                )
                dharma_task = evaluate_dharma_with_gemini(
                    original_prompts[i], generated_responses[i], current_prompt_meta, temperature
                )
                helpfulness_task = evaluate_helpfulness_with_gemini(
                    original_prompts[i], generated_responses[i], temperature
                )

                # Wait for all evaluations to complete
                ahimsa_result, dharma_result, helpfulness_result = await asyncio.gather(
                    ahimsa_task, dharma_task, helpfulness_task, return_exceptions=True
                )

                # Handle exceptions
                if isinstance(ahimsa_result, Exception):
                    ahimsa_result = {**DEFAULT_EVAL_RESPONSE, "error": f"Ahimsa eval failed: {str(ahimsa_result)}"}
                if isinstance(dharma_result, Exception):
                    dharma_result = {**DEFAULT_EVAL_RESPONSE, "error": f"Dharma eval failed: {str(dharma_result)}"}
                if isinstance(helpfulness_result, Exception):
                    helpfulness_result = {**DEFAULT_EVAL_RESPONSE, "error": f"Helpfulness eval failed: {str(helpfulness_result)}"}

                # Combine results
                combined_result = {
                    "prompt": original_prompts[i],
                    "response": generated_responses[i],
                    "scenario_id": scenario.get("scenario_id", f"scenario_{i}"),
                    **ahimsa_result,
                    **dharma_result,
                    **helpfulness_result
                }

                # Calculate combined score
                combined_score = calculate_combined_score(
                    ahimsa_result.get("ahimsa_score", 0.0),
                    dharma_result.get("dharma_score", 0.0),
                    helpfulness_result.get("helpfulness_score", 0.0)
                )
                combined_result["combined_score"] = combined_score

                evaluation_results.append(combined_result)

        else:
            # Batch evaluation mode for Gemini (existing logic)
            # Prepare payloads for batch processing
            ahimsa_payload = []
            dharma_payload = []
            helpfulness_payload = []

            for i, scenario in enumerate(scenarios):
                actual_tier = extract_tier_from_compound(scenario.get("tier", "C"))
                current_prompt_meta = {"tier": actual_tier, "scope": scenario.get("scope")}

                ahimsa_payload.append({
                    "prompt": original_prompts[i],
                    "model_response": generated_responses[i],
                    "original_prompt_meta": current_prompt_meta
                })
                dharma_payload.append({
                    "prompt": original_prompts[i],
                    "model_response": generated_responses[i],
                    "original_prompt_meta": current_prompt_meta
                })
                helpfulness_payload.append({
                    "prompt": original_prompts[i],
                    "model_response": generated_responses[i],
                    "original_prompt_meta": {}
                })

            # Run batch evaluations concurrently
            evaluation_tasks = []
            if ahimsa_payload:
                items_per_call = GRPO_CONFIG.get("gemini_ahimsa_items_per_call_eval", 10)
                max_concurrent = GRPO_CONFIG.get("gemini_ahimsa_max_concurrent_eval", 5)
                evaluation_tasks.append(
                    batch_process_ahimsa_evaluations_concurrently(
                        ahimsa_payload, items_per_call, max_concurrent, temperature
                    )
                )

            if dharma_payload:
                items_per_call = GRPO_CONFIG.get("gemini_dharma_items_per_call_eval", 10)
                max_concurrent = GRPO_CONFIG.get("gemini_dharma_max_concurrent_eval", 5)
                evaluation_tasks.append(
                    batch_process_dharma_evaluations_concurrently(
                        dharma_payload, items_per_call, max_concurrent, temperature
                    )
                )

            if helpfulness_payload:
                items_per_call = GRPO_CONFIG.get("gemini_helpfulness_items_per_call_eval", 10)
                max_concurrent = GRPO_CONFIG.get("gemini_helpfulness_max_concurrent_eval", 5)
                evaluation_tasks.append(
                    batch_process_helpfulness_evaluations_concurrently(
                        helpfulness_payload, items_per_call, max_concurrent, temperature
                    )
                )

            # Wait for all batch evaluations to complete
            if evaluation_tasks:
                batch_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

                # Process batch results
                ahimsa_results = batch_results[0] if len(batch_results) > 0 and not isinstance(batch_results[0], Exception) else []
                dharma_results = batch_results[1] if len(batch_results) > 1 and not isinstance(batch_results[1], Exception) else []
                helpfulness_results = batch_results[2] if len(batch_results) > 2 and not isinstance(batch_results[2], Exception) else []

                # Combine results for each scenario
                for i, scenario in enumerate(scenarios):
                    ahimsa_result = ahimsa_results[i] if i < len(ahimsa_results) else DEFAULT_EVAL_RESPONSE.copy()
                    dharma_result = dharma_results[i] if i < len(dharma_results) else DEFAULT_EVAL_RESPONSE.copy()
                    helpfulness_result = helpfulness_results[i] if i < len(helpfulness_results) else DEFAULT_EVAL_RESPONSE.copy()

                    # Combine results
                    combined_result = {
                        "prompt": original_prompts[i],
                        "response": generated_responses[i],
                        "scenario_id": scenario.get("scenario_id", f"scenario_{i}"),
                        **ahimsa_result,
                        **dharma_result,
                        **helpfulness_result
                    }

                    # Calculate combined score
                    combined_score = calculate_combined_score(
                        ahimsa_result.get("ahimsa_score", 0.0),
                        dharma_result.get("dharma_score", 0.0),
                        helpfulness_result.get("helpfulness_score", 0.0)
                    )
                    combined_result["combined_score"] = combined_score

                    evaluation_results.append(combined_result)

    else:
        logger.error(f"Unknown evaluator: {evaluator}. Supported evaluators: openai, gemini, anthropic")
        raise ValueError(f"Unknown evaluator: {evaluator}")

    # Calculate summary metrics
    from argen.evaluation.openai_evaluator import calculate_metrics
    final_metrics = calculate_metrics(evaluation_results)

    # Create the full output structure (similar to openai_evaluator.py)
    evaluator_info = f"{evaluator}"
    if evaluator == "gemini":
        from argen.reward_functions.gemini_rewards import GEMINI_EVAL_MODEL
        evaluator_info = f"gemini ({GEMINI_EVAL_MODEL})"
    elif evaluator == "openai" and openai_model:
        evaluator_info = f"openai ({openai_model})"
    elif evaluator == "anthropic" and anthropic_model:
        evaluator_info = f"anthropic ({anthropic_model})"

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
            "timestamp": datetime.datetime.now().isoformat()
        },
        "summary_metrics": final_metrics,
        "individual_results": evaluation_results
    }

    # Save the full evaluation results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    # Create evaluation summary
    create_evaluation_summary(evaluation_results, output_file)

    logger.info(f"Evaluation completed. Results saved to {output_file}")


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
