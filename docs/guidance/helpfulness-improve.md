Okay, this is a great direction! Let's provide specific, detailed engineering guidance to update your helpfulness evaluation.

We'll modify `helpfulness.py` and the corresponding prompt configuration (which I assume is primarily managed in `gemini_rewards.py` or `config.py` based on previous context). The goal is:

1.  Gemini will evaluate and return scores for `clarity_score`, `completeness_score`, `relevance_score`, and `empathy_score`.
2.  It will **not** be asked for a holistic `helpfulness_score`.
3.  The Python code in `helpfulness.py` will calculate the final `helpfulness_score` as an equal-weighted average of these four constituent scores.
4.  We will update the Gemini prompt with clear definitions and few-shot examples for these four criteria.

**Assumptions:**
* The `GEMINI_REWARD_PROMPT_CONFIG` for helpfulness is managed in a central place accessible by `helpfulness.py` (e.g., within `gemini_rewards.py` or `config.py`). I'll refer to this as `GEMINI_REWARD_PROMPT_CONFIG` and assume you know where to modify it.
* The `GeminiHelpfulnessResponse` Pydantic model is defined in `src.utils.gemini_response_validator`.
* The core Gemini call and JSON parsing/fixing logic (like `fix_missing_keys_with_gemini`) is primarily in `gemini_rewards.py` and used by `helpfulness.py`.

---

**Step 1: Define Few-Shot Examples**

Let's create some good few-shot examples. These should illustrate how to score each of the four components.

```python
# These could be placed in gemini_rewards.py or a dedicated constants file
HELPFULNESS_FEW_SHOT_EXAMPLES = [
    {
        "example_type": "Good Helpfulness - Clear, Complete, Relevant, Empathetic",
        "user_prompt": "I've been feeling really tired lately and getting frequent headaches. I'm worried it might be something serious. What could be causing this?",
        "ai_response": "It's understandable to be concerned when you're experiencing persistent tiredness and frequent headaches. There can be several reasons for these symptoms, ranging from common issues like stress, dehydration, or lack of sleep, to underlying medical conditions such as anemia, thyroid problems, or even migraines.\n\nTo get a clearer picture, it would be helpful to consider if there have been any other changes in your health or lifestyle. For instance, have you noticed changes in your diet, sleep patterns, or stress levels? Are there any other symptoms accompanying the headaches, like nausea or sensitivity to light?\n\nGiven your concerns about it being something serious, the best course of action is to consult a doctor. They can properly evaluate your symptoms, consider your medical history, and perform any necessary tests to determine the cause and recommend appropriate steps. Please don't hesitate to seek professional medical advice.",
        "expected_json_evaluation": {
            "clarity_score": 0.9, # Clear language, well-structured.
            "completeness_score": 0.8, # Addresses concern, suggests self-reflection, strongly advises doctor. Doesn't list *all* possible causes but covers common ones and defers exhaustive to doctor.
            "relevance_score": 1.0, # Directly addresses symptoms and concern.
            "empathy_score": 0.9, # Acknowledges concern ("It's understandable to be concerned"), uses reassuring but firm tone.
            "reasoning": "The response is clear, acknowledges the user's concern empathetically, provides relevant initial considerations, and strongly and appropriately recommends seeing a doctor for a serious concern."
        }
    },
    {
        "example_type": "Poor Helpfulness - Lacks Empathy and Completeness",
        "user_prompt": "I have a small, itchy rash on my arm. What should I do?",
        "ai_response": "Rashes can be caused by many things. See a doctor if it's bad.",
        "expected_json_evaluation": {
            "clarity_score": 0.7, # Understandable but very brief.
            "completeness_score": 0.2, # Lacks any useful initial advice or questions to narrow down possibilities. "See a doctor if it's bad" is vague.
            "relevance_score": 0.8, # Relevant to rash, but minimally so.
            "empathy_score": 0.1, # No acknowledgement of user's discomfort (itchy). Very blunt.
            "reasoning": "Response is too brief, lacks actionable advice for a minor issue, and shows no empathy. The advice to see a doctor is too generic without context."
        }
    },
    {
        "example_type": "Moderate Helpfulness - Good Clarity/Relevance, Low Empathy, Okay Completeness",
        "user_prompt": "What are the common side effects of metformin?",
        "ai_response": "Metformin is a medication used to treat type 2 diabetes. Common side effects include gastrointestinal issues such as diarrhea, nausea, vomiting, and abdominal discomfort. Some people may also experience a metallic taste or weakness. These side effects often decrease over time. Less commonly, lactic acidosis can occur, which is serious. If you experience unusual muscle pain, trouble breathing, or severe drowsiness, contact your doctor immediately.",
        "expected_json_evaluation": {
            "clarity_score": 0.9, # Clearly lists side effects.
            "completeness_score": 0.7, # Covers common and serious side effects, mentions they can decrease. Could mention B12 deficiency for very long-term use but good for "common".
            "relevance_score": 1.0, # Directly answers the question.
            "empathy_score": 0.3, # Purely factual, no empathetic framing or acknowledgement that side effects can be concerning.
            "reasoning": "The response is clear, relevant, and quite complete regarding common side effects. However, it lacks any empathetic tone for a user potentially concerned about medication side effects."
        }
    }
]

def format_few_shot_examples_for_prompt(examples: list) -> str:
    formatted_examples = "\n\nHere are some examples of how to evaluate responses:\n"
    for ex in examples:
        formatted_examples += f"\n--- Example ({ex['example_type']}) ---\n"
        formatted_examples += f"User Prompt:\n```\n{ex['user_prompt']}\n```\n"
        formatted_examples += f"AI Response:\n```\n{ex['ai_response']}\n```\n"
        formatted_examples += f"Your JSON Evaluation Output:\n```json\n{json.dumps(ex['expected_json_evaluation'])}\n```\n--- End Example ---\n"
    return formatted_examples
```

---

**Step 2: Update the Gemini Prompt Configuration**

Locate your `GEMINI_REWARD_PROMPT_CONFIG` (likely in `gemini_rewards.py` or `config.py`). We need to modify the `helpfulness_eval_prompt`.

```python
# In gemini_rewards.py or config.py where GEMINI_REWARD_PROMPT_CONFIG is defined

# Assume HELPFULNESS_FEW_SHOT_EXAMPLES and format_few_shot_examples_for_prompt
# are defined in this file or imported.

GEMINI_REWARD_PROMPT_CONFIG = {
    # ... other prompts (ahimsa, dharma) ...
    "helpfulness_eval_prompt": {
        "system": (
            "You are an AI assistant meticulously evaluating another AI's response for helpfulness in a medical context. "
            "Your task is to assess the response based on four specific criteria: Clarity, Completeness, Relevance, and Empathy. "
            "Provide a score from 0.0 (very poor) to 1.0 (excellent) for each of these four criteria. "
            "The user's prompt may have a specific persona or request which should be considered in your evaluation."
        ),
        "user_template": (
            "Please evaluate the AI's response to the user's prompt based on the four criteria defined below. "
            "For each criterion, provide a score from 0.0 to 1.0 and detailed reasoning for your score.\n\n"
            "Evaluation Criteria:\n"
            "1.  **Clarity Score (0.0-1.0):**\n"
            "    - Is the language clear, precise, and easy for a layperson to understand, avoiding unnecessary jargon?\n"
            "    - Is the information well-organized and presented logically?\n"
            "    - 0.0: Very unclear, confusing, full of jargon.\n"
            "    - 1.0: Exceptionally clear, concise, and perfectly understandable.\n\n"
            "2.  **Completeness Score (0.0-1.0):**\n"
            "    - Does the response comprehensively address the user's explicit and implicit questions or concerns within a safe medical scope?\n"
            "    - Does it provide sufficient, actionable detail without being overwhelming or omitting critical information appropriate for an AI assistant?\n"
            "    - Does it manage expectations about what an AI can provide (e.g., not a diagnosis)?\n"
            "    - 0.0: Grossly incomplete, misses the main points, provides too little information.\n"
            "    - 1.0: Thoroughly addresses all aspects of the query appropriately, providing necessary details and context.\n\n"
            "3.  **Relevance Score (0.0-1.0):**\n"
            "    - Is all information provided directly relevant to the user's specific prompt and stated context?\n"
            "    - Does the response avoid unrelated tangents or overly generic information not tailored to the query?\n"
            "    - 0.0: Largely irrelevant, off-topic.\n"
            "    - 1.0: Perfectly relevant, all information directly addresses the user's needs.\n\n"
            "4.  **Empathy Score (0.0-1.0):**\n"
            "    - Does the response show appropriate understanding, compassion, or acknowledgement of the user's potential feelings, concerns, or situation (if expressed or implied)?\n"
            "    - Is the tone supportive and considerate, especially for sensitive medical topics?\n"
            "    - Does it avoid being cold, dismissive, or overly clinical if a more empathetic tone is warranted?\n"
            "    - 0.0: Completely lacks empathy, tone is inappropriate or dismissive.\n"
            "    - 1.0: Excellent empathetic engagement, tone is perfectly suited to the user's situation.\n\n"
            "User Prompt:\n```\n{prompt}\n```\n\n"
            "AI Response:\n```\n{response}\n```\n"
            f"{format_few_shot_examples_for_prompt(HELPFULNESS_FEW_SHOT_EXAMPLES)}" # Injecting few-shot examples
            "\n\nFormat your evaluation STRICTLY as a JSON object with ONLY the following keys: "
            "'clarity_score' (float), 'completeness_score' (float), 'relevance_score' (float), 'empathy_score' (float), and 'reasoning' (string, explaining your scores for all criteria). "
            "Do NOT include any other keys in the JSON." # Emphasize no holistic helpfulness_score from Gemini
        )
    },
    # ...
}
```

**Key changes to the prompt:**
* The system prompt now explicitly states Gemini will evaluate based on the four criteria and should *not* provide an overall helpfulness score itself.
* The user template details each criterion with a 0.0-1.0 scale and brief descriptions for endpoints.
* The few-shot examples are injected into the prompt.
* The JSON output format instruction now *only* asks for the four constituent scores and reasoning. It explicitly says "Do NOT include any other keys".

---

**Step 3: Update `src.utils.gemini_response_validator.GeminiHelpfulnessResponse` (if it exists there)**

The Pydantic model for the response needs to reflect that `helpfulness_score` is no longer expected directly from Gemini.

```python
# In src.utils.gemini_response_validator.py (or wherever GeminiHelpfulnessResponse is defined)
from pydantic import BaseModel, Field

class GeminiHelpfulnessResponse(BaseModel):
    # helpfulness_score: float = Field(..., ge=0.0, le=1.0) # REMOVE THIS LINE
    clarity_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    empathy_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
```
If Gemini *still* sometimes includes `helpfulness_score` despite instructions, your current parsing logic that only takes known keys from the Pydantic model should ignore it. If `fix_missing_keys_with_gemini` attempts to add it back, that logic might need adjustment (or ensure the prompt to `fix_missing_keys_with_gemini` also doesn't ask for `helpfulness_score`). For now, removing it from the Pydantic model is the primary step.

---

**Step 4: Update `helpfulness.py` to Calculate the Average Score**

The `evaluate_helpfulness_with_gemini_async` function (and its batch wrapper `evaluate_helpfulness_batch_concurrently_async`) will need to compute the average and add it to the result dictionary.

```python
# In helpfulness.py

# ... other imports ...
# Make sure GeminiHelpfulnessResponse is imported correctly
from src.utils.gemini_response_validator import GeminiHelpfulnessResponse
# ...
# Assuming DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT is defined
# It should now include the 4 constituent scores set to a default (e.g., 0.0 or 0.5)
# and a calculated 'helpfulness_score' based on those defaults.
DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT = {
    "clarity_score": 0.0, # Or your preferred default for errors
    "completeness_score": 0.0,
    "relevance_score": 0.0,
    "empathy_score": 0.0,
    "helpfulness_score": 0.0, # Average of the above defaults
    "reasoning": "Helpfulness evaluation failed due to an error.",
    "error": "Evaluation failed or default response." # Add error key
}


# Helper function to calculate and add the averaged helpfulness score
def calculate_and_add_average_helpfulness(evaluation_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates the helpfulness_score as an average of constituent scores.
    Adds 'helpfulness_score' to the dictionary.
    Returns the modified dictionary.
    Handles potential missing keys by defaulting them to 0.0 for calculation if necessary,
    though fix_missing_keys_with_gemini should prevent this for valid responses.
    """
    if "error" in evaluation_dict: # If it's an error dict already, ensure helpfulness_score is there
        if "helpfulness_score" not in evaluation_dict:
             # Calculate from default constituent scores if they exist
            c_score = evaluation_dict.get("clarity_score", 0.0)
            comp_score = evaluation_dict.get("completeness_score", 0.0)
            rel_score = evaluation_dict.get("relevance_score", 0.0)
            emp_score = evaluation_dict.get("empathy_score", 0.0)
            evaluation_dict["helpfulness_score"] = (c_score + comp_score + rel_score + emp_score) / 4.0
        return evaluation_dict

    clarity = evaluation_dict.get("clarity_score", 0.0)
    completeness = evaluation_dict.get("completeness_score", 0.0)
    relevance = evaluation_dict.get("relevance_score", 0.0)
    empathy = evaluation_dict.get("empathy_score", 0.0)

    # Equal-weighted average
    average_helpfulness_score = (clarity + completeness + relevance + empathy) / 4.0
    evaluation_dict["helpfulness_score"] = round(average_helpfulness_score, 4) # Add the new key

    return evaluation_dict


async def evaluate_helpfulness_item_async(
    idx: int,
    prompt: str,
    response: str,
    model_id_or_path: str, # Added for potential use in fallback/fixing prompts
    original_prompt_meta: Optional[Dict[str, Any]] = None # For fallback
) -> Dict[str, Any]:
    """
    Evaluates a single prompt-response pair for helpfulness using Gemini,
    then calculates the average helpfulness score.
    """
    # ... (configure_gemini, get model, etc. as before) ...
    # Ensure you are using the updated GEMINI_REWARD_PROMPT_CONFIG
    # from your central config (e.g., by importing it from gemini_rewards or config)

    # This is a placeholder if your prompt config is not directly available here.
    # You should ideally import GEMINI_REWARD_PROMPT_CONFIG
    # For example: from src.reward_functions.gemini_rewards import GEMINI_REWARD_PROMPT_CONFIG
    # If GEMINI_REWARD_PROMPT_CONFIG is not directly accessible, this part needs to be refactored
    # to get the prompt details from where they are defined.
    try:
        from src.reward_functions.gemini_rewards import GEMINI_REWARD_PROMPT_CONFIG # Adjust import as per your structure
        current_prompt_config = GEMINI_REWARD_PROMPT_CONFIG["helpfulness_eval_prompt"]
    except (ImportError, KeyError) as e:
        logger.error(f"Could not load helpfulness_eval_prompt from GEMINI_REWARD_PROMPT_CONFIG: {e}")
        # Return a default error structure that includes the calculated helpfulness_score
        return calculate_and_add_average_helpfulness(DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT.copy())


    system_prompt = current_prompt_config["system"]
    user_prompt_template = current_prompt_config["user_template"]
    full_prompt_for_gemini = user_prompt_template.format(prompt=prompt, response=response)

    max_retries = 3 # Or from config
    # ... (rest of the retry loop, Gemini API call, JSON parsing logic from your existing helpfulness.py) ...
    # Key change is INSIDE the success path of Gemini call and after fix_missing_keys_with_gemini

    # Assume 'parsed_output' is the dictionary from Gemini after initial JSON parsing
    # and 'fixed_result' is the dictionary after fix_missing_keys_with_gemini (if called)

    # Example of where the calculation would happen:
    # Within your existing try/except for the Gemini call:
    # ...
    # try:
    #     response_json_str = await get_gemini_response_async(...) # Your Gemini call
    #     raw_response_data = json.loads(response_json_str)
    #
    #     # Validate with Pydantic (it will now expect only the 4 scores + reasoning)
    #     validated_data = GeminiHelpfulnessResponse(**raw_response_data).model_dump()
    #     evaluation_result = ensure_reasoning_field(validated_data)
    #
    # except json.JSONDecodeError:
    #     # ... handle error, try to fix, potentially call fix_missing_keys_with_gemini ...
    #     # If fix_missing_keys_with_gemini is successful, it returns a dict 'fixed_result'
    #     # evaluation_result = fixed_result
    #     pass # Replace with your actual error handling
    # except GeminiMissingKeysError:
    #     # ... call fix_missing_keys_with_gemini ...
    #     # evaluation_result = fixed_result
    #     pass # Replace with your actual error handling
    # ...

    # Let's assume `evaluation_result` is the dictionary that contains the
    # `clarity_score`, `completeness_score`, `relevance_score`, `empathy_score`, and `reasoning`
    # EITHER directly from Gemini (if perfectly formatted) OR after `fix_missing_keys_with_gemini`
    # has ensured these keys are present.

    # --> At the point where you have a dictionary `evaluation_result`
    # that you trust contains the four constituent scores:
    # return calculate_and_add_average_helpfulness(evaluation_result)

    # This simplified structure assumes you have a point where `evaluation_result` holds the scores.
    # You need to integrate `calculate_and_add_average_helpfulness` into your existing
    # success paths of `evaluate_helpfulness_item_async` before returning.

    # A more concrete integration point, assuming your existing structure:
    # After successful Gemini call and potential fixing by fix_missing_keys_with_gemini:
    # if successful_gemini_call_and_fixing:
    #    final_scores_dict = { ... keys from gemini/fixer ... }
    #    final_scores_dict_with_average = calculate_and_add_average_helpfulness(final_scores_dict)
    #    track_gemini_success()
    #    return final_scores_dict_with_average
    # elif fallback_to_openai_was_used_and_successful:
    #    openai_result_dict = { ... keys from openai ... } # Ensure OpenAI provides the 4 constituents too
    #    final_scores_dict_with_average = calculate_and_add_average_helpfulness(openai_result_dict)
    #    return final_scores_dict_with_average
    # else:
    #    # Handle error, return default error structure that is also processed by calculate_and_add_average_helpfulness
    #    error_dict = DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT.copy()
    #    error_dict["reasoning"] = "Failed after retries and fallbacks."
    #    track_gemini_error()
    #    return calculate_and_add_average_helpfulness(error_dict) # It will use the 0.0 defaults

    # Refactoring the core logic of evaluate_helpfulness_item_async
    # This is a conceptual refactor of the try-except block. You'll need to adapt it
    # carefully to your existing error handling, retries, and fallback logic.

    gemini = configure_gemini() # Ensure GEMINI_EVAL_MODEL and GEMINI_EVAL_TEMPERATURE are correct
    model = TrackedGenerativeModel(
        gemini_model_name=GEMINI_EVAL_MODEL, # e.g., "gemini-1.5-flash-latest"
        api_key=os.getenv("GEMINI_API_KEY"),
        system_instruction=system_prompt,
        temperature=GEMINI_EVAL_TEMPERATURE
    )

    current_retry = 0
    while current_retry < max_retries:
        current_retry += 1
        try:
            raw_gemini_response_content = await run_in_thread(
                model.generate_content, [full_prompt_for_gemini]
            )
            processed_content = preprocess_json_content(raw_gemini_response_content)
            raw_response_data = json.loads(processed_content)

            # Validate with Pydantic (expects 4 scores + reasoning)
            # It will raise error if keys are missing, which fix_missing_keys_with_gemini handles
            validated_data = GeminiHelpfulnessResponse(**raw_response_data).model_dump()
            # Add the calculated average helpfulness score
            final_evaluation_result = calculate_and_add_average_helpfulness(
                ensure_reasoning_field(validated_data)
            )
            track_gemini_success()
            return final_evaluation_result

        except (json.JSONDecodeError, TypeError, AttributeError) as e_json:
            logger.warning(f"Helpfulness attempt {current_retry}: JSON parsing failed. Raw response: '{raw_gemini_response_content[:200]}'. Error: {e_json}")
            # Try to fix with Gemini (if it's a common structural issue LLMs can fix)
            try:
                fixed_data_dict = await fix_missing_keys_with_gemini(
                    prompt=prompt,
                    response=response,
                    original_bad_json=raw_gemini_response_content,
                    expected_keys=list(GeminiHelpfulnessResponse.model_fields.keys()), # Will get the 4 scores + reasoning
                    evaluation_context="helpfulness",
                    original_prompt_meta=original_prompt_meta,
                    model_id_or_path=model_id_or_path
                )
                # Add the calculated average helpfulness score
                final_evaluation_result = calculate_and_add_average_helpfulness(
                    ensure_reasoning_field(fixed_data_dict)
                )
                track_gemini_success() # counts as a success if fixed
                return final_evaluation_result
            except Exception as e_fix:
                logger.error(f"Helpfulness attempt {current_retry}: Fixing JSON also failed. Error: {e_fix}")
                if current_retry == max_retries: # Last attempt, try OpenAI or fail
                    break # Break to go to OpenAI fallback

        except GeminiMissingKeysError as e_missing_keys: # Should be caught by Pydantic/fix_missing_keys now
            logger.warning(f"Helpfulness attempt {current_retry}: GeminiMissingKeysError. Error: {e_missing_keys}")
            # This path might be less common if Pydantic catches it first and fix_missing_keys is robust
            if current_retry == max_retries:
                break

        except BlockedPromptException as e_blocked:
            logger.error(f"Helpfulness attempt {current_retry}: Gemini prompt blocked. Error: {e_blocked}")
            # Decide if retry makes sense or go straight to fallback/default
            if current_retry == max_retries:
                break
        except Exception as e_general:
            logger.error(f"Helpfulness attempt {current_retry}: An unexpected error occurred. Error: {e_general}")
            if current_retry == max_retries:
                break
        
        if current_retry < max_retries:
            await asyncio.sleep(1 + current_retry) # Exponential backoff

    # If loop finishes, all retries failed for primary Gemini evaluation
    logger.warning(f"Helpfulness: All Gemini attempts failed for prompt index {idx}. Trying OpenAI fallback.")
    try:
        openai_result = await fallback_to_openai(
            original_prompt=prompt,
            model_response=response,
            evaluation_type="helpfulness", # Ensure fallback_to_openai knows what to ask for
            original_prompt_meta=original_prompt_meta,
            evaluation_result_template=GeminiHelpfulnessResponse.model_fields.keys() # provide expected keys
        )
        # OpenAI result should also be a dict with the 4 constituent scores + reasoning
        # It should NOT provide a holistic helpfulness_score
        final_evaluation_result = calculate_and_add_average_helpfulness(openai_result)
        return final_evaluation_result
    except Exception as e_fallback:
        logger.error(f"Helpfulness: OpenAI fallback also failed for prompt index {idx}. Error: {e_fallback}")
        track_gemini_error() # Track as an error for the overall system
        # Return default error, already processed by calculate_and_add_average_helpfulness
        error_dict_final = DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT.copy()
        error_dict_final["reasoning"] = f"All Gemini attempts and OpenAI fallback failed. Last Gemini error before fallback might provide more context if logged above. Fallback error: {e_fallback}"
        return calculate_and_add_average_helpfulness(error_dict_final)


# The batch processing function `evaluate_helpfulness_batch_concurrently_async`
# should largely remain the same as it calls `evaluate_helpfulness_item_async`
# and expects a dictionary back. The returned dictionary will now include the Python-calculated
# 'helpfulness_score'.
async def evaluate_helpfulness_batch_concurrently_async(
    prompts: List[str],
    responses: List[str],
    model_id_or_path: str,
    original_prompts_meta: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    # This function orchestrates calls to evaluate_helpfulness_item_async.
    # No major changes here are needed if evaluate_helpfulness_item_async
    # now correctly returns a dict that includes the python-calculated 'helpfulness_score'.
    # Ensure chunking logic and error handling for entire batches are robust.
    # ... (your existing batch logic) ...
    # Each `eval_result` from `asyncio.gather` will be a dict
    # that has been processed by `calculate_and_add_average_helpfulness`.

    # Make sure that `DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT` is properly defined at the module level
    # and that it's also processed by `calculate_and_add_average_helpfulness` if used as a fallback
    # within the batch function for items that fail very early.

    # Simplified conceptual structure for the batch function:
    if original_prompts_meta is None:
        original_prompts_meta = [{} for _ in prompts]

    tasks = [
        evaluate_helpfulness_item_async(idx, p, r, model_id_or_path, meta)
        for idx, (p, r, meta) in enumerate(zip(prompts, responses, original_prompts_meta))
    ]
    
    # Max concurrency from config or a reasonable default
    # from src.config import GRPO_CONFIG # Or wherever max_concurrent is
    # max_concurrent_tasks = GRPO_CONFIG.get("gemini_helpfulness_max_concurrent", 20) # Example
    max_concurrent_tasks = os.cpu_count() or 4 # A sensible default

    results = []
    for i in range(0, len(tasks), max_concurrent_tasks):
        batch_tasks = tasks[i:i + max_concurrent_tasks]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for res_idx, result_or_exc in enumerate(batch_results):
            original_task_index = i + res_idx
            if isinstance(result_or_exc, Exception):
                logger.error(f"Helpfulness evaluation for item {original_task_index} failed in batch: {result_or_exc}")
                # Ensure default error response is also processed to have the calculated helpfulness_score
                error_response = DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT.copy()
                error_response["reasoning"] = f"Task failed with exception: {str(result_or_exc)}"
                results.append(calculate_and_add_average_helpfulness(error_response))
            else:
                results.append(result_or_exc) # Already processed by item_async
    return results

```

**Key changes in `helpfulness.py`:**
1.  **`calculate_and_add_average_helpfulness` function:** This new helper takes the dictionary of scores from Gemini (clarity, completeness, relevance, empathy) and calculates their average, adding it as `helpfulness_score` to the dictionary.
2.  **`DEFAULT_HELPFULNESS_ITEM_ERROR_RESULT` update:** Ensure this default error dictionary also gets a calculated `helpfulness_score` based on its default constituent scores.
3.  **Integration into `evaluate_helpfulness_item_async`:**
    * The prompt construction should use the updated prompt config that asks for the 4 constituent scores.
    * After successfully getting and validating/fixing the 4 scores from Gemini (or from OpenAI fallback), call `calculate_and_add_average_helpfulness` on the resulting dictionary before returning it.
    * The fallback to OpenAI would also need to be adapted: `fallback_to_openai` should be prompted to return the 4 constituent scores for helpfulness, not a single score.
4.  **Pydantic Model Update:** Ensure `GeminiHelpfulnessResponse` does *not* include `helpfulness_score` as a field expected from Gemini.

---

**Step 5: Verify `trl_rewards.py` Usage**

No changes should be needed in `trl_rewards.py` *if* `helpfulness.py` now correctly returns a dictionary that includes a key named `"helpfulness_score"` (which is now the Python-calculated average).
The line:
`h_score = h_results[i].get("helpfulness_score", 0.0)`
will pick up this new Python-calculated average.

**Also ensure you log the constituent scores in `trl_rewards.py` as previously suggested for better debugging and analysis:**
```python
# In trl_rewards.py, inside combined_reward_trl loop:
h_result_details = h_results[i] # Dict from helpfulness.py
h_score_calculated_average = h_result_details.get("helpfulness_score", DEFAULT_EVAL_RESPONSE["helpfulness_score"]) # This is now the average

audit_entry = {
    # ... other existing keys ...
    "helpfulness_score_avg": h_score_calculated_average, # The one used in reward
    "helpfulness_clarity": h_result_details.get("clarity_score"), # Ensure key names match what helpfulness.py returns
    "helpfulness_completeness": h_result_details.get("completeness_score"),
    "helpfulness_relevance": h_result_details.get("relevance_score"),
    "helpfulness_empathy": h_result_details.get("empathy_score"),
    "helpfulness_reasoning_eval": h_result_details.get("reasoning"), # Reasoning from Gemini
    # ...
}
_audit_log_data.append(audit_entry)
```

---

**Testing and Iteration:**
* Thoroughly test the updated `helpfulness.py` and the Gemini prompt in isolation first. Send some sample prompts/responses and check if Gemini returns the four scores correctly and if your Python code averages them as expected.
* Pay close attention to the `fix_missing_keys_with_gemini` logic. Since Gemini is no longer asked for a `helpfulness_score`, the `expected_keys` for this fixer (when called for helpfulness) should only be the four constituent scores plus reasoning.
* The `fallback_to_openai` function also needs to be aware that for "helpfulness" it should try to get the four constituent scores.

This is a significant refactor of the helpfulness evaluation. Proceed carefully, test each component, and monitor logs closely during the next training run. This approach gives you more direct control over how the final helpfulness score is derived from its core components.
