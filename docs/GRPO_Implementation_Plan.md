
Okay, let's create a detailed task list for implementing GRPO-style training using Unsloth, your OpenAI+OPA-based evaluation as a reward signal, and wandb for tracking. We'll aim to adapt the core idea of GRPO (optimizing based on relative rewards within a group, or simply using direct rewards if GRPO implementation is complex) using standard tools like `trl`.

**Objective:** Fine-tune `unsloth/Llama-3.2-1B-Instruct` using RL (PPO, approximating GRPO's goal) to improve adherence to professional dharma (staying in the healthcare domain), guided by rewards derived from the existing OpenAI evaluation pipeline, and track the process with wandb.

---

### Task List: Implementing GRPO-style Alignment Training

**Phase 1: Setup and Preparation**

1.  **Environment & Dependencies:**
    *   Ensure necessary libraries are installed in the `argen-eval` conda environment:
        *   `trl`: For RL fine-tuning (`pip install trl`)
        *   `wandb`: For experiment tracking (`pip install wandb`)
        *   `datasets`: For handling training data (`pip install datasets`)
        *   `peft`: For LoRA (`pip install peft`)
        *   Confirm `unsloth`, `transformers`, `torch`, `openai`, `asyncio` are present.
    *   Update `requirements.txt` if necessary.

2.  **Weights & Biases (Wandb) Setup:**
    *   Create a wandb account if you don't have one.
    *   Log in to wandb in your terminal: `wandb login`.
    *   Define a wandb project name (e.g., `argen-grpo-alignment`).

3.  **Training Dataset Creation:**
    *   Create a new dataset file (e.g., `data/training_prompts.jsonl`).
    *   Populate it with prompts designed to test domain adherence, based on `demo-plan-v2.md`:
        *   Include clear in-domain medical questions.
        *   Include clear out-of-domain questions (finance, fashion, legal, etc.).
        *   Include mixed-domain questions.
        *   Aim for variety in phrasing.
        *   Start with a modest number (e.g., 50-100 prompts) for faster iteration.
    *   Format: Each line should be a JSON object with at least a `"prompt"` key: `{"prompt": "User query text..."}`.

4.  **Review OPA / Evaluation Prompts:**
    *   Briefly review the system prompts used in `src/reward_functions/openai_rewards.py` (`evaluate_ahimsa_with_openai` and `evaluate_dharma_with_openai`) to ensure they accurately capture the "stay in domain" (Dharma) requirement when evaluating responses. *Correction: The primary focus is Dharma, but Ahimsa might still be relevant as part of the overall score.*

**Phase 2: Reward Function Implementation**

5.  **Create Reward Calculation Module:**
    *   Create a new file, e.g., `src/training/reward_calculation.py`.
    *   Implement an `async` function `calculate_rewards_for_batch(prompts: List[str], responses: List[str], openai_api_key: str) -> List[float]`.
        *   This function will iterate through the prompts and responses.
        *   For each pair, it needs to call the existing *async* evaluation functions: `evaluate_ahimsa_with_openai` and `evaluate_dharma_with_openai`.
        *   Use `asyncio.gather` to run these evaluations concurrently for efficiency *per prompt-response pair*.
        *   Handle potential errors during OpenAI calls (e.g., rate limits, API errors) – return a default low reward (e.g., -1.0 or 0.0) in case of failure.
        *   **Define Reward Logic:** Based on the results from the OpenAI evaluations (specifically `dharma_score`, `dharma_violation`, potentially `ahimsa_score`), calculate a single scalar reward value for each response.
            *   *Initial Proposal:* Prioritize Dharma strongly. `reward = 1.0 if not dharma_violation else -1.0`.
            *   *Refinement (Optional):* Modulate based on score, e.g., `reward = dharma_score if not dharma_violation else -1.0`. Could also incorporate Ahimsa: e.g., `reward = (dharma_score + ahimsa_score) / 2 if not dharma_violation and not ahimsa_violation else -1.0`. *Start simple.*
        *   Return a list containing the scalar reward for each response in the batch.
    *   Implement a synchronous wrapper function if needed by the `trl` trainer (e.g., a function that takes the batch and uses `asyncio.run()` to execute the async calculation).

**Phase 3: GRPO/PPO Training Script Implementation**

6.  **Create Training Script:**
    *   Create a new main training script, e.g., `examples/train_grpo.py`.
    *   Import necessary modules: `torch`, `unsloth`, `transformers`, `trl`, `wandb`, `datasets`, `peft`, `asyncio`, and your `reward_calculation` functions.

7.  **Configuration & Initialization:**
    *   Use `argparse` to handle command-line arguments (model name, dataset path, output dir, wandb project, training hyperparameters).
    *   Initialize wandb: `wandb.init(project=WANDB_PROJECT_NAME, config=args)`.
    *   Load OpenAI API Key (from env vars).

8.  **Load Model and Tokenizer:**
    *   Use Unsloth's `FastLanguageModel.from_pretrained` to load the *baseline* model (`unsloth/Llama-3.2-1B-Instruct`).
    *   Ensure `load_in_4bit=True` (or 8bit) is used for efficiency if desired (as discussed for speedups).
    *   Load the corresponding tokenizer. Set padding token if necessary (`tokenizer.pad_token = tokenizer.eos_token`).
    *   Add LoRA adapters using `peft` and Unsloth's helpers: `model = FastLanguageModel.get_peft_model(...)`. Configure LoRA target modules, rank (`r`), alpha (`lora_alpha`), dropout.

9.  **Load Dataset:**
    *   Use `datasets.load_dataset("json", data_files=args.dataset_path, split="train")`.
    *   Preprocess/tokenize if needed, though `trl` trainers often handle this.

10. **Setup `trl` Trainer (Using PPO):**
    *   *Note:* `trl` lacks a native `GRPOTrainer`. We will use `PPOTrainer` as a practical alternative for RL-based alignment.
    *   Define `PPOConfig`:
        *   `model_name`, `learning_rate`, `batch_size`, `mini_batch_size`, `gradient_accumulation_steps`, `ppo_epochs`.
        *   `kl_penalty`: Use `kl_penalty="kl"`, set `init_kl_coef` (e.g., 0.05 - 0.2, needs tuning) and `target_kl` (e.g., 6.0). This controls how much the policy deviates from the base model.
        *   `log_with="wandb"`.
        *   Other PPO parameters (`clip_epsilon`, `vf_coef`, etc. - start with `trl` defaults).
    *   Instantiate `PPOTrainer`:
        ```python
        ppo_trainer = trl.PPOTrainer(
            config=ppo_config,
            model=model, # The LoRA-adapted model acts as the policy
            ref_model=None, # PPOTrainer creates its own reference
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=collator_function_if_needed
        )
        ```

11. **Implement Training Loop:**
    *   Iterate for a defined number of steps or epochs.
    *   Inside the loop:
        *   Get a batch of prompts: `batch = ppo_trainer.dataloader.sample()`. Extract prompt tensors.
        *   Generate responses from the policy model:
            ```python
            response_tensors = ppo_trainer.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_prompt=False,
                length_sampler=output_length_sampler_if_needed,
                **generation_kwargs # e.g., max_new_tokens, temperature
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            ```
        *   Calculate rewards for the generated responses using your `calculate_rewards_for_batch` function (handling the async call). Convert rewards to tensors.
            ```python
            rewards = calculate_rewards_for_batch(batch["prompt_text"], batch["response"], openai_api_key) # Adapt inputs as needed
            reward_tensors = [torch.tensor(r) for r in rewards]
            ```
        *   Perform PPO optimization step:
            ```python
            stats = ppo_trainer.step(
                query_tensors=batch["input_ids"],
                response_tensors=response_tensors,
                rewards=reward_tensors
            )
            ```
        *   Log statistics to wandb: `ppo_trainer.log_stats(stats, batch, reward_tensors)`. This should automatically log to wandb if configured in `PPOConfig`. Add custom logging if needed: `wandb.log({"custom_metric": ...})`.
        *   Periodically save model checkpoints (LoRA adapters): `ppo_trainer.save_pretrained(save_path)`.

12. **Implement Generation Kwargs & Utilities:**
    *   Define `generation_kwargs` dictionary (e.g., `max_new_tokens`, `temperature`, `do_sample`, `pad_token_id`). Tune `temperature` for exploration during training.
    *   Handle potential device placement issues (CPU/GPU). Unsloth + `trl` usually manage this well.

**Phase 4: Execution and Monitoring**

13. **Run Training:**
    *   Execute the script: `python examples/train_grpo.py --args...`.
    *   Monitor the training process via the terminal output and the wandb dashboard.
    *   Watch key metrics:
        *   `objective/reward`: Should ideally increase.
        *   `ppo/loss/policy`, `ppo/loss/value`: Should decrease and stabilize.
        *   `ppo/policy/kl`: Should stay close to the `target_kl` (controlled by `init_kl_coef`). If it grows too large, the model might be deviating too much, potentially losing capabilities.
        *   `ppo/returns/mean`, `ppo/returns/var`.

14. **Hyperparameter Tuning (If Necessary):**
    *   If training is unstable or not converging, adjust `learning_rate`, `init_kl_coef`, `batch_size`, or reward scaling in `calculate_rewards_for_batch`.
    *   Use wandb sweeps for systematic tuning if needed, but start with manual adjustments.

15. **Save Final Model:**
    *   Once training is satisfactory, ensure the final LoRA adapters are saved from the last checkpoint.

**Phase 5: Evaluation and Comparison**

16. **Merge Adapters:**
    *   Write a small utility or add code to `train_grpo.py` (or a new script `merge_adapters.py`) to merge the trained LoRA adapters into the base model for standalone evaluation.
    *   Load the base model using Unsloth.
    *   Load the PEFT model (adapters) from the training output directory.
    *   Merge them: `model = peft_model.merge_and_unload()` (check Unsloth/PEFT docs for the exact method).
    *   Save the merged model: `model.save_pretrained("models/aligned_model")`, `tokenizer.save_pretrained("models/aligned_model")`.

17. **Evaluate Aligned Model:**
    *   Copy `examples/evaluate_baseline.py` to `examples/evaluate_aligned.py`.
    *   Modify `evaluate_aligned.py` to load the merged model from `"models/aligned_model"` instead of the original baseline name.
    *   Run the evaluation script: `python examples/evaluate_aligned.py`. This will generate a new results JSON file using the same OpenAI+OPA evaluation process.

18. **Compare Results:**
    *   Analyze the output JSON files from the baseline evaluation and the aligned evaluation.
    *   Compare key metrics:
        *   `dharma_violation` counts/rates.
        *   Average `dharma_score`.
        *   Average `combined_score`.
        *   `generation_time_seconds`, `concurrent_eval_time_seconds` (check for significant changes).
    *   Qualitatively compare the `generated_response` examples for in-domain, out-of-domain, and mixed prompts. Does the aligned model correctly refuse out-of-domain queries while still answering in-domain ones appropriately?
    *   Document findings.

**Phase 6: Documentation**

19. **Create Markdown Document:**
    *   Create a new file, e.g., `GRPO_Implementation_Plan.md`.
    *   Copy this task list into the document.
    *   Add sections for:
        *   Objectives Recap
        *   Key Decisions (e.g., using PPO instead of literal GRPO, reward function logic)
        *   File Structure Overview (where new scripts/data reside)
        *   Wandb Project Link
        *   Instructions for running training and evaluation.
        *   Summary of expected outcomes and evaluation metrics.

---

This task list provides a step-by-step guide. Remember that RL fine-tuning can involve iteration and tuning, especially around reward shaping and hyperparameters like the KL coefficient. Start simple, get the pipeline working, and then refine.
