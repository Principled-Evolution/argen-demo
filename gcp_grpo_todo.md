# ArGen GCP GRPO Fine-tuning Plan & TODO List

## Context & Objectives

This document tracks the progress of adapting the ArGen GRPO fine-tuning workflow from its Predibase implementation to run directly on Google Cloud Platform (GCP) using Unsloth.

**Core Objectives:**

1.  **Migrate Training:** Move the fine-tuning process from Predibase to a GCP Compute Engine VM (Spot instances preferred for cost-efficiency).
2.  **Use Unsloth:** Leverage the Unsloth library for efficient fine-tuning of a Llama-3.2-1B model (or similar) on a single GCP GPU (L4, A10, A100).
3.  **Apply Dharma Alignment:** Focus the alignment goal on the "Dharma" principle (professional duty) for a conceptual healthcare agent, ensuring it stays within its domain and refuses out-of-scope requests (e.g., financial, fashion advice), as detailed in `demo-plan-v2.md`.
4.  **Reuse Reward Mechanism:** Integrate the existing Gemini-based reward functions (`gemini_ahimsa_reward`, `gemini_dharma_reward` from `src/reward_functions/gemini_rewards.py`) into the Unsloth `GRPOTrainer` workflow.
5.  **Robust Checkpointing:** Implement reliable checkpoint saving to Google Cloud Storage (GCS) to mitigate data loss from Spot VM preemptions.
6.  **Evaluate Alignment:** Ensure the process allows for comparison between a baseline model's behavior and the post-GRPO aligned model's behavior using the same Gemini/OPA-inspired evaluation logic.

**Key Reference Files & Discussions:**

*   **Target Architecture:** Based on the GCP+Unsloth recipe provided in the initial user query.
*   **Alignment Goal:** Defined primarily in `demo-plan-v2.md`.
*   **Original Plan Context:** `demo-plan.md`.
*   **Source Implementation (Predibase):** `scripts/run_fixed_json_gemini_opa_grpo_job.py`.
*   **Reward Functions:** `src/reward_functions/gemini_rewards.py`.
*   **Baseline Evaluation Example:** `examples/evaluate_baseline_with_gemini.py` and resulting `data/baseline_gemini_results_*.json` files.

---

**Goal:** Run GRPO fine-tuning for a model like `llama-3-2-1b-instruct` on a GCP Spot VM (L4, A10, or A100) using Unsloth, adapting the existing data and Gemini-based reward functions, with reliable checkpointing to GCS.

---

## Phase 1: Preparation & Setup

**Status: DONE**

*   **P1.1: Create New Branch:**
    *   **Task:** Create and switch to a new Git branch named `gcp-grpo-tests`. (**DONE**)
*   **P1.2: GCP Project Setup:**
    *   **Task:** Confirm GCP Project ID (`argen-grpo-opa-eval`). (**DONE**)
    *   **Task:** Confirm GCP Zone (`us-central1-a` selected). (**DONE**)
    *   **Task:** Confirm and create GCS Bucket (`gs://argen-grpo-gcp-checkpoints`). (**DONE**)
*   **P1.3: W&B Setup:**
    *   **Task:** Confirm W&B Project name (`argen-grpo-gcp`). (**DONE**)
*   **P1.4: Environment Dependencies:**
    *   **Task:** Identify Python dependencies (Unsloth, HF libs, google-generativeai, etc.). (**DONE**)
    *   **Task:** Create `requirements_gcp.txt`. (**DONE**)

---

## Phase 2: Script Adaptation (`scripts/train_gcp_grpo.py`)

**Status: DONE**

*   **P2.1: Create New Training Script:**
    *   **Task:** Create `scripts/train_gcp_grpo.py` with basic structure. (**DONE**)
*   **P2.2: Argument Parsing:**
    *   **Task:** Add `argparse` for command-line arguments. (**DONE**)
*   **P2.3: Data Loading:**
    *   **Task:** Implement loading from `.jsonl` path, keep 'prompt' column. (**DONE**)
*   **P2.4: Reward Function Integration:**
    *   **Task:** Import Gemini reward functions, add wrapper `get_grpo_reward`. (**DONE**)
*   **P2.5: Unsloth GRPOTrainer Initialization:**
    *   **Task:** Update to use `FastLanguageModel` and PEFT adapter setup. (**DONE**)
*   **P2.6: Training Loop & Checkpointing:**
    *   **Task:** Add `GCSCheckpointCallback` for periodic GCS saves. (**DONE**)
*   **P2.7: Final Save/Push:**
    *   **Task:** Add final local save and GCS copy of the adapter. (**DONE**)

---

## Phase 3: GCP VM Execution

**Status: IN PROGRESS**

*   **P3.1: VM Creation:**
    *   **Task:** Define and run `gcloud compute instances create` command. (**DONE**)
*   **P3.2: VM Setup:**
    *   **Status:** **TODO**
    *   **Tasks:**
        *   SSH into the VM (`gcloud compute ssh argen-grpo-vm --zone=us-central1-a --project=argen-grpo-opa-eval`).
        *   Create conda environment (`conda create -n grpo python=3.11 -y && conda activate grpo`).
        *   Install dependencies (`pip install -r requirements_gcp.txt`).
        *   Clone repository (`git clone <your-repo-url>`) and checkout branch (`cd <repo> && git checkout gcp-grpo-tests`).
        *   Configure `accelerate` (`accelerate config default`).
        *   Set up credentials:
            *   Provide `GEMINI_API_KEY` (e.g., `export GEMINI_API_KEY='...'`).
            *   Log in to W&B (`wandb login`).
            *   Authenticate `gcloud` for `gsutil` (e.g., `gcloud auth application-default login`).
*   **P3.3: Run Training:**
    *   **Status:** **TODO**
    *   **Tasks:**
        *   Execute `scripts/train_gcp_grpo.py` using `accelerate launch`, providing necessary arguments (dataset path, GCS path, etc.) and environment variables (WANDB\_PROJECT, GEMINI\_API\_KEY).
*   **P3.4: Monitoring & Resuming:**
    *   **Status:** **TODO** (As needed)
    *   **Tasks:**
        *   Monitor training progress via W&B and terminal logs.
        *   If preempted: Recreate VM, repeat setup (P3.2), download latest checkpoint from GCS, resume training using `--resume_from_checkpoint`.

---

## Phase 4: Evaluation & Cleanup

**Status: TODO**

*   **P4.1: Sanity Check:**
    *   **Task:** Load the trained adapter and test with a sample prompt.
*   **P4.2: GGUF Conversion (Optional):**
    *   **Task:** Convert adapter to GGUF format if needed.
*   **P4.3: VM Cleanup:**
    *   **Task:** Delete the GCP VM (`gcloud compute instances delete argen-grpo-vm --zone=us-central1-a`).

---

## Next Steps (Next Session)

1.  Complete **P3.2: VM Setup**.
2.  Perform **P3.3: Run Training**.
3.  Monitor and handle potential preemptions (**P3.4**).
4.  Proceed to **Phase 4** if training is successful. 