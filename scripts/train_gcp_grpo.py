#!/usr/bin/env python3
"""
GRPO Training Script for GCP using Unsloth.

Adapts the GCP+Unsloth recipe for the argen-demo project,
using Gemini-based reward functions.
"""

import torch, os
import argparse
import logging
import subprocess
from unsloth import GRPOTrainer, FastLanguageModel # Check if GRPOTrainer is the right class from unsloth, might need modification later
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from datasets import load_dataset

# Import custom reward functions and utilities
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add project root
from src.reward_functions.openai_rewards import gemini_ahimsa_reward, gemini_dharma_reward
from src.utils.env import load_env_vars, get_gemini_api_key

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model using GRPO on GCP with Unsloth.")

    # Paths and Model ID
    parser.add_argument("--model_id", type=str, default="unsloth/Llama-3.2-1B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset (.jsonl file)")
    parser.add_argument("--output_dir", type=str, default="llama32-grpo-local", help="Local directory to save checkpoints")
    parser.add_argument("--gcs_checkpoint_dir", type=str, required=True, help="GCS path for backing up checkpoints (e.g., gs://bucket/checkpoints)")

    # Reward Function
    parser.add_argument("--reward_mode", type=str, default="combined", choices=["ahimsa", "dharma", "combined"], help="Which reward function to use or combine")

    # Training Hyperparameters
    parser.add_argument("--seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_steps", type=int, default=1200, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--group_size", type=int, default=4, help="Group size for GRPO reward calculation")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision instead of float16")

    # LoRA Configuration
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r value (rank)")
    # Add other LoRA params like alpha, dropout, target_modules if needed

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save local checkpoint every N steps")
    parser.add_argument("--gcs_save_steps", type=int, default=200, help="Backup checkpoint to GCS every N steps")
    parser.add_argument("--wandb_project", type=str, default="argen-grpo-gcp", help="WandB project name")
    # Add wandb entity if needed

    return parser.parse_args()

# Placeholder for the reward function wrapper - will be defined inside main
# def get_grpo_reward(prompt: str, completion: str, context: dict) -> float:
#     pass

# --- Custom Callback for GCS Checkpointing ---
class GCSCheckpointCallback(TrainerCallback):
    def __init__(self, gcs_bucket_path: str, save_every_n_steps: int):
        super().__init__()
        self.gcs_bucket_path = gcs_bucket_path.rstrip('/') # Ensure no trailing slash
        self.save_every = save_every_n_steps
        # Basic check for gsutil availability
        try:
            subprocess.run(["gsutil", "--version"], check=True, capture_output=True)
            logger.info("gsutil found.")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning(f"gsutil command not found or failed. GCS checkpointing will be disabled. Error: {e}")
            self.gcs_bucket_path = None # Disable GCS saving

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event called after a checkpoint save."""
        if not self.gcs_bucket_path:
            return # GCS saving disabled

        # Check if the current step is a multiple of gcs_save_steps
        if state.global_step > 0 and state.global_step % self.save_every == 0:
            # The checkpoint saved is in a subdirectory named checkpoint-<global_step>
            local_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            
            if os.path.isdir(local_checkpoint_dir):
                gcs_dest_path = f"{self.gcs_bucket_path}/checkpoint-{state.global_step}/"
                command = ["gsutil", "-m", "cp", "-r", local_checkpoint_dir, gcs_dest_path]
                
                logger.info(f"Copying checkpoint {state.global_step} from {local_checkpoint_dir} to {gcs_dest_path}...")
                try:
                    # Run gsutil command
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    logger.info(f"Successfully copied checkpoint {state.global_step} to GCS.")
                    logger.debug(f"gsutil stdout: {result.stdout}")
                except FileNotFoundError:
                     logger.error("gsutil command not found. Please ensure Google Cloud SDK is installed and configured.")
                     self.gcs_bucket_path = None # Disable further attempts
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to copy checkpoint {state.global_step} to GCS. Error: {e}")
                    logger.error(f"gsutil stderr: {e.stderr}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during GCS copy: {e}")
            else:
                logger.warning(f"Local checkpoint directory not found, cannot copy to GCS: {local_checkpoint_dir}")
# --- End Custom Callback ---

# TODO: Define GCS Checkpointing Logic (P2.6)

def main():
    args = parse_arguments()
    logger.info(f"Starting GRPO training with arguments: {args}")

    # Load environment variables (.env file for GEMINI_API_KEY)
    logger.info("Loading environment variables from .env file if present...")
    load_env_vars()
    # Verify Gemini API key is accessible after loading env vars
    try:
        if not get_gemini_api_key():
            raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
        logger.info("Gemini API key found.")
    except Exception as e:
        logger.error(f"Failed to get Gemini API key: {e}")
        sys.exit(1)

    # --- Define Reward Function Wrapper --- 
    # Defined inside main to capture args
    def get_grpo_reward(prompt: str, completion: str, context: dict) -> float:
        """Wrapper to call Gemini-based reward functions for GRPOTrainer."""
        # Use separate log statements for multi-line clarity
        logger.debug("Getting reward...")
        logger.debug(f"  Prompt: {prompt[:100]}...")
        logger.debug(f"  Completion: {completion[:100]}...")
        
        # The context dict from GRPOTrainer might contain useful info, but
        # our underlying reward functions expect `example: dict`. We pass empty.
        example_arg = {}
        
        score = 0.0
        try:
            if args.reward_mode == "ahimsa":
                score = gemini_ahimsa_reward(prompt, completion, example_arg)
                logger.debug(f"Ahimsa reward: {score}")
            elif args.reward_mode == "dharma":
                score = gemini_dharma_reward(prompt, completion, example_arg)
                logger.debug(f"Dharma reward: {score}")
            elif args.reward_mode == "combined":
                ahimsa_score = gemini_ahimsa_reward(prompt, completion, example_arg)
                dharma_score = gemini_dharma_reward(prompt, completion, example_arg)
                score = (ahimsa_score + dharma_score) / 2.0 # Simple averaging
                logger.debug(f"Combined reward: (A: {ahimsa_score}, D: {dharma_score}) -> Avg: {score}")
            else:
                logger.error(f"Unknown reward mode: {args.reward_mode}")
                return 0.0 # Default score on error
                
            # Ensure score is float
            score = float(score)
            
        except Exception as e:
            logger.error(f"Error calculating reward for completion: {e}", exc_info=True)
            return 0.0 # Return default score on error
            
        return score
    # --- End Reward Function Wrapper ---

    # Update placeholder values with args
    MODEL_ID = args.model_id
    OUT_DIR  = args.output_dir
    SEQ_LEN  = args.seq_len
    GROUP    = args.group_size
    LR       = args.learning_rate
    STEPS    = args.max_steps
    DATASET_PATH = args.dataset_path
    REWARD_FN = get_grpo_reward # Use the actual reward function wrapper
    LORA_R = args.lora_r
    USE_FP16 = not args.use_bf16 # Use FP16 if BF16 is not specified
    LOGGING_STEPS = args.logging_steps
    SAVE_STEPS = args.save_steps
    GCS_CHECKPOINT_DIR = args.gcs_checkpoint_dir
    GCS_SAVE_STEPS = args.gcs_save_steps
    BATCH_SIZE = args.batch_size
    # --- End Placeholder values ---

    # Set WANDB_PROJECT environment variable
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # 1) Load data
    logger.info(f"Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset file not found: {DATASET_PATH}")
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    try:
        # Load the dataset assuming it's JSON Lines format
        ds = load_dataset("json", data_files=DATASET_PATH, split="train")
        logger.info(f"Dataset loaded with {len(ds)} rows.")

        # Verify required columns and select/rename if necessary
        # GRPOTrainer likely primarily needs 'prompt'
        required_columns = ['prompt']
        if not all(col in ds.column_names for col in required_columns):
            logger.error(f"Dataset missing required columns. Found: {ds.column_names}, Required: {required_columns}")
            # Attempt common renames or raise error
            # Example rename: if 'text' in ds.column_names:
            #    ds = ds.rename_column('text', 'prompt')
            raise ValueError(f"Dataset at {DATASET_PATH} must contain columns: {required_columns}")

        # Keep only the 'prompt' column for the trainer
        # GRPOTrainer will generate completions based on these prompts
        logger.info(f"Original columns: {ds.column_names}")
        columns_to_keep = ['prompt']
        columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)
            logger.info(f"Keeping only columns: {ds.column_names}")

    except Exception as e:
        logger.error(f"Failed to load or process dataset: {e}", exc_info=True)
        raise

    # 2) Load tokenizer
    logger.info(f"Loading tokenizer for: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        # Set padding side for Llama models to prevent warnings
        tok.padding_side = "left"
        logger.info("Setting pad_token to eos_token and padding_side to left")
        tok.pad_token = tok.eos_token
    logger.info("Tokenizer loaded.")

    # 3) Load Model and add LoRA adapters
    logger.info(f"Loading model: {MODEL_ID}")
    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = SEQ_LEN,
        dtype = torch.bfloat16 if args.use_bf16 else torch.float16,
        load_in_4bit = True, # Use 4bit QLoRA for efficiency
    )
    logger.info("Model loaded. Applying PEFT LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"], # Common Llama modules
        lora_alpha = LORA_R * 2, # General recommendation
        lora_dropout = 0, # Minimal dropout for GRPO
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
        max_seq_length = SEQ_LEN,
    )
    logger.info("PEFT LoRA model configured.")

    # Create GCS callback instance
    gcs_callback = GCSCheckpointCallback(
        gcs_bucket_path=GCS_CHECKPOINT_DIR,
        save_every_n_steps=GCS_SAVE_STEPS
    )

    # 4) Init GRPO trainer
    logger.info("Initializing GRPOTrainer...")
    # GRPOTrainer requires the model object, not the name
    trainer = GRPOTrainer(
        model             = model, # Pass the model object
        tokenizer         = tok,
        dataset           = ds,
        reward_fn         = REWARD_FN,
        group_size        = GROUP,
        batch_size        = BATCH_SIZE, # Updated to use args
        lr                = LR,
        seq_length        = SEQ_LEN,
        use_peft_lora     = True,
        lora_r            = LORA_R,
        fp16              = USE_FP16, # Updated to use args logic
        bf16              = args.use_bf16, # Updated to use args
        logging_steps     = LOGGING_STEPS,
        output_dir        = OUT_DIR,
        save_steps        = SAVE_STEPS,
        report_to         = "wandb", # TODO: Ensure WANDB_PROJECT env var is set correctly using args.wandb_project
        callbacks         = [gcs_callback] # Add the custom callback
    )
    logger.info("GRPOTrainer initialized.")

    # 5) Train
    logger.info(f"Starting training for {STEPS} steps...")
    # TODO: Add GCS checkpointing logic within/around train (P2.6)
    trainer.train(max_steps = STEPS)
    logger.info("Training finished.")

    # 6) Save final model locally
    logger.info(f"Saving final model adapter to {OUT_DIR}...")
    trainer.save_model(OUT_DIR) # Use save_model to save only the adapter
    # trainer.save_pretrained_local(OUT_DIR) # This might save the full model? Check Unsloth docs.
    logger.info("Final adapter saved locally.")

    # 7) Copy final adapter to GCS
    if gcs_callback.gcs_bucket_path: # Check if GCS saving is enabled
        logger.info(f"Copying final adapter from {OUT_DIR} to {gcs_callback.gcs_bucket_path}/final_adapter/...")
        # We copy the whole output dir which should contain the final adapter files
        command = ["gsutil", "-m", "cp", "-r", OUT_DIR, f"{gcs_callback.gcs_bucket_path}/final_adapter/"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info("Successfully copied final adapter to GCS.")
            logger.debug(f"gsutil stdout: {result.stdout}")
        except Exception as e:
             logger.error(f"Failed to copy final adapter to GCS. Error: {e}")
             if isinstance(e, subprocess.CalledProcessError):
                 logger.error(f"gsutil stderr: {e.stderr}")
    else:
        logger.info("GCS checkpointing was disabled, skipping final GCS copy.")

    # TODO: Optional push_to_hub (P2.7)
    # Example: trainer.push_to_hub(f"your-hf-handle/{args.output_dir}")

if __name__ == "__main__":
    # Logging is configured globally now
    main() 