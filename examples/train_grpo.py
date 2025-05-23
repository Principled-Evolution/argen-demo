#!/usr/bin/env python3
"""
Script to train a model using GRPO with TRL, matching the evaluation parameters
of the baseline model.

Key features:
- Configurable reward function (single combined or separate components)
- Training with basic or enhanced system prompt
- W&B integration for tracking metrics and experiments
- Evaluation during training with configurable frequency
- Early stopping based on evaluation metrics
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional
import datetime
import logging
import wandb
import asyncio
import tempfile
from pathlib import Path # Import Path
import numpy as np # Make sure numpy is imported
import tabulate
import colorama
from colorama import Fore, Style
import random # Import random for sampling
import torch
import torch.distributed as dist # Import the distributed module

# --- Preprocess command line arguments to fix en-dash issue in scientific notation ---
for i in range(len(sys.argv)):
    if isinstance(sys.argv[i], str):
        sys.argv[i] = sys.argv[i].replace('‑', '-')  # Replace en-dash with hyphen
# ---

# Import our logging fix before any TRL imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import logging_fix
except ImportError:
    print("WARNING: logging_fix.py not found. Logging may not work correctly.")
    print("Please copy logging_fix.py to the project root directory.")

# Initialize colorama for colored console output
# Force colors even when output is piped (for tee, tail, etc.)
colorama.init(strip=False, convert=False, wrap=True)

# Configure logging
# --- Create logs directory ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# --- Set up logging with our helper function ---
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = log_dir / f"train_grpo_{run_timestamp}.log"

GLOBAL_TRAIN_GRPO_TEMPLATE_DEBUG = False

try:
    # Use our logging fix if available
    logging_fix.setup_logging(log_file_path)
except (NameError, AttributeError):
    # Fall back to standard logging setup if logging_fix is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
        force=True  # Force this configuration to override any previous settings
    )
    # Add file handler to the root logger
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__) # Get logger for this module
logger.info(f"Console and File logging initialized. Log file: {log_file_path}")

# Set WANDB_IGNORE_GLOBS via os.environ *after* logger is configured
# This ensures the variable is set before wandb.init is called.
os.environ["WANDB_IGNORE_GLOBS"] = "rewards/*"

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define a function to reset logging configuration
def reset_logging_config():
    """
    Reset logging configuration after TRL imports.
    TRL's init_zero_verbose function may have changed the root logger level to ERROR.
    """
    try:
        # Use our logging fix if available
        logging_fix.reset_logging_config(log_file_path)
    except (NameError, AttributeError):
        # Fall back to manual reset if logging_fix is not available
        root_logger = logging.getLogger()
        main_logger = logging.getLogger(__name__) # Ensure we target the __main__ logger from train_grpo.py

        # Explicitly set levels again
        root_logger.setLevel(logging.DEBUG)
        main_logger.setLevel(logging.INFO) # For __main__ logger

        # Check for StreamHandler on root, add if missing, ensure its level is INFO
        has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        if not has_stream_handler:
            print("!!! reset_logging_config: Root logger MISSING StreamHandler, re-adding. !!!")
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            sh.setFormatter(formatter)
            root_logger.addHandler(sh)
        else:
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    print(f"!!! reset_logging_config: Ensuring StreamHandler {handler} is at INFO level. Current: {handler.level} !!!")
                    handler.setLevel(logging.INFO)

        # Make sure our file handler is still attached and at the correct level
        file_handler_exists = False
        for h in root_logger.handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path):
                h.setLevel(logging.DEBUG) # Ensure file handler is at DEBUG
                file_handler_exists = True
                break

        if not file_handler_exists:
            # This part might be tricky if file_handler variable from global scope is not accessible or correctly defined here
            # For simplicity, assuming it might need re-adding if truly gone, though re-adding needs the original 'file_handler' object.
            # This fallback might be incomplete if file_handler object isn't available.
            print("!!! reset_logging_config: File handler was removed. Re-adding might be needed but is complex in fallback. !!!")
            # logger.warning("File handler was removed and re-adding in fallback is not fully implemented here.")

        # Log a test message to verify logging is working
        # Use main_logger explicitly to ensure it's the one from __main__ with the .info level set
        main_logger.info("Logging configuration reset (from train_grpo.py internal reset_logging_config fallback)")

# Check if init_zero_verbose is being imported and potentially executed
try:
    # Try to access the function directly to see if it's already been imported
    from trl.scripts.utils import init_zero_verbose
    print("WARNING: init_zero_verbose is available in the namespace")

    # Check if it's been monkey-patched to prevent it from running
    def safe_init_zero_verbose():
        print("Prevented init_zero_verbose from running")
        return

    # Replace the function with our safe version
    import trl.scripts.utils
    original_init = trl.scripts.utils.init_zero_verbose
    trl.scripts.utils.init_zero_verbose = safe_init_zero_verbose
    print(f"Replaced init_zero_verbose function: {original_init} -> {trl.scripts.utils.init_zero_verbose}")
except (ImportError, AttributeError) as e:
    print(f"init_zero_verbose not directly accessible: {e}")

# Import TRL components
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.integrations import WandbCallback
from transformers.trainer_utils import is_main_process
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import Dataset, load_dataset

from src.utils.env import load_env_vars, get_openai_api_key, get_gemini_api_key
from src.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_SCENARIOS_PATH,
    DEFAULT_VALIDATION_SCENARIOS_PATH,
    get_system_prompt,
    get_grpo_config,
    REWARD_WEIGHTS,
    DEFAULT_TEMPERATURE,
    GRPO_TRAINING_TEMPERATURE,
    PENALTY_CONFIG
)
from src.reward_functions.trl_rewards import (
    ahimsa_reward_trl,
    dharma_reward_trl,
    combined_reward_trl,
    helpfulness_reward_trl,
)
# CRITICAL FIX: Import the module itself to get the exact same reference
import src.reward_functions.trl_rewards as trl_rewards_module
from src.callbacks.adaptive_weights import MeanAdaptiveWeights
from examples.evaluate_trained_model import perform_evaluation

# Enhanced metrics tracking
class RewardMetricsTracker:
    """
    Tracks exponential moving averages and standard deviations for reward components.
    Provides fast EMAs for both values and their standard deviations.
    """
    def __init__(self, alpha=0.1):
        """
        Initialize the metrics tracker.

        Args:
            alpha (float): EMA smoothing factor (0 < alpha <= 1).
                          Lower values = more smoothing, higher values = more responsive.
        """
        self.alpha = alpha

        # EMAs for reward values
        self.ema_ahimsa = None
        self.ema_dharma = None
        self.ema_helpfulness = None
        self.ema_combined = None

        # EMAs for standard deviations
        self.ema_ahimsa_std = None
        self.ema_dharma_std = None
        self.ema_helpfulness_std = None
        self.ema_combined_std = None

        # Track raw values for std calculation
        self._recent_values = {
            'ahimsa': [],
            'dharma': [],
            'helpfulness': [],
            'combined': []
        }
        self._window_size = 20  # Keep last N values for std calculation

    def update(self, ahimsa_values, dharma_values, helpfulness_values, combined_values):
        """
        Update EMAs with new batch of values.

        Args:
            ahimsa_values (list): List of ahimsa scores from current batch
            dharma_values (list): List of dharma scores from current batch
            helpfulness_values (list): List of helpfulness scores from current batch
            combined_values (list): List of combined scores from current batch
        """
        import numpy as np

        # Calculate batch means
        batch_mean_ahimsa = np.mean(ahimsa_values) if ahimsa_values else 0.0
        batch_mean_dharma = np.mean(dharma_values) if dharma_values else 0.0
        batch_mean_helpfulness = np.mean(helpfulness_values) if helpfulness_values else 0.0
        batch_mean_combined = np.mean(combined_values) if combined_values else 0.0

        # Calculate batch standard deviations
        batch_std_ahimsa = np.std(ahimsa_values) if len(ahimsa_values) > 1 else 0.0
        batch_std_dharma = np.std(dharma_values) if len(dharma_values) > 1 else 0.0
        batch_std_helpfulness = np.std(helpfulness_values) if len(helpfulness_values) > 1 else 0.0
        batch_std_combined = np.std(combined_values) if len(combined_values) > 1 else 0.0

        # Update EMAs for values
        self.ema_ahimsa = self._update_ema(self.ema_ahimsa, batch_mean_ahimsa)
        self.ema_dharma = self._update_ema(self.ema_dharma, batch_mean_dharma)
        self.ema_helpfulness = self._update_ema(self.ema_helpfulness, batch_mean_helpfulness)
        self.ema_combined = self._update_ema(self.ema_combined, batch_mean_combined)

        # Update EMAs for standard deviations
        self.ema_ahimsa_std = self._update_ema(self.ema_ahimsa_std, batch_std_ahimsa)
        self.ema_dharma_std = self._update_ema(self.ema_dharma_std, batch_std_dharma)
        self.ema_helpfulness_std = self._update_ema(self.ema_helpfulness_std, batch_std_helpfulness)
        self.ema_combined_std = self._update_ema(self.ema_combined_std, batch_std_combined)

        # Update recent values for alternative std calculation
        self._recent_values['ahimsa'].extend(ahimsa_values)
        self._recent_values['dharma'].extend(dharma_values)
        self._recent_values['helpfulness'].extend(helpfulness_values)
        self._recent_values['combined'].extend(combined_values)

        # Keep only recent values
        for key in self._recent_values:
            if len(self._recent_values[key]) > self._window_size:
                self._recent_values[key] = self._recent_values[key][-self._window_size:]

    def _update_ema(self, current_ema, new_value):
        """Update exponential moving average with new value."""
        if current_ema is None:
            return float(new_value)
        return self.alpha * new_value + (1 - self.alpha) * current_ema

    def get_metrics(self):
        """
        Get current EMA metrics.

        Returns:
            dict: Dictionary containing all current EMA values and standard deviations
        """
        return {
            # EMA values
            'ema_ahimsa': self.ema_ahimsa,
            'ema_dharma': self.ema_dharma,
            'ema_helpfulness': self.ema_helpfulness,
            'ema_combined': self.ema_combined,

            # EMA standard deviations
            'ema_ahimsa_std': self.ema_ahimsa_std,
            'ema_dharma_std': self.ema_dharma_std,
            'ema_helpfulness_std': self.ema_helpfulness_std,
            'ema_combined_std': self.ema_combined_std,
        }

# Global metrics tracker instance
_reward_metrics_tracker = RewardMetricsTracker(alpha=0.1)

# Reset logging configuration after all imports
reset_logging_config()

# --- ADDED: Import hash verification utility ---
try:
    from src.utils.data_integrity import verify_prompt_tier_hash, _DELIMITER
except ImportError as e:
    print(f"Error importing hashing utilities: {e}")
    print("Ensure you are running this script from the project root directory or have the 'src' directory in your PYTHONPATH.")
    sys.exit(1)
# --- END ADDED ---

def check_dependencies():
    """Check if necessary dependencies for GRPO training are installed."""
    try:
        import torch
        import transformers
        import trl
        print("torch, transformers, and trl libraries found.")
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, will use CPU (might be slow).")

        # Check for console formatting dependencies
        try:
            import tabulate
            import colorama
            print("Console formatting libraries (tabulate, colorama) found.")
        except ImportError as e:
            print(f"Warning: Missing formatting dependency - {e.name}. Pretty console logs will be disabled.")
            print("Hint: pip install tabulate colorama")
            # Don't fail on these optional dependencies

        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}. Please install required libraries.")
        print("Hint: pip install torch transformers trl")
        return False

def load_scenarios(file_path: str) -> List[Dict]:
    """
    Load scenarios from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of scenarios
    """
    scenarios = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    return scenarios

def prepare_dataset(scenarios_path: str, verify_hashes: bool = True, prompt_format: str = "chat",
                system_prompt: Optional[str] = None, verbose_logging: bool = False) -> 'Dataset':
    """
    Prepare dataset for GRPO training by loading scenarios from a JSONL file.
    Assumes the 'tier' field already contains the compound hash (Tier:Hash).
    Optionally performs sampled hash verification on load.

    Args:
        scenarios_path: Path to the scenarios file (processed with hashes).
        verify_hashes: Whether to perform sampled hash verification.
        prompt_format: Format to use for prompts: 'chat' or 'instruct'.
        system_prompt: System prompt to use for formatting prompts.
        verbose_logging: Whether to enable verbose logging.

    Returns:
        Dataset ready for GRPO training.
    """
    logger.info(f"Loading dataset from: {scenarios_path}")
    # Load scenarios using existing function
    scenarios = load_scenarios(scenarios_path)
    if not scenarios:
        logger.error("No scenarios loaded. Check the input file.")
        # Handle error appropriately, maybe raise exception
        raise ValueError(f"Failed to load any scenarios from {scenarios_path}")

    # --- Perform Verification (Optional) ---
    if verify_hashes:
        logger.info("Performing sampled hash verification on loaded scenarios...")
        num_items = len(scenarios)
        sample_rate = 0.1 # Or make configurable
        num_to_sample = max(1, min(num_items, int(num_items * sample_rate))) # Ensure sample size <= population size
        sample_indices = random.sample(range(num_items), num_to_sample)
        verification_failed = False
        verified_count = 0
        skipped_format_count = 0

        for i in sample_indices:
            scenario = scenarios[i]
            prompt = scenario.get("prompt")
            compound_tier = scenario.get("tier")
            if prompt is None or compound_tier is None:
                logger.error(f"Missing prompt or tier in file '{scenarios_path}' at loaded index {i}.")
                logger.error(f"Problematic scenario data: {json.dumps(scenario, indent=2)}")
                raise ValueError(f"Invalid scenario format in file '{scenarios_path}' at loaded index {i}: missing required fields 'prompt' or 'tier'. Scenario: {json.dumps(scenario)}")

            if _DELIMITER not in str(compound_tier):
                error_message = (
                    f"Invalid tier format in file '{scenarios_path}' at loaded index {i}. "
                    f"Tier '{compound_tier}' does not contain the expected hash delimiter ('{_DELIMITER}')."
                )
                logger.error(error_message)
                logger.error(f"Problematic scenario data: {json.dumps(scenario, indent=2)}")
                raise ValueError(error_message)

            if not verify_prompt_tier_hash(prompt, str(compound_tier)):
                error_message = (
                    f"Hash verification failed for scenario in file '{scenarios_path}' at loaded index {i}. "
                    f"Prompt: '{prompt[:100]}...', Tier with hash: '{compound_tier}'."
                )
                logger.error(error_message)
                logger.error(f"Problematic scenario data: {json.dumps(scenario, indent=2)}")
                raise ValueError(error_message)

            verified_count += 1

        if verification_failed:
            logger.critical("Input data hash verification failed for some samples during dataset preparation.")
            # Decide action: raise error or just warn and proceed? Let's raise.
            raise ValueError("Dataset hash verification failed. Check data integrity.")
        else:
             logger.info(f"Sampled hash verification passed ({verified_count} verified, {skipped_format_count} skipped due to format).")
    # --- End Verification ---

    # Extract columns directly assuming they exist (including compound 'tier')
    dataset_dict = {}
    # Get keys from the first scenario, assuming structure is consistent
    all_keys = list(scenarios[0].keys())

    # Initialize dataset dictionary with all keys except 'prompt' (we'll handle that separately)
    for key in all_keys:
        if key != 'prompt':
            dataset_dict[key] = [scenario.get(key) for scenario in scenarios]

    # Initialize prompt list
    dataset_dict['prompt'] = []

    # Format prompts based on the specified format
    logger.info(f"Formatting prompts using {prompt_format} format with system prompt")

    for scenario in scenarios:
        user_question = scenario.get("prompt")

        if prompt_format == "chat":
            # Chat format: List of message dictionaries
            formatted_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
        else:
            # Instruct format: For GRPO, we still need to use the chat format structure
            # but we'll make the tokenizer apply the instruct template
            formatted_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]

            # Log that we're using chat format structure even for instruct mode
            if verbose_logging and scenario == scenarios[0]:
                logger.info(f"Note: Using chat format structure for instruct mode to ensure compatibility with GRPOTrainer")

        dataset_dict["prompt"].append(formatted_prompt)

    # Log sample prompts if verbose logging is enabled
    if verbose_logging:
        logger.info(f"Sample formatted prompt (format={prompt_format}):")
        logger.info(json.dumps(dataset_dict["prompt"][0], indent=2))

    # Create and return the dataset
    logger.info(f"Dataset prepared with columns: {list(dataset_dict.keys())}")
    return Dataset.from_dict(dataset_dict)

# Placeholder for the evaluation function - NOW IMPLEMENTED
# This function should load the model from checkpoint_dir, run evaluation,
# and return the combined score.
def run_benchmark_eval(checkpoint_dir: str, eval_temperature: float = DEFAULT_TEMPERATURE) -> Optional[float]:
    """
    Runs evaluation using the evaluate_trained_model.py logic.

    Args:
        checkpoint_dir: The directory of the checkpoint to evaluate.
        eval_temperature: Temperature to use for evaluation calls.

    Returns:
        The combined evaluation score, or None if evaluation fails.
    """
    logger.info(f"Running benchmark evaluation for checkpoint: {checkpoint_dir}")

    # Ensure env vars are loaded if not already done globally
    # This might be redundant if load_env_vars() is called early in main()
    # but ensures availability within the callback context.
    load_env_vars()

    # Get the appropriate API key based on the evaluator
    # Default to OpenAI for backward compatibility
    evaluator = os.environ.get("ARGEN_EVALUATOR", "openai")
    openai_api_key = None
    gemini_api_key = None

    if evaluator == "openai":
        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found for evaluation.")
            return None
    else:  # evaluator == "gemini"
        try:
            gemini_api_key = get_gemini_api_key()
            if not gemini_api_key:
                logger.error("GEMINI_API_KEY not found for evaluation.")
                return None
        except ImportError:
            logger.error("google-generativeai package not installed. Please install it with 'pip install google-generativeai'")
            return None

    # Determine which scenarios file to use for this validation
    # Using a defined path for the validation set
    validation_scenarios_path = DEFAULT_VALIDATION_SCENARIOS_PATH
    if not os.path.exists(validation_scenarios_path):
         logger.error(f"Validation scenarios file not found: {validation_scenarios_path}")
         return None

    # Determine system prompt - Defaulting to ENHANCED for now.
    # TODO: Consider passing the actual prompt type used in training if necessary.
    system_prompt_type = 'ENHANCED'

    # Create a temporary file for results to avoid cluttering
    temp_dir = tempfile.gettempdir()
    # Add a timestamp to the temp file name for uniqueness in concurrent runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_output_file = os.path.join(temp_dir, f"eval_results_{os.path.basename(checkpoint_dir)}_{timestamp}.json")

    # Explicitly get default penalty settings from config to pass down
    default_med_disclaimer_penalty = PENALTY_CONFIG.get("apply_medical_disclaimer_penalty", False)
    default_prof_referral_penalty = PENALTY_CONFIG.get("apply_professional_referral_penalty", True)

    try:
        logger.info(f"Starting evaluation via perform_evaluation function...")
        # Run the evaluation function (needs to be async, use asyncio.run)
        evaluation_results = asyncio.run(perform_evaluation(
            model_path=checkpoint_dir,
            scenarios_path=validation_scenarios_path,
            output_file=temp_output_file,
            temperature=eval_temperature, # Use eval_temperature parameter
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            evaluator=evaluator,
            system_prompt_type=system_prompt_type,
            test_mode=False, # Ensure test_mode is False for actual evaluation
            # --- Explicitly pass default penalty flags ---
            apply_medical_disclaimer_penalty=default_med_disclaimer_penalty,
            apply_professional_referral_penalty=default_prof_referral_penalty
            # ---
        ))

        if evaluation_results and "summary_metrics" in evaluation_results:
            combined_score = evaluation_results["summary_metrics"].get("average_combined_score")
            if combined_score is not None:
                logger.info(f"Benchmark evaluation successful. Combined Score: {combined_score:.4f}")
                return float(combined_score)
            else:
                logger.warning("Evaluation completed but 'average_combined_score' not found in summary metrics.")
                # Log the structure for debugging if needed
                # logger.debug(f"Evaluation results structure: {evaluation_results}")
                return None
        else:
            logger.warning(f"Evaluation function call completed but did not return valid results dictionary.")
            # Log the raw return for debugging if needed
            # logger.debug(f"Return value from perform_evaluation: {evaluation_results}")
            return None

    except Exception as e:
        logger.error(f"Benchmark evaluation failed during execution for {checkpoint_dir}: {e}", exc_info=True)
        return None
    finally:
        # Clean up temporary file regardless of success or failure
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
                logger.info(f"Cleaned up temporary eval file: {temp_output_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary evaluation file {temp_output_file}: {e}")

class CustomWandbLoggingCallback(TrainerCallback):
    """Logs additional metrics to W&B during training steps and logs audit table."""

    def __init__(self, trainer_instance=None):
        super().__init__()
        self._trainer = trainer_instance # Store trainer reference if needed

    def set_trainer(self, trainer):
        self._trainer = trainer # Method to set trainer after initialization if needed

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Log combined reward to W&B if using separate reward components."""
        if state.is_local_process_zero and logs is not None:
            # Heuristic to check if separate rewards mode is active by looking for trainer-logged component means
            is_separate_rewards_mode = any(k.startswith("rewards/") and k.endswith("_reward_trl/mean") for k in logs)

            if is_separate_rewards_mode and "reward" in logs:
                # The 'reward' metric from the trainer is the weighted sum when separate rewards are used.
                wandb.log({"rewards/combined_from_components": logs["reward"]}, step=state.global_step, commit=False)
                logger.debug(f"Logged rewards/combined_from_components: {logs['reward']:.4f} at step {state.global_step}")

    def on_step_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, optimizer=None, lr_scheduler=None, **kwargs):
        """Log metrics and audit table at the end of a training step."""
        if state.is_local_process_zero and state.global_step > 0:
            # CRITICAL DEBUG: Always log audit data status for diagnosis
            logger.info(f"🔍 DEBUG Step {state.global_step}: CustomWandbLoggingCallback.on_step_end() called")
            logger.info(f"🔍 DEBUG Step {state.global_step}: _audit_log_data length = {len(trl_rewards_module._audit_log_data)}")
            logger.info(f"🔍 DEBUG Step {state.global_step}: _audit_log_data id = {id(trl_rewards_module._audit_log_data)}")

            # --- Log Average Reward Components ---
            # Calculate averages from the module-level _audit_log_data
            # This ensures averages are calculated from the data processed in the preceding step
            if trl_rewards_module._audit_log_data: # Check if list is populated
                try:
                    # ENHANCED DEBUG: Log detailed audit data structure
                    logger.info(f"🔍 DEBUG Step {state.global_step}: _audit_log_data has {len(trl_rewards_module._audit_log_data)} entries")
                    if trl_rewards_module._audit_log_data:
                        logger.info(f"🔍 DEBUG Step {state.global_step}: Sample audit data keys: {list(trl_rewards_module._audit_log_data[0].keys())}")
                        logger.info(f"🔍 DEBUG Step {state.global_step}: Sample audit data values: {trl_rewards_module._audit_log_data[0]}")

                        # Check all entries for consistency
                        all_keys = set()
                        for entry in trl_rewards_module._audit_log_data:
                            all_keys.update(entry.keys())
                        logger.info(f"🔍 DEBUG Step {state.global_step}: All unique keys across entries: {sorted(all_keys)}")

                    avg_ahimsa = float(np.mean([d.get("ahimsa_score", 0.0) for d in trl_rewards_module._audit_log_data]))
                    avg_dharma = float(np.mean([d.get("dharma_score", 0.0) for d in trl_rewards_module._audit_log_data]))
                    avg_helpfulness = float(np.mean([d.get("helpfulness_score", 0.0) for d in trl_rewards_module._audit_log_data]))
                    avg_combined = float(np.mean([d.get("combined_reward", 0.0) for d in trl_rewards_module._audit_log_data]))
                    avg_penalty = float(np.mean([d.get("penalty", 0.0) for d in trl_rewards_module._audit_log_data])) # Assuming penalty is stored

                    logger.info(f"🔍 DEBUG Step {state.global_step}: Calculated averages - Ahimsa: {avg_ahimsa:.4f}, Dharma: {avg_dharma:.4f}, Helpfulness: {avg_helpfulness:.4f}, Combined: {avg_combined:.4f}")

                    # --- Update Enhanced Metrics Tracker ---
                    # Extract individual values for EMA tracking
                    ahimsa_values = [d.get("ahimsa_score", 0.0) for d in trl_rewards_module._audit_log_data]
                    dharma_values = [d.get("dharma_score", 0.0) for d in trl_rewards_module._audit_log_data]
                    helpfulness_values = [d.get("helpfulness_score", 0.0) for d in trl_rewards_module._audit_log_data]
                    combined_values = [d.get("combined_reward", 0.0) for d in trl_rewards_module._audit_log_data]

                    # Update the global metrics tracker
                    _reward_metrics_tracker.update(ahimsa_values, dharma_values, helpfulness_values, combined_values)

                    # Get EMA metrics
                    ema_metrics = _reward_metrics_tracker.get_metrics()

                    # Prepare log data with both averages and EMAs
                    reward_log_data = {
                        "rewards/avg_ahimsa": avg_ahimsa,
                        "rewards/avg_dharma": avg_dharma,
                        "rewards/avg_helpfulness": avg_helpfulness,
                        "rewards/avg_combined": avg_combined,
                        "rewards/avg_penalty": avg_penalty,

                        # Add EMA values
                        "rewards/ema_ahimsa": ema_metrics['ema_ahimsa'],
                        "rewards/ema_dharma": ema_metrics['ema_dharma'],
                        "rewards/ema_helpfulness": ema_metrics['ema_helpfulness'],
                        "rewards/ema_combined": ema_metrics['ema_combined'],

                        # Add EMA standard deviations
                        "rewards/ema_ahimsa_std": ema_metrics['ema_ahimsa_std'],
                        "rewards/ema_dharma_std": ema_metrics['ema_dharma_std'],
                        "rewards/ema_helpfulness_std": ema_metrics['ema_helpfulness_std'],
                        "rewards/ema_combined_std": ema_metrics['ema_combined_std'],
                    }

                    # Log using wandb.log with commit=False, Trainer will handle commit
                    wandb.log(reward_log_data, step=state.global_step, commit=False)

                    # CRITICAL: Store metrics for console callback access
                    # The console callback runs after this and needs access to these metrics
                    self._latest_reward_metrics = reward_log_data

                    logger.debug(f"Logged enhanced reward metrics at step {state.global_step}: {len(reward_log_data)} metrics")

                except Exception as e:
                    logger.error(f"Failed to calculate or log enhanced reward metrics at step {state.global_step}: {e}", exc_info=True)
            else:
                # CRITICAL DEBUG: Log when _audit_log_data is empty
                logger.warning(f"🚨 DEBUG Step {state.global_step}: _audit_log_data is EMPTY! No individual metrics will be logged.")
                logger.warning(f"🚨 DEBUG Step {state.global_step}: This means the reward function didn't populate audit data or it was cleared.")
            # ---

            # --- Log Audit Table every 500 steps ---
            # Use module-level _audit_log_data populated by the reward function
            # Check if wandb is active and if this is the main process before logging
            # Check if it's time to log the table AND if there's data
            if state.global_step % 500 == 0 and trl_rewards_module._audit_log_data and wandb.run:
                 try:
                     logger.info(f"Logging audit table for step {state.global_step} with {len(trl_rewards_module._audit_log_data)} entries...")
                     # Ensure all keys exist in the first item before creating columns
                     if trl_rewards_module._audit_log_data[0]:
                         table_columns = list(trl_rewards_module._audit_log_data[0].keys())
                         table_data = [list(d.values()) for d in trl_rewards_module._audit_log_data]
                         table = wandb.Table(columns=table_columns, data=table_data)
                         # Log audit table with commit=False
                         wandb.log({"audit_table": table}, step=state.global_step, commit=False)
                     else:
                         logger.warning(f"Audit log data found but first item is empty at step {state.global_step}, skipping table log.")
                 except Exception as e:
                     logger.error(f"Failed to log audit table for step {state.global_step} from rank {os.getenv('RANK', 'N/A')}: {e}", exc_info=True)
            # ---

            # Clear audit log data *after* potential logging
            # This ensures it's ready for the next step's reward calculation
            if trl_rewards_module._audit_log_data: # Clear only if it wasn't empty
                trl_rewards_module._audit_log_data.clear()
            # ---

# --- ADD New Callback to Log Manual Eval Command ---
class LogEvalCommandCallback(TrainerCallback):
    """Logs the command needed to run manual evaluation when a checkpoint is saved."""
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log evaluation command after a checkpoint is saved."""
        if not state.is_local_process_zero:
            return

        last_checkpoint_dir = kwargs.get("output_dir")

        if not last_checkpoint_dir:
            # This on_save event likely came from a final save_model() call
            # or another call path that doesn't specify a particular checkpoint directory.
            # We are interested in specific checkpoint-xxx directories for this callback.
            logger.debug("LogEvalCommandCallback: 'output_dir' not in kwargs for this on_save event. Skipping eval command logging for this event.")
            return

        # Check if the path is a directory and optionally if it looks like a checkpoint
        if os.path.isdir(last_checkpoint_dir):
            # Optional: Add more specific check if it's a checkpoint-XXX directory.
            # This helps ensure we only log for actual runnable checkpoints if desired.
            # For example:
            # if not os.path.basename(last_checkpoint_dir).startswith("checkpoint-"):
            #     logger.debug(f"LogEvalCommandCallback: Path {last_checkpoint_dir} does not appear to be a standard checkpoint-XXX directory. Skipping.")
            #     return

            eval_scenarios_path = DEFAULT_VALIDATION_SCENARIOS_PATH
            # Get the evaluator from environment variable or default to "openai"
            evaluator = os.environ.get("ARGEN_EVALUATOR", "openai")
            eval_command = (
                f"python examples/evaluate_trained_model.py "
                f"--model {last_checkpoint_dir} "
                f"--scenarios \"{eval_scenarios_path}\" "
                f"--evaluator {evaluator}"
            )
            logger.info(f"\n*** Checkpoint saved: {last_checkpoint_dir} ***\n"
                        f"To evaluate this checkpoint manually, run:\n"
                        f"{eval_command}\n")
        else:
            # This case means output_dir was in kwargs, but it's not a directory (or doesn't exist at the moment of check)
            logger.warning(f"LogEvalCommandCallback: Checkpoint path '{last_checkpoint_dir}' provided in on_save kwargs is not a valid directory.")

class ConsoleMetricsCallback(TrainerCallback):
    """
    Enhanced console metrics logger with better readability and trend indicators.
    Shows key metrics in a tabular format with color coding for changes.
    """
    def __init__(self):
        super().__init__()
        self.last_metrics = {}
        self.most_recent_eval_metrics = {}
        self.history = {
            "step": [],
            "train/reward": [],
            "train/kl": [],
            "eval/combined_mean": [],
        }
        self.log_eval_summary = True  # Set to True to log eval summary when detected
        self.trainer = None  # Will be set when callback is added to trainer

        # Check if formatting libraries are available
        self.has_tabulate = False
        self.has_colorama = False
        try:
            import tabulate
            self.has_tabulate = True
        except ImportError:
            logger.warning("tabulate not found. Using simplified console output.")

        try:
            import colorama
            from colorama import Fore, Style
            self.has_colorama = True
            # Force colorama to work even when piped (for tee, tail, etc.)
            # Check if user wants to force colors via environment variable
            if os.environ.get("FORCE_COLOR", "").lower() in ("1", "true", "yes"):
                self.has_colorama = True
                logger.info("Forcing colorama colors due to FORCE_COLOR environment variable")
        except ImportError:
            logger.warning("colorama not found. Console output will be monochrome.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Process logs to create more readable console output."""
        if not state.is_local_process_zero or not logs:
            return

        # Get the actual step from state rather than logs
        step = state.global_step

        # CRITICAL: Inject reward metrics from CustomWandbLoggingCallback
        # Look for the callback instance and get its latest metrics
        injected_metrics = False
        for callback in self.trainer.callback_handler.callbacks:
            if hasattr(callback, '_latest_reward_metrics'):
                logger.info(f"🔍 DEBUG Step {step}: Found CustomWandbLoggingCallback with _latest_reward_metrics")
                if callback._latest_reward_metrics:
                    logger.info(f"🔍 DEBUG Step {step}: Injecting {len(callback._latest_reward_metrics)} metrics into logs")
                    logger.info(f"🔍 DEBUG Step {step}: Injected metrics keys: {list(callback._latest_reward_metrics.keys())}")
                    logs.update(callback._latest_reward_metrics)
                    injected_metrics = True
                else:
                    logger.warning(f"🚨 DEBUG Step {step}: _latest_reward_metrics is empty or None")
                break

        if not injected_metrics:
            logger.warning(f"🚨 DEBUG Step {step}: No CustomWandbLoggingCallback found or no metrics to inject")

        # ENHANCED DEBUG: Always log what keys are available in logs
        logger.info(f"🔍 DEBUG Step {step}: Available log keys at step {step}: {list(logs.keys())}")

        # Check specifically for reward metrics
        reward_keys = [k for k in logs.keys() if 'reward' in k.lower() or 'ahimsa' in k.lower() or 'dharma' in k.lower() or 'helpfulness' in k.lower()]
        logger.info(f"🔍 DEBUG Step {step}: Reward-related keys: {reward_keys}")

        # Skip non-metrics logs
        if 'loss' not in logs and 'learning_rate' not in logs and 'reward' not in logs:
            return

        # Determine if this is training or evaluation metrics
        is_eval = any(k.startswith("eval/") for k in logs)

        if is_eval:
            self._process_eval_metrics(step, logs)
        else:
            self._process_training_metrics(step, logs)

    def _process_training_metrics(self, step, logs):
        """Format and display training metrics in a concise table."""
        # Store history
        self.history["step"].append(step)
        self.history["train/reward"].append(logs.get("reward", float('nan')))
        self.history["train/kl"].append(logs.get("kl_div", logs.get("kl", float('nan'))))

        # If we don't have tabulate, use a simpler format
        if not self.has_tabulate:
            self._process_training_metrics_simple(step, logs)
            return

        # Build metrics table
        table_data = []
        headers = ["Metric", "Value", "Change", "Trend"]

        # Key metrics to display prominently - "Reward" will be handled contextually
        key_metrics = [
            # Remove Step from table since it's in the title
            # ("Step", step, None, None),
            # ("Reward", logs.get("reward", float('nan')), "reward", "↑"), # Removed old generic reward
            ("Loss", logs.get("loss", float('nan')), "loss", "↓"),
            ("KL Div", logs.get("kl_div", logs.get("kl", float('nan'))), "kl", "↓"),
            # Format LR with higher precision to avoid showing 0
            ("LR", logs.get("learning_rate", float('nan')), None, None),
        ]

        # Determine if we are in separate rewards mode based on logs
        # Heuristic: check for trainer-logged component means
        is_separate_rewards_mode_console = any(k.startswith("rewards/") and k.endswith("_reward_trl/mean") for k in logs)

        if is_separate_rewards_mode_console:
            if "reward" in logs: # This is the weighted sum from Trainer
                key_metrics.insert(1, ("Combined Reward", logs.get("reward"), "reward", "↑"))

            # Display component rewards from Trainer logs
            component_map = {
                "ahimsa": "rewards/ahimsa_reward_trl/mean",
                "dharma": "rewards/dharma_reward_trl/mean",
                "helpfulness": "rewards/helpfulness_reward_trl/mean"
            }
            for name, log_key in component_map.items():
                if log_key in logs:
                    key_metrics.append((
                        name.capitalize(),
                        logs.get(log_key),
                        log_key,
                        "↑"
                    ))
        else:
            # Handling for single combined_reward_trl
            # Trainer's "reward" is the direct output of combined_reward_trl for the batch
            if "reward" in logs:
                 key_metrics.insert(1, ("Reward (Batch Direct)", logs.get("reward"), "reward", "↑"))

            # Combined reward from _audit_log_data (averaged by CustomWandbLoggingCallback)
            if "rewards/avg_combined" in logs:
                key_metrics.append(("Combined Reward", logs.get("rewards/avg_combined"), "rewards/avg_combined", "↑"))

            # Component rewards from _audit_log_data
            for component in ["ahimsa", "dharma", "helpfulness", "penalty"]:
                metric_key = f"rewards/avg_{component}"
                if metric_key in logs:
                    key_metrics.append((
                        component.capitalize(),
                        logs.get(metric_key),
                        metric_key,
                        "↑" if component != "penalty" else "↓" # Penalties are better if lower
                    ))

        # Add EMA metrics if available
        ema_metrics_available = any(k.startswith("rewards/ema_") and not k.endswith("_std") for k in logs)
        if ema_metrics_available:
            # Add EMA values
            for component in ["ahimsa", "dharma", "helpfulness", "combined"]:
                ema_key = f"rewards/ema_{component}"
                ema_std_key = f"rewards/ema_{component}_std"
                if ema_key in logs:
                    # Display EMA value with std deviation in parentheses if available
                    ema_value = logs.get(ema_key)
                    ema_std = logs.get(ema_std_key)
                    if ema_std is not None:
                        display_name = f"EMA {component.capitalize()} (±{ema_std:.3f})"
                    else:
                        display_name = f"EMA {component.capitalize()}"

                    key_metrics.append((
                        display_name,
                        ema_value,
                        ema_key,
                        "↑"
                    ))

        # Format each metric row with trend indicators
        for name, value, key, trend_dir in key_metrics:
            # Special handling for LR to avoid showing 0 due to precision issues
            if name == "LR" and isinstance(value, float) and value < 0.0001:
                row = [name, f"{value:.8f}"]  # Use higher precision for very small LR values
            else:
                row = [name, f"{value:.4f}" if isinstance(value, float) else value]

            # Add change and trend indicators if we have historical data
            if key and key in self.last_metrics:
                last_val = self.last_metrics.get(key, 0)
                if isinstance(value, (int, float)) and isinstance(last_val, (int, float)):
                    change = value - last_val
                    change_str = f"{change:+.4f}" if isinstance(change, float) else f"{change:+d}"

                    # FIXED: Arrow direction always shows actual change direction
                    # Color shows whether the change is good (green) or bad (red)

                    # Determine arrow direction based on actual change
                    if change > 0:
                        arrow = "↑"  # Value increased
                    elif change < 0:
                        arrow = "↓"  # Value decreased
                    else:
                        arrow = "−"  # No change

                    # Determine color based on whether change is good or bad
                    if self.has_colorama:
                        from colorama import Fore, Style
                        if trend_dir == "↑":  # Higher is better (rewards, etc.)
                            color = Fore.GREEN if change > 0 else Fore.RED if change < 0 else ""
                        else:  # Lower is better (loss, KL div, etc.)
                            color = Fore.GREEN if change < 0 else Fore.RED if change > 0 else ""

                        trend = f"{color}{arrow}{Style.RESET_ALL}"
                    else:
                        # No colorama - just show the arrow direction
                        trend = arrow
                    row.extend([change_str, trend])
                else:
                    row.extend(["", ""])
            else:
                row.extend(["", ""])

            table_data.append(row)

        # Update last seen metrics
        for name, value, key, _ in key_metrics:
            if key:
                self.last_metrics[key] = value

        # Generate table string using tabulate
        import tabulate as tabulate_lib
        table_str = tabulate_lib.tabulate(table_data, headers=headers, tablefmt="simple")

        # Add latest eval metrics if available
        eval_summary = ""
        if self.most_recent_eval_metrics and self.log_eval_summary:
            eval_step = self.most_recent_eval_metrics.get("step", "N/A")
            eval_combined = self.most_recent_eval_metrics.get("eval/combined_mean", float('nan'))

            if self.has_colorama:
                from colorama import Fore, Style
                eval_summary = (
                    f"\n{Fore.CYAN}Last Eval (Step {eval_step}): "
                    f"Combined={eval_combined:.4f}{Style.RESET_ALL}"
                )
            else:
                eval_summary = f"\nLast Eval (Step {eval_step}): Combined={eval_combined:.4f}"

            # Only show this once after each eval
            self.log_eval_summary = False

        # Log the formatted table
        if self.has_colorama:
            from colorama import Fore, Style
            logger.info(f"\n{Fore.BLUE}Training Metrics{Style.RESET_ALL} - Step {step}:\n{table_str}{eval_summary}\n")
        else:
            logger.info(f"\nTraining Metrics - Step {step}:\n{table_str}{eval_summary}\n")

    def _process_training_metrics_simple(self, step, logs):
        """Format training metrics in a simple way when tabulate is not available."""
        metrics_str = f"Step {step} | "

        # Add key metrics
        if "loss" in logs:
            metrics_str += f"Loss: {logs['loss']:.4f} | "
        if "kl_div" in logs or "kl" in logs:
            kl = logs.get("kl_div", logs.get("kl", float('nan')))
            metrics_str += f"KL: {kl:.4f} | "

        # Determine if we are in separate rewards mode based on logs
        is_separate_rewards_mode_console_simple = any(k.startswith("rewards/") and k.endswith("_reward_trl/mean") for k in logs)

        if is_separate_rewards_mode_console_simple:
            if "reward" in logs: # This is the weighted sum from Trainer
                metrics_str += f"Combined Reward: {logs['reward']:.4f} | "

            component_map_simple = {
                "Ahimsa": "rewards/ahimsa_reward_trl/mean",
                "Dharma": "rewards/dharma_reward_trl/mean",
                "Helpfulness": "rewards/helpfulness_reward_trl/mean"
            }
            for name, log_key in component_map_simple.items():
                if log_key in logs:
                    metrics_str += f"{name}: {logs[log_key]:.4f} | "
        else:
            # Handling for single combined_reward_trl
            if "reward" in logs: # Direct output from combined_reward_trl for the batch
                 metrics_str += f"Reward (Batch Direct): {logs['reward']:.4f} | "

            if "rewards/avg_combined" in logs: # From _audit_log_data
                metrics_str += f"Combined Reward: {logs['rewards/avg_combined']:.4f} | "

            for component in ["ahimsa", "dharma", "helpfulness", "penalty"]: # From _audit_log_data
                metric_key = f"rewards/avg_{component}"
                if metric_key in logs:
                    metrics_str += f"{component.capitalize()}: {logs[metric_key]:.4f} | "

            # Add EMA metrics if available
            ema_metrics_available = any(k.startswith("rewards/ema_") and not k.endswith("_std") for k in logs)
            if ema_metrics_available:
                for component in ["ahimsa", "dharma", "helpfulness", "combined"]:
                    ema_key = f"rewards/ema_{component}"
                    ema_std_key = f"rewards/ema_{component}_std"
                    if ema_key in logs:
                        ema_value = logs.get(ema_key)
                        ema_std = logs.get(ema_std_key)
                        if ema_std is not None:
                            metrics_str += f"EMA {component.capitalize()}: {ema_value:.4f}(±{ema_std:.3f}) | "
                        else:
                            metrics_str += f"EMA {component.capitalize()}: {ema_value:.4f} | "

        # Add last eval summary if available
        if self.most_recent_eval_metrics and self.log_eval_summary:
            eval_step = self.most_recent_eval_metrics.get("step", "N/A")
            eval_combined = self.most_recent_eval_metrics.get("eval/combined_mean", float('nan'))
            metrics_str += f"[Last Eval Combined: {eval_combined:.4f}]"
            self.log_eval_summary = False

        logger.info(metrics_str)

    def _process_eval_metrics(self, step, logs):
        """Process and display evaluation metrics."""
        # Store latest eval metrics
        self.most_recent_eval_metrics = logs.copy()
        self.most_recent_eval_metrics["step"] = step

        # If we have eval/combined_mean, add to history
        if "eval/combined_mean" in logs:
            # Pad history with NaN values if steps don't align
            while len(self.history["step"]) > 0 and len(self.history["eval/combined_mean"]) < len(self.history["step"]) - 1:
                self.history["eval/combined_mean"].append(float('nan'))

            self.history["eval/combined_mean"].append(logs["eval/combined_mean"])

        # Use a simpler format if tabulate is not available
        if not self.has_tabulate:
            self._process_eval_metrics_simple(step, logs)
            return

        # Build metrics table
        table_data = []
        headers = ["Eval Metric", "Value"]

        # Display all eval metrics
        for key, value in sorted(logs.items()):
            if key.startswith("eval/"):
                display_name = key.replace("eval/", "")
                formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                table_data.append([display_name, formatted_value])

        # Generate table string
        if table_data:
            import tabulate as tabulate_lib
            table_str = tabulate_lib.tabulate(table_data, headers=headers, tablefmt="simple")

            if self.has_colorama:
                from colorama import Fore, Style
                logger.info(f"\n{Fore.CYAN}Evaluation Metrics{Style.RESET_ALL} - Step {step}:\n{table_str}\n")
            else:
                logger.info(f"\nEvaluation Metrics - Step {step}:\n{table_str}\n")

            # Enable summary display on next training log
            self.log_eval_summary = True

    def _process_eval_metrics_simple(self, step, logs):
        """Format evaluation metrics in a simple way when tabulate is not available."""
        metrics_str = f"Evaluation Metrics (Step {step}):\n"

        for key, value in sorted(logs.items()):
            if key.startswith("eval/"):
                display_name = key.replace("eval/", "")
                formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                metrics_str += f"  {display_name}: {formatted_value}\n"

        logger.info(metrics_str)
        self.log_eval_summary = True

def main():
    """Run the GRPO training script."""

    # === Force Re-apply Logging Config ===
    # Reset logging configuration again at the start of main
    # This ensures our desired logging setup is active
    try:
        # Use our logging fix if available
        logging_fix.reset_logging_config(log_file_path)
    except (NameError, AttributeError):
        # Fall back to our internal reset function if logging_fix is not available
        reset_logging_config()

    # Add a test log message to verify logging is working
    logger.info("Starting GRPO training script with logging verified")
    print("Direct print: Starting GRPO training script")
    # === End Force Re-apply ===

    parser = argparse.ArgumentParser(description="Train a model using GRPO with TRL.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Name/identifier of the model to fine-tune (e.g., HF identifier)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=DEFAULT_SCENARIOS_PATH,
        help="Path to the scenarios file"
    )
    parser.add_argument(
        "--eval_scenarios",
        type=str,
        help="Path to the evaluation scenarios file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="argen-grpo",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--use_basic_prompt",
        action="store_true",
        help="Use the basic system prompt instead of the enhanced one"
    )
    parser.add_argument(
        "--use_separate_rewards",
        action="store_true",
        help="Use separate Ahimsa and Dharma reward functions instead of the combined one"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="Number of generations per iteration in GRPO"
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint dir to resume training")
    parser.add_argument("--kl_beta_start", type=float, default=None,
                        help="Override initial KL beta")
    parser.add_argument("--save_strategy", type=str, choices=["no", "epoch", "steps"],
                        default=None, help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every X updates steps. Used if save_strategy is 'steps'.")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="How many checkpoints to keep")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                        help="Batch size per GPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=18,
                        help="Batch size per GPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Number of updates steps to accumulate gradients before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                      help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=None,
                      help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--bf16", type=lambda x: x.lower() == 'true', default=None,
                      help="Whether to use bf16 precision")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == 'true', default=None,
                      help="Whether to use gradient checkpointing")
    parser.add_argument("--kl_beta_end", type=float, default=None,
                      help="Final KL beta value")
    parser.add_argument("--beta_schedule", type=str, default=None,
                      help="KL beta schedule type")
    parser.add_argument("--logging_steps", type=int, default=None,
                      help="Log every X updates steps")
    parser.add_argument("--report_to", type=str, nargs="+", default=None,
                      help="Integration to report results and logs to")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for initialization")
    # Add evaluation strategy arguments
    parser.add_argument("--evaluation_strategy", type=str,
                      choices=["no", "steps", "epoch"], default="no",
                      help="Evaluation strategy to use")
    parser.add_argument("--eval_steps", type=int, default=None,
                      help="Run evaluation every X steps. Only applies when evaluation_strategy='steps'")
    # Add early stopping parameters
    parser.add_argument("--early_stopping", action="store_true",
                      help="Enable early stopping based on evaluation metrics")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                      help="Number of evaluations with no improvement after which training will be stopped")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001,
                      help="Minimum change to qualify as improvement for early stopping")
    # Add the three parameters required by the command
    parser.add_argument("--minimum_lr", type=float, default=None,
                      help="Minimum learning rate for scheduler (formerly min_lr)")
    parser.add_argument("--kl_penalty", type=str, choices=["fixed", "adaptive"], default=None,
                      help="KL penalty type (fixed or adaptive)")
    parser.add_argument("--target_kl", type=float, default=None,
                      help="Target KL divergence for adaptive KL penalty")
    # Add evaluator parameter
    parser.add_argument("--evaluator", type=str, choices=["openai", "gemini"], default="gemini",
                      help="Which LLM to use for evaluation (openai or gemini)")
    # Add prompt format parameter
    parser.add_argument("--prompt_format", type=str, choices=["chat", "instruct"], default="chat",
                      help="Format to use for prompts: 'chat' (structured with roles) or 'instruct' (plain text)")
    # Add verbose logging parameter
    parser.add_argument("--verbose_logging", action="store_true",
                      help="Enable verbose logging of prompts and responses")
    # Add adaptive weights parameter
    parser.add_argument("--adaptive_weights", type=str,
                      choices=["none", "mean_inverse", "equal_contribution", "target_floor", "target_weighted"],
                      default="target_floor",
                      help="Strategy for dynamically adjusting reward weights: 'none' (static weights), 'mean_inverse' (weights inversely proportional to means), 'equal_contribution' (weights scaled to ensure proportional contributions), 'target_floor' (weights start at target values and only increase when needed), or 'target_weighted' (weights proportional to target/mean without floor constraint)")
    # Add adaptive weights gamma parameter
    parser.add_argument("--adaptive_weights_gamma", type=float, default=0.05,
                      help="Exponential moving average factor for adaptive weights (default: 0.05)")
    # Add adaptive weights minimum weight parameter
    parser.add_argument("--adaptive_weights_min", type=float, default=0.1,
                      help="Minimum weight for any component in adaptive weighting (default: 0.1)")
    # Add log_eval_reasoning parameter
    parser.add_argument("--log_eval_reasoning", action="store_true",
                      help="Include reasoning field in Gemini evaluations during GRPO training (default: False)")
    # Add use_batch_gemini_calls parameter
    parser.add_argument("--use_batch_gemini_calls", action="store_true",
                      help="Use batch Gemini API calls instead of single calls for training (default: False)")
    # Add temperature parameters
    parser.add_argument("--temperature", type=float, default=None,
                      help="Temperature for model generation during training (overrides config default)")
    parser.add_argument("--eval_temperature", type=float, default=None,
                      help="Temperature for evaluation calls (overrides config default)")

    print("--- RIGHT BEFORE PARSE_ARGS (direct print) ---")
    logger.info("--- RIGHT BEFORE PARSE_ARGS (logger.info) ---")
    args = parser.parse_args()
    print("--- RIGHT AFTER PARSE_ARGS (direct print) ---")
    logger.info("--- RIGHT AFTER PARSE_ARGS (logger.info) ---")

    # logger.info(f"POST ARGPARSE - Using evaluator: {args.evaluator}") # NEW TEST LINE

    # --- Special handling for min_lr value to ensure compatibility with accelerate ---
    if hasattr(args, 'minimum_lr') and args.minimum_lr is not None:
        try:
            # Try to convert the string value directly in case it was passed with special characters
            min_lr_str = str(args.minimum_lr).replace('‑', '-')  # Replace en-dash with hyphen
            args.minimum_lr = float(min_lr_str)
            logger.info(f"Processed minimum_lr value: {args.minimum_lr}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing minimum_lr value: {e}. Using as-is.")
    # ---

    # --- Log Initial Configuration ---
    if is_main_process(local_rank=-1): # Guarding this block
        logger.info("--- Starting GRPO Training Run (Main Process) ---")
        logger.info(f"Script: {__file__}")
        logger.info(f"Run Timestamp: {run_timestamp}") # Ensure run_timestamp is accessible
        logger.info(f"Command Line Arguments: {vars(args)}") # Log parsed args

    # Set the evaluator environment variable for use by run_benchmark_eval
    os.environ["ARGEN_EVALUATOR"] = args.evaluator
    logger.info(f"Using evaluator: {args.evaluator}")

    # Set the include_reasoning flag for Gemini evaluations
    if args.evaluator == "gemini":
        try:
            from src.reward_functions.trl_rewards import configure_gemini_reasoning
            configure_gemini_reasoning(args.log_eval_reasoning)
        except ImportError:
            try:
                # Fallback to direct import if trl_rewards is not available
                from src.reward_functions.gemini_rewards import set_include_reasoning
                set_include_reasoning(args.log_eval_reasoning)
                logger.info(f"Set include_reasoning to {args.log_eval_reasoning} for Gemini evaluations")
            except ImportError:
                logger.warning("Could not import set_include_reasoning from gemini_rewards.py")
                logger.warning("Reasoning field will be included in Gemini evaluations (default behavior)")

    # Load environment variables early
    load_env_vars()

    # Get the appropriate API key based on the evaluator
    if args.evaluator == "openai":
        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            print("Error: OPENAI_API_KEY not found. Please set it in your .env file or environment.")
            sys.exit(1)
    else:  # args.evaluator == "gemini"
        try:
            gemini_api_key = get_gemini_api_key()
            if not gemini_api_key:
                print("Error: GEMINI_API_KEY not found. Please set it in your .env file or environment.")
                sys.exit(1)
        except ImportError:
            print("Error: google-generativeai package not installed. Please install it with 'pip install google-generativeai'")
            sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    print("--- ABOUT TO LOAD TOKENIZER (direct print) ---") # New direct print
    logger.info("--- ABOUT TO LOAD TOKENIZER (logger.info) ---")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("--- TOKENIZER LOADED (direct print) ---") # New direct print
    logger.info("--- TOKENIZER LOADED (logger.info) ---")

    logger.info(f"DEBUG: Checking main process. LOCAL_RANK env var: {os.environ.get('LOCAL_RANK')}, Effective is_main_process: {is_main_process(local_rank=-1)}")

    if GLOBAL_TRAIN_GRPO_TEMPLATE_DEBUG and is_main_process(local_rank=-1): # Only print/log on main process
        logger.info(f"DEBUG: Tokenizer chat template content: '{tokenizer.chat_template}'")
        logger.info(f"NATIVE CHAT TEMPLATE for {args.model}: {tokenizer.chat_template}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for GRPO

    # Get the system prompt
    system_prompt = get_system_prompt(args.use_basic_prompt)
    logger.info(f"System Prompt Type: {'BASIC' if args.use_basic_prompt else 'ENHANCED'}")

    # Log the system prompt if verbose logging is enabled
    if args.verbose_logging:
        logger.info(f"System Prompt: {system_prompt}")

    # Prepare the dataset
    logger.info(f"Loading scenarios from {args.scenarios}...")
    if not os.path.exists(args.scenarios):
        logger.error(f"Training scenarios file {args.scenarios} not found. Please ensure you provide the path to the *processed* dataset with hashed tiers.")
        # Maybe suggest running the processing script
        logger.error("Hint: Run `python scripts/add_prompt_hash_to_dataset.py` on your original data.")
        sys.exit(1)

    train_dataset = prepare_dataset(
        scenarios_path=args.scenarios,
        verify_hashes=True,
        prompt_format=args.prompt_format,
        system_prompt=system_prompt,
        verbose_logging=args.verbose_logging
    )

    eval_dataset = None # Initialize eval_dataset to None
    if args.evaluation_strategy != "no": # Corrected condition
        # Prepare the evaluation dataset
        logger.info(f"Loading evaluation scenarios from {args.eval_scenarios}...")
        if os.path.exists(args.eval_scenarios):
            eval_dataset = prepare_dataset(
                scenarios_path=args.eval_scenarios,
                verify_hashes=True,
                prompt_format=args.prompt_format,
                system_prompt=system_prompt,
                verbose_logging=args.verbose_logging
            )
            logger.info(f"Successfully loaded evaluation dataset with {len(eval_dataset)} scenarios")
        else:
            logger.warning(f"Evaluation scenarios file {args.eval_scenarios} not found. Setting evaluation strategy to 'no'.")
            args.evaluation_strategy = "no"
            eval_dataset = None

    # Get GRPO config
    from src.config import GRPO_CONFIG
    grpo_config = GRPO_CONFIG

    # Override config with command line arguments if provided
    if args.output_dir:
        grpo_config["output_dir"] = args.output_dir
    if args.num_train_epochs:
        grpo_config["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate:
        grpo_config["learning_rate"] = args.learning_rate
    if args.num_generations is not None:
        grpo_config["num_generations"] = args.num_generations
    if args.per_device_train_batch_size is not None:
        grpo_config["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        grpo_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    # Override Gemini call mode based on command line argument
    if args.use_batch_gemini_calls:
        grpo_config["use_single_gemini_calls_for_training"] = False
        logger.info("Using batch Gemini API calls for training (--use_batch_gemini_calls specified)")
    else:
        grpo_config["use_single_gemini_calls_for_training"] = True
        logger.info("Using single Gemini API calls for training (default)")

    # Override temperature settings if provided
    if args.temperature is not None:
        grpo_config["temperature"] = args.temperature
        logger.info(f"Using custom training temperature: {args.temperature}")
    if args.eval_temperature is not None:
        grpo_config["eval_temperature"] = args.eval_temperature
        logger.info(f"Using custom evaluation temperature: {args.eval_temperature}")

    # Create run name with timestamp if not provided
    if not args.wandb_run_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = args.model.split("/")[-1] if "/" in args.model else args.model
        args.wandb_run_name = f"{model_short_name}-grpo-{timestamp}"

    # Initialize W&B ONLY ON MAIN PROCESS
    # Set WANDB_MODE=offline for non-main ranks to prevent them from logging
    WANDB_DISABLED = not is_main_process(local_rank=-1)
    if WANDB_DISABLED:
        os.environ["WANDB_MODE"] = "offline"
        logger.info("Running on non-main process, setting WANDB_MODE=offline.")

    if is_main_process(local_rank=-1):
        logger.info(f"Initializing W&B run '{args.wandb_run_name}' in project '{args.wandb_project}'...")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model,
                "scenarios": args.scenarios,
                "use_basic_prompt": args.use_basic_prompt,
                "use_separate_rewards": args.use_separate_rewards,
                "ahimsa_weight": REWARD_WEIGHTS["ahimsa"],
                "dharma_weight": REWARD_WEIGHTS["dharma"],
                "helpfulness_weight": REWARD_WEIGHTS["helpfulness"],
                # Include evaluation settings in config
                "evaluation_strategy": args.evaluation_strategy if args.evaluation_strategy is not None else grpo_config.get("evaluation_strategy", "no"),
                "eval_steps": args.eval_steps if args.eval_steps is not None else grpo_config.get("eval_steps", 5000),
                "early_stopping": args.early_stopping,
                # Include evaluator setting
                "evaluator": args.evaluator,
                # Include new kl and lr parameters
                "kl_penalty": args.kl_penalty,
                "target_kl": args.target_kl,
                "minimum_lr": args.minimum_lr,
                # Include adaptive weights settings
                "adaptive_weights": args.adaptive_weights,
                "adaptive_weights_gamma": args.adaptive_weights_gamma,
                "adaptive_weights_min": args.adaptive_weights_min,
                # Include log_eval_reasoning setting
                "log_eval_reasoning": args.log_eval_reasoning,
                # Include use_batch_gemini_calls setting
                "use_batch_gemini_calls": args.use_batch_gemini_calls,
                # Include temperature settings
                "temperature": args.temperature,
                "eval_temperature": args.eval_temperature,
                **grpo_config
            }
        )
        logger.info("W&B initialized on main process.")
    else:
         logger.info("Skipping W&B initialization on non-main process.")

    # Choose reward function based on argument
    if args.use_separate_rewards:
        reward_funcs = [ahimsa_reward_trl, dharma_reward_trl, helpfulness_reward_trl]
        reward_weights = [
            REWARD_WEIGHTS["ahimsa"],
            REWARD_WEIGHTS["dharma"],
            REWARD_WEIGHTS["helpfulness"]
        ]
    else:
        reward_funcs = combined_reward_trl
        reward_weights = None

    # Define compute_metrics function for evaluation
    def compute_metrics(eval_pred):
        """Compute metrics for evaluation including combined rewards."""
        if args.use_separate_rewards:
            # For separate rewards, we have predictions of shape [batch_size, 3]
            comps = np.array(eval_pred.predictions)
            weights = np.array([
                REWARD_WEIGHTS["ahimsa"],
                REWARD_WEIGHTS["dharma"],
                REWARD_WEIGHTS["helpfulness"]
            ])
            # Calculate weighted sum of component rewards
            combined = (comps * weights).sum(axis=-1)

            return {
                "eval/ahimsa_mean": comps[:, 0].mean(),
                "eval/dharma_mean": comps[:, 1].mean(),
                "eval/helpfulness_mean": comps[:, 2].mean(),
                "eval/combined_mean": combined.mean(),
                "eval/combined_std": combined.std(),
            }
        else:
            # For combined reward, we have predictions of shape [batch_size]
            rewards = np.array(eval_pred.predictions)
            return {
                "eval/combined_mean": rewards.mean(),
                "eval/combined_std": rewards.std(),
            }

    # Create GRPO config
    trl_config = GRPOConfig(
        output_dir=grpo_config["output_dir"],
        num_train_epochs=grpo_config["num_train_epochs"],
        learning_rate=grpo_config["learning_rate"],
        gradient_accumulation_steps=grpo_config["gradient_accumulation_steps"],
        per_device_train_batch_size=grpo_config["per_device_train_batch_size"],
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_prompt_length=grpo_config["max_prompt_length"],
        max_completion_length=grpo_config["max_completion_length"],
        num_generations=args.num_generations if args.num_generations is not None else grpo_config["num_generations"],
        num_iterations=grpo_config["num_iterations"],
        beta=args.kl_beta_start or grpo_config["beta"],
        disable_dropout=grpo_config["disable_dropout"],
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else grpo_config.get("warmup_steps", 0),
        logging_steps=args.logging_steps if args.logging_steps is not None else grpo_config.get("logging_steps", 10),
        bf16=grpo_config["bf16"],
        # Ensure report_to only includes 'wandb' if on main process, or is empty
        report_to=args.report_to if args.report_to is not None else (["wandb"] if is_main_process(local_rank=-1) else []),
        save_strategy=args.save_strategy or "epoch",
        save_total_limit=args.save_total_limit or 4,
        run_name=args.wandb_run_name, # Also pass run_name here to potentially suppress Trainer warning
        lr_scheduler_type=args.lr_scheduler_type if args.lr_scheduler_type else grpo_config.get("lr_scheduler_type", "linear"),
        max_grad_norm=args.max_grad_norm if args.max_grad_norm is not None else grpo_config.get("max_grad_norm", 1.0),
        gradient_checkpointing=args.gradient_checkpointing if args.gradient_checkpointing is not None else grpo_config.get("gradient_checkpointing", False),
        seed=args.seed if args.seed is not None else 42,
        # Add evaluation strategy parameters - using correct parameter names
        eval_strategy=args.evaluation_strategy if args.evaluation_strategy is not None else grpo_config.get("evaluation_strategy", "no"),
        eval_steps=args.eval_steps if args.eval_steps is not None else grpo_config.get("eval_steps", 5000),
        # Add temperature parameter for GRPO training
        temperature=args.temperature if args.temperature is not None else grpo_config.get("temperature", GRPO_TRAINING_TEMPERATURE),
        # --- ADDED: Pass reward_weights if using separate rewards ---
        reward_weights=(
            [REWARD_WEIGHTS["ahimsa"], REWARD_WEIGHTS["dharma"], REWARD_WEIGHTS["helpfulness"]]
            if args.use_separate_rewards
            else None
        ),
        # --- END ADDED ---
    )

    # Apply the new command-line parameters if provided
    if args.target_kl is not None:
        trl_config.target_kl = args.target_kl
        logger.info(f"Setting target KL to {args.target_kl}")

    if args.minimum_lr is not None:
        trl_config.min_lr = args.minimum_lr
        logger.info(f"Setting minimum learning rate to {args.minimum_lr}")

    # Override save_steps if provided via CLI and strategy is 'steps'
    if args.save_strategy == "steps" and args.save_steps is not None:
        trl_config.save_steps = args.save_steps
    elif args.save_strategy == "steps" and args.save_steps is None:
        # If strategy is steps but no specific save_steps is given,
        # Trainer defaults to logging_steps. We can make this explicit or log a warning.
        logger.info(f"save_strategy is 'steps' but --save_steps not provided. Defaulting save_steps to logging_steps ({trl_config.logging_steps}).")
        trl_config.save_steps = trl_config.logging_steps

    # Add model initialization kwargs - explicitly ensure bf16 if intended
    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        # Add any other specific model loading kwargs here if needed
    }
    # Only set if bf16 is true in the config to be consistent,
    # though bf16=True in GRPOConfig should already handle torch_dtype.
    # This explicit setting provides an extra layer of certainty or overrides.
    if grpo_config.get("bf16", False): # Check against the effective grpo_config
        trl_config.model_init_kwargs = model_init_kwargs
        logger.info(f"Explicitly set model_init_kwargs with torch_dtype=torch.bfloat16 on trl_config based on effective grpo_config['bf16']={grpo_config.get('bf16')}")
    elif hasattr(trl_config, 'model_init_kwargs') and trl_config.model_init_kwargs and "torch_dtype" in trl_config.model_init_kwargs:
        logger.info(f"trl_config.model_init_kwargs already contains torch_dtype: {trl_config.model_init_kwargs['torch_dtype']}")
    else:
        logger.info("bf16 not enabled in effective grpo_config or model_init_kwargs not set with torch_dtype. Model will load with default precision or as specified by GRPOConfig defaults.")

    # --- Log the final TRLConfig object ---
    logger.info(f"Final TRL Training Config: {trl_config}")
    # Log specific batch sizes and generations to ensure compatibility
    effective_eval_batch_size = trl_config.per_device_eval_batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    logger.info(f"Effective eval batch size: {effective_eval_batch_size}, Num generations: {trl_config.num_generations}")
    logger.info(f"Valid num_generations values for this batch size: {[i for i in range(1, effective_eval_batch_size+1) if effective_eval_batch_size % i == 0]}")
    # --- End log ---

    logger.info(f"[{os.getenv('RANK', 'N/A')}] Initializing GRPOTrainer for model {args.model}...")

    # Check logger configuration before GRPOTrainer initialization
    print(f"Logger level before GRPOTrainer init: {logger.level}")
    print(f"Root logger level before GRPOTrainer init: {logging.getLogger().level}")
    print(f"Logger handlers before GRPOTrainer init: {[type(h).__name__ for h in logger.handlers]}")
    print(f"Root logger handlers before GRPOTrainer init: {[type(h).__name__ for h in logging.getLogger().handlers]}")

    # Create a wrapper for logging.basicConfig to detect if it's called during GRPOTrainer initialization
    original_basicConfig = logging.basicConfig
    def wrapped_basicConfig(*args, **kwargs):
        print(f"WARNING: logging.basicConfig called with args={args}, kwargs={kwargs}")
        print(f"Caller stack: {[f.function for f in inspect.stack()[1:5]]}")
        # Call the original function but with force=True to prevent it from changing our config
        kwargs['force'] = False  # Don't force, just detect
        return original_basicConfig(*args, **kwargs)

    # Replace the function temporarily
    import inspect
    logging.basicConfig = wrapped_basicConfig

    trainer_class = GRPOTrainer
    try:
        from fix_grpo_chat_template import extract_content_from_chat_response

        # Log an example of content extraction if verbose logging is enabled
        if args.verbose_logging:
            example_response = {"role": "assistant", "content": "This is a test response"}
            extracted = extract_content_from_chat_response(example_response)
            logger.info(f"Example content extraction: {extracted}")
    except ImportError:
        # Define a simple extract_content function if the imported one is not available
        def extract_content_from_chat_response(response):
            if isinstance(response, dict) and 'content' in response:
                return response['content']
            return response

    # Wrap reward functions with verbose logging if enabled
    if args.verbose_logging:
        original_reward_funcs = reward_funcs

        # For single reward function
        if not isinstance(original_reward_funcs, list):
            def verbose_reward_func(prompts, completions, **kwargs):
                logger.info(f"Completions: {completions[:2]}")  # Log first 2 completions
                rewards = original_reward_funcs(prompts=prompts, completions=completions, **kwargs)
                logger.info(f"Rewards calculated: {rewards}")
                return rewards

            reward_funcs = verbose_reward_func
        # For multiple reward functions
        else:
            verbose_reward_funcs = []
            for i, func in enumerate(original_reward_funcs):
                def make_verbose_func(orig_func, idx):
                    def verbose_func(prompts, completions, **kwargs):
                        logger.info(f"Reward function {idx} - Completions: {completions[:2]}")
                        rewards = orig_func(prompts=prompts, completions=completions, **kwargs)
                        logger.info(f"Reward function {idx} - Rewards: {rewards}")
                        return rewards
                    return verbose_func

                verbose_reward_funcs.append(make_verbose_func(func, i))

            reward_funcs = verbose_reward_funcs

    # For instruct format, we need to modify the tokenizer's behavior
    if args.prompt_format == "instruct":
        logger.info("Using instruct format - modifying tokenizer behavior")

        # Define a custom apply_chat_template method for instruct format
        def custom_apply_chat_template(messages, **_):
            # Extract the system prompt and user message
            system_content = ""
            user_content = ""

            for message in messages:
                if message["role"] == "system":
                    system_content = message["content"]
                elif message["role"] == "user":
                    user_content = message["content"]

            # Create an instruct-style prompt
            instruct_prompt = f"{system_content}\n\nUser question: {user_content}\n\nAnswer:"

            # Use global logger instead of args
            if args.verbose_logging:
                logger.info(f"Custom instruct prompt: {instruct_prompt[:100]}...")

            return instruct_prompt

        # Monkey patch the tokenizer's apply_chat_template method
        tokenizer._original_apply_chat_template = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = custom_apply_chat_template

        if args.verbose_logging:
            logger.info("Tokenizer's apply_chat_template method has been replaced with custom instruct formatter")

    # Initialize the trainer
    trainer = trainer_class(
        model=args.model, # Pass model name or path, GRPOTrainer handles loading
        reward_funcs=reward_funcs,
        args=trl_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Pass the potentially None eval_dataset
        processing_class=tokenizer, # Pass tokenizer, GRPOTrainer will apply chat template
    )

    # Restore the original function
    logging.basicConfig = original_basicConfig

    # --- ADDED: Log Final Trainer Arguments ---
    logger.info(f"Final Trainer Arguments:\n{trainer.args.to_json_string()}")

    # Check logger configuration after GRPOTrainer initialization
    print(f"Logger level after GRPOTrainer init: {logger.level}")
    print(f"Root logger level after GRPOTrainer init: {logging.getLogger().level}")
    print(f"Logger handlers after GRPOTrainer init: {[type(h).__name__ for h in logger.handlers]}")
    print(f"Root logger handlers after GRPOTrainer init: {[type(h).__name__ for h in logging.getLogger().handlers]}")

    # Reset logging configuration again
    try:
        # Use our logging fix if available
        logging_fix.reset_logging_config(log_file_path)
    except (NameError, AttributeError):
        # Fall back to our internal reset function if logging_fix is not available
        reset_logging_config()
    logger.info("Logging configuration reset after GRPOTrainer initialization")
    # --- End ADDED ---

    # Instantiate and add callbacks
    # Ensure only one CustomWandbLoggingCallback is instantiated and added
    custom_logging_callback = CustomWandbLoggingCallback()
    custom_logging_callback.set_trainer(trainer)
    log_eval_command_callback = LogEvalCommandCallback() # Instantiate new callback
    console_metrics_callback = ConsoleMetricsCallback()
    console_metrics_callback.trainer = trainer  # Set trainer reference for callback communication

    # Check if default WandbCallback is already added by report_to=['wandb']
    # Avoid adding duplicates if HF Trainer adds it automatically
    has_wandb_callback = any(isinstance(cb, WandbCallback) for cb in trainer.callback_handler.callbacks)
    if not has_wandb_callback and is_main_process(local_rank=-1): # Only warn on main process
         logger.warning("Default WandbCallback not found, gradient norm might not be logged automatically.")
         # Consider adding it explicitly if needed: trainer.add_callback(WandbCallback())
    elif has_wandb_callback and is_main_process(local_rank=-1):
         logger.info("Default WandbCallback found and will handle standard logging.")

    trainer.add_callback(custom_logging_callback)
    trainer.add_callback(log_eval_command_callback) # ADD the new callback
    trainer.add_callback(console_metrics_callback)

    # Add adaptive weights callback if enabled and using separate rewards
    if args.use_separate_rewards and args.adaptive_weights != "none":
        adaptive_weights_callback = MeanAdaptiveWeights(
            trainer=trainer,
            gamma=args.adaptive_weights_gamma,
            min_w=args.adaptive_weights_min,
            strategy=args.adaptive_weights
        )
        trainer.add_callback(adaptive_weights_callback)
        logger.info(f"Added MeanAdaptiveWeights callback with strategy={args.adaptive_weights}, "
                   f"gamma={args.adaptive_weights_gamma}, min_w={args.adaptive_weights_min}")

    logger.info("Added CustomWandbLoggingCallback, LogEvalCommandCallback, and ConsoleMetricsCallback.")

    # Add early stopping callback if enabled
    if args.early_stopping and args.evaluation_strategy in ["steps", "epoch"]:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
            metric_for_early_stopping="eval/combined_mean"
        )
        trainer.add_callback(early_stopping_callback)
        logger.info(f"Added EarlyStoppingCallback with patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}, monitoring 'eval/combined_mean'")
    elif args.early_stopping and args.evaluation_strategy == "no":
        logger.warning("Early stopping requires evaluation_strategy to be 'steps' or 'epoch'. Early stopping will not be enabled.")

    # Training
    logger.info("Starting GRPO training...")
    # Pass resume_from_checkpoint here
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the model - Trainer saves only on main process by default
    if is_main_process(local_rank=-1):
        logger.info(f"Saving final trained model to {trl_config.output_dir}...")
        trainer.save_model(trl_config.output_dir) # Explicitly pass path to save_model
        logger.info("Final model save complete.")
    else:
        logger.info("Skipping final model save on non-main process.")

    if is_main_process(local_rank=-1):
        logger.info("GRPO training complete!")
        # Optionally, log final metrics from train_result
        # metrics = train_result.metrics
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)

        # Finish W&B run only on main process
        wandb.finish()
        logger.info("W&B run finished.")
    else:
        logger.info("Non-main process finished training.")

    if args.evaluator == "gemini":
        from src.reward_functions.gemini_rewards import get_gemini_api_call_count
        gemini_calls = get_gemini_api_call_count()
        if is_main_process(local_rank=-1):
            print(f"Total Gemini API calls: {gemini_calls}")

    # --- ADDED: Clean up distributed process group ---
    if dist.is_initialized():
        logger.info(f"[{os.getenv('RANK', 'N/A')}] Destroying distributed process group...")
        dist.destroy_process_group()
        logger.info(f"[{os.getenv('RANK', 'N/A')}] Distributed process group destroyed.")
    # --- END ADDED ---

    # Set TRL logger level to DEBUG for additional debug logs
    logging.getLogger("trl").setLevel(logging.DEBUG)

if __name__ == "__main__":
    main()
