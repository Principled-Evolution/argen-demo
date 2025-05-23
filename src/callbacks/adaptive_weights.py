"""
Adaptive weights callback for GRPO training.

This callback dynamically adjusts the weights of reward functions based on their
running means to ensure all components contribute meaningfully to the gradient.
"""

import logging
import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

class MeanAdaptiveWeights(TrainerCallback):
    """
    Callback that dynamically adjusts reward weights based on their running means.

    This addresses the issue where reward components with low means contribute little
    to the gradient despite having high variance and potentially important signal.

    Four strategies are implemented:
    1. Mean-inverse weighting: Weights are inversely proportional to running means
    2. Equal-contribution rule: Weights are scaled to ensure contributions proportional to targets
    3. Target-floor: Weights start at target values and only increase when needed (clamped to target floor)
    4. Target-weighted: Weights proportional to target/mean without floor constraint

    Args:
        trainer: The GRPO trainer instance
        gamma: Exponential moving average factor (default: 0.05)
        min_w: Minimum weight for any component (default: 0.1)
        strategy: Weighting strategy, either "mean_inverse" or "equal_contribution" (default: "mean_inverse")
        target_weights: Target weight proportions for equal_contribution strategy (default: None)
    """

    def __init__(
        self,
        trainer,
        gamma: float = 0.05,
        min_w: float = 0.1,
        strategy: str = "mean_inverse",
        target_weights: Optional[list] = None
    ):
        self.trainer = trainer
        self.gamma = gamma
        self.min_w = min_w
        self.strategy = strategy

        # Initialize running mean with a reasonable starting value
        self.running_mean = torch.zeros_like(trainer.reward_weights) + 0.3

        # Set target weights for equal_contribution strategy
        if strategy == "equal_contribution" and target_weights is not None:
            self.target_weights = torch.tensor(target_weights, device=self.running_mean.device)
        else:
            # Default to original weights if not specified
            self.target_weights = torch.clone(trainer.reward_weights)

        logger.info(f"Initialized MeanAdaptiveWeights callback with strategy={strategy}, gamma={gamma}, min_w={min_w}")
        logger.info(f"Initial weights: {trainer.reward_weights.tolist()}")
        logger.info(f"Target weights: {self.target_weights.tolist() if hasattr(self, 'target_weights') else 'None'}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kw):
        """
        Update weights based on the running means of reward components.

        This is called whenever the trainer logs metrics, which happens at regular intervals
        determined by the trainer's logging_steps parameter.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            logs: Dictionary of logged metrics
            **kw: Additional keyword arguments
        """
        # Add debug logging to see if this method is being called
        logger.info(f"MeanAdaptiveWeights.on_log called at step {state.global_step if state else 'unknown'}")

        if logs is None:
            logger.warning("MeanAdaptiveWeights.on_log: logs is None, returning early")
            return

        # Log the keys in the logs dictionary to help diagnose issues
        logger.info(f"MeanAdaptiveWeights.on_log: logs keys: {sorted(logs.keys())}")

        # Define possible key patterns for each reward component
        ahimsa_patterns = ["train/rewards/ahimsa_reward_trl/mean", "rewards/ahimsa_reward_trl/mean", "rewards/avg_ahimsa"]
        dharma_patterns = ["train/rewards/dharma_reward_trl/mean", "rewards/dharma_reward_trl/mean", "rewards/avg_dharma"]
        helpfulness_patterns = ["train/rewards/helpfulness_reward_trl/mean", "rewards/helpfulness_reward_trl/mean", "rewards/avg_helpfulness"]

        # Find the first matching key for each component
        ahimsa_key = next((key for key in ahimsa_patterns if key in logs), None)
        dharma_key = next((key for key in dharma_patterns if key in logs), None)
        helpfulness_key = next((key for key in helpfulness_patterns if key in logs), None)

        # Check if we found all required keys
        if not all([ahimsa_key, dharma_key, helpfulness_key]):
            missing_components = []
            if not ahimsa_key: missing_components.append("ahimsa")
            if not dharma_key: missing_components.append("dharma")
            if not helpfulness_key: missing_components.append("helpfulness")

            logger.warning(f"MeanAdaptiveWeights.on_log: Missing reward components: {missing_components}, returning early")

            # Check if there are any reward-related keys that might be useful
            similar_keys = [key for key in logs.keys() if any(term in key.lower() for term in ["reward", "ahimsa", "dharma", "help"])]
            if similar_keys:
                logger.info(f"MeanAdaptiveWeights.on_log: Found similar keys: {similar_keys}")

            return

        # Log which keys we're using
        logger.info(f"MeanAdaptiveWeights.on_log: Using keys: ahimsa={ahimsa_key}, dharma={dharma_key}, helpfulness={helpfulness_key}")

        # Extract batch means from logs using the keys we found
        batch_mean = torch.tensor([
            logs[ahimsa_key],
            logs[dharma_key],
            logs[helpfulness_key],
        ], device=self.running_mean.device)

        # Update running mean with exponential moving average
        self.running_mean.mul_(1-self.gamma).add_(self.gamma * batch_mean)

        # Calculate new weights based on strategy
        if self.strategy == "mean_inverse":
            # Mean-inverse weighting: weights inversely proportional to means
            inv = 1 / (self.running_mean + 1e-5)
            new_w = inv / inv.sum()
        elif self.strategy == "target_floor":
            # Target Floor Mean Inverse: weights start at target values and only increase when needed
            # Calculate raw weights using target-weighted inverse mean
            raw = self.target_weights / (self.running_mean + 1e-5)
            raw = raw / raw.sum()  # normalize

            # Clamp to floor = target
            clamped = torch.maximum(raw, self.target_weights)
            new_w = clamped / clamped.sum()  # renormalize

            # Log the intermediate values for debugging
            if state.is_local_process_zero:
                logger.info(f"Target Floor calculation - Raw weights: {raw.tolist()}, Clamped: {clamped.tolist()}")
        elif self.strategy == "target_weighted":
            # Target-Weighted Mean Inverse: weights proportional to target/mean without floor constraint
            # This allows weights to go below their target values if their means are high
            raw = self.target_weights / (self.running_mean + 1e-5)
            new_w = raw / raw.sum()  # normalize

            # Log the raw weights for debugging
            if state.is_local_process_zero:
                logger.info(f"Target Weighted calculation - Raw weights: {raw.tolist()}")
        else:  # equal_contribution
            # Equal-contribution rule: scale weights to ensure proportional contributions
            # Define possible std key patterns for each reward component
            ahimsa_std_patterns = ["train/rewards/ahimsa_reward_trl/std", "rewards/ahimsa_reward_trl/std"]
            dharma_std_patterns = ["train/rewards/dharma_reward_trl/std", "rewards/dharma_reward_trl/std"]
            helpfulness_std_patterns = ["train/rewards/helpfulness_reward_trl/std", "rewards/helpfulness_reward_trl/std"]

            # Find the first matching std key for each component or use default values
            ahimsa_std = next((logs[key] for key in ahimsa_std_patterns if key in logs), 0.1)
            dharma_std = next((logs[key] for key in dharma_std_patterns if key in logs), 0.1)
            helpfulness_std = next((logs[key] for key in helpfulness_std_patterns if key in logs), 0.1)

            # Create tensor with std values
            sigma = torch.sqrt(torch.tensor(
                [ahimsa_std**2, dharma_std**2, helpfulness_std**2],
                device=self.running_mean.device))

            # Scale weights so that w_k * sigma_k ≈ target_k * Σ w_j * sigma_j
            new_w = self.target_weights * (sigma.mean() / (sigma + 1e-5))
            new_w = new_w / new_w.sum()

        # Apply minimum weight constraint and renormalize
        new_w = torch.clamp(new_w, self.min_w)
        new_w = new_w / new_w.sum()

        # Update trainer's reward weights
        self.trainer.reward_weights.copy_(new_w)

        # Skip broadcasting entirely - each process will update its own weights independently
        # This is simpler and avoids device placement issues
        # The weights should converge to similar values across processes anyway
        # If you need synchronized weights, you can re-enable this code

        # Commented out to avoid distributed training issues:
        # if dist.is_initialized() and hasattr(self.trainer.args, 'local_rank') and self.trainer.args.local_rank != -1:
        #     try:
        #         logger.info(f"Broadcasting weights from rank 0 to all processes")
        #         dist.broadcast(self.trainer.reward_weights, src=0)
        #     except Exception as e:
        #         logger.warning(f"Error broadcasting reward weights: {e}")

        # Log that we're skipping broadcasting
        if dist.is_initialized():
            logger.info("Skipping weight broadcasting to avoid device placement issues")

        # Log the new weights
        if state.is_local_process_zero:
            weight_names = ["ahimsa", "dharma", "helpfulness"]
            weight_values = {f"adaptive_weights/{name}": val.item() for name, val in zip(weight_names, new_w)}

            # Add running means for monitoring
            mean_values = {f"adaptive_weights/mean_{name}": val.item() for name, val in zip(weight_names, self.running_mean)}

            # Add target weights for reference
            target_values = {f"adaptive_weights/target_{name}": val.item() for name, val in zip(weight_names, self.target_weights)}

            # Add raw and clamped weights for target_floor and target_weighted strategies
            raw_values = {}
            clamped_values = {}
            if 'raw' in locals():
                raw_values = {f"adaptive_weights/raw_{name}": val.item() for name, val in zip(weight_names, raw)}

                if self.strategy == "target_floor" and 'clamped' in locals():
                    clamped_values = {f"adaptive_weights/clamped_{name}": val.item() for name, val in zip(weight_names, clamped)}

            # Combine and log
            log_data = {**weight_values, **mean_values, **target_values, **raw_values, **clamped_values}

            # Log to wandb if available
            try:
                import wandb
                if wandb.run:
                    wandb.log(log_data, step=state.global_step)
            except ImportError:
                pass

            # Always log to console with a more visible format
            log_message = (
                f"\n==== ADAPTIVE WEIGHTS UPDATED AT STEP {state.global_step} (STRATEGY: {self.strategy}) ====\n"
                f"New weights: {', '.join([f'{name}={val:.3f}' for name, val in zip(weight_names, new_w)])}\n"
                f"Running means: {', '.join([f'{name}={val:.3f}' for name, val in zip(weight_names, self.running_mean)])}\n"
                f"Target weights: {', '.join([f'{name}={val:.3f}' for name, val in zip(weight_names, self.target_weights)])}\n"
                f"=================================================="
            )
            logger.info(log_message)

            # Also print directly to stdout to ensure visibility
            print(log_message)
