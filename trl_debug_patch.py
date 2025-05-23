#!/usr/bin/env python3
"""
Script to patch the TRL library with additional debugging information.
"""

import os
import sys
import logging
import inspect
import importlib
import types
from pathlib import Path

# Configure logging
log_dir = Path("debug_logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "trl_patch.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def patch_grpo_trainer():
    """Patch the GRPOTrainer class with additional debugging."""
    try:
        import trl
        from trl import GRPOTrainer
        
        logger.info(f"Patching GRPOTrainer from TRL version {trl.__version__}")
        
        # Get the original _generate_and_score_completions method
        original_generate = GRPOTrainer._generate_and_score_completions
        
        # Define the patched method
        def patched_generate_and_score_completions(self, accumulated_local_batch):
            """Patched version with additional debugging."""
            logger.info("=== ENTERING _generate_and_score_completions ===")
            logger.info(f"Batch keys: {accumulated_local_batch.keys()}")
            
            if "input_ids" in accumulated_local_batch:
                input_shape = accumulated_local_batch["input_ids"].shape
                logger.info(f"Input IDs shape: {input_shape}")
                
                # Log a sample of the input IDs
                if input_shape[0] > 0:
                    sample_input = accumulated_local_batch["input_ids"][0]
                    logger.info(f"Sample input ID length: {len(sample_input)}")
                    
                    # Decode the sample input
                    try:
                        tokenizer = self.processing_class
                        decoded = tokenizer.decode(sample_input)
                        logger.info(f"Sample decoded input:\n{decoded}")
                        
                        # Save to file
                        with open(log_dir / "sample_input_decoded.txt", "w") as f:
                            f.write(decoded)
                    except Exception as e:
                        logger.error(f"Error decoding sample input: {e}")
            
            # Call the original method
            logger.info("Calling original _generate_and_score_completions method")
            try:
                result = original_generate(self, accumulated_local_batch)
                
                # Log the result
                logger.info(f"Result keys: {result.keys()}")
                
                if "completions" in result:
                    completions = result["completions"]
                    logger.info(f"Generated {len(completions)} completions")
                    
                    # Log a few completions
                    for i, completion in enumerate(completions[:2]):
                        logger.info(f"Completion {i}: {completion}")
                        
                        # Save to file
                        with open(log_dir / f"completion_{i}.txt", "w") as f:
                            f.write(completion)
                
                logger.info("=== EXITING _generate_and_score_completions ===")
                return result
            
            except Exception as e:
                logger.error(f"Error in original _generate_and_score_completions: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        
        # Replace the original method with the patched one
        GRPOTrainer._generate_and_score_completions = patched_generate_and_score_completions
        logger.info("Successfully patched _generate_and_score_completions method")
        
        # Patch the get_train_dataloader method
        original_get_dataloader = GRPOTrainer.get_train_dataloader
        
        def patched_get_train_dataloader(self):
            """Patched version with additional debugging."""
            logger.info("=== ENTERING get_train_dataloader ===")
            
            # Log dataset information
            if hasattr(self, "train_dataset"):
                logger.info(f"Train dataset type: {type(self.train_dataset)}")
                logger.info(f"Train dataset length: {len(self.train_dataset)}")
                
                # Log a few examples
                for i, example in enumerate(self.train_dataset[:2]):
                    logger.info(f"Dataset example {i}: {example}")
            
            # Call the original method
            result = original_get_dataloader(self)
            logger.info(f"Dataloader batch size: {result.batch_size}")
            logger.info("=== EXITING get_train_dataloader ===")
            return result
        
        # Replace the original method with the patched one
        GRPOTrainer.get_train_dataloader = patched_get_train_dataloader
        logger.info("Successfully patched get_train_dataloader method")
        
        return True
    
    except Exception as e:
        logger.error(f"Error patching GRPOTrainer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def patch_data_utils():
    """Patch the data_utils module with additional debugging."""
    try:
        import trl
        
        logger.info(f"Patching data_utils from TRL version {trl.__version__}")
        
        # Get the original apply_chat_template function
        original_apply_chat_template = trl.data_utils.apply_chat_template
        
        # Define the patched function
        def patched_apply_chat_template(example, tokenizer, tools=None):
            """Patched version with additional debugging."""
            logger.info("=== ENTERING apply_chat_template ===")
            logger.info(f"Example keys: {example.keys()}")
            
            if "prompt" in example:
                logger.info(f"Prompt type: {type(example['prompt'])}")
                logger.info(f"Prompt content: {example['prompt']}")
            
            logger.info(f"Tokenizer: {tokenizer.__class__.__name__}")
            logger.info(f"Chat template: {tokenizer.chat_template}")
            
            # Call the original function
            try:
                result = original_apply_chat_template(example, tokenizer, tools)
                
                # Log the result
                logger.info(f"Result keys: {result.keys()}")
                
                if "prompt" in result:
                    logger.info(f"Formatted prompt:\n{result['prompt']}")
                    
                    # Save to file
                    with open(log_dir / "formatted_prompt.txt", "w") as f:
                        f.write(result["prompt"])
                
                logger.info("=== EXITING apply_chat_template ===")
                return result
            
            except Exception as e:
                logger.error(f"Error in original apply_chat_template: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        
        # Replace the original function with the patched one
        trl.data_utils.apply_chat_template = patched_apply_chat_template
        logger.info("Successfully patched apply_chat_template function")
        
        return True
    
    except Exception as e:
        logger.error(f"Error patching data_utils: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("Starting TRL patching")
    
    # Patch the GRPOTrainer class
    success1 = patch_grpo_trainer()
    
    # Patch the data_utils module
    success2 = patch_data_utils()
    
    if success1 and success2:
        logger.info("TRL patching completed successfully")
    else:
        logger.error("TRL patching failed")

if __name__ == "__main__":
    main()
