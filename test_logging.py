#!/usr/bin/env python3
"""
Test script to diagnose logging issues with TRL.
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Basic Config (StreamHandler for Console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
    force=True  # Force this configuration to override any previous settings
)
logger = logging.getLogger(__name__)  # Get logger for this module

# File Handler Setup
run_timestamp = "test"
log_file_path = log_dir / f"test_logging_{run_timestamp}.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)  # Log DEBUG level and up to file
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add File Handler to the root logger
logging.getLogger().addHandler(file_handler)

logger.info(f"Console and File logging initialized. Log file: {log_file_path}")

# Define a function to reset logging configuration
def reset_logging_config():
    """
    Reset logging configuration after TRL imports.
    TRL's init_zero_verbose function may have changed the root logger level to ERROR.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Set root logger level back to DEBUG to allow all handlers to work
    root_logger.setLevel(logging.DEBUG)
    
    # Make sure our file handler is still attached
    file_handler_exists = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path)
        for h in root_logger.handlers
    )
    
    if not file_handler_exists:
        # Re-add our file handler if it was removed
        root_logger.addHandler(file_handler)
        logger.warning("File handler was removed and has been re-added")
    
    # Ensure __main__ logger is at INFO level
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    
    # Log a test message to verify logging is working
    logger.info("Logging configuration reset")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

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

# Reset logging configuration after all imports
reset_logging_config()

def main():
    """Run the test script."""
    # Reset logging configuration again at the start of main
    reset_logging_config()
    
    # Add a test log message to verify logging is working
    logger.info("Starting test script with logging verified")
    print("Direct print: Starting test script")
    
    # Check logger level and handlers
    print(f"Logger level: {logger.level}")
    print(f"Root logger level: {logging.getLogger().level}")
    print(f"Logger handlers: {[type(h).__name__ for h in logger.handlers]}")
    print(f"Root logger handlers: {[type(h).__name__ for h in logging.getLogger().handlers]}")
    
    # Test logging at different levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Create a wrapper for logging.basicConfig to detect if it's called
    import inspect
    original_basicConfig = logging.basicConfig
    def wrapped_basicConfig(*args, **kwargs):
        print(f"WARNING: logging.basicConfig called with args={args}, kwargs={kwargs}")
        print(f"Caller stack: {[f.function for f in inspect.stack()[1:5]]}")
        # Call the original function but with force=True to prevent it from changing our config
        kwargs['force'] = False  # Don't force, just detect
        return original_basicConfig(*args, **kwargs)
    
    # Replace the function temporarily
    logging.basicConfig = wrapped_basicConfig
    
    # Create a GRPOConfig object
    config = GRPOConfig(
        output_dir="./output",
        num_train_epochs=1,
        learning_rate=1e-5,
    )
    
    # Log after creating GRPOConfig
    logger.info("Created GRPOConfig object")
    
    # Restore the original function
    logging.basicConfig = original_basicConfig
    
    # Reset logging configuration again
    reset_logging_config()
    
    # Final log message
    logger.info("Test script completed")

if __name__ == "__main__":
    main()
