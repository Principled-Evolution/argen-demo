#!/usr/bin/env python3
"""
Test script to verify our logging fix.
"""

import sys
import os
import logging
from pathlib import Path

# Import our logging fix before any TRL imports
import logging_fix

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Set up logging with our helper function
log_file_path = log_dir / f"test_logging_fix.log"
logging_fix.setup_logging(log_file_path)

logger = logging.getLogger(__name__)  # Get logger for this module
logger.info(f"Console and File logging initialized. Log file: {log_file_path}")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import TRL components - this should not silence our loggers
from trl import GRPOTrainer, GRPOConfig

# Reset logging configuration after all imports
logging_fix.reset_logging_config(log_file_path)

def main():
    """Run the test script."""
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
    
    # Create a GRPOConfig object
    config = GRPOConfig(
        output_dir="./output",
        num_train_epochs=1,
        learning_rate=1e-5,
    )
    
    # Log after creating GRPOConfig
    logger.info("Created GRPOConfig object")
    
    # Reset logging configuration again
    logging_fix.reset_logging_config(log_file_path)
    
    # Final log message
    logger.info("Test script completed")

if __name__ == "__main__":
    main()
