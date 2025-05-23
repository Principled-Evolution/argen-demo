#!/usr/bin/env python3
"""
Monkey patch for TRL's init_zero_verbose function to prevent it from silencing loggers.
This file should be imported before any TRL imports.

Usage:
import logging_fix  # Import this before any TRL imports
from trl import GRPOTrainer, GRPOConfig
"""

import logging
import sys
import importlib.util

# First, set up our own logging configuration
def setup_logging(log_file_path=None):
    """Set up logging with both console and file handlers."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root to DEBUG to allow all handlers
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if a path is provided
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Now, monkey patch the TRL init_zero_verbose function before it's imported
def patch_trl_init_zero_verbose():
    """
    Monkey patch TRL's init_zero_verbose function to prevent it from silencing loggers.
    """
    # Check if trl.scripts.utils is already imported
    if 'trl.scripts.utils' in sys.modules:
        # It's already imported, so we need to patch the existing module
        trl_scripts_utils = sys.modules['trl.scripts.utils']
        
        # Save the original function
        if hasattr(trl_scripts_utils, 'init_zero_verbose'):
            original_init = trl_scripts_utils.init_zero_verbose
            
            # Define our replacement function
            def safe_init_zero_verbose():
                print("Prevented TRL's init_zero_verbose from running and silencing loggers")
                return
            
            # Replace the function
            trl_scripts_utils.init_zero_verbose = safe_init_zero_verbose
            print(f"Patched existing init_zero_verbose function")
    else:
        # It's not imported yet, so we'll try to load it and patch it
        try:
            # Find the module spec
            spec = importlib.util.find_spec('trl.scripts.utils')
            if spec is not None:
                # Load the module
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Save the original function
                if hasattr(module, 'init_zero_verbose'):
                    original_init = module.init_zero_verbose
                    
                    # Define our replacement function
                    def safe_init_zero_verbose():
                        print("Prevented TRL's init_zero_verbose from running and silencing loggers")
                        return
                    
                    # Replace the function
                    module.init_zero_verbose = safe_init_zero_verbose
                    
                    # Update sys.modules
                    sys.modules['trl.scripts.utils'] = module
                    print(f"Pre-emptively patched init_zero_verbose function")
        except (ImportError, AttributeError) as e:
            print(f"Could not patch TRL's init_zero_verbose: {e}")

# Apply the patch
patch_trl_init_zero_verbose()

# Also patch the TRL module's __init__ to prevent it from importing and running init_zero_verbose
def patch_trl_init():
    """
    Patch TRL's __init__.py to prevent it from importing and running init_zero_verbose.
    """
    if 'trl' in sys.modules:
        # It's already imported, so we need to patch the existing module
        trl_module = sys.modules['trl']
        
        # Check if _import_structure exists and contains scripts
        if hasattr(trl_module, '_import_structure') and 'scripts' in trl_module._import_structure:
            # Remove init_zero_verbose from the import structure
            if 'init_zero_verbose' in trl_module._import_structure['scripts']:
                trl_module._import_structure['scripts'].remove('init_zero_verbose')
                print("Removed init_zero_verbose from TRL's import structure")

# Apply the TRL init patch
patch_trl_init()

# Function to reset logging configuration if needed
def reset_logging_config(log_file_path=None):
    """
    Reset logging configuration after TRL imports.
    TRL's init_zero_verbose function may have changed the root logger level to ERROR.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Set root logger level back to DEBUG to allow all handlers to work
    root_logger.setLevel(logging.DEBUG)
    
    # If a file path is provided, check if the file handler is still attached
    if log_file_path:
        file_handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path)
            for h in root_logger.handlers
        )
        
        if not file_handler_exists:
            # Re-add our file handler if it was removed
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.getLogger(__name__).warning("File handler was removed and has been re-added")
    
    # Log a test message to verify logging is working
    logging.getLogger(__name__).info("Logging configuration reset")

# Export the functions
__all__ = ['setup_logging', 'reset_logging_config']
