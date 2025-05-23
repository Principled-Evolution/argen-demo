#!/usr/bin/env python3
"""
Script to compare the behavior of the PyPI version of TRL with a local clone.
"""

import os
import sys
import logging
import importlib.util
import inspect
from pathlib import Path

# Configure logging
log_dir = Path("debug_logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "trl_comparison.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_trl_version():
    """Check the installed TRL version."""
    try:
        import trl
        logger.info(f"Installed TRL version: {trl.__version__}")
        return trl.__version__
    except (ImportError, AttributeError):
        logger.error("Could not determine TRL version")
        return None

def load_local_trl(local_path):
    """Load the local TRL module from the specified path."""
    try:
        # Add the local TRL path to sys.path
        local_path = os.path.abspath(local_path)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)
        
        # Try to import the local TRL module
        import trl as local_trl
        logger.info(f"Local TRL loaded from: {local_path}")
        logger.info(f"Local TRL version: {getattr(local_trl, '__version__', 'unknown')}")
        
        return local_trl
    except Exception as e:
        logger.error(f"Error loading local TRL: {e}")
        return None

def compare_grpo_trainer(pypi_trl, local_trl):
    """Compare the GRPOTrainer implementation between PyPI and local TRL."""
    if not pypi_trl or not local_trl:
        logger.error("Cannot compare: one or both TRL modules not available")
        return
    
    try:
        # Get the GRPOTrainer classes
        pypi_grpo_trainer = pypi_trl.GRPOTrainer
        local_grpo_trainer = local_trl.GRPOTrainer
        
        logger.info("Comparing GRPOTrainer implementations...")
        
        # Compare the class definitions
        pypi_source = inspect.getsource(pypi_grpo_trainer)
        local_source = inspect.getsource(local_grpo_trainer)
        
        if pypi_source == local_source:
            logger.info("GRPOTrainer implementations are identical")
        else:
            logger.info("GRPOTrainer implementations differ")
            
            # Save the sources to files for comparison
            with open(log_dir / "pypi_grpo_trainer.py", "w") as f:
                f.write(pypi_source)
            
            with open(log_dir / "local_grpo_trainer.py", "w") as f:
                f.write(local_source)
            
            logger.info(f"Saved source code to {log_dir}/pypi_grpo_trainer.py and {log_dir}/local_grpo_trainer.py")
            
            # Compare specific methods
            compare_methods(pypi_grpo_trainer, local_grpo_trainer)
    
    except Exception as e:
        logger.error(f"Error comparing GRPOTrainer: {e}")

def compare_methods(pypi_class, local_class):
    """Compare specific methods between the PyPI and local implementations."""
    methods_to_compare = [
        "_generate_and_score_completions",
        "compute_loss",
        "get_train_dataloader",
        "prediction_step"
    ]
    
    for method_name in methods_to_compare:
        try:
            pypi_method = getattr(pypi_class, method_name)
            local_method = getattr(local_class, method_name)
            
            pypi_source = inspect.getsource(pypi_method)
            local_source = inspect.getsource(local_method)
            
            if pypi_source == local_source:
                logger.info(f"Method '{method_name}' is identical")
            else:
                logger.info(f"Method '{method_name}' differs")
                
                # Save the method sources to files
                with open(log_dir / f"pypi_{method_name}.py", "w") as f:
                    f.write(pypi_source)
                
                with open(log_dir / f"local_{method_name}.py", "w") as f:
                    f.write(local_source)
                
                logger.info(f"Saved method source to {log_dir}/pypi_{method_name}.py and {log_dir}/local_{method_name}.py")
        
        except (AttributeError, TypeError) as e:
            logger.error(f"Error comparing method '{method_name}': {e}")

def compare_data_utils(pypi_trl, local_trl):
    """Compare the data_utils module between PyPI and local TRL."""
    if not pypi_trl or not local_trl:
        logger.error("Cannot compare: one or both TRL modules not available")
        return
    
    try:
        # Get the data_utils modules
        pypi_data_utils = pypi_trl.data_utils
        local_data_utils = local_trl.data_utils
        
        logger.info("Comparing data_utils implementations...")
        
        # Compare specific functions
        functions_to_compare = [
            "apply_chat_template",
            "maybe_apply_chat_template",
            "is_conversational"
        ]
        
        for func_name in functions_to_compare:
            try:
                pypi_func = getattr(pypi_data_utils, func_name)
                local_func = getattr(local_data_utils, func_name)
                
                pypi_source = inspect.getsource(pypi_func)
                local_source = inspect.getsource(local_func)
                
                if pypi_source == local_source:
                    logger.info(f"Function '{func_name}' is identical")
                else:
                    logger.info(f"Function '{func_name}' differs")
                    
                    # Save the function sources to files
                    with open(log_dir / f"pypi_{func_name}.py", "w") as f:
                        f.write(pypi_source)
                    
                    with open(log_dir / f"local_{func_name}.py", "w") as f:
                        f.write(local_source)
                    
                    logger.info(f"Saved function source to {log_dir}/pypi_{func_name}.py and {log_dir}/local_{func_name}.py")
            
            except (AttributeError, TypeError) as e:
                logger.error(f"Error comparing function '{func_name}': {e}")
    
    except Exception as e:
        logger.error(f"Error comparing data_utils: {e}")

def main():
    logger.info("Starting TRL version comparison")
    
    # Check the installed TRL version
    pypi_version = check_trl_version()
    
    # Ask for the local TRL path
    local_trl_path = input("Enter the path to your local TRL repository: ")
    
    # Load the local TRL module
    local_trl_module = load_local_trl(local_trl_path)
    
    # Import the PyPI TRL module
    import trl as pypi_trl_module
    
    # Compare the GRPOTrainer implementations
    compare_grpo_trainer(pypi_trl_module, local_trl_module)
    
    # Compare the data_utils implementations
    compare_data_utils(pypi_trl_module, local_trl_module)
    
    logger.info("TRL comparison completed")

if __name__ == "__main__":
    main()
