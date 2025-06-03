"""
Weave-specific configuration for ArGen evaluations.

This module contains configuration settings specific to WANDB Weave integration,
including project settings, evaluation naming conventions, and feature flags.
"""

import os
from typing import Dict, Optional, Any
from datetime import datetime

# Weave availability check
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None

# Default Weave project configuration
DEFAULT_WEAVE_PROJECT = "argen-evaluations"
DEFAULT_WEAVE_ENTITY = None  # Will use default WANDB entity

# Weave evaluation configuration
WEAVE_CONFIG = {
    # Project settings
    "default_project": DEFAULT_WEAVE_PROJECT,
    "default_entity": DEFAULT_WEAVE_ENTITY,
    
    # Evaluation naming
    "evaluation_name_prefix": "argen-eval",
    "auto_generate_names": True,
    "include_timestamp": True,
    
    # Feature flags
    "enable_model_versioning": True,
    "enable_dataset_versioning": True,
    "enable_automatic_tagging": True,
    "enable_cost_tracking": True,
    
    # Performance settings
    "batch_size": 10,
    "max_concurrent_evaluations": 5,
    "timeout_seconds": 300,
    
    # Output settings
    "save_traditional_output": True,  # Always save traditional JSON output
    "weave_only_mode": False,  # When True, skip traditional evaluation
}

def ensure_weave_available():
    """
    Ensure Weave is available for use.
    
    Raises:
        ImportError: If Weave is not installed
    """
    if not WEAVE_AVAILABLE:
        raise ImportError(
            "Weave is not installed. Install with: poetry install -E weave"
        )

def get_weave_project_name(custom_name: Optional[str] = None) -> str:
    """
    Get the Weave project name to use.
    
    Args:
        custom_name: Custom project name override
        
    Returns:
        Project name to use for Weave initialization
    """
    if custom_name:
        return custom_name
    
    # Check environment variable
    env_project = os.getenv("WEAVE_PROJECT")
    if env_project:
        return env_project
    
    return WEAVE_CONFIG["default_project"]

def get_evaluation_name(
    model_name: str,
    custom_name: Optional[str] = None,
    include_timestamp: bool = None
) -> str:
    """
    Generate a standardized evaluation name.
    
    Args:
        model_name: Name of the model being evaluated
        custom_name: Custom evaluation name override
        include_timestamp: Whether to include timestamp in name
        
    Returns:
        Standardized evaluation name
    """
    if custom_name:
        return custom_name
    
    if include_timestamp is None:
        include_timestamp = WEAVE_CONFIG["include_timestamp"]
    
    # Clean model name for use in evaluation name
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    base_name = f"{WEAVE_CONFIG['evaluation_name_prefix']}-{clean_model_name}"
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}-{timestamp}"
    
    return base_name

def get_display_name(
    model_name: str,
    scenario_count: int,
    custom_name: Optional[str] = None
) -> str:
    """
    Generate a display name for evaluation runs.
    
    Args:
        model_name: Name of the model being evaluated
        scenario_count: Number of scenarios in evaluation
        custom_name: Custom display name override
        
    Returns:
        Display name for the evaluation run
    """
    if custom_name:
        return custom_name
    
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return f"{clean_model_name} ({scenario_count} scenarios) - {timestamp}"

def get_weave_config() -> Dict[str, Any]:
    """
    Get the complete Weave configuration.
    
    Returns:
        Dictionary containing all Weave configuration settings
    """
    return WEAVE_CONFIG.copy()

def update_weave_config(**kwargs) -> None:
    """
    Update Weave configuration settings.
    
    Args:
        **kwargs: Configuration key-value pairs to update
    """
    WEAVE_CONFIG.update(kwargs)

def is_weave_enabled() -> bool:
    """
    Check if Weave integration is available and enabled.
    
    Returns:
        True if Weave is available and can be used
    """
    return WEAVE_AVAILABLE

def get_weave_tags(
    model_name: str,
    evaluator: str,
    scenario_count: int,
    **kwargs
) -> Dict[str, str]:
    """
    Generate standardized tags for Weave evaluations.
    
    Args:
        model_name: Name of the model being evaluated
        evaluator: Evaluator used (openai, gemini)
        scenario_count: Number of scenarios
        **kwargs: Additional tag key-value pairs
        
    Returns:
        Dictionary of tags for the evaluation
    """
    tags = {
        "model": model_name,
        "evaluator": evaluator,
        "scenario_count": str(scenario_count),
        "framework": "argen",
        "version": "1.0",
    }
    
    # Add any additional tags
    tags.update(kwargs)
    
    return tags
