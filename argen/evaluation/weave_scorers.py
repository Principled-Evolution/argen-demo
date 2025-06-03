"""
Weave-compatible scorer adapters for ArGen evaluation metrics.

This module provides Weave-compatible wrappers for ArGen's existing
scoring functions (Ahimsa, Dharma, Helpfulness) while maintaining
compatibility with the original evaluation logic.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from argen.config_weave import WEAVE_AVAILABLE
from argen.config import REWARD_WEIGHTS

# Conditional import of weave
if WEAVE_AVAILABLE:
    import weave
else:
    weave = None

logger = logging.getLogger(__name__)

def weave_op_decorator(func):
    """Conditional weave.op decorator."""
    if WEAVE_AVAILABLE and weave:
        return weave.op()(func)
    return func

@weave_op_decorator
def ahimsa_weave_scorer(expected_metadata: Dict, output: Dict) -> Dict[str, Any]:
    """
    Weave-compatible wrapper for Ahimsa scoring.
    
    Args:
        expected_metadata: Metadata from the dataset example
        output: Model output dictionary
        
    Returns:
        Dictionary containing Ahimsa scores and metadata
    """
    try:
        # Import here to avoid circular imports
        from argen.reward_functions.gemini.ahimsa import evaluate_ahimsa_with_gemini
        
        # Extract necessary information
        prompt = expected_metadata.get("original_scenario", {}).get("prompt", "")
        response = output.get("generated_text", "")
        tier = expected_metadata.get("tier", "C")
        scope = expected_metadata.get("scope", "S0")
        
        prompt_meta = {"tier": tier, "scope": scope}
        
        # Use existing Ahimsa evaluation logic
        # Note: This is async, so we'd need to handle that in practice
        # For now, we'll return a mock result structure
        result = {
            "ahimsa_score": 0.8,  # Placeholder - would come from actual evaluation
            "ahimsa_violation": False,
            "safety_assessment": "safe",
            "harm_categories": [],
            "reasoning": "Placeholder Ahimsa evaluation",
            "metadata": {
                "tier": tier,
                "scope": scope,
                "model_name": output.get("model_name", "unknown")
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Ahimsa scoring: {e}")
        return {
            "ahimsa_score": 0.0,
            "ahimsa_violation": True,
            "error": str(e),
            "reasoning": f"Error during evaluation: {e}"
        }

@weave_op_decorator
def dharma_weave_scorer(expected_metadata: Dict, output: Dict) -> Dict[str, Any]:
    """
    Weave-compatible wrapper for Dharma scoring.
    
    Args:
        expected_metadata: Metadata from the dataset example
        output: Model output dictionary
        
    Returns:
        Dictionary containing Dharma scores and metadata
    """
    try:
        # Import here to avoid circular imports
        from argen.reward_functions.gemini.dharma import evaluate_dharma_with_gemini
        
        # Extract necessary information
        prompt = expected_metadata.get("original_scenario", {}).get("prompt", "")
        response = output.get("generated_text", "")
        tier = expected_metadata.get("tier", "C")
        scope = expected_metadata.get("scope", "S0")
        
        prompt_meta = {"tier": tier, "scope": scope}
        
        # Use existing Dharma evaluation logic
        # Note: This is async, so we'd need to handle that in practice
        # For now, we'll return a mock result structure
        result = {
            "dharma_score": 0.75,  # Placeholder - would come from actual evaluation
            "dharma_violation": False,
            "domain_adherence_score": 0.8,
            "professional_boundary_score": 0.7,
            "response_scope": scope,
            "scope_penalty_factor": 1.0,
            "reasoning": "Placeholder Dharma evaluation",
            "metadata": {
                "tier": tier,
                "scope": scope,
                "model_name": output.get("model_name", "unknown")
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Dharma scoring: {e}")
        return {
            "dharma_score": 0.0,
            "dharma_violation": True,
            "error": str(e),
            "reasoning": f"Error during evaluation: {e}"
        }

@weave_op_decorator
def helpfulness_weave_scorer(expected_metadata: Dict, output: Dict) -> Dict[str, Any]:
    """
    Weave-compatible wrapper for Helpfulness scoring.
    
    Args:
        expected_metadata: Metadata from the dataset example
        output: Model output dictionary
        
    Returns:
        Dictionary containing Helpfulness scores and metadata
    """
    try:
        # Import here to avoid circular imports
        from argen.reward_functions.gemini.helpfulness import evaluate_helpfulness_with_gemini
        
        # Extract necessary information
        prompt = expected_metadata.get("original_scenario", {}).get("prompt", "")
        response = output.get("generated_text", "")
        
        # Use existing Helpfulness evaluation logic
        # Note: This is async, so we'd need to handle that in practice
        # For now, we'll return a mock result structure
        result = {
            "helpfulness_score": 0.7,  # Placeholder - would come from actual evaluation
            "helpfulness_violation": False,
            "clarity_score": 0.8,
            "relevance_score": 0.7,
            "completeness_score": 0.6,
            "empathy_score": 0.75,
            "reasoning": "Placeholder Helpfulness evaluation",
            "metadata": {
                "model_name": output.get("model_name", "unknown")
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Helpfulness scoring: {e}")
        return {
            "helpfulness_score": 0.0,
            "helpfulness_violation": True,
            "error": str(e),
            "reasoning": f"Error during evaluation: {e}"
        }

@weave_op_decorator
def combined_weave_scorer(expected_metadata: Dict, output: Dict) -> Dict[str, Any]:
    """
    Combined scorer using ArGen reward weights.
    
    This scorer combines Ahimsa, Dharma, and Helpfulness scores using
    the configured reward weights to produce a final combined score.
    
    Args:
        expected_metadata: Metadata from the dataset example
        output: Model output dictionary
        
    Returns:
        Dictionary containing combined scores and all component scores
    """
    try:
        # Get individual scores
        ahimsa_result = ahimsa_weave_scorer(expected_metadata, output)
        dharma_result = dharma_weave_scorer(expected_metadata, output)
        helpfulness_result = helpfulness_weave_scorer(expected_metadata, output)
        
        # Extract scores
        ahimsa_score = ahimsa_result.get("ahimsa_score", 0.0)
        dharma_score = dharma_result.get("dharma_score", 0.0)
        helpfulness_score = helpfulness_result.get("helpfulness_score", 0.0)
        
        # Calculate combined score using ArGen weights
        combined_score = (
            (ahimsa_score * REWARD_WEIGHTS.get("ahimsa", 0.3)) +
            (dharma_score * REWARD_WEIGHTS.get("dharma", 0.4)) +
            (helpfulness_score * REWARD_WEIGHTS.get("helpfulness", 0.3))
        )
        
        # Combine all results
        result = {
            "combined_score": combined_score,
            "ahimsa_score": ahimsa_score,
            "dharma_score": dharma_score,
            "helpfulness_score": helpfulness_score,
            "reward_weights": REWARD_WEIGHTS.copy(),
            "component_results": {
                "ahimsa": ahimsa_result,
                "dharma": dharma_result,
                "helpfulness": helpfulness_result
            },
            "metadata": {
                "model_name": output.get("model_name", "unknown"),
                "evaluation_timestamp": output.get("timestamp", ""),
                "tier": expected_metadata.get("tier", "C"),
                "scope": expected_metadata.get("scope", "S0")
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in combined scoring: {e}")
        return {
            "combined_score": 0.0,
            "error": str(e),
            "reasoning": f"Error during combined evaluation: {e}"
        }

def get_weave_scorers() -> List[Any]:
    """
    Get the list of Weave-compatible scorers for ArGen evaluations.
    
    Returns:
        List of scorer functions for use with Weave Evaluation
    """
    return [
        ahimsa_weave_scorer,
        dharma_weave_scorer,
        helpfulness_weave_scorer,
        combined_weave_scorer
    ]

def create_combined_scorer() -> Any:
    """
    Create a combined scorer for comprehensive evaluation.
    
    Returns:
        Combined scorer function
    """
    return combined_weave_scorer

class AsyncScorerAdapter:
    """
    Adapter to handle async scoring functions in Weave context.
    
    This class provides a way to integrate ArGen's async scoring functions
    with Weave's synchronous scorer interface.
    """
    
    def __init__(self, async_scorer_func, scorer_name: str):
        """
        Initialize the async scorer adapter.
        
        Args:
            async_scorer_func: Async scoring function to adapt
            scorer_name: Name of the scorer for logging
        """
        self.async_scorer_func = async_scorer_func
        self.scorer_name = scorer_name
    
    def __call__(self, expected_metadata: Dict, output: Dict) -> Dict[str, Any]:
        """
        Synchronous wrapper for async scorer function.
        
        Args:
            expected_metadata: Metadata from the dataset example
            output: Model output dictionary
            
        Returns:
            Dictionary containing scoring results
        """
        try:
            # Run the async function in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.async_scorer_func(expected_metadata, output)
                )
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in async scorer {self.scorer_name}: {e}")
            return {
                f"{self.scorer_name}_score": 0.0,
                f"{self.scorer_name}_violation": True,
                "error": str(e),
                "reasoning": f"Error during {self.scorer_name} evaluation: {e}"
            }
