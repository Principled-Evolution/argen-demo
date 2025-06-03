"""
WANDB Weave integration for ArGen evaluations.

This module provides the core integration between ArGen's evaluation system
and WANDB Weave, enabling enhanced tracking, visualization, and comparison
of model evaluations while maintaining backward compatibility.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from argen.config_weave import (
    ensure_weave_available,
    get_weave_project_name,
    get_evaluation_name,
    get_display_name,
    get_weave_tags,
    WEAVE_AVAILABLE
)

# Conditional import of weave
if WEAVE_AVAILABLE:
    import weave
    from weave import Model, Evaluation
else:
    weave = None
    Model = object
    Evaluation = object

logger = logging.getLogger(__name__)

class WeaveIntegrationError(Exception):
    """Exception raised for Weave integration errors."""
    pass

def init_weave(project_name: Optional[str] = None) -> bool:
    """
    Initialize Weave for the current session.
    
    Args:
        project_name: Optional project name override
        
    Returns:
        True if initialization successful, False otherwise
    """
    if not WEAVE_AVAILABLE:
        logger.warning("Weave is not available. Skipping Weave initialization.")
        return False
    
    try:
        project = get_weave_project_name(project_name)
        weave.init(project)
        logger.info(f"Weave initialized with project: {project}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Weave: {e}")
        return False

class ArGenWeaveModel(Model):
    """
    Weave-compatible wrapper for ArGen models.
    
    This class adapts ArGen's model evaluation interface to work with
    Weave's Model abstraction, enabling automatic tracking and versioning.
    """
    
    model_name: str
    temperature: float
    system_prompt_type: str
    generation_batch_size: int
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.9,
        system_prompt_type: str = "ENHANCED",
        generation_batch_size: int = 12,
        **kwargs
    ):
        if not WEAVE_AVAILABLE:
            raise WeaveIntegrationError("Weave is not available")
        
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            system_prompt_type=system_prompt_type,
            generation_batch_size=generation_batch_size,
            **kwargs
        )
    
    @weave.op() if WEAVE_AVAILABLE else lambda x: x
    def predict(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response for the given prompt.
        
        This method delegates to ArGen's existing generation logic
        while providing Weave-compatible tracking.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        # Import here to avoid circular imports
        from argen.evaluation.openai_evaluator import generate_responses_locally
        
        try:
            # Use ArGen's existing generation logic
            # This is a simplified interface - in practice, we'd need to
            # adapt the full generation pipeline
            responses = generate_responses_locally(
                model_name=self.model_name,
                prompts=[prompt],
                temperature=self.temperature,
                system_prompt_type=self.system_prompt_type,
                generation_batch_size=1,
                test_mode=kwargs.get('test_mode', False)
            )
            
            if responses and len(responses) > 0:
                return {
                    "generated_text": responses[0],
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "system_prompt_type": self.system_prompt_type,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "generated_text": "",
                    "error": "No response generated",
                    "model_name": self.model_name,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return {
                "generated_text": "",
                "error": str(e),
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat()
            }

def create_weave_dataset(scenarios: List[Dict]) -> List[Dict]:
    """
    Convert ArGen scenarios to Weave-compatible dataset format.
    
    Args:
        scenarios: List of ArGen scenario dictionaries
        
    Returns:
        List of Weave-compatible dataset examples
    """
    weave_examples = []
    
    for i, scenario in enumerate(scenarios):
        # Extract the core fields needed for evaluation
        example = {
            "prompt": scenario.get("prompt", ""),
            "scenario_id": scenario.get("scenario_id", f"scenario_{i}"),
            "expected_metadata": {
                "tier": scenario.get("tier", "C"),
                "scope": scenario.get("scope", "S0"),
                "category": scenario.get("category", ""),
                "subcategory": scenario.get("subcategory", ""),
                "original_scenario": scenario  # Preserve full scenario for reference
            }
        }
        weave_examples.append(example)
    
    return weave_examples

class WeaveEvaluationManager:
    """
    Manages Weave evaluations for ArGen models.
    
    This class orchestrates the evaluation process using Weave's evaluation
    framework while maintaining compatibility with ArGen's existing evaluation logic.
    """
    
    def __init__(
        self,
        project_name: Optional[str] = None,
        scenarios: Optional[List[Dict]] = None
    ):
        """
        Initialize the Weave evaluation manager.
        
        Args:
            project_name: Weave project name
            scenarios: List of evaluation scenarios
        """
        ensure_weave_available()
        
        self.project_name = get_weave_project_name(project_name)
        self.scenarios = scenarios or []
        self.dataset = None
        self.evaluation = None
        
        # Initialize Weave
        if not init_weave(self.project_name):
            raise WeaveIntegrationError("Failed to initialize Weave")
    
    def setup_dataset(self, scenarios: Optional[List[Dict]] = None) -> None:
        """
        Setup the Weave dataset from scenarios.
        
        Args:
            scenarios: Optional scenarios to use (overrides constructor scenarios)
        """
        if scenarios:
            self.scenarios = scenarios
        
        if not self.scenarios:
            raise ValueError("No scenarios provided for evaluation")
        
        self.dataset = create_weave_dataset(self.scenarios)
        logger.info(f"Created Weave dataset with {len(self.dataset)} examples")
    
    def get_evaluation_url(self) -> Optional[str]:
        """
        Get the Weave UI URL for the current evaluation.
        
        Returns:
            URL string if available, None otherwise
        """
        # This would need to be implemented based on Weave's URL structure
        # For now, return a placeholder
        if self.project_name:
            return f"https://wandb.ai/weave/{self.project_name}/evaluations"
        return None
    
    async def run_evaluation(
        self,
        model: ArGenWeaveModel,
        evaluation_name: Optional[str] = None,
        display_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a Weave evaluation for the given model.
        
        Args:
            model: ArGenWeaveModel instance to evaluate
            evaluation_name: Optional evaluation name
            display_name: Optional display name for the run
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        if not self.dataset:
            self.setup_dataset()
        
        # Import scorers here to avoid circular imports
        from argen.evaluation.weave_scorers import (
            get_weave_scorers,
            create_combined_scorer
        )
        
        # Get ArGen-specific scorers
        scorers = get_weave_scorers()
        
        # Create evaluation name
        eval_name = evaluation_name or get_evaluation_name(model.model_name)
        
        # Create display name
        disp_name = display_name or get_display_name(
            model.model_name,
            len(self.dataset)
        )
        
        # Create Weave evaluation
        evaluation = Evaluation(
            dataset=self.dataset,
            scorers=scorers,
            evaluation_name=eval_name
        )
        
        # Run the evaluation
        try:
            logger.info(f"Starting Weave evaluation: {eval_name}")
            results = await evaluation.evaluate(
                model,
                __weave={"display_name": disp_name}
            )
            
            logger.info(f"Weave evaluation completed: {eval_name}")
            return {
                "evaluation_name": eval_name,
                "display_name": disp_name,
                "results": results,
                "url": self.get_evaluation_url(),
                "dataset_size": len(self.dataset)
            }
            
        except Exception as e:
            logger.error(f"Weave evaluation failed: {e}")
            raise WeaveIntegrationError(f"Evaluation failed: {e}")

def harmonize_results(
    weave_results: Dict[str, Any],
    traditional_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Combine Weave and traditional evaluation results.
    
    Args:
        weave_results: Results from Weave evaluation
        traditional_results: Results from traditional ArGen evaluation
        
    Returns:
        Harmonized results dictionary
    """
    harmonized = {
        "evaluation_type": "weave_integrated",
        "timestamp": datetime.now().isoformat(),
        "weave_results": weave_results
    }
    
    if traditional_results:
        harmonized["traditional_results"] = traditional_results
        harmonized["evaluation_type"] = "hybrid"
    
    return harmonized
