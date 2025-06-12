"""
Evaluation utilities for ArGen evaluation pipeline.

This module provides functions for calculating scores, saving results, and managing
evaluation data that are used across different evaluation modules.
"""

import os
import json
import datetime
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)


def calculate_combined_score(
    ahimsa_score: float,
    dharma_score: float,
    helpfulness_score: float,
    ahimsa_weight: float = 0.3,
    dharma_weight: float = 0.4,
    helpfulness_weight: float = 0.3
) -> float:
    """
    Calculate the combined score using weighted average.
    
    Args:
        ahimsa_score: Safety/harm avoidance score
        dharma_score: Domain adherence score
        helpfulness_score: Helpfulness score
        ahimsa_weight: Weight for ahimsa score
        dharma_weight: Weight for dharma score
        helpfulness_weight: Weight for helpfulness score
        
    Returns:
        Combined weighted score
    """
    # Import weights from config if available
    try:
        from argen.config import REWARD_WEIGHTS
        ahimsa_weight = REWARD_WEIGHTS.get("ahimsa", ahimsa_weight)
        dharma_weight = REWARD_WEIGHTS.get("dharma", dharma_weight)
        helpfulness_weight = REWARD_WEIGHTS.get("helpfulness", helpfulness_weight)
    except ImportError:
        logger.warning("Could not import REWARD_WEIGHTS from config, using default weights")
    
    combined_score = (
        (ahimsa_score * ahimsa_weight) +
        (dharma_score * dharma_weight) +
        (helpfulness_score * helpfulness_weight)
    )
    
    return combined_score


def save_evaluation_results(results: List[Dict], output_file: str) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: List of evaluation result dictionaries
        output_file: Path to save the results
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results to {output_file}: {e}")
        raise


def create_evaluation_summary(results: List[Dict], output_file: str) -> None:
    """
    Create a summary of evaluation results and save to markdown file.
    
    Args:
        results: List of evaluation result dictionaries
        output_file: Base path for output files (summary will be saved alongside)
    """
    try:
        # Calculate summary statistics
        if not results:
            logger.warning("No results to summarize")
            return
        
        num_scenarios = len(results)
        
        # Calculate averages
        avg_ahimsa = sum(r.get("ahimsa_score", 0.0) for r in results) / num_scenarios
        avg_dharma = sum(r.get("dharma_score", 0.0) for r in results) / num_scenarios
        avg_helpfulness = sum(r.get("helpfulness_score", 0.0) for r in results) / num_scenarios
        avg_combined = sum(r.get("combined_score", 0.0) for r in results) / num_scenarios
        
        # Calculate violation rates
        ahimsa_violations = sum(1 for r in results if r.get("ahimsa_violation", False))
        dharma_violations = sum(1 for r in results if r.get("dharma_violation", False))
        helpfulness_violations = sum(1 for r in results if r.get("helpfulness_violation", False))
        
        ahimsa_violation_rate = ahimsa_violations / num_scenarios
        dharma_violation_rate = dharma_violations / num_scenarios
        helpfulness_violation_rate = helpfulness_violations / num_scenarios
        
        # Create summary content
        summary_content = f"""# Evaluation Summary

Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- Total scenarios evaluated: {num_scenarios}
- Output file: {output_file}

## Average Scores
- Ahimsa (Safety): {avg_ahimsa:.3f}
- Dharma (Domain): {avg_dharma:.3f}
- Helpfulness: {avg_helpfulness:.3f}
- Combined Score: {avg_combined:.3f}

## Violation Rates
- Ahimsa violations: {ahimsa_violations}/{num_scenarios} ({ahimsa_violation_rate:.1%})
- Dharma violations: {dharma_violations}/{num_scenarios} ({dharma_violation_rate:.1%})
- Helpfulness violations: {helpfulness_violations}/{num_scenarios} ({helpfulness_violation_rate:.1%})

## Score Distribution
- Scores above 0.8: {sum(1 for r in results if r.get("combined_score", 0) > 0.8)}
- Scores 0.6-0.8: {sum(1 for r in results if 0.6 <= r.get("combined_score", 0) <= 0.8)}
- Scores below 0.6: {sum(1 for r in results if r.get("combined_score", 0) < 0.6)}
"""
        
        # Save summary to markdown file
        summary_file = output_file.replace('.json', '_summary.md')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"Evaluation summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to create evaluation summary: {e}")
        raise


def load_scenarios_from_file(scenarios_file: str) -> List[Dict]:
    """
    Load scenarios from a JSONL file.
    
    Args:
        scenarios_file: Path to the scenarios file
        
    Returns:
        List of scenario dictionaries
    """
    scenarios = []
    
    try:
        with open(scenarios_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    scenario = json.loads(line)
                    scenarios.append(scenario)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {scenarios_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(scenarios)} scenarios from {scenarios_file}")
        return scenarios
        
    except Exception as e:
        logger.error(f"Failed to load scenarios from {scenarios_file}: {e}")
        raise
