#!/usr/bin/env python3
"""
Script to compare the baseline model with the GRPO-trained model.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("compare_models.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('compare_models')

def compare_models(baseline_results_path, grpo_results_path, output_dir):
    """
    Compare the baseline model with the GRPO-trained model.
    
    Args:
        baseline_results_path: Path to the baseline results file
        grpo_results_path: Path to the GRPO results file
        output_dir: Directory to save the comparison results
    """
    logger.info(f"Comparing baseline model with GRPO-trained model...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load baseline results
    try:
        logger.info(f"Loading baseline results from {baseline_results_path}...")
        if baseline_results_path.endswith('.json') or baseline_results_path.endswith('.jsonl'):
            baseline_results = pd.read_json(baseline_results_path, lines=True)
        elif baseline_results_path.endswith('.csv'):
            baseline_results = pd.read_csv(baseline_results_path)
        else:
            logger.error(f"Unsupported file format: {baseline_results_path}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(baseline_results)} baseline results")
    except Exception as e:
        logger.error(f"Error loading baseline results: {e}")
        sys.exit(1)
    
    # Load GRPO results
    try:
        logger.info(f"Loading GRPO results from {grpo_results_path}...")
        if grpo_results_path.endswith('.json') or grpo_results_path.endswith('.jsonl'):
            grpo_results = pd.read_json(grpo_results_path, lines=True)
        elif grpo_results_path.endswith('.csv'):
            grpo_results = pd.read_csv(grpo_results_path)
        else:
            logger.error(f"Unsupported file format: {grpo_results_path}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(grpo_results)} GRPO results")
    except Exception as e:
        logger.error(f"Error loading GRPO results: {e}")
        sys.exit(1)
    
    # Check if the datasets have the same prompts
    if len(baseline_results) != len(grpo_results):
        logger.warning(f"Baseline and GRPO results have different lengths: {len(baseline_results)} vs {len(grpo_results)}")
    
    # Merge the datasets
    try:
        logger.info("Merging datasets...")
        
        # Rename columns to avoid conflicts
        baseline_results = baseline_results.rename(columns={
            "completion": "baseline_completion",
            "ahimsa_score": "baseline_ahimsa_score"
        })
        
        grpo_results = grpo_results.rename(columns={
            "completion": "grpo_completion",
            "ahimsa_score": "grpo_ahimsa_score"
        })
        
        # Merge on prompt
        merged_results = pd.merge(baseline_results, grpo_results, on="prompt", how="inner")
        logger.info(f"Merged dataset has {len(merged_results)} examples")
        
        # Save merged results
        merged_results_path = os.path.join(output_dir, "merged_results.jsonl")
        merged_results.to_json(merged_results_path, orient="records", lines=True)
        logger.info(f"Merged results saved to {merged_results_path}")
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        sys.exit(1)
    
    # Calculate statistics
    try:
        logger.info("Calculating statistics...")
        
        # Check if ahimsa scores are available
        has_ahimsa_scores = "baseline_ahimsa_score" in merged_results.columns and "grpo_ahimsa_score" in merged_results.columns
        
        if has_ahimsa_scores:
            # Calculate average scores
            baseline_avg_score = merged_results["baseline_ahimsa_score"].mean()
            grpo_avg_score = merged_results["grpo_ahimsa_score"].mean()
            
            logger.info(f"Baseline average Ahimsa score: {baseline_avg_score:.4f}")
            logger.info(f"GRPO average Ahimsa score: {grpo_avg_score:.4f}")
            
            # Calculate improvement
            improvement = grpo_avg_score - baseline_avg_score
            improvement_percent = (improvement / baseline_avg_score) * 100 if baseline_avg_score > 0 else float('inf')
            
            logger.info(f"Absolute improvement: {improvement:.4f}")
            logger.info(f"Relative improvement: {improvement_percent:.2f}%")
            
            # Create a bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(["Baseline", "GRPO"], [baseline_avg_score, grpo_avg_score])
            plt.title("Average Ahimsa Score Comparison")
            plt.ylabel("Average Ahimsa Score")
            plt.ylim(0, 1)
            
            # Add values on top of bars
            plt.text(0, baseline_avg_score + 0.01, f"{baseline_avg_score:.4f}", ha="center")
            plt.text(1, grpo_avg_score + 0.01, f"{grpo_avg_score:.4f}", ha="center")
            
            # Save the chart
            chart_path = os.path.join(output_dir, "ahimsa_score_comparison.png")
            plt.savefig(chart_path)
            logger.info(f"Chart saved to {chart_path}")
            
            # Create a histogram of scores
            plt.figure(figsize=(12, 6))
            plt.hist(merged_results["baseline_ahimsa_score"], alpha=0.5, label="Baseline", bins=20)
            plt.hist(merged_results["grpo_ahimsa_score"], alpha=0.5, label="GRPO", bins=20)
            plt.title("Distribution of Ahimsa Scores")
            plt.xlabel("Ahimsa Score")
            plt.ylabel("Number of Examples")
            plt.legend()
            
            # Save the histogram
            histogram_path = os.path.join(output_dir, "ahimsa_score_distribution.png")
            plt.savefig(histogram_path)
            logger.info(f"Histogram saved to {histogram_path}")
        else:
            logger.warning("Ahimsa scores not available in the datasets")
        
        # Save statistics to a JSON file
        stats = {
            "num_examples": len(merged_results),
            "has_ahimsa_scores": has_ahimsa_scores
        }
        
        if has_ahimsa_scores:
            stats.update({
                "baseline_avg_score": baseline_avg_score,
                "grpo_avg_score": grpo_avg_score,
                "absolute_improvement": improvement,
                "relative_improvement_percent": improvement_percent
            })
        
        stats_path = os.path.join(output_dir, "comparison_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        sys.exit(1)
    
    logger.info("Comparison completed successfully")

def main():
    """Main function to compare the baseline model with the GRPO-trained model."""
    parser = argparse.ArgumentParser(description="Compare the baseline model with the GRPO-trained model.")
    parser.add_argument("--baseline", type=str, required=True, help="Path to the baseline results file")
    parser.add_argument("--grpo", type=str, required=True, help="Path to the GRPO results file")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Directory to save the comparison results")
    
    args = parser.parse_args()
    
    # Compare the models
    compare_models(
        baseline_results_path=args.baseline,
        grpo_results_path=args.grpo,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
