"""
Script to compare baseline and fine-tuned model results.

This script:
1. Loads baseline and fine-tuned model evaluation results
2. Compares metrics between the two models
3. Generates a detailed comparison report
4. Creates visualizations of the comparison
"""

import sys
import os
import json
import argparse
import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing evaluation results
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_metrics(baseline_results: Dict[str, Any], finetuned_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare metrics between baseline and fine-tuned models.
    
    Args:
        baseline_results: Baseline model evaluation results
        finetuned_results: Fine-tuned model evaluation results
        
    Returns:
        Dictionary containing comparison metrics
    """
    # Extract metrics
    metrics = {
        "model_name": {
            "baseline": baseline_results.get("model_name", "Unknown"),
            "finetuned": finetuned_results.get("model_name", "Unknown")
        },
        "temperature": {
            "baseline": baseline_results.get("temperature", 0.0),
            "finetuned": finetuned_results.get("temperature", 0.0)
        },
        "num_scenarios": {
            "baseline": baseline_results.get("num_scenarios", 0),
            "finetuned": finetuned_results.get("num_scenarios", 0)
        },
        "ahimsa_violations": {
            "baseline": baseline_results.get("ahimsa_violations", 0),
            "finetuned": finetuned_results.get("ahimsa_violations", 0)
        },
        "ahimsa_violation_rate": {
            "baseline": baseline_results.get("ahimsa_violation_rate", 0.0),
            "finetuned": finetuned_results.get("ahimsa_violation_rate", 0.0)
        },
        "average_ahimsa_score": {
            "baseline": baseline_results.get("average_ahimsa_score", 0.0),
            "finetuned": finetuned_results.get("average_ahimsa_score", 0.0)
        },
        "dharma_violations": {
            "baseline": baseline_results.get("dharma_violations", 0),
            "finetuned": finetuned_results.get("dharma_violations", 0)
        },
        "dharma_violation_rate": {
            "baseline": baseline_results.get("dharma_violation_rate", 0.0),
            "finetuned": finetuned_results.get("dharma_violation_rate", 0.0)
        },
        "average_dharma_score": {
            "baseline": baseline_results.get("average_dharma_score", 0.0),
            "finetuned": finetuned_results.get("average_dharma_score", 0.0)
        },
        "average_combined_score": {
            "baseline": baseline_results.get("average_combined_score", 0.0),
            "finetuned": finetuned_results.get("average_combined_score", 0.0)
        }
    }
    
    # Calculate improvements
    improvements = {}
    for metric, values in metrics.items():
        if metric in ["model_name", "temperature", "num_scenarios"]:
            continue
        
        baseline_value = values["baseline"]
        finetuned_value = values["finetuned"]
        
        if isinstance(baseline_value, (int, float)) and isinstance(finetuned_value, (int, float)):
            if "violation" in metric:
                # For violation metrics, lower is better
                improvement = baseline_value - finetuned_value
                improvement_percent = (improvement / baseline_value) * 100 if baseline_value != 0 else float('inf')
            else:
                # For score metrics, higher is better
                improvement = finetuned_value - baseline_value
                improvement_percent = (improvement / baseline_value) * 100 if baseline_value != 0 else float('inf')
            
            improvements[metric] = {
                "absolute": improvement,
                "percent": improvement_percent
            }
    
    # Combine metrics and improvements
    comparison = {
        "metrics": metrics,
        "improvements": improvements
    }
    
    return comparison


def generate_comparison_report(comparison: Dict[str, Any], output_file: str) -> None:
    """
    Generate a detailed comparison report.
    
    Args:
        comparison: Comparison metrics
        output_file: Path to save the report
    """
    metrics = comparison["metrics"]
    improvements = comparison["improvements"]
    
    # Create report
    report = []
    report.append("# Baseline vs. Fine-tuned Model Comparison Report")
    report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## Model Information")
    report.append(f"- Baseline Model: {metrics['model_name']['baseline']}")
    report.append(f"- Fine-tuned Model: {metrics['model_name']['finetuned']}")
    report.append(f"- Temperature: {metrics['temperature']['baseline']}")
    report.append(f"- Number of Scenarios: {metrics['num_scenarios']['baseline']}\n")
    
    report.append("## Performance Metrics")
    
    # Ahimsa metrics
    report.append("### Ahimsa (Non-harm) Metrics")
    report.append("| Metric | Baseline | Fine-tuned | Improvement | % Change |")
    report.append("| ------ | -------- | ---------- | ----------- | -------- |")
    
    # Ahimsa violations
    baseline_violations = metrics["ahimsa_violations"]["baseline"]
    finetuned_violations = metrics["ahimsa_violations"]["finetuned"]
    violation_improvement = improvements["ahimsa_violations"]["absolute"]
    violation_percent = improvements["ahimsa_violations"]["percent"]
    report.append(f"| Violations | {baseline_violations} | {finetuned_violations} | {violation_improvement:.2f} | {violation_percent:.2f}% |")
    
    # Ahimsa violation rate
    baseline_rate = metrics["ahimsa_violation_rate"]["baseline"]
    finetuned_rate = metrics["ahimsa_violation_rate"]["finetuned"]
    rate_improvement = improvements["ahimsa_violation_rate"]["absolute"]
    rate_percent = improvements["ahimsa_violation_rate"]["percent"]
    report.append(f"| Violation Rate | {baseline_rate:.2%} | {finetuned_rate:.2%} | {rate_improvement:.2%} | {rate_percent:.2f}% |")
    
    # Ahimsa score
    baseline_score = metrics["average_ahimsa_score"]["baseline"]
    finetuned_score = metrics["average_ahimsa_score"]["finetuned"]
    score_improvement = improvements["average_ahimsa_score"]["absolute"]
    score_percent = improvements["average_ahimsa_score"]["percent"]
    report.append(f"| Average Score | {baseline_score:.4f} | {finetuned_score:.4f} | {score_improvement:.4f} | {score_percent:.2f}% |")
    
    report.append("")
    
    # Dharma metrics
    report.append("### Dharma (Domain Adherence) Metrics")
    report.append("| Metric | Baseline | Fine-tuned | Improvement | % Change |")
    report.append("| ------ | -------- | ---------- | ----------- | -------- |")
    
    # Dharma violations
    baseline_violations = metrics["dharma_violations"]["baseline"]
    finetuned_violations = metrics["dharma_violations"]["finetuned"]
    violation_improvement = improvements["dharma_violations"]["absolute"]
    violation_percent = improvements["dharma_violations"]["percent"]
    report.append(f"| Violations | {baseline_violations} | {finetuned_violations} | {violation_improvement:.2f} | {violation_percent:.2f}% |")
    
    # Dharma violation rate
    baseline_rate = metrics["dharma_violation_rate"]["baseline"]
    finetuned_rate = metrics["dharma_violation_rate"]["finetuned"]
    rate_improvement = improvements["dharma_violation_rate"]["absolute"]
    rate_percent = improvements["dharma_violation_rate"]["percent"]
    report.append(f"| Violation Rate | {baseline_rate:.2%} | {finetuned_rate:.2%} | {rate_improvement:.2%} | {rate_percent:.2f}% |")
    
    # Dharma score
    baseline_score = metrics["average_dharma_score"]["baseline"]
    finetuned_score = metrics["average_dharma_score"]["finetuned"]
    score_improvement = improvements["average_dharma_score"]["absolute"]
    score_percent = improvements["average_dharma_score"]["percent"]
    report.append(f"| Average Score | {baseline_score:.4f} | {finetuned_score:.4f} | {score_improvement:.4f} | {score_percent:.2f}% |")
    
    report.append("")
    
    # Combined score
    report.append("### Combined Score")
    report.append("| Metric | Baseline | Fine-tuned | Improvement | % Change |")
    report.append("| ------ | -------- | ---------- | ----------- | -------- |")
    
    baseline_score = metrics["average_combined_score"]["baseline"]
    finetuned_score = metrics["average_combined_score"]["finetuned"]
    score_improvement = improvements["average_combined_score"]["absolute"]
    score_percent = improvements["average_combined_score"]["percent"]
    report.append(f"| Average Score | {baseline_score:.4f} | {finetuned_score:.4f} | {score_improvement:.4f} | {score_percent:.2f}% |")
    
    report.append("")
    
    # Summary
    report.append("## Summary")
    
    # Determine overall improvement
    ahimsa_score_improvement = improvements["average_ahimsa_score"]["absolute"]
    dharma_score_improvement = improvements["average_dharma_score"]["absolute"]
    combined_score_improvement = improvements["average_combined_score"]["absolute"]
    
    if combined_score_improvement > 0:
        report.append("The fine-tuned model shows overall improvement compared to the baseline model.")
        
        if ahimsa_score_improvement > 0:
            report.append(f"- Ahimsa (non-harm) score improved by {ahimsa_score_improvement:.4f} ({improvements['average_ahimsa_score']['percent']:.2f}%).")
        else:
            report.append(f"- Ahimsa (non-harm) score decreased by {abs(ahimsa_score_improvement):.4f} ({abs(improvements['average_ahimsa_score']['percent']):.2f}%).")
        
        if dharma_score_improvement > 0:
            report.append(f"- Dharma (domain adherence) score improved by {dharma_score_improvement:.4f} ({improvements['average_dharma_score']['percent']:.2f}%).")
        else:
            report.append(f"- Dharma (domain adherence) score decreased by {abs(dharma_score_improvement):.4f} ({abs(improvements['average_dharma_score']['percent']):.2f}%).")
    else:
        report.append("The fine-tuned model does not show overall improvement compared to the baseline model.")
        report.append(f"- Combined score decreased by {abs(combined_score_improvement):.4f} ({abs(improvements['average_combined_score']['percent']):.2f}%).")
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Comparison report saved to {output_file}")


def create_visualizations(comparison: Dict[str, Any], output_dir: str) -> None:
    """
    Create visualizations of the comparison.
    
    Args:
        comparison: Comparison metrics
        output_dir: Directory to save the visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = comparison["metrics"]
    
    # Create bar chart of violation rates
    plt.figure(figsize=(10, 6))
    
    labels = ['Ahimsa Violation Rate', 'Dharma Violation Rate']
    baseline_values = [metrics["ahimsa_violation_rate"]["baseline"], metrics["dharma_violation_rate"]["baseline"]]
    finetuned_values = [metrics["ahimsa_violation_rate"]["finetuned"], metrics["dharma_violation_rate"]["finetuned"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline')
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned')
    
    plt.xlabel('Metric')
    plt.ylabel('Violation Rate')
    plt.title('Violation Rates: Baseline vs. Fine-tuned')
    plt.xticks(x, labels)
    plt.ylim(0, max(max(baseline_values), max(finetuned_values)) * 1.2)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    violation_rates_file = os.path.join(output_dir, "violation_rates.png")
    plt.savefig(violation_rates_file)
    plt.close()
    
    # Create bar chart of average scores
    plt.figure(figsize=(10, 6))
    
    labels = ['Ahimsa Score', 'Dharma Score', 'Combined Score']
    baseline_values = [
        metrics["average_ahimsa_score"]["baseline"],
        metrics["average_dharma_score"]["baseline"],
        metrics["average_combined_score"]["baseline"]
    ]
    finetuned_values = [
        metrics["average_ahimsa_score"]["finetuned"],
        metrics["average_dharma_score"]["finetuned"],
        metrics["average_combined_score"]["finetuned"]
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline')
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned')
    
    plt.xlabel('Metric')
    plt.ylabel('Average Score')
    plt.title('Average Scores: Baseline vs. Fine-tuned')
    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    average_scores_file = os.path.join(output_dir, "average_scores.png")
    plt.savefig(average_scores_file)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    """Run the comparison script."""
    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned model results.")
    parser.add_argument("--baseline", type=str, default="data/baseline_gemini_results.json", help="Path to baseline results")
    parser.add_argument("--finetuned", type=str, default="data/finetuned_gemini_results.json", help="Path to fine-tuned results")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save the report and visualizations")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline results file not found: {args.baseline}")
        sys.exit(1)
    
    if not os.path.exists(args.finetuned):
        print(f"Error: Fine-tuned results file not found: {args.finetuned}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading baseline results from {args.baseline}...")
    baseline_results = load_results(args.baseline)
    
    print(f"Loading fine-tuned results from {args.finetuned}...")
    finetuned_results = load_results(args.finetuned)
    
    # Compare metrics
    print("Comparing metrics...")
    comparison = compare_metrics(baseline_results, finetuned_results)
    
    # Generate comparison report
    report_file = os.path.join(args.output_dir, "comparison_report.md")
    print(f"Generating comparison report...")
    generate_comparison_report(comparison, report_file)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(comparison, args.output_dir)
    
    # Save comparison data
    comparison_file = os.path.join(args.output_dir, "comparison_data.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison data saved to {comparison_file}")
    print("Comparison completed successfully!")


if __name__ == "__main__":
    main()
