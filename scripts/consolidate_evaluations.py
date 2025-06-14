#!/usr/bin/env python3
"""
Evaluation Results Consolidation Script

This script parses, consolidates, and compares evaluation results from different
LLM evaluators (Gemini, Claude, etc.) to address potential evaluation circularity
in ArGen research paper benchmarks.

Usage:
    python consolidate_evaluations.py --files file1.json file2.json ... --output results.md
    python consolidate_evaluations.py --directory path/to/evals/ --output results.md
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Consolidate and compare evaluation results from multiple LLM evaluators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="List of JSON evaluation files to process"
    )
    input_group.add_argument(
        "--directory",
        type=str,
        help="Directory containing JSON evaluation files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output markdown file path"
    )
    
    parser.add_argument(
        "--json-output",
        type=str,
        help="Optional JSON output file for programmatic access"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*eval*.json",
        help="File pattern to match when using --directory"
    )
    
    return parser.parse_args()


def clean_model_name(model_path: str) -> str:
    """
    Clean and standardize model names for display.
    
    Args:
        model_path: Raw model name/path from evaluation config
        
    Returns:
        Clean, human-readable model name
    """
    # Handle checkpoint paths
    if "checkpoint" in model_path.lower():
        return f"ArGen-{os.path.basename(model_path)}"
    
    # Handle grpo paths
    if "grpo" in model_path.lower():
        basename = os.path.basename(model_path)
        return f"ArGen-{basename}"
    
    # Handle standard model names
    if "llama" in model_path.lower():
        if "3.2-1B" in model_path:
            return "Llama-3.2-1B"
        elif "3.2-3B" in model_path:
            return "Llama-3.2-3B"
        else:
            return "Llama-" + model_path.split("/")[-1]
    
    # Handle other models
    if "medalpaca" in model_path.lower():
        return "MedAlpaca-7B"
    
    # Default: use basename
    return os.path.basename(model_path).replace("/", "-")


def clean_evaluator_name(evaluator_str: str) -> Tuple[str, str]:
    """
    Extract clean evaluator name and type.
    
    Args:
        evaluator_str: Raw evaluator string from config
        
    Returns:
        Tuple of (clean_name, evaluator_type)
    """
    evaluator_lower = evaluator_str.lower()
    
    if "gemini" in evaluator_lower:
        if "2.0-flash" in evaluator_lower:
            return "Gemini 2.0 Flash", "gemini"
        else:
            return "Gemini", "gemini"
    
    elif "claude" in evaluator_lower:
        if "3-5-sonnet" in evaluator_lower or "3.5-sonnet" in evaluator_lower:
            return "Claude 3.5 Sonnet", "anthropic"
        elif "3-opus" in evaluator_lower:
            return "Claude 3 Opus", "anthropic"
        else:
            return "Claude", "anthropic"
    
    elif "openai" in evaluator_lower or "gpt" in evaluator_lower:
        if "gpt-4o" in evaluator_lower:
            return "GPT-4o", "openai"
        elif "o3" in evaluator_lower:
            return "OpenAI o3", "openai"
        else:
            return "OpenAI", "openai"
    
    else:
        return evaluator_str, "unknown"


def parse_evaluation_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single evaluation JSON file and extract relevant metrics.
    
    Args:
        file_path: Path to the JSON evaluation file
        
    Returns:
        Dictionary with parsed evaluation data or None if parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate required sections
        if 'evaluation_config' not in data or 'summary_metrics' not in data:
            print(f"Warning: {file_path} missing required sections")
            return None
        
        config = data['evaluation_config']
        metrics = data['summary_metrics']
        
        # Extract basic info
        raw_model_name = config.get('model_name', 'Unknown')
        raw_evaluator = config.get('evaluator', 'Unknown')
        
        # Clean names
        clean_model = clean_model_name(raw_model_name)
        clean_eval, eval_type = clean_evaluator_name(raw_evaluator)
        
        # Build result dictionary
        result = {
            'file_path': file_path,
            'model_name': clean_model,
            'raw_model_name': raw_model_name,
            'evaluator_name': clean_eval,
            'evaluator_type': eval_type,
            'raw_evaluator': raw_evaluator,
            'timestamp': config.get('timestamp', ''),
            'num_scenarios': config.get('num_scenarios', 0),
            
            # Core metrics
            'ahimsa_violation_rate': metrics.get('ahimsa_violation_rate', 0),
            'average_ahimsa_score': metrics.get('average_ahimsa_score', 0),
            'dharma_violation_rate': metrics.get('dharma_violation_rate', 0),
            'average_dharma_score': metrics.get('average_dharma_score', 0),
            'helpfulness_violation_rate': metrics.get('helpfulness_violation_rate', 0),
            'average_helpfulness_score': metrics.get('average_helpfulness_score', 0),
            'average_combined_score': metrics.get('average_combined_score', 0),
            
            # Additional metrics
            'average_clarity_score': metrics.get('average_clarity_score', 0),
            'average_relevance_score': metrics.get('average_relevance_score', 0),
            'average_completeness_score': metrics.get('average_completeness_score', 0),
            'average_scope_penalty_factor': metrics.get('average_scope_penalty_factor', 0),
            'severe_scope_penalty_rate': metrics.get('severe_scope_penalty_rate', 0),
            
            # Counts
            'ahimsa_violations': metrics.get('ahimsa_violations', 0),
            'dharma_violations': metrics.get('dharma_violations', 0),
            'helpfulness_violations': metrics.get('helpfulness_violations', 0),
            'severe_scope_penalties': metrics.get('severe_scope_penalties', 0),
        }
        
        return result
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def collect_evaluation_files(args) -> List[str]:
    """
    Collect evaluation files based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List of file paths to process
    """
    if args.files:
        # Validate that all files exist
        missing_files = [f for f in args.files if not os.path.exists(f)]
        if missing_files:
            print(f"Error: The following files do not exist: {missing_files}")
            sys.exit(1)
        return args.files
    
    elif args.directory:
        # Find files matching pattern in directory
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Error: Directory {args.directory} does not exist")
            sys.exit(1)
        
        files = list(directory.glob(args.pattern))
        if not files:
            print(f"Error: No files matching pattern '{args.pattern}' found in {args.directory}")
            sys.exit(1)
        
        return [str(f) for f in files]
    
    else:
        print("Error: Must specify either --files or --directory")
        sys.exit(1)


def consolidate_evaluations(file_paths: List[str]) -> pd.DataFrame:
    """
    Parse and consolidate evaluation results from multiple files.
    
    Args:
        file_paths: List of JSON evaluation file paths
        
    Returns:
        DataFrame with consolidated evaluation data
    """
    parsed_data = []
    
    print(f"Processing {len(file_paths)} evaluation files...")
    
    for file_path in file_paths:
        print(f"  Parsing: {os.path.basename(file_path)}")
        result = parse_evaluation_file(file_path)
        if result:
            parsed_data.append(result)
        else:
            print(f"  Warning: Skipped {file_path} due to parsing errors")
    
    if not parsed_data:
        print("Error: No valid evaluation files found")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(parsed_data)
    
    print(f"\nSuccessfully processed {len(df)} evaluation results:")
    print(f"  Models: {sorted(df['model_name'].unique())}")
    print(f"  Evaluators: {sorted(df['evaluator_name'].unique())}")
    
    return df


def calculate_improvement(baseline_val: float, argen_val: float,
                         lower_is_better: bool = False) -> str:
    """
    Calculate percentage improvement between baseline and ArGen models.

    Args:
        baseline_val: Baseline model value
        argen_val: ArGen model value
        lower_is_better: If True, lower values are better (e.g., violation rates)

    Returns:
        Formatted improvement string with arrow and percentage
    """
    if baseline_val == 0:
        return "N/A"

    pct_change = ((argen_val - baseline_val) / baseline_val) * 100

    if lower_is_better:
        # For metrics where lower is better (violation rates)
        if pct_change < 0:
            return f"↓ {abs(pct_change):.1f}%"
        else:
            return f"↑ {pct_change:.1f}%"
    else:
        # For metrics where higher is better (scores)
        if pct_change > 0:
            return f"↑ {pct_change:.1f}%"
        else:
            return f"↓ {abs(pct_change):.1f}%"


def format_percentage(value: float) -> str:
    """Format a decimal value as a percentage."""
    return f"{value * 100:.1f}%"


def format_score(value: float) -> str:
    """Format a score value to 3 decimal places."""
    return f"{value:.3f}"


def generate_markdown_report(df: pd.DataFrame, output_path: str):
    """
    Generate a comprehensive markdown report comparing evaluation results.

    Args:
        df: DataFrame with consolidated evaluation data
        output_path: Path to save the markdown report
    """
    # Identify baseline and ArGen models
    baseline_models = df[df['model_name'].str.contains('Llama', case=False, na=False)]
    argen_models = df[df['model_name'].str.contains('ArGen', case=False, na=False)]

    # Get unique evaluators
    evaluators = sorted(df['evaluator_name'].unique())

    # Start building markdown content
    lines = []
    lines.append("# Model Evaluation Results Comparison")
    lines.append("")
    lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Add metadata
    lines.append("## Evaluation Overview")
    lines.append("")
    lines.append(f"- **Total Evaluations**: {len(df)}")
    lines.append(f"- **Models Evaluated**: {', '.join(sorted(df['model_name'].unique()))}")
    lines.append(f"- **Evaluators Used**: {', '.join(evaluators)}")
    lines.append(f"- **Scenarios per Evaluation**: {df['num_scenarios'].iloc[0] if len(df) > 0 else 'N/A'}")
    lines.append("")

    # Define key metrics for comparison
    key_metrics = [
        ('dharma_violation_rate', 'Dharma Violation Rate', True, format_percentage),
        ('ahimsa_violation_rate', 'Ahimsa Violation Rate', True, format_percentage),
        ('helpfulness_violation_rate', 'Helpfulness Violation Rate', True, format_percentage),
        ('average_combined_score', 'Average Combined Score', False, format_score),
        ('average_dharma_score', 'Average Dharma Score', False, format_score),
        ('average_ahimsa_score', 'Average Ahimsa Score', False, format_score),
        ('average_helpfulness_score', 'Average Helpfulness Score', False, format_score),
        ('severe_scope_penalty_rate', 'Severe Scope Penalty Rate', True, format_percentage),
    ]

    # Create comparison table
    lines.append("## Key Metrics Comparison")
    lines.append("")

    # Table header
    header_cols = ["Metric", "Evaluator"]

    # Add model columns dynamically
    baseline_name = baseline_models['model_name'].iloc[0] if len(baseline_models) > 0 else "Baseline"
    argen_name = argen_models['model_name'].iloc[0] if len(argen_models) > 0 else "ArGen"

    header_cols.extend([f"Baseline ({baseline_name})", f"ArGen ({argen_name})", "Improvement"])

    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    # Add rows for each metric and evaluator combination
    for metric_key, metric_name, lower_is_better, formatter in key_metrics:
        for i, evaluator in enumerate(evaluators):
            # Get baseline value
            baseline_row = df[(df['evaluator_name'] == evaluator) &
                            (df['model_name'].str.contains('Llama', case=False, na=False))]
            baseline_val = baseline_row[metric_key].iloc[0] if len(baseline_row) > 0 else 0

            # Get ArGen value
            argen_row = df[(df['evaluator_name'] == evaluator) &
                          (df['model_name'].str.contains('ArGen', case=False, na=False))]
            argen_val = argen_row[metric_key].iloc[0] if len(argen_row) > 0 else 0

            # Calculate improvement
            improvement = calculate_improvement(baseline_val, argen_val, lower_is_better)

            # Format values
            baseline_formatted = formatter(baseline_val)
            argen_formatted = formatter(argen_val) if len(argen_row) > 0 else "N/A"

            # Create row
            if i == 0:  # First evaluator for this metric
                metric_cell = metric_name
            else:
                metric_cell = ""  # Empty for subsequent evaluators

            row = [metric_cell, evaluator, baseline_formatted, argen_formatted, improvement]
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Add detailed metrics table
    lines.append("## Detailed Metrics by Model and Evaluator")
    lines.append("")

    # Create detailed table
    detail_cols = ["Model", "Evaluator", "Combined Score", "Dharma Score", "Ahimsa Score",
                   "Helpfulness Score", "Dharma Violations", "Ahimsa Violations"]

    lines.append("| " + " | ".join(detail_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(detail_cols)) + " |")

    # Sort by model name and evaluator for consistent ordering
    df_sorted = df.sort_values(['model_name', 'evaluator_name'])

    for _, row in df_sorted.iterrows():
        detail_row = [
            row['model_name'],
            row['evaluator_name'],
            format_score(row['average_combined_score']),
            format_score(row['average_dharma_score']),
            format_score(row['average_ahimsa_score']),
            format_score(row['average_helpfulness_score']),
            format_percentage(row['dharma_violation_rate']),
            format_percentage(row['ahimsa_violation_rate'])
        ]
        lines.append("| " + " | ".join(detail_row) + " |")

    lines.append("")

    # Add notes section
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Violation Rates**: Lower percentages indicate better performance")
    lines.append("- **Scores**: Higher values indicate better performance (scale: 0.0 to 1.0)")
    lines.append("- **Improvement**: ↑ indicates improvement, ↓ indicates decline")
    lines.append("- **Combined Score**: Weighted average of Dharma, Ahimsa, and Helpfulness scores")
    lines.append("")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nMarkdown report saved to: {output_path}")


def export_to_json(df: pd.DataFrame, output_path: str):
    """
    Export consolidated data to JSON format for programmatic access.

    Args:
        df: DataFrame with consolidated evaluation data
        output_path: Path to save the JSON file
    """
    # Convert DataFrame to records and handle pandas data types
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            # Convert pandas/numpy types to native Python types
            if hasattr(val, 'item'):  # numpy scalar
                record[col] = val.item()
            elif pd.isna(val):  # Handle NaN values
                record[col] = None
            else:
                record[col] = val
        records.append(record)

    # Create structured output
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_evaluations': int(len(df)),
            'models': sorted([str(x) for x in df['model_name'].unique()]),
            'evaluators': sorted([str(x) for x in df['evaluator_name'].unique()]),
            'num_scenarios': int(df['num_scenarios'].iloc[0]) if len(df) > 0 else 0
        },
        'evaluations': records
    }

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"JSON data exported to: {output_path}")


def main():
    """Main function to orchestrate the consolidation process."""
    args = parse_arguments()

    print("ArGen Evaluation Results Consolidation")
    print("=" * 50)

    # Collect input files
    file_paths = collect_evaluation_files(args)

    # Parse and consolidate data
    df = consolidate_evaluations(file_paths)

    # Generate markdown report
    generate_markdown_report(df, args.output)

    # Export JSON if requested
    if args.json_output:
        export_to_json(df, args.json_output)

    print("\n" + "=" * 50)
    print("Consolidation completed successfully!")
    print(f"Results available in: {args.output}")
    if args.json_output:
        print(f"JSON data available in: {args.json_output}")


if __name__ == "__main__":
    main()
