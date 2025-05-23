#!/usr/bin/env python3
"""
Log condensation script for GRPO training logs.
Removes repetitive content while preserving key metrics and trends.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

def parse_log_line(line: str) -> Dict[str, Any]:
    """Parse a log line and extract key information."""
    result = {
        'timestamp': None,
        'level': None,
        'module': None,
        'message': line.strip(),
        'type': 'other',
        'line_number': None
    }

    # Parse standard log format: YYYY-MM-DD HH:MM:SS,mmm - module - LEVEL - message
    log_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([^-]+) - (\w+) - (.+)', line)
    if log_match:
        result['timestamp'] = log_match.group(1)
        result['module'] = log_match.group(2).strip()
        result['level'] = log_match.group(3)
        result['message'] = log_match.group(4)

    # Classify line types
    message = result['message']

    # Training metrics and progress (CRITICAL - always keep)
    if re.search(r"'loss':|'learning_rate':|'grad_norm':|'reward':|'epoch':", message):
        result['type'] = 'training_metrics'

    # Progress bars with percentages (keep key milestones)
    elif re.search(r'\d+%\|.*\| \d+/\d+', message):
        result['type'] = 'progress_bar'
        # Extract percentage for trend tracking
        pct_match = re.search(r'(\d+)%', message)
        if pct_match:
            result['progress_pct'] = int(pct_match.group(1))

    # Combined reward summaries (important for trend)
    elif 'Combined reward:' in message:
        result['type'] = 'reward_summary'
        # Extract reward value
        reward_match = re.search(r'Combined reward: ([\d.]+)', message)
        if reward_match:
            result['reward_value'] = float(reward_match.group(1))

    # Gemini API evaluations (very repetitive - heavily sample)
    elif any(keyword in message for keyword in ['Gemini Ahimsa Penalty:', 'Gemini Dharma Scope Penalty:']):
        result['type'] = 'gemini_evaluation'

    # Configuration and setup (always keep)
    elif any(keyword in message for keyword in ['Config', 'Arguments', 'Initializing', 'Loading', 'Starting GRPO']):
        result['type'] = 'configuration'

    # Checkpoints and saves (always keep)
    elif any(keyword in message for keyword in ['checkpoint', 'save', 'Saving']):
        result['type'] = 'checkpoint'

    # Errors and warnings (always keep)
    elif result['level'] in ['ERROR', 'WARNING']:
        result['type'] = 'error_warning'

    # W&B and logging setup
    elif any(keyword in message for keyword in ['wandb:', 'W&B', 'Tracking run']):
        result['type'] = 'wandb'

    return result

def extract_training_metrics(line: str) -> Optional[Dict[str, float]]:
    """Extract numerical training metrics from a log line."""
    metrics = {}

    # Look for the main training metrics dictionary
    if "{'loss':" in line or "'loss':" in line:
        # Common patterns for metrics
        patterns = {
            'loss': r"'loss': ([\d.]+)",
            'learning_rate': r"'learning_rate': ([\d.e-]+)",
            'grad_norm': r"'grad_norm': ([\d.]+)",
            'reward': r"'reward': ([\d.]+)",
            'reward_std': r"'reward_std': ([\d.]+)",
            'kl': r"'kl': ([\d.e-]+)",
            'epoch': r"'epoch': ([\d.]+)",
            'num_tokens': r"'num_tokens': ([\d.]+)",
            'completions/mean_length': r"'completions/mean_length': ([\d.]+)",
            'rewards/combined_reward_trl/mean': r"'rewards/combined_reward_trl/mean': ([\d.]+)",
            'rewards/combined_reward_trl/std': r"'rewards/combined_reward_trl/std': ([\d.]+)"
        }

        for metric, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                try:
                    metrics[metric] = float(match.group(1))
                except ValueError:
                    pass

    return metrics if metrics else None

def should_keep_line(parsed_line: Dict[str, Any], line_idx: int, total_lines: int,
                    last_progress_pct: int, last_reward_sample: int) -> tuple[bool, int, int]:
    """Determine if a line should be kept in the condensed log."""
    line_type = parsed_line['type']

    # Always keep these critical types
    if line_type in ['configuration', 'checkpoint', 'error_warning', 'training_metrics']:
        return True, last_progress_pct, last_reward_sample

    # Keep W&B setup but not all messages
    if line_type == 'wandb':
        if any(keyword in parsed_line['message'] for keyword in ['Tracking run', 'View run at', 'View project at']):
            return True, last_progress_pct, last_reward_sample
        return False, last_progress_pct, last_reward_sample

    # Keep progress bars at key milestones
    if line_type == 'progress_bar':
        current_pct = parsed_line.get('progress_pct', 0)
        # Keep if it's a significant jump in progress (every 5%) or major milestones
        if (current_pct >= last_progress_pct + 5 or
            current_pct in [1, 5, 10, 25, 50, 75, 90, 95, 100] or
            current_pct == 0):
            return True, current_pct, last_reward_sample
        return False, last_progress_pct, last_reward_sample

    # Sample reward summaries to show trends (keep every 20th)
    if line_type == 'reward_summary':
        if line_idx >= last_reward_sample + 20:
            return True, last_progress_pct, line_idx
        return False, last_progress_pct, last_reward_sample

    # Heavily sample gemini evaluations (keep 1 in every 100)
    if line_type == 'gemini_evaluation':
        return line_idx % 100 == 0, last_progress_pct, last_reward_sample

    # Keep first and last portions of the log
    if line_idx < 200 or line_idx > total_lines - 200:
        return True, last_progress_pct, last_reward_sample

    return False, last_progress_pct, last_reward_sample

def create_metrics_trend_summary(metrics_history: List[Dict]) -> str:
    """Create a summary of metrics trends over time."""
    if not metrics_history:
        return "No metrics data found.\n"

    summary = []
    summary.append("TRAINING METRICS PROGRESSION:")
    summary.append("=" * 60)
    summary.append(f"{'Step':<8} {'Time':<12} {'Loss':<8} {'LR':<10} {'Reward':<8} {'Epoch':<6}")
    summary.append("-" * 60)

    # Sample key points from the metrics history
    sample_indices = []
    if len(metrics_history) <= 20:
        sample_indices = list(range(len(metrics_history)))
    else:
        # Always include first, last, and evenly spaced samples
        sample_indices = [0]
        step = len(metrics_history) // 18  # 18 middle points + first + last = 20 total
        for i in range(step, len(metrics_history) - 1, step):
            sample_indices.append(i)
        sample_indices.append(len(metrics_history) - 1)

    for idx in sample_indices:
        entry = metrics_history[idx]
        metrics = entry['metrics']

        step = entry['line']
        time_str = entry['timestamp'][:12] if entry['timestamp'] else "N/A"
        loss = f"{metrics.get('loss', 0):.4f}" if 'loss' in metrics else "N/A"
        lr = f"{metrics.get('learning_rate', 0):.2e}" if 'learning_rate' in metrics else "N/A"
        reward = f"{metrics.get('reward', 0):.3f}" if 'reward' in metrics else "N/A"
        epoch = f"{metrics.get('epoch', 0):.2f}" if 'epoch' in metrics else "N/A"

        summary.append(f"{step:<8} {time_str:<12} {loss:<8} {lr:<10} {reward:<8} {epoch:<6}")

    summary.append("-" * 60)
    summary.append("")

    # Add trend analysis
    if len(metrics_history) >= 2:
        first_metrics = metrics_history[0]['metrics']
        last_metrics = metrics_history[-1]['metrics']

        summary.append("TREND ANALYSIS:")
        summary.append("-" * 30)

        for metric in ['loss', 'reward', 'learning_rate']:
            if metric in first_metrics and metric in last_metrics:
                start_val = first_metrics[metric]
                end_val = last_metrics[metric]
                change = end_val - start_val
                change_pct = (change / start_val * 100) if start_val != 0 else 0

                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                summary.append(f"{metric.capitalize()}: {start_val:.4f} → {end_val:.4f} ({direction} {change_pct:+.1f}%)")

        summary.append("")

    return "\n".join(summary)

def condense_log(input_file: Path, output_file: Path) -> None:
    """Condense a training log file while preserving key metrics trends."""

    print(f"Reading log file: {input_file}")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total lines: {total_lines:,}")

    condensed_lines = []
    metrics_history = []
    last_progress_pct = -1
    last_reward_sample = -1

    print("Processing lines...")
    for i, line in enumerate(lines):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{total_lines:,} lines ({i/total_lines:.1%})")

        parsed = parse_log_line(line)
        parsed['line_number'] = i + 1

        # Extract and track training metrics
        metrics = extract_training_metrics(line)
        if metrics:
            metrics_history.append({
                'line': i + 1,
                'timestamp': parsed['timestamp'],
                'metrics': metrics
            })

        # Decide whether to keep this line
        keep, last_progress_pct, last_reward_sample = should_keep_line(
            parsed, i, total_lines, last_progress_pct, last_reward_sample
        )

        if keep:
            condensed_lines.append(line)

    print(f"Writing condensed log: {output_file}")

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header with summary
        f.write("=" * 80 + "\n")
        f.write("CONDENSED GRPO TRAINING LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Original file: {input_file}\n")
        f.write(f"Original lines: {total_lines:,}\n")
        f.write(f"Condensed lines: {len(condensed_lines):,}\n")
        f.write(f"Compression ratio: {len(condensed_lines)/total_lines:.1%}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Write metrics trend summary
        if metrics_history:
            f.write(create_metrics_trend_summary(metrics_history))
            f.write("\n")

        # Write condensed log content
        f.write("CONDENSED LOG CONTENT:\n")
        f.write("=" * 80 + "\n")
        f.writelines(condensed_lines)

    print(f"\nCondensation complete!")
    print(f"Reduced from {total_lines:,} to {len(condensed_lines):,} lines")
    print(f"Compression ratio: {len(condensed_lines)/total_lines:.1%}")
    print(f"Tracked {len(metrics_history)} metric snapshots")

def main():
    if len(sys.argv) != 3:
        print("Usage: python condense_log.py <input_log_file> <output_log_file>")
        print("Example: python condense_log.py logs/run.log logs/run_condensed.log")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    try:
        condense_log(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
