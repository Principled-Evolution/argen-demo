#!/usr/bin/env python3
"""
Median Seed Analysis - Final Evaluations

This script identifies the Median Seed for ablation studies.
The Median Seed is the seed whose peak performance represents the median
of all three seeds' peak performances (Claude-judged).

Usage: python median_seed_analysis_final.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class MedianSeedAnalyzer:
    def __init__(self):
        """Initialize the Median Seed analyzer."""
        self.data_path = Path("../data")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        self.consolidated_data = None
        self.claude_data = None
        self.argen_data = None

    def load_consolidated_data(self) -> None:
        """Load the consolidated evaluation data."""
        print("📊 Loading consolidated evaluation data...")
        
        json_path = self.data_path / "consolidated_final_evals.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Consolidated data not found at {json_path}. Run final_evals_consolidator.py first.")
        
        with open(json_path, 'r') as f:
            self.consolidated_data = json.load(f)
        
        print(f"✓ Loaded {len(self.consolidated_data['evaluations'])} evaluations")

    def filter_claude_argen_data(self) -> None:
        """Filter to Claude-only ArGen evaluations."""
        print("🔍 Filtering Claude ArGen evaluations...")
        
        all_evals = self.consolidated_data['evaluations']
        self.claude_data = [eval for eval in all_evals if eval['evaluator_type'] == 'claude']
        self.argen_data = [eval for eval in self.claude_data if eval['model_family'] == 'argen']
        
        print(f"✓ Claude ArGen evaluations: {len(self.argen_data)}")

    def calculate_seed_peak_performances(self) -> Dict:
        """Calculate peak performance for each seed."""
        print("📈 Calculating peak performance for each seed...")
        
        seed_data = {}
        
        # Group by seed
        for eval in self.argen_data:
            seed = eval['seed']
            if seed not in seed_data:
                seed_data[seed] = []
            seed_data[seed].append(eval)
        
        # Find peak performance for each seed
        seed_peaks = {}
        for seed, evaluations in seed_data.items():
            # Sort by combined score (descending)
            sorted_evals = sorted(evaluations, key=lambda x: x['average_combined_score'], reverse=True)
            peak_eval = sorted_evals[0]
            
            seed_peaks[seed] = {
                'peak_score': peak_eval['average_combined_score'],
                'peak_model': peak_eval,
                'all_evaluations': evaluations,
                'num_checkpoints': len(evaluations)
            }
            
            print(f"✓ Seed {seed}: Peak score {peak_eval['average_combined_score']:.4f} "
                  f"({peak_eval['model_type']} {peak_eval['checkpoint']})")
        
        return seed_peaks

    def identify_median_seed(self, seed_peaks: Dict) -> Tuple[int, Dict]:
        """Identify the seed with median peak performance."""
        print("🎯 Identifying median seed...")
        
        # Extract peak scores
        peak_scores = [(seed, data['peak_score']) for seed, data in seed_peaks.items()]
        peak_scores.sort(key=lambda x: x[1])  # Sort by score
        
        print("Peak scores by seed:")
        for seed, score in peak_scores:
            print(f"  Seed {seed}: {score:.4f}")
        
        # Find median (middle value)
        if len(peak_scores) == 3:
            median_seed, median_score = peak_scores[1]  # Middle value
        else:
            # Fallback for different number of seeds
            median_idx = len(peak_scores) // 2
            median_seed, median_score = peak_scores[median_idx]
        
        print(f"✓ Median Seed: {median_seed} (score: {median_score:.4f})")
        
        return median_seed, seed_peaks[median_seed]

    def analyze_seed_performance_distributions(self, seed_peaks: Dict) -> Dict:
        """Analyze performance distributions for each seed."""
        seed_stats = {}
        
        for seed, data in seed_peaks.items():
            scores = [eval['average_combined_score'] for eval in data['all_evaluations']]
            
            seed_stats[seed] = {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'range': np.max(scores) - np.min(scores),
                'count': len(scores),
                'peak_score': data['peak_score']
            }
        
        return seed_stats

    def generate_median_seed_report(self, median_seed: int, median_data: Dict, 
                                   seed_peaks: Dict, seed_stats: Dict) -> str:
        """Generate comprehensive Median Seed analysis report."""
        
        median_model = median_data['peak_model']
        
        report = f"""# Median Seed Analysis Report - Final Evaluations

## Executive Summary

**Median Seed Identified**: Seed {median_seed}
- **Peak Performance**: {median_data['peak_score']:.4f}
- **Peak Model**: {median_model['model_type'].upper()} ({median_model['checkpoint']})
- **Purpose**: Representative seed for ablation studies (reward-only, policy-only)
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Methodology

The Median Seed is selected to ensure fair and representative ablation studies:

1. **Calculate Peak Performance**: For each seed, find the highest `average_combined_score` across all checkpoints
2. **Rank Seeds**: Order the three seeds by their peak performance scores
3. **Select Median**: Choose the seed with the middle (median) peak performance
4. **Scientific Rationale**: Avoids cherry-picking the best or worst performing seed

## Seed Peak Performance Analysis

### Peak Scores Summary
| Rank | Seed | Peak Score | Peak Model | Checkpoint | Model Type |
|------|------|------------|------------|------------|------------|
"""
        
        # Sort seeds by peak performance for ranking
        sorted_seeds = sorted(seed_peaks.items(), key=lambda x: x[1]['peak_score'], reverse=True)
        
        for rank, (seed, data) in enumerate(sorted_seeds, 1):
            peak_model = data['peak_model']
            marker = " 🎯" if seed == median_seed else ""
            report += f"| {rank} | {seed}{marker} | {data['peak_score']:.4f} | {peak_model['model_type'].upper()} | {peak_model['checkpoint']} | {peak_model['model_name'].split('/')[-1] if '/' in peak_model['model_name'] else peak_model['model_name']} |\n"

        report += f"""
### Median Seed Selection
- **Selected Seed**: {median_seed} (middle-ranked performance)
- **Peak Score**: {median_data['peak_score']:.4f}
- **Rationale**: Represents typical ArGen performance, avoiding extremes

## Detailed Seed Analysis

"""
        
        # Detailed analysis for each seed
        for seed in sorted([1, 2, 3]):
            if seed not in seed_peaks:
                continue
                
            data = seed_peaks[seed]
            stats = seed_stats[seed]
            peak_model = data['peak_model']
            
            marker = "🎯 **MEDIAN SEED**" if seed == median_seed else ""
            
            report += f"""### Seed {seed} {marker}

#### Peak Performance Model
- **Model**: {peak_model['model_type'].upper()} ({peak_model['checkpoint']})
- **Combined Score**: {peak_model['average_combined_score']:.4f}
- **Ahimsa**: {peak_model['average_ahimsa_score']:.4f}
- **Dharma**: {peak_model['average_dharma_score']:.4f}
- **Helpfulness**: {peak_model['average_helpfulness_score']:.4f}

#### Performance Distribution ({stats['count']} checkpoints)
| Metric | Value |
|--------|-------|
| Mean | {stats['mean']:.4f} |
| Median | {stats['median']:.4f} |
| Std Dev | {stats['std']:.4f} |
| Min | {stats['min']:.4f} |
| Max | {stats['max']:.4f} |
| Range | {stats['range']:.4f} |

"""

        # Ablation study recommendations
        report += f"""## Ablation Study Recommendations

### Using Median Seed {median_seed} for Ablations

The Median Seed should be used to train the following ablation models:

1. **Reward-Only Model**: Train using only reward optimization (no policy constraints)
2. **Policy-Only Model**: Train using only policy optimization (no reward shaping)

### Training Configuration
- **Base Seed**: {median_seed}
- **Peak Checkpoint Reference**: {median_model['checkpoint']} ({median_data['peak_score']:.4f} score)
- **Training Steps**: Match the peak checkpoint's training duration
- **Evaluation**: Use same 100-scenario benchmark with Claude-3.5-Sonnet

### Scientific Justification
- **Avoids Cherry-Picking**: Not using the best-performing seed prevents bias
- **Representative Performance**: Median seed represents typical ArGen behavior
- **Fair Comparison**: Ablations will be compared against a representative baseline
- **Reproducible**: Clear methodology for seed selection

## Performance Context

### Seed Performance Spread
- **Highest Peak**: {max(seed_peaks.values(), key=lambda x: x['peak_score'])['peak_score']:.4f}
- **Median Peak**: {median_data['peak_score']:.4f}
- **Lowest Peak**: {min(seed_peaks.values(), key=lambda x: x['peak_score'])['peak_score']:.4f}
- **Peak Range**: {max(seed_peaks.values(), key=lambda x: x['peak_score'])['peak_score'] - min(seed_peaks.values(), key=lambda x: x['peak_score'])['peak_score']:.4f}

### Cross-Seed Consistency
"""
        
        # Calculate cross-seed statistics
        all_peak_scores = [data['peak_score'] for data in seed_peaks.values()]
        report += f"""- **Mean Peak Score**: {np.mean(all_peak_scores):.4f}
- **Peak Score Std Dev**: {np.std(all_peak_scores):.4f}
- **Coefficient of Variation**: {(np.std(all_peak_scores) / np.mean(all_peak_scores)) * 100:.1f}%

## Data Provenance

### Median Seed Model Source
- **Evaluation File**: `{median_model['file_path']}`
- **Evaluator**: {median_model['evaluator']}
- **Timestamp**: {median_model['timestamp']}
- **Scenarios**: {median_model['num_scenarios']}

### Analysis Methodology
1. **Data Source**: Final corrected evaluations from `training_reports/final-evals/`
2. **Evaluator Filter**: Claude-3.5-Sonnet only (consistent judgment)
3. **Scope**: All ArGen checkpoints across 3 seeds
4. **Selection**: Mathematical median of peak performances

### Next Steps
1. **Train Ablation Models**: Use Seed {median_seed} configuration
2. **Evaluate Ablations**: Same evaluation protocol as main models
3. **Compare Results**: Ablation performance vs. Median Seed peak performance

---
*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*Analysis Script*: `median_seed_analysis_final.py`  
*Data Source*: `../data/consolidated_final_evals.json`  
*Total Models Analyzed*: {sum(len(data['all_evaluations']) for data in seed_peaks.values())} across {len(seed_peaks)} seeds
"""
        
        return report

    def run_analysis(self) -> None:
        """Run the complete Median Seed analysis."""
        print("🎯 Starting Median Seed Analysis")
        print("=" * 40)
        
        # Load and filter data
        self.load_consolidated_data()
        self.filter_claude_argen_data()
        
        # Calculate seed peak performances
        seed_peaks = self.calculate_seed_peak_performances()
        
        # Identify median seed
        median_seed, median_data = self.identify_median_seed(seed_peaks)
        
        # Analyze distributions
        seed_stats = self.analyze_seed_performance_distributions(seed_peaks)
        
        # Generate report
        report = self.generate_median_seed_report(median_seed, median_data, seed_peaks, seed_stats)
        
        # Save report
        report_path = self.reports_path / "median_seed_final_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to: {report_path}")
        print(f"\n🎉 Median Seed analysis complete!")
        print(f"📊 Median Seed {median_seed} identified for ablation studies")

if __name__ == "__main__":
    analyzer = MedianSeedAnalyzer()
    analyzer.run_analysis()
