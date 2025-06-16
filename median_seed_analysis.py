#!/usr/bin/env python3
"""
Median Seed Analysis Script

This script identifies the Median Seed for ArGen ablation studies.
The Median Seed is the seed whose peak performance represents the median
of all three seeds' peak performances, ensuring fair ablation comparisons.

Usage: python median_seed_analysis.py
"""

import json
import os
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def load_evaluation_data() -> List[Dict]:
    """Load all evaluation JSON files and extract key metrics."""
    
    evaluation_files = [
        # GRPO5 (Seed 1) evaluations
        "training_reports/grpo5-evals/eval_grpo_5_seed_1_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo5-evals/eval_checkpoint-1000_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo5-evals/eval_checkpoint-2000_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo5-evals/eval_checkpoint-2600_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo5-evals/eval_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        
        # GRPO6 (Seed 2) evaluations
        "training_reports/grpo6-grpo7-evals/eval_grpo_6_seed_2_4_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo6-grpo7-evals/eval_checkpoint-1000_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo6-grpo7-evals/eval_checkpoint-2000_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo6-grpo7-evals/eval_checkpoint-3000_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        "training_reports/grpo6-grpo7-evals/eval_checkpoint-4000_benchmarking_20250510_135534-cleanprep-hashprompt.json",
        
        # GRPO7 (Seed 3) evaluations
        "training_reports/grpo6-grpo7-evals/eval_grpo_7_seed_3_3_benchmarking_20250510_135534-cleanprep-hashprompt.json",
    ]
    
    all_data = []
    
    for file_path in evaluation_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract key information
                model_info = {
                    'file_path': file_path,
                    'model_name': data['evaluation_config']['model_name'],
                    'evaluator': data['evaluation_config']['evaluator'],
                    'timestamp': data['evaluation_config']['timestamp'],
                    'average_combined_score': data['summary_metrics']['average_combined_score'],
                    'average_ahimsa_score': data['summary_metrics']['average_ahimsa_score'],
                    'average_dharma_score': data['summary_metrics']['average_dharma_score'],
                    'average_helpfulness_score': data['summary_metrics']['average_helpfulness_score'],
                }
                
                # Parse seed and checkpoint info from model name
                model_info.update(parse_model_info(model_info['model_name']))
                
                all_data.append(model_info)
                print(f"✓ Loaded: {model_info['model_name']}")
                
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
        else:
            print(f"✗ File not found: {file_path}")
    
    return all_data

def parse_model_info(model_name: str) -> Dict:
    """Parse seed and checkpoint information from model name."""
    info = {
        'seed': None,
        'checkpoint': None,
        'model_type': 'unknown'
    }
    
    if 'grpo_5_seed_1' in model_name:
        info['seed'] = 1
        info['model_type'] = 'grpo5'
        if 'checkpoint-' in model_name:
            checkpoint = model_name.split('checkpoint-')[1].split('/')[0]
            info['checkpoint'] = int(checkpoint)
        else:
            info['checkpoint'] = 'final'
            
    elif 'grpo_6_seed_2_4' in model_name:
        info['seed'] = 2
        info['model_type'] = 'grpo6'
        if 'checkpoint-' in model_name:
            checkpoint = model_name.split('checkpoint-')[1].split('/')[0]
            info['checkpoint'] = int(checkpoint)
        else:
            info['checkpoint'] = 'final'
            
    elif 'grpo_7_seed_3_3' in model_name:
        info['seed'] = 3
        info['model_type'] = 'grpo7'
        if 'checkpoint-' in model_name:
            checkpoint = model_name.split('checkpoint-')[1].split('/')[0]
            info['checkpoint'] = int(checkpoint)
        else:
            info['checkpoint'] = 'final'
    
    return info

def calculate_seed_peak_performances(data: List[Dict]) -> Dict[int, Dict]:
    """Calculate peak performance for each seed."""
    
    seed_data = {}
    
    # Group data by seed
    for item in data:
        seed = item['seed']
        if seed not in seed_data:
            seed_data[seed] = []
        seed_data[seed].append(item)
    
    # Calculate peak performance for each seed
    seed_peaks = {}
    for seed, models in seed_data.items():
        # Find the model with highest combined score for this seed
        best_model = max(models, key=lambda x: x['average_combined_score'])
        
        seed_peaks[seed] = {
            'peak_score': best_model['average_combined_score'],
            'best_model': best_model,
            'all_models': models,
            'num_models': len(models)
        }
    
    return seed_peaks

def identify_median_seed(seed_peaks: Dict[int, Dict]) -> Tuple[int, Dict]:
    """Identify the median seed based on peak performances."""
    
    # Extract peak scores
    peak_scores = [(seed, data['peak_score']) for seed, data in seed_peaks.items()]
    peak_scores.sort(key=lambda x: x[1])  # Sort by score
    
    # Find median (middle value for 3 seeds)
    median_idx = len(peak_scores) // 2
    median_seed, median_score = peak_scores[median_idx]
    
    return median_seed, seed_peaks[median_seed]

def generate_median_seed_report(median_seed: int, median_data: Dict, seed_peaks: Dict[int, Dict]) -> str:
    """Generate a detailed report about the Median Seed selection."""
    
    report = f"""# Median Seed Analysis Report

## Median Seed Identification

**Median Seed**: Seed {median_seed}
- **Peak Combined Score**: {median_data['peak_score']:.4f}
- **Best Model**: {median_data['best_model']['model_name']}
- **Best Checkpoint**: {median_data['best_model']['checkpoint']}
- **Total Models Evaluated**: {median_data['num_models']}

## Seed Performance Comparison

| Seed | Peak Score | Best Checkpoint | Model Type | Rank |
|------|------------|-----------------|------------|------|
"""
    
    # Sort seeds by peak performance for ranking
    sorted_seeds = sorted(seed_peaks.items(), key=lambda x: x[1]['peak_score'], reverse=True)
    
    for rank, (seed, data) in enumerate(sorted_seeds, 1):
        marker = " ← **MEDIAN**" if seed == median_seed else ""
        report += f"| {seed} | {data['peak_score']:.4f} | {data['best_model']['checkpoint']} | {data['best_model']['model_type'].upper()} | {rank}{marker} |\n"
    
    report += f"""
## Detailed Seed Analysis

"""
    
    for seed in sorted([1, 2, 3]):
        if seed in seed_peaks:
            data = seed_peaks[seed]
            report += f"""### Seed {seed} Performance
- **Peak Score**: {data['peak_score']:.4f}
- **Best Model**: {data['best_model']['model_name']}
- **Best Checkpoint**: {data['best_model']['checkpoint']}
- **Ahimsa Score**: {data['best_model']['average_ahimsa_score']:.4f}
- **Dharma Score**: {data['best_model']['average_dharma_score']:.4f}
- **Helpfulness Score**: {data['best_model']['average_helpfulness_score']:.4f}

"""
    
    report += f"""## Median Seed Selection Rationale

The Median Seed (Seed {median_seed}) was selected using the following methodology:

1. **Calculate Peak Performance**: For each seed, identify the checkpoint with the highest Claude-judged `average_combined_score`
2. **Rank Seeds**: Order the three seeds by their peak performance scores
3. **Select Median**: Choose the seed with the middle (median) peak performance

This approach ensures:
- **Scientific Rigor**: Avoids cherry-picking the best-performing seed for ablations
- **Representative Results**: Uses a seed that represents typical (not exceptional) performance
- **Fair Comparisons**: Ablation studies will be based on a representative training run

## Recommended Usage

Use Seed {median_seed} for:
1. **Reward-Only Ablation**: Train from scratch with only reward model, no policy optimization
2. **Policy-Only Ablation**: Train from scratch with only policy optimization, no reward model
3. **Component Analysis**: Any ablation study requiring fair baseline comparison

**Important**: Do NOT use the Champion Model's seed for ablations, as this would constitute cherry-picking and compromise the scientific validity of the comparison.

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Seeds analyzed: {len(seed_peaks)}*
"""

    return report

if __name__ == "__main__":
    print("🎯 Median Seed Analysis")
    print("=" * 50)
    
    # Load all evaluation data
    print("\n📊 Loading evaluation data...")
    data = load_evaluation_data()
    print(f"✓ Loaded {len(data)} evaluation files")
    
    # Calculate seed peak performances
    print("\n📈 Calculating seed peak performances...")
    seed_peaks = calculate_seed_peak_performances(data)
    
    for seed, data in seed_peaks.items():
        print(f"✓ Seed {seed}: Peak score {data['peak_score']:.4f}")
    
    # Identify median seed
    print("\n🎯 Identifying Median Seed...")
    median_seed, median_data = identify_median_seed(seed_peaks)
    
    print(f"✓ Median Seed: {median_seed}")
    print(f"✓ Peak Score: {median_data['peak_score']:.4f}")
    
    # Generate report
    print("\n📝 Generating report...")
    report = generate_median_seed_report(median_seed, median_data, seed_peaks)
    
    # Save report
    with open('median_seed_report.md', 'w') as f:
        f.write(report)
    
    print("✓ Report saved to: median_seed_report.md")
    print("\n🎉 Median Seed analysis complete!")
