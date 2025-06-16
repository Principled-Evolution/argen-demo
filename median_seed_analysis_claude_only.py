#!/usr/bin/env python3
"""
Median Seed Analysis Script - Claude Evaluations Only

This script identifies the Median Seed for ArGen ablation studies using
CLAUDE-ONLY evaluation data for scientific rigor.

Usage: python median_seed_analysis_claude_only.py
"""

import json
import os
import glob
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def load_claude_evaluation_data() -> List[Dict]:
    """Load all Claude evaluation JSON files from clean structure."""
    
    # Find all Claude evaluation files
    claude_files = glob.glob('training_reports_clean/claude_evaluations/**/*.json', recursive=True)
    
    all_data = []
    
    print(f"Found {len(claude_files)} Claude evaluation files...")
    
    for file_path in sorted(claude_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Verify this is a Claude evaluation
            evaluator = data.get('evaluation_config', {}).get('evaluator', '')
            if 'claude' not in evaluator.lower() and 'anthropic' not in evaluator.lower():
                print(f"⚠️  Skipping non-Claude file: {file_path}")
                continue
                
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
            
            # Only include ArGen models (not baseline)
            if model_info['model_type'] != 'baseline':
                all_data.append(model_info)
                print(f"✓ Loaded: {model_info['model_name']} (Claude)")
            
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
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
            
    elif 'meta-llama/Llama-3.2-1B-Instruct' in model_name:
        info['model_type'] = 'baseline'
        info['seed'] = 'baseline'
        info['checkpoint'] = 'baseline'
    
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
    
    report = f"""# Median Seed Analysis Report (Claude Evaluations Only)

## Median Seed Identification

**Median Seed**: Seed {median_seed}
- **Peak Combined Score**: {median_data['peak_score']:.4f}
- **Best Model**: {median_data['best_model']['model_name']}
- **Best Checkpoint**: {median_data['best_model']['checkpoint']}
- **Total Models Evaluated**: {median_data['num_models']}
- **Evaluator**: Claude-3.5-Sonnet (consistent)

## Seed Performance Comparison (Claude Only)

| Seed | Peak Score | Best Checkpoint | Model Type | Rank |
|------|------------|-----------------|------------|------|
"""
    
    # Sort seeds by peak performance for ranking
    sorted_seeds = sorted(seed_peaks.items(), key=lambda x: x[1]['peak_score'], reverse=True)
    
    for rank, (seed, data) in enumerate(sorted_seeds, 1):
        marker = " ← **MEDIAN**" if seed == median_seed else ""
        report += f"| {seed} | {data['peak_score']:.4f} | {data['best_model']['checkpoint']} | {data['best_model']['model_type'].upper()} | {rank}{marker} |\n"
    
    report += f"""
## Detailed Seed Analysis (Claude Evaluations)

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

1. **Claude-Only Evaluations**: Used exclusively Claude-3.5-Sonnet evaluations for consistency
2. **Calculate Peak Performance**: For each seed, identify the checkpoint with the highest Claude-judged `average_combined_score`
3. **Rank Seeds**: Order the three seeds by their peak performance scores
4. **Select Median**: Choose the seed with the middle (median) peak performance

This approach ensures:
- **Scientific Rigor**: Consistent evaluator (Claude) across all comparisons
- **Representative Results**: Uses a seed that represents typical (not exceptional) performance
- **Fair Comparisons**: Ablation studies will be based on a representative training run

## Recommended Usage

Use Seed {median_seed} for:
1. **Reward-Only Ablation**: Train from scratch with only reward model, no policy optimization
2. **Policy-Only Ablation**: Train from scratch with only policy optimization, no reward model
3. **Component Analysis**: Any ablation study requiring fair baseline comparison

**Important**: 
- Do NOT use the Champion Model's seed for ablations (would constitute cherry-picking)
- All ablation comparisons should use Claude evaluations for consistency
- This analysis is based on {len(seed_peaks)} seeds with Claude evaluations

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Seeds analyzed: {len(seed_peaks)} (Claude-only)*
*Evaluator: Claude-3.5-Sonnet (consistent across all models)*
"""

    return report

if __name__ == "__main__":
    print("🎯 Median Seed Analysis (Claude Evaluations Only)")
    print("=" * 60)
    
    # Load Claude evaluation data
    print("\n📊 Loading Claude evaluation data...")
    data = load_claude_evaluation_data()
    print(f"✓ Loaded {len(data)} Claude evaluation files")
    
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
    with open('median_seed_report_claude_only.md', 'w') as f:
        f.write(report)
    
    print("✓ Report saved to: median_seed_report_claude_only.md")
    print("\n🎉 Median Seed analysis complete (Claude-only)!")
