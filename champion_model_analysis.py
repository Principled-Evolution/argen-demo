#!/usr/bin/env python3
"""
Champion Model Analysis Script

This script identifies the Champion Model from ArGen evaluation data.
The Champion Model is the single best-performing checkpoint across all seeds
based on Claude-3.5-Sonnet's average_combined_score.

Usage: python champion_model_analysis.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

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
        
        # Baseline evaluation
        "training_reports/meta-llama-3.2-1B-baseline-eval/eval_Llama-3.2-1B-Instruct_benchmarking_20250510_135534-cleanprep-hashprompt.json"
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
                    'average_clarity_score': data['summary_metrics']['average_clarity_score'],
                    'average_relevance_score': data['summary_metrics']['average_relevance_score'],
                    'average_completeness_score': data['summary_metrics']['average_completeness_score'],
                    'ahimsa_violation_rate': data['summary_metrics']['ahimsa_violation_rate'],
                    'dharma_violation_rate': data['summary_metrics']['dharma_violation_rate'],
                    'helpfulness_violation_rate': data['summary_metrics']['helpfulness_violation_rate']
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
            
    elif 'Llama-3.2-1B-Instruct' in model_name:
        info['model_type'] = 'baseline'
        info['seed'] = 'baseline'
        info['checkpoint'] = 'baseline'
    
    return info

def identify_champion_model(data: List[Dict]) -> Tuple[Dict, pd.DataFrame]:
    """Identify the Champion Model based on highest average_combined_score."""
    
    # Filter out baseline for champion selection (only ArGen models)
    argen_data = [d for d in data if d['model_type'] != 'baseline']
    
    # Sort by average_combined_score (descending)
    sorted_data = sorted(argen_data, key=lambda x: x['average_combined_score'], reverse=True)
    
    champion = sorted_data[0]
    
    # Create DataFrame for analysis
    df = pd.DataFrame(argen_data)
    df = df.sort_values('average_combined_score', ascending=False)
    
    return champion, df

def generate_champion_report(champion: Dict, df: pd.DataFrame, all_data: List[Dict]) -> str:
    """Generate a detailed report about the Champion Model."""
    
    # Find baseline for comparison
    baseline = next((d for d in all_data if d['model_type'] == 'baseline'), None)
    
    report = f"""# Champion Model Analysis Report

## Champion Model Identification

**Champion Model**: {champion['model_name']}
- **Seed**: {champion['seed']}
- **Checkpoint**: {champion['checkpoint']}
- **Model Type**: {champion['model_type'].upper()}
- **Average Combined Score**: {champion['average_combined_score']:.4f}

## Performance Breakdown

### Core Metrics (Champion Model)
- **Ahimsa Score**: {champion['average_ahimsa_score']:.4f}
- **Dharma Score**: {champion['average_dharma_score']:.4f}  
- **Helpfulness Score**: {champion['average_helpfulness_score']:.4f}
- **Clarity Score**: {champion['average_clarity_score']:.4f}
- **Relevance Score**: {champion['average_relevance_score']:.4f}
- **Completeness Score**: {champion['average_completeness_score']:.4f}

### Violation Rates (Champion Model)
- **Ahimsa Violations**: {champion['ahimsa_violation_rate']:.1%}
- **Dharma Violations**: {champion['dharma_violation_rate']:.1%}
- **Helpfulness Violations**: {champion['helpfulness_violation_rate']:.1%}

"""

    if baseline:
        improvement = champion['average_combined_score'] - baseline['average_combined_score']
        improvement_pct = (improvement / baseline['average_combined_score']) * 100
        
        report += f"""## Baseline Comparison

**Baseline Model**: {baseline['model_name']}
- **Baseline Combined Score**: {baseline['average_combined_score']:.4f}
- **Champion Combined Score**: {champion['average_combined_score']:.4f}
- **Absolute Improvement**: +{improvement:.4f}
- **Relative Improvement**: +{improvement_pct:.1f}%

"""

    # Top 5 models ranking
    report += """## Top 5 ArGen Models Ranking

| Rank | Model | Seed | Checkpoint | Combined Score |
|------|-------|------|------------|----------------|
"""
    
    for i, row in df.head(5).iterrows():
        report += f"| {df.index.get_loc(i) + 1} | {row['model_type'].upper()} | {row['seed']} | {row['checkpoint']} | {row['average_combined_score']:.4f} |\n"

    report += f"""
## Analysis Summary

The Champion Model represents the peak performance achieved by the ArGen framework across all training runs and checkpoints. This model should be used for:

1. **Main Results Tables**: Report these scores as "ArGen (Best)" or "MedGuide-AI (Champion)"
2. **Performance Claims**: "At its best, ArGen achieved {champion['average_combined_score']:.4f} combined score"
3. **Baseline Comparisons**: Demonstrate the maximum improvement over the baseline model

**Note**: For ablation studies, use the Median Seed (see median_seed_analysis.py) to ensure fair, non-cherry-picked comparisons.

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Total ArGen models analyzed: {len(df)}*
"""

    return report

if __name__ == "__main__":
    print("🔍 Champion Model Analysis")
    print("=" * 50)
    
    # Load all evaluation data
    print("\n📊 Loading evaluation data...")
    data = load_evaluation_data()
    print(f"✓ Loaded {len(data)} evaluation files")
    
    # Identify champion model
    print("\n🏆 Identifying Champion Model...")
    champion, df = identify_champion_model(data)
    
    print(f"✓ Champion Model: {champion['model_name']}")
    print(f"✓ Combined Score: {champion['average_combined_score']:.4f}")
    
    # Generate report
    print("\n📝 Generating report...")
    report = generate_champion_report(champion, df, data)
    
    # Save report
    with open('champion_model_report.md', 'w') as f:
        f.write(report)
    
    print("✓ Report saved to: champion_model_report.md")
    print("\n🎉 Champion Model analysis complete!")
