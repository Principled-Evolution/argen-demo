#!/usr/bin/env python3
"""
Helpful-Champion Model Analysis Script

This script identifies the Helpful-Champion Model from ArGen evaluation data.
The Helpful-Champion is the ArGen model that best maintains or improves 
helpfulness compared to the baseline, while still showing overall improvement.

Usage: python helpful_champion_analysis.py
"""

import json
import os
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

def identify_helpful_champion(data: List[Dict]) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Identify the Helpful-Champion Model based on best helpfulness preservation."""
    
    # Find baseline for comparison
    baseline = next((d for d in data if d['model_type'] == 'baseline'), None)
    if not baseline:
        raise ValueError("Baseline model not found!")
    
    baseline_helpfulness = baseline['average_helpfulness_score']
    baseline_combined = baseline['average_combined_score']
    
    # Filter ArGen models and calculate helpfulness metrics
    argen_data = [d for d in data if d['model_type'] != 'baseline']
    
    for model in argen_data:
        # Calculate helpfulness change vs baseline
        model['helpfulness_change'] = model['average_helpfulness_score'] - baseline_helpfulness
        model['helpfulness_change_pct'] = (model['helpfulness_change'] / baseline_helpfulness) * 100
        
        # Calculate combined score improvement (must be positive to be considered)
        model['combined_improvement'] = model['average_combined_score'] - baseline_combined
        model['combined_improvement_pct'] = (model['combined_improvement'] / baseline_combined) * 100
    
    # Filter models that show overall improvement (positive combined score change)
    improved_models = [m for m in argen_data if m['combined_improvement'] > 0]
    
    if not improved_models:
        raise ValueError("No ArGen models show improvement over baseline!")
    
    # Sort by helpfulness change (descending - best helpfulness preservation/improvement first)
    helpful_champion = max(improved_models, key=lambda x: x['helpfulness_change'])
    
    # Create DataFrame for analysis
    df = pd.DataFrame(improved_models)
    df = df.sort_values('helpfulness_change', ascending=False)
    
    return helpful_champion, df, baseline

def generate_helpful_champion_report(helpful_champion: Dict, df: pd.DataFrame, baseline: Dict, all_data: List[Dict]) -> str:
    """Generate a detailed report about the Helpful-Champion Model."""
    
    # Find overall champion for comparison
    argen_models = [d for d in all_data if d['model_type'] != 'baseline']
    overall_champion = max(argen_models, key=lambda x: x['average_combined_score'])
    
    report = f"""# Helpful-Champion Model Analysis Report

## Helpful-Champion Model Identification

**Helpful-Champion Model**: {helpful_champion['model_name']}
- **Seed**: {helpful_champion['seed']}
- **Checkpoint**: {helpful_champion['checkpoint']}
- **Model Type**: {helpful_champion['model_type'].upper()}
- **Combined Score**: {helpful_champion['average_combined_score']:.4f}
- **Helpfulness Score**: {helpful_champion['average_helpfulness_score']:.4f}

## Helpfulness Analysis

### Baseline Comparison (Helpfulness Focus)
- **Baseline Helpfulness**: {baseline['average_helpfulness_score']:.4f}
- **Helpful-Champion Helpfulness**: {helpful_champion['average_helpfulness_score']:.4f}
- **Helpfulness Change**: {helpful_champion['helpfulness_change']:+.4f}
- **Helpfulness Change %**: {helpful_champion['helpfulness_change_pct']:+.1f}%

### Overall Performance (Helpful-Champion)
- **Combined Score**: {helpful_champion['average_combined_score']:.4f}
- **Combined Improvement**: +{helpful_champion['combined_improvement']:.4f}
- **Combined Improvement %**: +{helpful_champion['combined_improvement_pct']:.1f}%

## Champion Model Comparison

### Overall Champion vs Helpful-Champion
| Metric | Overall Champion | Helpful-Champion | Difference |
|--------|------------------|------------------|------------|
| **Model** | {overall_champion['model_type'].upper()} Seed {overall_champion['seed']} | {helpful_champion['model_type'].upper()} Seed {helpful_champion['seed']} | - |
| **Combined Score** | {overall_champion['average_combined_score']:.4f} | {helpful_champion['average_combined_score']:.4f} | {helpful_champion['average_combined_score'] - overall_champion['average_combined_score']:+.4f} |
| **Helpfulness Score** | {overall_champion['average_helpfulness_score']:.4f} | {helpful_champion['average_helpfulness_score']:.4f} | {helpful_champion['average_helpfulness_score'] - overall_champion['average_helpfulness_score']:+.4f} |
| **Ahimsa Score** | {overall_champion['average_ahimsa_score']:.4f} | {helpful_champion['average_ahimsa_score']:.4f} | {helpful_champion['average_ahimsa_score'] - overall_champion['average_ahimsa_score']:+.4f} |
| **Dharma Score** | {overall_champion['average_dharma_score']:.4f} | {helpful_champion['average_dharma_score']:.4f} | {helpful_champion['average_dharma_score'] - overall_champion['average_dharma_score']:+.4f} |

## Top 5 Models by Helpfulness Preservation

| Rank | Model | Seed | Checkpoint | Helpfulness Score | Change vs Baseline | Combined Score |
|------|-------|------|------------|-------------------|-------------------|----------------|
"""
    
    for i, (idx, row) in enumerate(df.head(5).iterrows(), 1):
        report += f"| {i} | {row['model_type'].upper()} | {row['seed']} | {row['checkpoint']} | {row['average_helpfulness_score']:.4f} | {row['helpfulness_change']:+.4f} | {row['average_combined_score']:.4f} |\n"

    report += f"""
## Detailed Performance Breakdown (Helpful-Champion)

### Core Metrics
- **Ahimsa Score**: {helpful_champion['average_ahimsa_score']:.4f}
- **Dharma Score**: {helpful_champion['average_dharma_score']:.4f}
- **Helpfulness Score**: {helpful_champion['average_helpfulness_score']:.4f}
- **Clarity Score**: {helpful_champion['average_clarity_score']:.4f}
- **Relevance Score**: {helpful_champion['average_relevance_score']:.4f}
- **Completeness Score**: {helpful_champion['average_completeness_score']:.4f}

### Violation Rates
- **Ahimsa Violations**: {helpful_champion['ahimsa_violation_rate']:.1%}
- **Dharma Violations**: {helpful_champion['dharma_violation_rate']:.1%}
- **Helpfulness Violations**: {helpful_champion['helpfulness_violation_rate']:.1%}

## Analysis Summary

The Helpful-Champion Model represents the best balance between overall improvement and helpfulness preservation. Key insights:

1. **Helpfulness Preservation**: {helpful_champion['helpfulness_change']:+.4f} change vs baseline ({helpful_champion['helpfulness_change_pct']:+.1f}%)
2. **Overall Improvement**: +{helpful_champion['combined_improvement_pct']:.1f}% combined score improvement
3. **Use Case**: Ideal for scenarios where maintaining helpfulness is critical
4. **Trade-offs**: May sacrifice some safety gains for better user assistance

### Recommended Usage

Use the Helpful-Champion Model when:
- **User Experience Priority**: Helpfulness is the primary concern
- **Balanced Performance**: Need good overall improvement with minimal helpfulness loss
- **Comparative Studies**: Demonstrating that ArGen can maintain helpfulness while improving safety

**Note**: For maximum safety gains, use the Overall Champion Model. For fair ablations, use the Median Seed.

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Models analyzed: {len(df)} improved ArGen models*
"""

    return report

if __name__ == "__main__":
    print("🤝 Helpful-Champion Model Analysis")
    print("=" * 50)
    
    # Load all evaluation data
    print("\n📊 Loading evaluation data...")
    data = load_evaluation_data()
    print(f"✓ Loaded {len(data)} evaluation files")
    
    # Identify helpful champion model
    print("\n🤝 Identifying Helpful-Champion Model...")
    helpful_champion, df, baseline = identify_helpful_champion(data)
    
    print(f"✓ Helpful-Champion: {helpful_champion['model_name']}")
    print(f"✓ Helpfulness Score: {helpful_champion['average_helpfulness_score']:.4f}")
    print(f"✓ Helpfulness Change: {helpful_champion['helpfulness_change']:+.4f}")
    
    # Generate report
    print("\n📝 Generating report...")
    report = generate_helpful_champion_report(helpful_champion, df, baseline, data)
    
    # Save report
    with open('helpful_champion_report.md', 'w') as f:
        f.write(report)
    
    print("✓ Report saved to: helpful_champion_report.md")
    print("\n🎉 Helpful-Champion analysis complete!")
