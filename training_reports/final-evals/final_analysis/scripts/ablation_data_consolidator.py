#!/usr/bin/env python3
"""
Ablation Data Consolidator - Final Evaluations

This script consolidates reward-only and policy-only ablation evaluation data
from both Claude and Gemini evaluators for statistical analysis.

Usage: python ablation_data_consolidator.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class AblationDataConsolidator:
    def __init__(self):
        """Initialize the ablation data consolidator."""
        self.base_path = Path("../")
        self.data_path = Path("../data")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        # Ablation data paths
        self.claude_ablations_path = self.base_path / ".." / "reward-policy-ablations-claude"
        self.gemini_ablations_path = self.base_path / "reward-policy-ablations-gemini"
        
        self.consolidated_data = None
        self.ablation_evaluations = []

    def load_ablation_files(self) -> None:
        """Load all ablation evaluation files."""
        print("📊 Loading ablation evaluation files...")
        import sys
        sys.stdout.flush()
        
        # Expected ablation files
        ablation_files = {
            'claude_reward_only': self.claude_ablations_path / "eval_seed_3_108_ablation_reward_only_benchmarking_20250510_135534-cleanprep-hashprompt.json",
            'claude_policy_only': self.claude_ablations_path / "eval_seed_3_108_ablation_policy_only_benchmarking_20250510_135534-cleanprep-hashprompt.json",
            'gemini_reward_only': self.gemini_ablations_path / "eval_seed_3_108_ablation_reward_only_benchmarking_20250510_135534-cleanprep-hashprompt.json",
            'gemini_policy_only': self.gemini_ablations_path / "eval_seed_3_108_ablation_policy_only_benchmarking_20250510_135534-cleanprep-hashprompt.json"
        }
        
        loaded_files = {}
        for key, file_path in ablation_files.items():
            if file_path.exists():
                print(f"✓ Loading {key}: {file_path.name}")
                with open(file_path, 'r') as f:
                    loaded_files[key] = json.load(f)
            else:
                print(f"✗ Missing {key}: {file_path}")
        
        if len(loaded_files) != 4:
            raise FileNotFoundError(f"Expected 4 ablation files, found {len(loaded_files)}")
        
        self.ablation_files = loaded_files
        print(f"✓ Loaded {len(loaded_files)} ablation evaluation files")

    def extract_evaluation_data(self, eval_data: Dict, evaluator_type: str, ablation_type: str) -> Dict:
        """Extract key metrics from evaluation data."""
        config = eval_data['evaluation_config']
        summary = eval_data['summary_metrics']
        
        return {
            'evaluator_type': evaluator_type,
            'evaluator': config['evaluator'],
            'ablation_type': ablation_type,
            'model_name': config['model_name'],
            'num_scenarios': config['num_scenarios'],
            'timestamp': config['timestamp'],
            
            # Core metrics
            'average_combined_score': summary['average_combined_score'],
            'average_ahimsa_score': summary['average_ahimsa_score'],
            'average_dharma_score': summary['average_dharma_score'],
            'average_helpfulness_score': summary['average_helpfulness_score'],
            
            # Violation rates
            'ahimsa_violation_rate': summary['ahimsa_violation_rate'],
            'dharma_violation_rate': summary['dharma_violation_rate'],
            'helpfulness_violation_rate': summary['helpfulness_violation_rate'],
            
            # Additional metrics
            'average_clarity_score': summary['average_clarity_score'],
            'average_relevance_score': summary['average_relevance_score'],
            'average_completeness_score': summary['average_completeness_score'],
            'average_scope_penalty_factor': summary['average_scope_penalty_factor'],
            'severe_scope_penalty_rate': summary['severe_scope_penalty_rate'],
            
            # Individual results for detailed analysis
            'individual_results': eval_data['individual_results']
        }

    def consolidate_ablation_data(self) -> None:
        """Consolidate all ablation evaluation data."""
        print("🔄 Consolidating ablation evaluation data...")
        
        evaluations = []
        
        # Process each ablation file
        file_mapping = {
            'claude_reward_only': ('claude', 'reward_only'),
            'claude_policy_only': ('claude', 'policy_only'),
            'gemini_reward_only': ('gemini', 'reward_only'),
            'gemini_policy_only': ('gemini', 'policy_only')
        }
        
        for file_key, (evaluator, ablation_type) in file_mapping.items():
            eval_data = self.extract_evaluation_data(
                self.ablation_files[file_key], 
                evaluator, 
                ablation_type
            )
            evaluations.append(eval_data)
            print(f"✓ Processed {evaluator} {ablation_type}: {eval_data['average_combined_score']:.4f}")
        
        self.ablation_evaluations = evaluations

    def load_median_seed_baselines(self) -> Dict:
        """Load the median seed baselines for both evaluators."""
        print("📈 Loading median seed baselines...")

        baselines = {'claude': None, 'gemini': None}

        # Load consolidated data to find median seed baseline
        consolidated_path = self.data_path / "consolidated_final_evals.json"
        if not consolidated_path.exists():
            print("⚠️  Consolidated data not found. Run final_evals_consolidator.py first.")
            return baselines

        with open(consolidated_path, 'r') as f:
            consolidated_data = json.load(f)

        # Find median seed baselines (Seed 3, GRPO7, checkpoint-3000) for both evaluators
        for eval in consolidated_data['evaluations']:
            if (eval['model_family'] == 'argen' and
                eval['seed'] == 3 and
                eval['model_type'] == 'grpo7' and
                eval['checkpoint'] == 3000):

                evaluator = eval['evaluator_type']
                if evaluator in baselines:
                    baselines[evaluator] = eval
                    print(f"✓ Found {evaluator} baseline: {eval['average_combined_score']:.4f}")

        if baselines['claude'] is None or baselines['gemini'] is None:
            missing = [k for k, v in baselines.items() if v is None]
            print(f"⚠️  Missing baselines for: {', '.join(missing)}")

        return baselines

    def validate_data_completeness(self) -> Dict:
        """Validate completeness and consistency of ablation data."""
        print("🔍 Validating data completeness...")
        
        validation_results = {
            'total_evaluations': len(self.ablation_evaluations),
            'evaluators': set(),
            'ablation_types': set(),
            'scenario_counts': {},
            'missing_data': [],
            'data_quality': {}
        }
        
        for eval in self.ablation_evaluations:
            evaluator = eval['evaluator_type']
            ablation_type = eval['ablation_type']
            key = f"{evaluator}_{ablation_type}"
            
            validation_results['evaluators'].add(evaluator)
            validation_results['ablation_types'].add(ablation_type)
            validation_results['scenario_counts'][key] = eval['num_scenarios']
            
            # Check for complete 100-scenario evaluations
            if eval['num_scenarios'] != 100:
                validation_results['missing_data'].append(f"{key}: {eval['num_scenarios']} scenarios")
            
            # Check data quality
            validation_results['data_quality'][key] = {
                'has_individual_results': len(eval['individual_results']) == eval['num_scenarios'],
                'combined_score_valid': 0 <= eval['average_combined_score'] <= 1,
                'all_metrics_present': all(key in eval for key in [
                    'average_ahimsa_score', 'average_dharma_score', 'average_helpfulness_score'
                ])
            }
        
        # Summary
        validation_results['evaluators'] = list(validation_results['evaluators'])
        validation_results['ablation_types'] = list(validation_results['ablation_types'])
        validation_results['is_complete'] = len(validation_results['missing_data']) == 0
        validation_results['expected_evaluations'] = 4  # 2 evaluators × 2 ablation types
        
        print(f"✓ Validation complete: {validation_results['total_evaluations']}/4 evaluations")
        if validation_results['missing_data']:
            print(f"⚠️  Issues found: {validation_results['missing_data']}")
        
        return validation_results

    def save_consolidated_data(self) -> None:
        """Save consolidated ablation data."""
        print("💾 Saving consolidated ablation data...")

        # Load baselines for inclusion
        baselines = self.load_median_seed_baselines()

        consolidated_data = {
            'metadata': {
                'consolidation_timestamp': datetime.now().isoformat(),
                'total_ablation_evaluations': len(self.ablation_evaluations),
                'evaluators': list(set(eval['evaluator_type'] for eval in self.ablation_evaluations)),
                'ablation_types': list(set(eval['ablation_type'] for eval in self.ablation_evaluations)),
                'baselines_included': {k: v is not None for k, v in baselines.items()}
            },
            'baselines': baselines,
            'ablation_evaluations': self.ablation_evaluations,
            'validation': self.validation_results
        }
        
        # Save JSON
        output_path = self.data_path / "consolidated_ablation_evals.json"
        with open(output_path, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        
        print(f"✓ Consolidated data saved to: {output_path}")

    def generate_consolidation_report(self) -> str:
        """Generate data consolidation report."""
        baselines = self.load_median_seed_baselines()
        
        report = f"""# Ablation Data Consolidation Report

## Executive Summary

**Consolidation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Ablation Evaluations**: {len(self.ablation_evaluations)}
- **Evaluators**: {', '.join(set(eval['evaluator_type'] for eval in self.ablation_evaluations))}
- **Ablation Types**: {', '.join(set(eval['ablation_type'] for eval in self.ablation_evaluations))}
- **Baseline Reference**: {'✓ Loaded' if any(baselines.values()) else '✗ Missing'}

## Data Summary

### Ablation Evaluations
| Evaluator | Ablation Type | Combined Score | Scenarios | Status |
|-----------|---------------|----------------|-----------|--------|
"""
        
        for eval in sorted(self.ablation_evaluations, key=lambda x: (x['evaluator_type'], x['ablation_type'])):
            status = "✓ Complete" if eval['num_scenarios'] == 100 else f"⚠️  {eval['num_scenarios']} scenarios"
            report += f"| {eval['evaluator_type'].title()} | {eval['ablation_type'].replace('_', '-').title()} | {eval['average_combined_score']:.4f} | {eval['num_scenarios']} | {status} |\n"
        
        if any(baselines.values()):
            report += f"""
### Baseline Reference (Median Seed)
"""
            for evaluator, baseline in baselines.items():
                if baseline:
                    report += f"- **{evaluator.title()}**: {baseline['model_type'].upper()} Seed {baseline['seed']} ({baseline['checkpoint']}) - {baseline['average_combined_score']:.4f}\n"

            report += f"""- **Purpose**: Representative baseline for ablation comparisons

"""
        
        report += f"""## Validation Results

### Data Completeness
- **Expected Evaluations**: 4 (2 evaluators × 2 ablation types)
- **Found Evaluations**: {self.validation_results['total_evaluations']}
- **Completeness**: {'✓ Complete' if self.validation_results['is_complete'] else '⚠️  Issues found'}

### Data Quality Checks
"""
        
        for key, quality in self.validation_results['data_quality'].items():
            evaluator, ablation = key.split('_', 1)
            report += f"""
#### {evaluator.title()} {ablation.replace('_', '-').title()}
- Individual Results: {'✓' if quality['has_individual_results'] else '✗'}
- Valid Combined Score: {'✓' if quality['combined_score_valid'] else '✗'}
- All Metrics Present: {'✓' if quality['all_metrics_present'] else '✗'}
"""
        
        if self.validation_results['missing_data']:
            report += f"""
### Issues Found
{chr(10).join(f'- {issue}' for issue in self.validation_results['missing_data'])}
"""
        
        report += f"""
## Next Steps

1. **Statistical Analysis**: Run `ablation_statistical_analysis.py`
2. **Comparative Analysis**: Compare ablations against median seed baseline
3. **Cross-Evaluator Validation**: Analyze Claude vs Gemini consistency
4. **Report Generation**: Generate comprehensive ablation reports

## Data Files Generated

- `consolidated_ablation_evals.json`: Unified ablation dataset
- `ablation_data_consolidation_report.md`: This report

---
*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*Script*: `ablation_data_consolidator.py`  
*Data Sources*: Claude and Gemini ablation evaluations
"""
        
        return report

    def run_consolidation(self) -> None:
        """Run the complete ablation data consolidation."""
        print("🚀 Starting Ablation Data Consolidation")
        print("=" * 50)
        
        # Load ablation files
        self.load_ablation_files()
        
        # Consolidate data
        self.consolidate_ablation_data()
        
        # Validate data
        self.validation_results = self.validate_data_completeness()
        
        # Save consolidated data
        self.save_consolidated_data()
        
        # Generate report
        report = self.generate_consolidation_report()
        report_path = self.reports_path / "ablation_data_consolidation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Consolidation report saved to: {report_path}")
        print(f"\n🎉 Ablation data consolidation complete!")
        print(f"📊 Ready for statistical analysis")

if __name__ == "__main__":
    print("Starting ablation data consolidation...")
    try:
        consolidator = AblationDataConsolidator()
        consolidator.run_consolidation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
