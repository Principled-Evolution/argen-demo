#!/usr/bin/env python3
"""
Final Evaluations Data Consolidator

This script consolidates all Claude and Gemini evaluation data from the final-evals folder,
validates data integrity, and creates a unified dataset for analysis.

Usage: python final_evals_consolidator.py
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import numpy as np

class FinalEvalsConsolidator:
    def __init__(self, base_path: str = "../../"):
        """Initialize consolidator with base path to final-evals folder."""
        self.base_path = Path(base_path)
        self.claude_path = self.base_path / "claude-runs"
        self.gemini_path = self.base_path / "gemini-runs"
        self.data_path = Path("../data")
        
        # Ensure data directory exists
        self.data_path.mkdir(exist_ok=True)
        
        self.all_data = []
        self.validation_results = {
            'total_files': 0,
            'claude_files': 0,
            'gemini_files': 0,
            'baseline_files': 0,
            'argen_files': 0,
            'missing_files': [],
            'corrupted_files': [],
            'incomplete_evaluations': []
        }

    def load_evaluation_file(self, file_path: Path) -> Optional[Dict]:
        """Load and validate a single evaluation file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic validation
            if 'evaluation_config' not in data or 'summary_metrics' not in data:
                self.validation_results['corrupted_files'].append(str(file_path))
                return None
            
            # Check if evaluation is complete (100 scenarios expected)
            num_scenarios = data.get('evaluation_config', {}).get('num_scenarios', 0)
            if num_scenarios != 100:
                self.validation_results['incomplete_evaluations'].append({
                    'file': str(file_path),
                    'scenarios': num_scenarios
                })
            
            return data
            
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
            self.validation_results['corrupted_files'].append(str(file_path))
            return None

    def parse_model_info(self, model_name: str, file_path: str) -> Dict:
        """Parse seed, checkpoint, and model type information from model name and file path."""
        info = {
            'seed': None,
            'checkpoint': None,
            'model_type': 'unknown',
            'model_family': 'unknown'
        }
        
        # Handle baseline model
        if 'Llama-3.2-1B-Instruct' in model_name:
            info.update({
                'model_type': 'baseline',
                'model_family': 'baseline',
                'seed': 'baseline',
                'checkpoint': 'baseline'
            })
            return info
        
        # Parse ArGen models
        if 'grpo_5_seed_1' in model_name or 'grpo_5_seed_1' in file_path:
            info['seed'] = 1
            info['model_type'] = 'grpo5'
            info['model_family'] = 'argen'
        elif 'grpo_6_seed_2_4' in model_name or 'grpo_6_seed_2_4' in file_path:
            info['seed'] = 2
            info['model_type'] = 'grpo6'
            info['model_family'] = 'argen'
        elif 'grpo_7_seed_3_3' in model_name or 'grpo_7_seed_3_3' in file_path:
            info['seed'] = 3
            info['model_type'] = 'grpo7'
            info['model_family'] = 'argen'
        
        # Parse checkpoint information
        if 'checkpoint-' in file_path:
            try:
                checkpoint_str = file_path.split('checkpoint-')[1].split('_')[0]
                info['checkpoint'] = int(checkpoint_str)
            except (IndexError, ValueError):
                info['checkpoint'] = 'unknown'
        else:
            # Final model (no checkpoint specified)
            info['checkpoint'] = 'final'
        
        return info

    def extract_model_data(self, data: Dict, file_path: str) -> Dict:
        """Extract relevant metrics and metadata from evaluation data."""
        config = data['evaluation_config']
        metrics = data['summary_metrics']
        
        # Parse model information
        model_info = self.parse_model_info(config['model_name'], file_path)
        
        # Extract core data
        model_data = {
            # File metadata
            'file_path': file_path,
            'evaluator': config['evaluator'],
            'evaluator_type': 'claude' if 'claude' in config['evaluator'].lower() or 'anthropic' in config['evaluator'].lower() else 'gemini',
            'timestamp': config['timestamp'],
            'num_scenarios': config.get('num_scenarios', 0),
            
            # Model identification
            'model_name': config['model_name'],
            'model_type': model_info['model_type'],
            'model_family': model_info['model_family'],
            'seed': model_info['seed'],
            'checkpoint': model_info['checkpoint'],
            
            # Core performance metrics
            'average_combined_score': metrics['average_combined_score'],
            'average_ahimsa_score': metrics['average_ahimsa_score'],
            'average_dharma_score': metrics['average_dharma_score'],
            'average_helpfulness_score': metrics['average_helpfulness_score'],
            
            # Detailed helpfulness metrics
            'average_clarity_score': metrics['average_clarity_score'],
            'average_relevance_score': metrics['average_relevance_score'],
            'average_completeness_score': metrics['average_completeness_score'],
            
            # Violation rates
            'ahimsa_violation_rate': metrics['ahimsa_violation_rate'],
            'dharma_violation_rate': metrics['dharma_violation_rate'],
            'helpfulness_violation_rate': metrics['helpfulness_violation_rate'],
            
            # Additional metrics
            'ahimsa_violations': metrics['ahimsa_violations'],
            'dharma_violations': metrics['dharma_violations'],
            'helpfulness_violations': metrics['helpfulness_violations'],
        }
        
        return model_data

    def load_all_evaluations(self) -> List[Dict]:
        """Load all evaluation files from both Claude and Gemini directories."""
        print("📊 Loading evaluation files...")
        
        # Load Claude evaluations
        claude_files = list(self.claude_path.glob("*.json"))
        print(f"Found {len(claude_files)} Claude evaluation files")
        
        for file_path in sorted(claude_files):
            data = self.load_evaluation_file(file_path)
            if data:
                model_data = self.extract_model_data(data, str(file_path))
                self.all_data.append(model_data)
                self.validation_results['claude_files'] += 1
                
                if model_data['model_family'] == 'baseline':
                    self.validation_results['baseline_files'] += 1
                else:
                    self.validation_results['argen_files'] += 1
        
        # Load Gemini evaluations
        gemini_files = list(self.gemini_path.glob("*.json"))
        print(f"Found {len(gemini_files)} Gemini evaluation files")
        
        for file_path in sorted(gemini_files):
            data = self.load_evaluation_file(file_path)
            if data:
                model_data = self.extract_model_data(data, str(file_path))
                self.all_data.append(model_data)
                self.validation_results['gemini_files'] += 1
                
                if model_data['model_family'] == 'baseline':
                    self.validation_results['baseline_files'] += 1
                else:
                    self.validation_results['argen_files'] += 1
        
        self.validation_results['total_files'] = len(self.all_data)
        print(f"✓ Successfully loaded {len(self.all_data)} evaluation files")
        
        return self.all_data

    def validate_data_completeness(self) -> Dict:
        """Validate that all expected models and evaluations are present."""
        print("\n🔍 Validating data completeness...")
        
        # Expected models structure
        expected_models = {
            'baseline': ['baseline'],
            'seed_1': ['final', 1000, 2000, 2600, 3000],
            'seed_2': ['final', 1000, 2000, 3000, 4000],
            'seed_3': ['final', 1000, 2000, 3000, 4000]
        }
        
        # Check what we have
        found_models = {}
        for evaluator in ['claude', 'gemini']:
            found_models[evaluator] = {
                'baseline': set(),
                'seed_1': set(),
                'seed_2': set(), 
                'seed_3': set()
            }
        
        for item in self.all_data:
            evaluator = item['evaluator_type']
            if item['model_family'] == 'baseline':
                found_models[evaluator]['baseline'].add(item['checkpoint'])
            elif item['seed'] == 1:
                found_models[evaluator]['seed_1'].add(item['checkpoint'])
            elif item['seed'] == 2:
                found_models[evaluator]['seed_2'].add(item['checkpoint'])
            elif item['seed'] == 3:
                found_models[evaluator]['seed_3'].add(item['checkpoint'])
        
        # Report missing models
        missing_models = {}
        for evaluator in ['claude', 'gemini']:
            missing_models[evaluator] = {}
            
            # Check baseline
            if 'baseline' not in found_models[evaluator]['baseline']:
                missing_models[evaluator]['baseline'] = ['baseline']
            
            # Check each seed
            for seed_key, expected_checkpoints in [('seed_1', expected_models['seed_1']), 
                                                   ('seed_2', expected_models['seed_2']), 
                                                   ('seed_3', expected_models['seed_3'])]:
                missing = []
                for checkpoint in expected_checkpoints:
                    if checkpoint not in found_models[evaluator][seed_key]:
                        missing.append(checkpoint)
                if missing:
                    missing_models[evaluator][seed_key] = missing
        
        return {
            'found_models': found_models,
            'missing_models': missing_models,
            'expected_models': expected_models
        }

    def save_consolidated_data(self) -> None:
        """Save consolidated data to JSON and CSV formats."""
        print("\n💾 Saving consolidated data...")
        
        # Save as JSON
        json_path = self.data_path / "consolidated_final_evals.json"
        with open(json_path, 'w') as f:
            json.dump({
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_evaluations': len(self.all_data),
                    'validation_results': self.validation_results
                },
                'evaluations': self.all_data
            }, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(self.all_data)
        csv_path = self.data_path / "model_performance_matrix.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Saved JSON data to: {json_path}")
        print(f"✓ Saved CSV data to: {csv_path}")

    def generate_consolidation_report(self, completeness_check: Dict) -> str:
        """Generate a detailed consolidation report."""
        report = f"""# Final Evaluations Data Consolidation Report

## Summary Statistics

- **Total Evaluation Files**: {self.validation_results['total_files']}
- **Claude Evaluations**: {self.validation_results['claude_files']}
- **Gemini Evaluations**: {self.validation_results['gemini_files']}
- **Baseline Model Files**: {self.validation_results['baseline_files']}
- **ArGen Model Files**: {self.validation_results['argen_files']}

## Data Quality Assessment

### Successfully Loaded Files
- ✅ **{len(self.all_data)}** files loaded successfully

### Data Issues
- ❌ **Corrupted Files**: {len(self.validation_results['corrupted_files'])}
- ⚠️ **Incomplete Evaluations**: {len(self.validation_results['incomplete_evaluations'])}

"""
        
        if self.validation_results['corrupted_files']:
            report += "#### Corrupted Files:\n"
            for file in self.validation_results['corrupted_files']:
                report += f"- `{file}`\n"
            report += "\n"
        
        if self.validation_results['incomplete_evaluations']:
            report += "#### Incomplete Evaluations:\n"
            for item in self.validation_results['incomplete_evaluations']:
                report += f"- `{item['file']}` ({item['scenarios']} scenarios)\n"
            report += "\n"
        
        # Model completeness section
        report += "## Model Completeness Analysis\n\n"
        
        for evaluator in ['claude', 'gemini']:
            report += f"### {evaluator.title()} Evaluations\n\n"
            
            found = completeness_check['found_models'][evaluator]
            missing = completeness_check['missing_models'][evaluator]
            
            # Baseline
            baseline_status = "✅" if found['baseline'] else "❌"
            report += f"- **Baseline**: {baseline_status} {len(found['baseline'])} files\n"
            
            # Seeds
            for i, seed_key in enumerate(['seed_1', 'seed_2', 'seed_3'], 1):
                expected_count = len(completeness_check['expected_models'][seed_key])
                actual_count = len(found[seed_key])
                status = "✅" if actual_count == expected_count else "⚠️"
                report += f"- **Seed {i}**: {status} {actual_count}/{expected_count} files\n"
                
                if seed_key in missing and missing[seed_key]:
                    report += f"  - Missing: {missing[seed_key]}\n"
            
            report += "\n"
        
        # File provenance
        report += "## File Provenance\n\n"
        report += "### Claude Evaluations\n"
        report += f"- **Source Directory**: `{self.claude_path}`\n"
        report += f"- **File Count**: {self.validation_results['claude_files']}\n\n"
        
        report += "### Gemini Evaluations\n"
        report += f"- **Source Directory**: `{self.gemini_path}`\n"
        report += f"- **File Count**: {self.validation_results['gemini_files']}\n\n"
        
        report += f"""## Output Files

- **Consolidated JSON**: `../data/consolidated_final_evals.json`
- **Performance Matrix CSV**: `../data/model_performance_matrix.csv`

---
*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Base directory: `{self.base_path.absolute()}`*
"""
        
        return report

    def run_consolidation(self) -> None:
        """Run the complete consolidation process."""
        print("🚀 Starting Final Evaluations Data Consolidation")
        print("=" * 60)
        
        # Load all evaluation files
        self.load_all_evaluations()
        
        # Validate data completeness
        completeness_check = self.validate_data_completeness()
        
        # Save consolidated data
        self.save_consolidated_data()
        
        # Generate and save report
        report = self.generate_consolidation_report(completeness_check)
        report_path = Path("../reports/data_consolidation_report.md")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Consolidation report saved to: {report_path}")
        print("\n🎉 Data consolidation complete!")
        print(f"📊 Total evaluations processed: {len(self.all_data)}")

if __name__ == "__main__":
    consolidator = FinalEvalsConsolidator()
    consolidator.run_consolidation()
