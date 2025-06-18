#!/usr/bin/env python3
"""
Ablation Statistical Analysis - Final Evaluations

This script performs comprehensive statistical analysis of reward-only and 
policy-only ablations compared to the median seed baseline.

Usage: python ablation_statistical_analysis.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats

class AblationStatisticalAnalyzer:
    def __init__(self):
        """Initialize the ablation statistical analyzer."""
        self.data_path = Path("../data")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        self.consolidated_data = None
        self.baseline = None
        self.ablations = {}
        self.statistical_results = {}

    def load_consolidated_data(self) -> None:
        """Load consolidated ablation data."""
        print("📊 Loading consolidated ablation data...")
        
        data_path = self.data_path / "consolidated_ablation_evals.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Consolidated ablation data not found at {data_path}. Run ablation_data_consolidator.py first.")
        
        with open(data_path, 'r') as f:
            self.consolidated_data = json.load(f)
        
        self.baselines = self.consolidated_data['baselines']
        
        # Organize ablations by type and evaluator
        for eval in self.consolidated_data['ablation_evaluations']:
            evaluator = eval['evaluator_type']
            ablation_type = eval['ablation_type']
            
            if ablation_type not in self.ablations:
                self.ablations[ablation_type] = {}
            
            self.ablations[ablation_type][evaluator] = eval
        
        print(f"✓ Loaded baseline and {len(self.consolidated_data['ablation_evaluations'])} ablation evaluations")

    def calculate_effect_size(self, score1: float, score2: float, pooled_std: float = 0.1) -> Dict:
        """Calculate Cohen's d effect size and interpretation."""
        cohens_d = (score1 - score2) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'cohens_d': cohens_d,
            'interpretation': interpretation,
            'magnitude': abs(cohens_d)
        }

    def compare_ablation_to_baseline(self, ablation_data: Dict, evaluator: str) -> Dict:
        """Compare ablation performance to median seed baseline."""
        if evaluator not in self.baselines or not self.baselines[evaluator]:
            return {}

        baseline = self.baselines[evaluator]
        metrics = ['combined', 'ahimsa', 'dharma', 'helpfulness']
        comparison = {}

        for metric in metrics:
            ablation_score = ablation_data[f'average_{metric}_score']
            baseline_score = baseline[f'average_{metric}_score']

            difference = ablation_score - baseline_score
            relative_change = (difference / baseline_score) * 100
            effect_size = self.calculate_effect_size(ablation_score, baseline_score)

            comparison[metric] = {
                'ablation_score': ablation_score,
                'baseline_score': baseline_score,
                'difference': difference,
                'relative_change': relative_change,
                'effect_size': effect_size
            }

        return comparison

    def compare_ablation_types(self, evaluator: str) -> Dict:
        """Compare reward-only vs policy-only ablations."""
        if evaluator not in self.ablations['reward_only'] or evaluator not in self.ablations['policy_only']:
            return {}
        
        reward_only = self.ablations['reward_only'][evaluator]
        policy_only = self.ablations['policy_only'][evaluator]
        
        metrics = ['combined', 'ahimsa', 'dharma', 'helpfulness']
        comparison = {}
        
        for metric in metrics:
            reward_score = reward_only[f'average_{metric}_score']
            policy_score = policy_only[f'average_{metric}_score']
            
            difference = reward_score - policy_score
            relative_difference = (difference / policy_score) * 100 if policy_score != 0 else 0
            effect_size = self.calculate_effect_size(reward_score, policy_score)
            
            comparison[metric] = {
                'reward_only_score': reward_score,
                'policy_only_score': policy_score,
                'difference': difference,
                'relative_difference': relative_difference,
                'effect_size': effect_size,
                'better_performer': 'reward_only' if reward_score > policy_score else 'policy_only'
            }
        
        return comparison

    def analyze_cross_evaluator_consistency(self) -> Dict:
        """Analyze consistency between Claude and Gemini evaluations."""
        consistency_results = {}

        for ablation_type in ['reward_only', 'policy_only']:
            if 'claude' not in self.ablations[ablation_type] or 'gemini' not in self.ablations[ablation_type]:
                continue

            claude_data = self.ablations[ablation_type]['claude']
            gemini_data = self.ablations[ablation_type]['gemini']

            metrics = ['combined', 'ahimsa', 'dharma', 'helpfulness']
            ablation_consistency = {}

            # Calculate correlations using individual scenario scores
            claude_individual = claude_data['individual_results']
            gemini_individual = gemini_data['individual_results']

            for metric in metrics:
                claude_score = claude_data[f'average_{metric}_score']
                gemini_score = gemini_data[f'average_{metric}_score']

                difference = claude_score - gemini_score
                relative_difference = (difference / gemini_score) * 100 if gemini_score != 0 else 0

                # Calculate correlation from individual scores if available
                correlation = None
                if len(claude_individual) == len(gemini_individual) == 100:
                    try:
                        claude_scores = [r[f'{metric}_score'] for r in claude_individual]
                        gemini_scores = [r[f'{metric}_score'] for r in gemini_individual]
                        correlation, _ = stats.pearsonr(claude_scores, gemini_scores)
                    except (KeyError, ValueError):
                        correlation = None

                ablation_consistency[metric] = {
                    'claude_score': claude_score,
                    'gemini_score': gemini_score,
                    'difference': difference,
                    'relative_difference': relative_difference,
                    'correlation': correlation,
                    'agreement_level': 'high' if abs(relative_difference) < 5 else 'medium' if abs(relative_difference) < 10 else 'low'
                }

            consistency_results[ablation_type] = ablation_consistency

        return consistency_results

    def calculate_performance_degradation(self) -> Dict:
        """Calculate performance degradation from baseline for each ablation."""
        degradation_results = {}

        for ablation_type in self.ablations:
            degradation_results[ablation_type] = {}

            for evaluator in self.ablations[ablation_type]:
                if evaluator not in self.baselines or not self.baselines[evaluator]:
                    continue

                baseline_combined = self.baselines[evaluator]['average_combined_score']
                ablation_combined = self.ablations[ablation_type][evaluator]['average_combined_score']

                degradation = baseline_combined - ablation_combined
                degradation_percent = (degradation / baseline_combined) * 100

                degradation_results[ablation_type][evaluator] = {
                    'baseline_score': baseline_combined,
                    'ablation_score': ablation_combined,
                    'degradation': degradation,
                    'degradation_percent': degradation_percent,
                    'performance_retention': ((ablation_combined / baseline_combined) * 100)
                }

        return degradation_results

    def run_statistical_analysis(self) -> None:
        """Run comprehensive statistical analysis."""
        print("📈 Running statistical analysis...")
        
        # Baseline comparisons
        print("  Analyzing ablation vs baseline comparisons...")
        baseline_comparisons = {}
        for ablation_type in self.ablations:
            baseline_comparisons[ablation_type] = {}
            for evaluator in self.ablations[ablation_type]:
                baseline_comparisons[ablation_type][evaluator] = self.compare_ablation_to_baseline(
                    self.ablations[ablation_type][evaluator], evaluator
                )
        
        # Ablation type comparisons
        print("  Comparing reward-only vs policy-only...")
        ablation_comparisons = {}
        for evaluator in ['claude', 'gemini']:
            ablation_comparisons[evaluator] = self.compare_ablation_types(evaluator)
        
        # Cross-evaluator consistency
        print("  Analyzing cross-evaluator consistency...")
        consistency_analysis = self.analyze_cross_evaluator_consistency()
        
        # Performance degradation
        print("  Calculating performance degradation...")
        degradation_analysis = self.calculate_performance_degradation()
        
        # Store results
        self.statistical_results = {
            'baseline_comparisons': baseline_comparisons,
            'ablation_comparisons': ablation_comparisons,
            'consistency_analysis': consistency_analysis,
            'degradation_analysis': degradation_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        print("✓ Statistical analysis complete")

    def save_analysis_results(self) -> None:
        """Save statistical analysis results."""
        print("💾 Saving analysis results...")
        
        # Save detailed results as JSON
        results_path = self.data_path / "ablation_statistical_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.statistical_results, f, indent=2)
        
        # Create performance matrix CSV
        self.create_performance_matrix()
        
        print(f"✓ Analysis results saved to: {results_path}")

    def create_performance_matrix(self) -> None:
        """Create performance comparison matrix CSV."""
        matrix_data = []

        # Add baselines
        for evaluator, baseline in self.baselines.items():
            if baseline:
                matrix_data.append({
                    'Model_Type': 'Baseline',
                    'Evaluator': evaluator.title(),
                    'Ablation_Type': 'Full_Model',
                    'Combined_Score': baseline['average_combined_score'],
                    'Ahimsa_Score': baseline['average_ahimsa_score'],
                    'Dharma_Score': baseline['average_dharma_score'],
                    'Helpfulness_Score': baseline['average_helpfulness_score']
                })
        
        # Add ablations
        for ablation_type in self.ablations:
            for evaluator in self.ablations[ablation_type]:
                data = self.ablations[ablation_type][evaluator]
                matrix_data.append({
                    'Model_Type': 'Ablation',
                    'Evaluator': evaluator.title(),
                    'Ablation_Type': ablation_type.replace('_', '_').title(),
                    'Combined_Score': data['average_combined_score'],
                    'Ahimsa_Score': data['average_ahimsa_score'],
                    'Dharma_Score': data['average_dharma_score'],
                    'Helpfulness_Score': data['average_helpfulness_score']
                })
        
        df = pd.DataFrame(matrix_data)
        matrix_path = self.data_path / "ablation_performance_matrix.csv"
        df.to_csv(matrix_path, index=False)
        
        print(f"✓ Performance matrix saved to: {matrix_path}")

    def run_analysis(self) -> None:
        """Run the complete ablation statistical analysis."""
        print("🔬 Starting Ablation Statistical Analysis")
        print("=" * 50)
        
        # Load data
        self.load_consolidated_data()
        
        # Run statistical analysis
        self.run_statistical_analysis()
        
        # Save results
        self.save_analysis_results()
        
        print(f"\n🎉 Statistical analysis complete!")
        print(f"📊 Results ready for report generation")

if __name__ == "__main__":
    print("Starting ablation statistical analysis...")
    try:
        analyzer = AblationStatisticalAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
