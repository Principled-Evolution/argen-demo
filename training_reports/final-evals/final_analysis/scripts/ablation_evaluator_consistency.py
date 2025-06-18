#!/usr/bin/env python3
"""
Ablation Evaluator Consistency Analysis - Final Evaluations

This script analyzes consistency between Claude and Gemini evaluations
for reward-only and policy-only ablations.

Usage: python ablation_evaluator_consistency.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr

class AblationEvaluatorConsistencyAnalyzer:
    def __init__(self):
        """Initialize the evaluator consistency analyzer."""
        self.data_path = Path("../data")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        self.consolidated_data = None
        self.ablations = {}
        self.consistency_results = {}

    def load_consolidated_data(self) -> None:
        """Load consolidated ablation data."""
        print("📊 Loading consolidated ablation data...")
        
        data_path = self.data_path / "consolidated_ablation_evals.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Consolidated data not found. Run ablation_data_consolidator.py first.")
        
        with open(data_path, 'r') as f:
            self.consolidated_data = json.load(f)
        
        # Organize ablations by type and evaluator
        for eval in self.consolidated_data['ablation_evaluations']:
            evaluator = eval['evaluator_type']
            ablation_type = eval['ablation_type']
            
            if ablation_type not in self.ablations:
                self.ablations[ablation_type] = {}
            
            self.ablations[ablation_type][evaluator] = eval
        
        print(f"✓ Loaded {len(self.consolidated_data['ablation_evaluations'])} ablation evaluations")

    def calculate_scenario_level_correlations(self, claude_data: Dict, gemini_data: Dict) -> Dict:
        """Calculate correlations at individual scenario level."""
        claude_results = claude_data['individual_results']
        gemini_results = gemini_data['individual_results']
        
        if len(claude_results) != len(gemini_results):
            return {'error': 'Mismatched scenario counts'}
        
        metrics = ['combined_score', 'ahimsa_score', 'dharma_score', 'helpfulness_score']
        correlations = {}
        
        for metric in metrics:
            try:
                claude_scores = [r[metric] for r in claude_results]
                gemini_scores = [r[metric] for r in gemini_results]
                
                # Calculate correlations
                pearson_r, pearson_p = pearsonr(claude_scores, gemini_scores)
                spearman_r, spearman_p = spearmanr(claude_scores, gemini_scores)
                
                # Calculate mean absolute difference
                differences = [abs(c - g) for c, g in zip(claude_scores, gemini_scores)]
                mean_abs_diff = np.mean(differences)
                
                correlations[metric] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'mean_abs_diff': mean_abs_diff,
                    'claude_mean': np.mean(claude_scores),
                    'gemini_mean': np.mean(gemini_scores),
                    'claude_std': np.std(claude_scores),
                    'gemini_std': np.std(gemini_scores)
                }
                
            except (KeyError, ValueError) as e:
                correlations[metric] = {'error': str(e)}
        
        return correlations

    def analyze_systematic_bias(self, claude_data: Dict, gemini_data: Dict) -> Dict:
        """Analyze systematic differences between evaluators."""
        claude_results = claude_data['individual_results']
        gemini_results = gemini_data['individual_results']
        
        if len(claude_results) != len(gemini_results):
            return {'error': 'Mismatched scenario counts'}
        
        metrics = ['combined_score', 'ahimsa_score', 'dharma_score', 'helpfulness_score']
        bias_analysis = {}
        
        for metric in metrics:
            try:
                claude_scores = [r[metric] for r in claude_results]
                gemini_scores = [r[metric] for r in gemini_results]
                
                # Calculate differences (Claude - Gemini)
                differences = [c - g for c, g in zip(claude_scores, gemini_scores)]
                
                # Statistical test for systematic bias
                t_stat, t_p = stats.ttest_1samp(differences, 0)
                
                bias_analysis[metric] = {
                    'mean_difference': np.mean(differences),
                    'std_difference': np.std(differences),
                    'median_difference': np.median(differences),
                    'min_difference': np.min(differences),
                    'max_difference': np.max(differences),
                    't_statistic': t_stat,
                    't_p_value': t_p,
                    'significant_bias': t_p < 0.05,
                    'bias_direction': 'claude_higher' if np.mean(differences) > 0 else 'gemini_higher'
                }
                
            except (KeyError, ValueError) as e:
                bias_analysis[metric] = {'error': str(e)}
        
        return bias_analysis

    def analyze_ranking_agreement(self, claude_data: Dict, gemini_data: Dict) -> Dict:
        """Analyze ranking agreement between evaluators."""
        claude_results = claude_data['individual_results']
        gemini_results = gemini_data['individual_results']
        
        if len(claude_results) != len(gemini_results):
            return {'error': 'Mismatched scenario counts'}
        
        metrics = ['combined_score', 'ahimsa_score', 'dharma_score', 'helpfulness_score']
        ranking_analysis = {}
        
        for metric in metrics:
            try:
                claude_scores = [r[metric] for r in claude_results]
                gemini_scores = [r[metric] for r in gemini_results]
                
                # Create rankings (higher score = better rank)
                claude_ranks = stats.rankdata([-s for s in claude_scores])  # Negative for descending
                gemini_ranks = stats.rankdata([-s for s in gemini_scores])
                
                # Calculate rank correlation
                rank_correlation, rank_p = spearmanr(claude_ranks, gemini_ranks)
                
                # Calculate rank differences
                rank_differences = [abs(c - g) for c, g in zip(claude_ranks, gemini_ranks)]
                mean_rank_diff = np.mean(rank_differences)
                
                ranking_analysis[metric] = {
                    'rank_correlation': rank_correlation,
                    'rank_p_value': rank_p,
                    'mean_rank_difference': mean_rank_diff,
                    'max_rank_difference': np.max(rank_differences),
                    'agreement_strength': 'strong' if rank_correlation > 0.7 else 'moderate' if rank_correlation > 0.5 else 'weak'
                }
                
            except (KeyError, ValueError) as e:
                ranking_analysis[metric] = {'error': str(e)}
        
        return ranking_analysis

    def run_consistency_analysis(self) -> None:
        """Run comprehensive evaluator consistency analysis."""
        print("🔍 Running evaluator consistency analysis...")
        
        for ablation_type in ['reward_only', 'policy_only']:
            if 'claude' not in self.ablations[ablation_type] or 'gemini' not in self.ablations[ablation_type]:
                print(f"⚠️  Skipping {ablation_type}: missing evaluator data")
                continue
            
            print(f"  Analyzing {ablation_type} consistency...")
            
            claude_data = self.ablations[ablation_type]['claude']
            gemini_data = self.ablations[ablation_type]['gemini']
            
            # Scenario-level correlations
            correlations = self.calculate_scenario_level_correlations(claude_data, gemini_data)
            
            # Systematic bias analysis
            bias_analysis = self.analyze_systematic_bias(claude_data, gemini_data)
            
            # Ranking agreement
            ranking_analysis = self.analyze_ranking_agreement(claude_data, gemini_data)
            
            self.consistency_results[ablation_type] = {
                'correlations': correlations,
                'bias_analysis': bias_analysis,
                'ranking_analysis': ranking_analysis,
                'summary_comparison': {
                    'claude_combined': claude_data['average_combined_score'],
                    'gemini_combined': gemini_data['average_combined_score'],
                    'summary_difference': claude_data['average_combined_score'] - gemini_data['average_combined_score']
                }
            }
        
        print("✓ Consistency analysis complete")

    def generate_consistency_report(self) -> str:
        """Generate comprehensive evaluator consistency report."""
        report = f"""# Ablation Evaluator Consistency Analysis Report

## Executive Summary

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Evaluators Compared**: Claude-3.5-Sonnet vs Gemini-2.0-Flash
- **Ablation Types**: Reward-Only and Policy-Only
- **Analysis Level**: Individual scenario correlations and systematic bias

## Summary Comparison

### Combined Score Agreement
| Ablation Type | Claude Score | Gemini Score | Difference | Agreement |
|---------------|--------------|--------------|------------|-----------|
"""
        
        for ablation_type in self.consistency_results:
            summary = self.consistency_results[ablation_type]['summary_comparison']
            diff = summary['summary_difference']
            agreement = 'High' if abs(diff) < 0.02 else 'Medium' if abs(diff) < 0.05 else 'Low'
            
            report += f"| {ablation_type.replace('_', '-').title()} | {summary['claude_combined']:.4f} | {summary['gemini_combined']:.4f} | {diff:+.4f} | {agreement} |\n"
        
        report += """
## Detailed Consistency Analysis

"""
        
        for ablation_type in self.consistency_results:
            results = self.consistency_results[ablation_type]
            
            report += f"""### {ablation_type.replace('_', '-').title()} Ablation

#### Scenario-Level Correlations
| Metric | Pearson r | Spearman r | Mean Abs Diff | Agreement Strength |
|--------|-----------|------------|---------------|-------------------|
"""
            
            correlations = results['correlations']
            for metric in ['combined_score', 'ahimsa_score', 'dharma_score', 'helpfulness_score']:
                if metric in correlations and 'error' not in correlations[metric]:
                    corr = correlations[metric]
                    strength = 'Strong' if corr['pearson_r'] > 0.7 else 'Moderate' if corr['pearson_r'] > 0.5 else 'Weak'
                    report += f"| {metric.replace('_score', '').title()} | {corr['pearson_r']:.3f} | {corr['spearman_r']:.3f} | {corr['mean_abs_diff']:.4f} | {strength} |\n"
            
            report += f"""
#### Systematic Bias Analysis
| Metric | Mean Difference | Significant Bias | Bias Direction |
|--------|-----------------|------------------|----------------|
"""
            
            bias = results['bias_analysis']
            for metric in ['combined_score', 'ahimsa_score', 'dharma_score', 'helpfulness_score']:
                if metric in bias and 'error' not in bias[metric]:
                    b = bias[metric]
                    sig_bias = 'Yes' if b['significant_bias'] else 'No'
                    direction = b['bias_direction'].replace('_', ' ').title()
                    report += f"| {metric.replace('_score', '').title()} | {b['mean_difference']:+.4f} | {sig_bias} | {direction} |\n"
            
            report += f"""
#### Ranking Agreement
| Metric | Rank Correlation | Agreement Strength |
|--------|------------------|--------------------|
"""
            
            ranking = results['ranking_analysis']
            for metric in ['combined_score', 'ahimsa_score', 'dharma_score', 'helpfulness_score']:
                if metric in ranking and 'error' not in ranking[metric]:
                    r = ranking[metric]
                    report += f"| {metric.replace('_score', '').title()} | {r['rank_correlation']:.3f} | {r['agreement_strength'].title()} |\n"
        
        report += f"""
## Key Findings

### Overall Consistency
"""
        
        # Calculate overall consistency metrics
        all_pearson_rs = []
        significant_biases = 0
        total_metrics = 0
        
        for ablation_type in self.consistency_results:
            correlations = self.consistency_results[ablation_type]['correlations']
            bias = self.consistency_results[ablation_type]['bias_analysis']
            
            for metric in ['combined_score', 'ahimsa_score', 'dharma_score', 'helpfulness_score']:
                if metric in correlations and 'error' not in correlations[metric]:
                    all_pearson_rs.append(correlations[metric]['pearson_r'])
                
                if metric in bias and 'error' not in bias[metric]:
                    total_metrics += 1
                    if bias[metric]['significant_bias']:
                        significant_biases += 1
        
        avg_correlation = np.mean(all_pearson_rs) if all_pearson_rs else 0
        bias_rate = (significant_biases / total_metrics * 100) if total_metrics > 0 else 0
        
        report += f"""- **Average Correlation**: {avg_correlation:.3f} across all metrics
- **Systematic Bias Rate**: {bias_rate:.1f}% of metrics show significant bias
- **Overall Assessment**: {'High' if avg_correlation > 0.7 and bias_rate < 25 else 'Moderate' if avg_correlation > 0.5 else 'Low'} consistency

### Implications for Research
1. **Cross-Evaluator Validation**: {'Strong' if avg_correlation > 0.7 else 'Moderate' if avg_correlation > 0.5 else 'Weak'} agreement supports robustness of findings
2. **Bias Considerations**: {significant_biases} out of {total_metrics} metrics show systematic evaluator differences
3. **Recommendation**: {'Use both evaluators for validation' if avg_correlation > 0.5 else 'Consider evaluator-specific analysis'}

## Methodology

### Analysis Approach
1. **Scenario-Level Analysis**: Individual scenario score correlations
2. **Systematic Bias Testing**: Statistical tests for consistent evaluator differences  
3. **Ranking Agreement**: Spearman correlation of scenario rankings
4. **Statistical Significance**: p < 0.05 threshold for bias detection

### Data Sources
- **Claude Evaluations**: {len(self.ablations['reward_only']['claude']['individual_results']) if 'reward_only' in self.ablations and 'claude' in self.ablations['reward_only'] else 'N/A'} scenarios per ablation
- **Gemini Evaluations**: {len(self.ablations['reward_only']['gemini']['individual_results']) if 'reward_only' in self.ablations and 'gemini' in self.ablations['reward_only'] else 'N/A'} scenarios per ablation
- **Evaluation Protocol**: Same 100-scenario benchmark for both evaluators

---
*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*Analysis Script*: `ablation_evaluator_consistency.py`  
*Data Source*: `consolidated_ablation_evals.json`
"""
        
        return report

    def save_consistency_results(self) -> None:
        """Save consistency analysis results."""
        print("💾 Saving consistency analysis results...")

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj

        serializable_results = convert_numpy_types(self.consistency_results)

        # Save detailed results as JSON
        results_path = self.data_path / "ablation_consistency_results.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Consistency results saved to: {results_path}")

    def run_analysis(self) -> None:
        """Run the complete evaluator consistency analysis."""
        print("🔍 Starting Ablation Evaluator Consistency Analysis")
        print("=" * 60)
        
        # Load data
        self.load_consolidated_data()
        
        # Run consistency analysis
        self.run_consistency_analysis()
        
        # Save results
        self.save_consistency_results()
        
        # Generate report
        report = self.generate_consistency_report()
        report_path = self.reports_path / "ablation_evaluator_consistency_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Consistency report saved to: {report_path}")
        print(f"\n🎉 Evaluator consistency analysis complete!")

if __name__ == "__main__":
    analyzer = AblationEvaluatorConsistencyAnalyzer()
    analyzer.run_analysis()
