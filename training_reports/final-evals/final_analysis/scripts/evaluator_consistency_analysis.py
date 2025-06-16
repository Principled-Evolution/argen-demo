#!/usr/bin/env python3
"""
Evaluator Consistency Analysis

This script compares Claude and Gemini evaluations to assess consistency
and validate that Claude-based selections are robust across evaluators.

Usage: python evaluator_consistency_analysis.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import spearmanr, pearsonr

class EvaluatorConsistencyAnalyzer:
    def __init__(self):
        """Initialize the evaluator consistency analyzer."""
        self.data_path = Path("../data")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        self.consolidated_data = None
        self.claude_data = None
        self.gemini_data = None

    def load_consolidated_data(self) -> None:
        """Load the consolidated evaluation data."""
        print("📊 Loading consolidated evaluation data...")
        
        json_path = self.data_path / "consolidated_final_evals.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Consolidated data not found at {json_path}. Run final_evals_consolidator.py first.")
        
        with open(json_path, 'r') as f:
            self.consolidated_data = json.load(f)
        
        all_evals = self.consolidated_data['evaluations']
        self.claude_data = [eval for eval in all_evals if eval['evaluator_type'] == 'claude']
        self.gemini_data = [eval for eval in all_evals if eval['evaluator_type'] == 'gemini']
        
        print(f"✓ Claude evaluations: {len(self.claude_data)}")
        print(f"✓ Gemini evaluations: {len(self.gemini_data)}")

    def create_matched_pairs(self) -> List[Tuple[Dict, Dict]]:
        """Create matched pairs of Claude and Gemini evaluations for the same models."""
        print("🔗 Creating matched evaluation pairs...")
        
        # Create lookup for Claude evaluations
        claude_lookup = {}
        for eval in self.claude_data:
            key = (eval['model_family'], eval['seed'], eval['checkpoint'])
            claude_lookup[key] = eval
        
        # Find matching Gemini evaluations
        matched_pairs = []
        unmatched_gemini = []
        
        for gemini_eval in self.gemini_data:
            key = (gemini_eval['model_family'], gemini_eval['seed'], gemini_eval['checkpoint'])
            
            if key in claude_lookup:
                matched_pairs.append((claude_lookup[key], gemini_eval))
            else:
                unmatched_gemini.append(gemini_eval)
        
        print(f"✓ Found {len(matched_pairs)} matched pairs")
        if unmatched_gemini:
            print(f"⚠️ {len(unmatched_gemini)} Gemini evaluations without Claude matches")
        
        return matched_pairs

    def calculate_correlation_analysis(self, matched_pairs: List[Tuple[Dict, Dict]]) -> Dict:
        """Calculate correlation analysis between Claude and Gemini evaluations."""
        print("📈 Calculating correlation analysis...")
        
        metrics = ['average_combined_score', 'average_ahimsa_score', 'average_dharma_score', 'average_helpfulness_score']
        
        correlations = {}
        
        for metric in metrics:
            claude_scores = [pair[0][metric] for pair in matched_pairs]
            gemini_scores = [pair[1][metric] for pair in matched_pairs]
            
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
        
        return correlations

    def analyze_ranking_consistency(self, matched_pairs: List[Tuple[Dict, Dict]]) -> Dict:
        """Analyze ranking consistency between evaluators."""
        print("🏆 Analyzing ranking consistency...")
        
        # Create DataFrames for easier ranking
        claude_df = pd.DataFrame([pair[0] for pair in matched_pairs])
        gemini_df = pd.DataFrame([pair[1] for pair in matched_pairs])
        
        # Add model identifiers
        for i, (claude_eval, gemini_eval) in enumerate(matched_pairs):
            claude_df.loc[i, 'model_id'] = f"{claude_eval['model_type']}_s{claude_eval['seed']}_c{claude_eval['checkpoint']}"
            gemini_df.loc[i, 'model_id'] = f"{gemini_eval['model_type']}_s{gemini_eval['seed']}_c{gemini_eval['checkpoint']}"
        
        # Rank by combined score
        claude_df['rank'] = claude_df['average_combined_score'].rank(ascending=False)
        gemini_df['rank'] = gemini_df['average_combined_score'].rank(ascending=False)
        
        # Find top models according to each evaluator
        claude_top5 = set(claude_df.nsmallest(5, 'rank')['model_id'])
        gemini_top5 = set(gemini_df.nsmallest(5, 'rank')['model_id'])
        
        claude_top10 = set(claude_df.nsmallest(10, 'rank')['model_id'])
        gemini_top10 = set(gemini_df.nsmallest(10, 'rank')['model_id'])
        
        # Calculate overlap
        top5_overlap = len(claude_top5.intersection(gemini_top5))
        top10_overlap = len(claude_top10.intersection(gemini_top10))
        
        # Identify champions according to each evaluator
        claude_champion = claude_df.loc[claude_df['rank'].idxmin()]
        gemini_champion = gemini_df.loc[gemini_df['rank'].idxmin()]
        
        return {
            'top5_overlap': top5_overlap,
            'top5_overlap_pct': (top5_overlap / 5) * 100,
            'top10_overlap': top10_overlap,
            'top10_overlap_pct': (top10_overlap / 10) * 100,
            'claude_champion': claude_champion.to_dict(),
            'gemini_champion': gemini_champion.to_dict(),
            'same_champion': claude_champion['model_id'] == gemini_champion['model_id'],
            'claude_rankings': claude_df[['model_id', 'rank', 'average_combined_score']].sort_values('rank'),
            'gemini_rankings': gemini_df[['model_id', 'rank', 'average_combined_score']].sort_values('rank')
        }

    def analyze_systematic_differences(self, matched_pairs: List[Tuple[Dict, Dict]]) -> Dict:
        """Analyze systematic differences between evaluators."""
        print("🔍 Analyzing systematic differences...")
        
        # Calculate differences (Claude - Gemini) for each metric
        metrics = ['average_combined_score', 'average_ahimsa_score', 'average_dharma_score', 'average_helpfulness_score']
        
        differences = {}
        
        for metric in metrics:
            diffs = [pair[0][metric] - pair[1][metric] for pair in matched_pairs]
            
            # Statistical tests
            t_stat, t_p = stats.ttest_1samp(diffs, 0)  # Test if mean difference is significantly different from 0
            
            differences[metric] = {
                'mean_diff': np.mean(diffs),
                'std_diff': np.std(diffs),
                'median_diff': np.median(diffs),
                'min_diff': np.min(diffs),
                'max_diff': np.max(diffs),
                't_statistic': t_stat,
                't_p_value': t_p,
                'significant_bias': t_p < 0.05
            }
        
        return differences

    def generate_consistency_report(self, matched_pairs: List[Tuple[Dict, Dict]], 
                                  correlations: Dict, ranking_analysis: Dict, 
                                  systematic_differences: Dict) -> str:
        """Generate comprehensive evaluator consistency report."""
        
        report = f"""# Evaluator Consistency Analysis Report

## Executive Summary

This report analyzes the consistency between Claude-3.5-Sonnet and Gemini-2.0-Flash evaluations to validate the robustness of Claude-based model selections.

### Key Findings
- **Matched Evaluations**: {len(matched_pairs)} model pairs analyzed
- **Top-5 Overlap**: {ranking_analysis['top5_overlap']}/5 ({ranking_analysis['top5_overlap_pct']:.0f}%)
- **Top-10 Overlap**: {ranking_analysis['top10_overlap']}/10 ({ranking_analysis['top10_overlap_pct']:.0f}%)
- **Same Champion**: {'✅ Yes' if ranking_analysis['same_champion'] else '❌ No'}

---

## 1. Correlation Analysis

### Metric Correlations (Claude vs Gemini)
| Metric | Pearson r | p-value | Spearman ρ | p-value | Mean Abs Diff |
|--------|-----------|---------|------------|---------|---------------|
"""
        
        for metric, corr in correlations.items():
            metric_name = metric.replace('average_', '').replace('_score', '').title()
            report += f"| {metric_name} | {corr['pearson_r']:.3f} | {corr['pearson_p']:.3f} | {corr['spearman_r']:.3f} | {corr['spearman_p']:.3f} | {corr['mean_abs_diff']:.4f} |\n"

        report += f"""
### Interpretation
- **Strong Correlation**: r > 0.7 indicates good agreement
- **Moderate Correlation**: 0.4 < r < 0.7 indicates reasonable agreement  
- **Weak Correlation**: r < 0.4 indicates poor agreement

### Score Distributions
| Metric | Claude Mean (±SD) | Gemini Mean (±SD) |
|--------|-------------------|-------------------|
"""
        
        for metric, corr in correlations.items():
            metric_name = metric.replace('average_', '').replace('_score', '').title()
            report += f"| {metric_name} | {corr['claude_mean']:.4f} (±{corr['claude_std']:.4f}) | {corr['gemini_mean']:.4f} (±{corr['gemini_std']:.4f}) |\n"

        report += f"""
---

## 2. Ranking Consistency Analysis

### Champion Model Comparison
| Evaluator | Champion Model | Combined Score |
|-----------|----------------|----------------|
| Claude | {ranking_analysis['claude_champion']['model_id']} | {ranking_analysis['claude_champion']['average_combined_score']:.4f} |
| Gemini | {ranking_analysis['gemini_champion']['model_id']} | {ranking_analysis['gemini_champion']['average_combined_score']:.4f} |

**Same Champion**: {'✅ Yes - Both evaluators agree' if ranking_analysis['same_champion'] else '❌ No - Different champions selected'}

### Top Model Overlap
- **Top-5 Models**: {ranking_analysis['top5_overlap']}/5 overlap ({ranking_analysis['top5_overlap_pct']:.0f}%)
- **Top-10 Models**: {ranking_analysis['top10_overlap']}/10 overlap ({ranking_analysis['top10_overlap_pct']:.0f}%)

### Top 10 Rankings Comparison
| Rank | Claude Model | Claude Score | Gemini Model | Gemini Score |
|------|--------------|--------------|--------------|--------------|
"""
        
        claude_top10 = ranking_analysis['claude_rankings'].head(10)
        gemini_top10 = ranking_analysis['gemini_rankings'].head(10)
        
        for i in range(min(10, len(claude_top10), len(gemini_top10))):
            claude_row = claude_top10.iloc[i]
            gemini_row = gemini_top10.iloc[i]
            report += f"| {i+1} | {claude_row['model_id']} | {claude_row['average_combined_score']:.4f} | {gemini_row['model_id']} | {gemini_row['average_combined_score']:.4f} |\n"

        report += f"""
---

## 3. Systematic Differences Analysis

### Mean Differences (Claude - Gemini)
| Metric | Mean Diff | Std Dev | Median | Range | t-test p | Significant Bias |
|--------|-----------|---------|--------|-------|----------|------------------|
"""
        
        for metric, diff in systematic_differences.items():
            metric_name = metric.replace('average_', '').replace('_score', '').title()
            bias_status = "⚠️ Yes" if diff['significant_bias'] else "✅ No"
            report += f"| {metric_name} | {diff['mean_diff']:+.4f} | {diff['std_diff']:.4f} | {diff['median_diff']:+.4f} | [{diff['min_diff']:+.4f}, {diff['max_diff']:+.4f}] | {diff['t_p_value']:.3f} | {bias_status} |\n"

        report += f"""
### Interpretation
- **Positive values**: Claude scores higher than Gemini
- **Negative values**: Gemini scores higher than Claude
- **Significant bias**: p < 0.05 indicates systematic difference

---

## 4. Validation of Claude-Based Selections

### Robustness Assessment
"""
        
        # Assess robustness based on correlations and ranking consistency
        combined_corr = correlations['average_combined_score']['pearson_r']
        
        if combined_corr > 0.7 and ranking_analysis['top5_overlap_pct'] >= 60:
            robustness = "✅ **HIGH ROBUSTNESS**"
            recommendation = "Claude-based selections are well-validated by Gemini agreement."
        elif combined_corr > 0.5 and ranking_analysis['top5_overlap_pct'] >= 40:
            robustness = "⚠️ **MODERATE ROBUSTNESS**"
            recommendation = "Claude-based selections show reasonable consistency with Gemini."
        else:
            robustness = "❌ **LOW ROBUSTNESS**"
            recommendation = "Consider additional validation or multi-evaluator consensus."
        
        report += f"""
**Overall Assessment**: {robustness}

### Evidence
- **Combined Score Correlation**: {combined_corr:.3f} ({'Strong' if combined_corr > 0.7 else 'Moderate' if combined_corr > 0.5 else 'Weak'})
- **Top-5 Agreement**: {ranking_analysis['top5_overlap_pct']:.0f}%
- **Champion Agreement**: {'Yes' if ranking_analysis['same_champion'] else 'No'}

### Recommendation
{recommendation}

---

## 5. Implications for Research

### For Champion Model Selection
- **Primary Analysis**: Continue using Claude-only for consistency
- **Validation**: {'Strong' if combined_corr > 0.7 else 'Moderate' if combined_corr > 0.5 else 'Weak'} cross-evaluator validation
- **Confidence**: {'High' if ranking_analysis['same_champion'] else 'Moderate'} confidence in champion selection

### For Publication
1. **Report Claude Results**: Use Claude-based champion as primary result
2. **Cross-Validation**: {'Mention strong Gemini agreement' if combined_corr > 0.7 else 'Note moderate Gemini consistency' if combined_corr > 0.5 else 'Acknowledge evaluator differences'}
3. **Transparency**: Report both evaluator results for completeness

---

## 6. Data Provenance

### Analysis Details
- **Matched Pairs**: {len(matched_pairs)} models evaluated by both systems
- **Claude Evaluations**: {len([pair[0] for pair in matched_pairs])} files
- **Gemini Evaluations**: {len([pair[1] for pair in matched_pairs])} files
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Statistical Methods
- **Correlation**: Pearson (linear) and Spearman (rank) correlations
- **Ranking**: Top-k overlap analysis
- **Bias Testing**: One-sample t-tests for systematic differences

---
*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*Analysis Script*: `evaluator_consistency_analysis.py`  
*Data Source*: `../data/consolidated_final_evals.json`
"""
        
        return report

    def run_consistency_analysis(self) -> None:
        """Run the complete evaluator consistency analysis."""
        print("🔄 Starting Evaluator Consistency Analysis")
        print("=" * 50)
        
        # Load data
        self.load_consolidated_data()
        
        # Create matched pairs
        matched_pairs = self.create_matched_pairs()
        
        if len(matched_pairs) < 5:
            print("⚠️ Warning: Too few matched pairs for reliable analysis")
            return
        
        # Calculate correlations
        correlations = self.calculate_correlation_analysis(matched_pairs)
        
        # Analyze ranking consistency
        ranking_analysis = self.analyze_ranking_consistency(matched_pairs)
        
        # Analyze systematic differences
        systematic_differences = self.analyze_systematic_differences(matched_pairs)
        
        # Generate report
        report = self.generate_consistency_report(matched_pairs, correlations, 
                                                ranking_analysis, systematic_differences)
        
        # Save report
        report_path = self.reports_path / "evaluator_consistency_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Consistency report saved to: {report_path}")
        print(f"\n🎉 Evaluator consistency analysis complete!")
        print(f"📊 Analyzed {len(matched_pairs)} matched model pairs")

if __name__ == "__main__":
    analyzer = EvaluatorConsistencyAnalyzer()
    analyzer.run_consistency_analysis()
