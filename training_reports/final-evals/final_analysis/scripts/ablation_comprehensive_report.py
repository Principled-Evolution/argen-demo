#!/usr/bin/env python3
"""
Ablation Comprehensive Report Generator - Final Evaluations

This script generates comprehensive reports for reward-only and policy-only
ablation analysis with statistical comparisons.

Usage: python ablation_comprehensive_report.py
"""

print("Script starting...")
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class AblationReportGenerator:
    def __init__(self):
        """Initialize the ablation report generator."""
        self.data_path = Path("../data")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        self.consolidated_data = None
        self.statistical_results = None
        self.baseline = None
        self.ablations = {}

    def load_analysis_data(self) -> None:
        """Load consolidated data and statistical results."""
        print("📊 Loading analysis data...")
        import sys
        sys.stdout.flush()
        
        # Load consolidated data
        consolidated_path = self.data_path / "consolidated_ablation_evals.json"
        if not consolidated_path.exists():
            raise FileNotFoundError(f"Consolidated data not found. Run ablation_data_consolidator.py first.")
        
        with open(consolidated_path, 'r') as f:
            self.consolidated_data = json.load(f)
        
        # Load statistical results
        results_path = self.data_path / "ablation_statistical_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Statistical results not found. Run ablation_statistical_analysis.py first.")
        
        with open(results_path, 'r') as f:
            self.statistical_results = json.load(f)
        
        self.baseline = self.consolidated_data['baseline']
        
        # Organize ablations
        for eval in self.consolidated_data['ablation_evaluations']:
            ablation_type = eval['ablation_type']
            evaluator = eval['evaluator_type']
            
            if ablation_type not in self.ablations:
                self.ablations[ablation_type] = {}
            
            self.ablations[ablation_type][evaluator] = eval
        
        print("✓ Analysis data loaded successfully")

    def generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        if not self.baseline:
            return "## Executive Summary\n\n*Baseline data not available*\n"
        
        # Get key findings
        degradation = self.statistical_results['degradation_analysis']
        
        # Find best performing ablation
        best_ablation = None
        best_score = 0
        for ablation_type in self.ablations:
            for evaluator in self.ablations[ablation_type]:
                score = self.ablations[ablation_type][evaluator]['average_combined_score']
                if score > best_score:
                    best_score = score
                    best_ablation = (ablation_type, evaluator)
        
        summary = f"""## Executive Summary

### Key Findings
- **Baseline (Median Seed)**: {self.baseline['average_combined_score']:.4f} combined score
- **Best Ablation**: {best_ablation[0].replace('_', '-').title()} ({best_ablation[1].title()}) - {best_score:.4f}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}

### Performance Overview
| Ablation Type | Claude Score | Gemini Score | Avg Degradation |
|---------------|--------------|--------------|-----------------|
"""
        
        for ablation_type in ['reward_only', 'policy_only']:
            claude_score = self.ablations[ablation_type]['claude']['average_combined_score']
            gemini_score = self.ablations[ablation_type]['gemini']['average_combined_score']
            
            # Calculate average degradation
            claude_deg = degradation[ablation_type]['claude']['degradation_percent']
            gemini_deg = degradation[ablation_type]['gemini']['degradation_percent']
            avg_deg = (claude_deg + gemini_deg) / 2
            
            summary += f"| {ablation_type.replace('_', '-').title()} | {claude_score:.4f} | {gemini_score:.4f} | {avg_deg:.1f}% |\n"
        
        return summary

    def generate_baseline_comparison_section(self) -> str:
        """Generate baseline comparison section."""
        if not self.baseline:
            return "## Baseline Comparison\n\n*Baseline data not available*\n"
        
        section = f"""## Baseline Comparison

### Median Seed Baseline Reference
- **Model**: {self.baseline['model_type'].upper()} Seed {self.baseline['seed']} ({self.baseline['checkpoint']})
- **Combined Score**: {self.baseline['average_combined_score']:.4f}
- **Purpose**: Representative baseline for fair ablation comparisons

### Ablation vs Baseline Performance

#### Combined Score Comparison
| Ablation | Evaluator | Score | vs Baseline | Relative Change | Effect Size |
|----------|-----------|-------|-------------|-----------------|-------------|
"""
        
        baseline_comparisons = self.statistical_results['baseline_comparisons']
        
        for ablation_type in ['reward_only', 'policy_only']:
            for evaluator in ['claude', 'gemini']:
                if evaluator in baseline_comparisons[ablation_type]:
                    comp = baseline_comparisons[ablation_type][evaluator]['combined']
                    effect = comp['effect_size']
                    
                    section += f"| {ablation_type.replace('_', '-').title()} | {evaluator.title()} | {comp['ablation_score']:.4f} | {comp['difference']:+.4f} | {comp['relative_change']:+.1f}% | {effect['cohens_d']:.3f} ({effect['interpretation']}) |\n"
        
        section += f"""
#### Detailed Metric Breakdown

##### Reward-Only Ablation
"""
        
        # Add detailed breakdowns for each ablation type
        for ablation_type in ['reward_only', 'policy_only']:
            section += f"""
{'##### Policy-Only Ablation' if ablation_type == 'policy_only' else ''}

**Claude Evaluation**:
"""
            if 'claude' in baseline_comparisons[ablation_type]:
                comp = baseline_comparisons[ablation_type]['claude']
                for metric in ['ahimsa', 'dharma', 'helpfulness']:
                    m = comp[metric]
                    section += f"- **{metric.title()}**: {m['ablation_score']:.4f} vs {m['baseline_score']:.4f} ({m['relative_change']:+.1f}%)\n"
            
            section += f"""
**Gemini Evaluation**:
"""
            if 'gemini' in baseline_comparisons[ablation_type]:
                comp = baseline_comparisons[ablation_type]['gemini']
                for metric in ['ahimsa', 'dharma', 'helpfulness']:
                    m = comp[metric]
                    section += f"- **{metric.title()}**: {m['ablation_score']:.4f} vs {m['baseline_score']:.4f} ({m['relative_change']:+.1f}%)\n"
        
        return section

    def generate_ablation_comparison_section(self) -> str:
        """Generate reward-only vs policy-only comparison section."""
        section = """## Reward-Only vs Policy-Only Comparison

### Direct Performance Comparison
"""
        
        ablation_comparisons = self.statistical_results['ablation_comparisons']
        
        for evaluator in ['claude', 'gemini']:
            if evaluator in ablation_comparisons:
                comp = ablation_comparisons[evaluator]
                section += f"""
#### {evaluator.title()} Evaluation
| Metric | Reward-Only | Policy-Only | Difference | Better Performer |
|--------|-------------|-------------|------------|------------------|
"""
                for metric in ['combined', 'ahimsa', 'dharma', 'helpfulness']:
                    if metric in comp:
                        m = comp[metric]
                        section += f"| {metric.title()} | {m['reward_only_score']:.4f} | {m['policy_only_score']:.4f} | {m['difference']:+.4f} | {m['better_performer'].replace('_', '-').title()} |\n"
        
        section += """
### Statistical Significance
"""
        
        # Add effect size analysis
        for evaluator in ['claude', 'gemini']:
            if evaluator in ablation_comparisons:
                comp = ablation_comparisons[evaluator]
                section += f"""
#### {evaluator.title()} Effect Sizes
| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
"""
                for metric in ['combined', 'ahimsa', 'dharma', 'helpfulness']:
                    if metric in comp:
                        effect = comp[metric]['effect_size']
                        section += f"| {metric.title()} | {effect['cohens_d']:.3f} | {effect['interpretation'].title()} |\n"
        
        return section

    def generate_evaluator_consistency_section(self) -> str:
        """Generate cross-evaluator consistency section."""
        section = """## Cross-Evaluator Consistency

### Claude vs Gemini Agreement
"""
        
        consistency = self.statistical_results['consistency_analysis']
        
        for ablation_type in ['reward_only', 'policy_only']:
            if ablation_type in consistency:
                section += f"""
#### {ablation_type.replace('_', '-').title()} Ablation
| Metric | Claude Score | Gemini Score | Difference | Agreement Level |
|--------|--------------|--------------|------------|-----------------|
"""
                cons = consistency[ablation_type]
                for metric in ['combined', 'ahimsa', 'dharma', 'helpfulness']:
                    if metric in cons:
                        m = cons[metric]
                        section += f"| {metric.title()} | {m['claude_score']:.4f} | {m['gemini_score']:.4f} | {m['difference']:+.4f} | {m['agreement_level'].title()} |\n"
        
        return section

    def generate_performance_degradation_section(self) -> str:
        """Generate performance degradation analysis section."""
        section = """## Performance Degradation Analysis

### Degradation from Baseline
"""
        
        degradation = self.statistical_results['degradation_analysis']
        
        section += """
| Ablation Type | Evaluator | Baseline | Ablation | Degradation | Performance Retention |
|---------------|-----------|----------|----------|-------------|----------------------|
"""
        
        for ablation_type in ['reward_only', 'policy_only']:
            for evaluator in ['claude', 'gemini']:
                if evaluator in degradation[ablation_type]:
                    deg = degradation[ablation_type][evaluator]
                    section += f"| {ablation_type.replace('_', '-').title()} | {evaluator.title()} | {deg['baseline_score']:.4f} | {deg['ablation_score']:.4f} | {deg['degradation_percent']:.1f}% | {deg['performance_retention']:.1f}% |\n"
        
        section += """
### Key Insights

#### Component Contribution Analysis
"""
        
        # Analyze which component contributes more
        reward_avg_retention = np.mean([degradation['reward_only'][ev]['performance_retention'] for ev in degradation['reward_only']])
        policy_avg_retention = np.mean([degradation['policy_only'][ev]['performance_retention'] for ev in degradation['policy_only']])

        better_component = "reward optimization" if reward_avg_retention > policy_avg_retention else "policy optimization"
        worse_component = "policy optimization" if reward_avg_retention > policy_avg_retention else "reward optimization"

        section += f"""
- **{better_component.title()}** appears more critical: {max(reward_avg_retention, policy_avg_retention):.1f}% average retention
- **{worse_component.title()}** shows greater impact when removed: {min(reward_avg_retention, policy_avg_retention):.1f}% average retention
- **Performance Gap**: {abs(reward_avg_retention - policy_avg_retention):.1f} percentage points difference
"""

        return section, reward_avg_retention, policy_avg_retention

    def generate_comprehensive_report(self) -> str:
        """Generate the complete comprehensive report."""

        # Generate sections and get retention values
        degradation_section, reward_avg_retention, policy_avg_retention = self.generate_performance_degradation_section()

        report = f"""# Comprehensive Ablation Analysis Report

{self.generate_executive_summary()}

{self.generate_baseline_comparison_section()}

{self.generate_ablation_comparison_section()}

{self.generate_evaluator_consistency_section()}

{degradation_section}

## Methodology and Data Provenance

### Analysis Methodology
1. **Baseline Selection**: Median Seed 3 (GRPO7 checkpoint-3000) for representative comparison
2. **Statistical Analysis**: Cohen's d effect sizes and relative performance changes
3. **Cross-Validation**: Claude and Gemini evaluator consistency analysis
4. **Complete Evaluations**: 100 scenarios per ablation model

### Data Sources
- **Ablation Models**: Seed 3 reward-only and policy-only ablations
- **Baseline Model**: {self.baseline['model_type'].upper()} Seed {self.baseline['seed']} ({self.baseline['checkpoint']})
- **Evaluators**: Claude-3.5-Sonnet and Gemini-2.0-Flash
- **Evaluation Protocol**: Same 100-scenario benchmark as main analysis

### File References
"""
        
        # Add file references
        for eval in self.consolidated_data['ablation_evaluations']:
            model_path = eval['model_name'].split('/')[-1] if '/' in eval['model_name'] else eval['model_name']
            report += f"- **{eval['evaluator_type'].title()} {eval['ablation_type'].replace('_', '-').title()}**: `{model_path}`\n"
        
        report += f"""
## Recommendations

### For Publication
1. **Report both ablation types** to demonstrate component importance
2. **Use cross-evaluator consistency** to validate findings robustness  
3. **Emphasize performance retention** rather than just degradation
4. **Include effect size interpretations** for statistical rigor

### For Model Development
1. **{'Reward optimization' if reward_avg_retention > policy_avg_retention else 'Policy optimization'} appears more critical** based on retention analysis
2. **Both components contribute significantly** to overall performance
3. **Consider hybrid approaches** that balance both components

---
*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*Analysis Scripts*: `ablation_*.py`  
*Data Source*: `consolidated_ablation_evals.json`  
*Statistical Results*: `ablation_statistical_results.json`
"""
        
        return report

    def run_report_generation(self) -> None:
        """Generate all ablation reports."""
        print("📝 Generating comprehensive ablation reports...")
        
        # Load data
        self.load_analysis_data()
        
        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report()
        
        # Save comprehensive report
        report_path = self.reports_path / "ablation_comprehensive_report.md"
        with open(report_path, 'w') as f:
            f.write(comprehensive_report)
        
        print(f"✓ Comprehensive report saved to: {report_path}")
        print(f"\n🎉 Report generation complete!")

if __name__ == "__main__":
    print("Starting ablation report generation...")
    try:
        generator = AblationReportGenerator()
        generator.run_report_generation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
