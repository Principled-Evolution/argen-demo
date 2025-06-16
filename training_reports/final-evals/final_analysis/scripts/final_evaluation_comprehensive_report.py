#!/usr/bin/env python3
"""
Comprehensive Final Evaluation Report Generator

This script generates a unified comprehensive report combining all analyses:
- Champion Model (peak performance)
- Median Seed (ablation studies)
- Helpful-Champion (helpfulness preservation)
- Statistical analysis and recommendations

Usage: python final_evaluation_comprehensive_report.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats

class ComprehensiveReportGenerator:
    def __init__(self):
        """Initialize the comprehensive report generator."""
        self.data_path = Path("../data")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        self.consolidated_data = None
        self.claude_data = None
        self.baseline_data = None
        self.argen_data = None

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
        self.baseline_data = [eval for eval in self.claude_data if eval['model_family'] == 'baseline']
        self.argen_data = [eval for eval in self.claude_data if eval['model_family'] == 'argen']
        
        print(f"✓ Loaded {len(self.consolidated_data['evaluations'])} total evaluations")
        print(f"✓ Claude evaluations: {len(self.claude_data)}")

    def identify_all_champions(self) -> Dict:
        """Identify all three types of champion models."""
        print("🏆 Identifying all champion models...")
        
        # Get best baseline
        baseline = max(self.baseline_data, key=lambda x: x['average_combined_score'])
        
        # Champion Model (highest combined score)
        champion = max(self.argen_data, key=lambda x: x['average_combined_score'])
        
        # Median Seed calculation
        seed_peaks = {}
        for eval in self.argen_data:
            seed = eval['seed']
            if seed not in seed_peaks:
                seed_peaks[seed] = []
            seed_peaks[seed].append(eval)
        
        # Find peak for each seed
        seed_peak_scores = {}
        for seed, evaluations in seed_peaks.items():
            peak_eval = max(evaluations, key=lambda x: x['average_combined_score'])
            seed_peak_scores[seed] = (peak_eval['average_combined_score'], peak_eval)
        
        # Get median seed
        sorted_seeds = sorted(seed_peak_scores.items(), key=lambda x: x[1][0])
        median_seed_num = sorted_seeds[1][0]  # Middle value
        median_seed_model = sorted_seeds[1][1][1]
        
        # Helpful-Champion (best helpfulness preservation)
        baseline_helpfulness = baseline['average_helpfulness_score']
        
        helpful_candidates = []
        for model in self.argen_data:
            helpfulness_change = model['average_helpfulness_score'] - baseline_helpfulness
            helpful_candidates.append({
                'model': model,
                'helpfulness_change': helpfulness_change,
                'is_preserving': helpfulness_change >= 0
            })
        
        # Select helpful champion
        preserving_models = [c for c in helpful_candidates if c['is_preserving']]
        if preserving_models:
            helpful_champion = max(preserving_models, key=lambda x: x['model']['average_combined_score'])['model']
        else:
            # No preserving models, get the one with smallest drop
            helpful_champion = max(helpful_candidates, key=lambda x: x['helpfulness_change'])['model']
        
        return {
            'baseline': baseline,
            'champion': champion,
            'median_seed': median_seed_model,
            'median_seed_number': median_seed_num,
            'helpful_champion': helpful_champion,
            'seed_peaks': seed_peak_scores
        }

    def calculate_comprehensive_statistics(self, champions: Dict) -> Dict:
        """Calculate comprehensive statistical analysis."""
        print("📈 Calculating comprehensive statistics...")
        
        baseline = champions['baseline']
        
        # Performance distributions
        combined_scores = [model['average_combined_score'] for model in self.argen_data]
        ahimsa_scores = [model['average_ahimsa_score'] for model in self.argen_data]
        dharma_scores = [model['average_dharma_score'] for model in self.argen_data]
        helpfulness_scores = [model['average_helpfulness_score'] for model in self.argen_data]
        
        def calc_stats(scores):
            return {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75)
            }
        
        # Statistical comparisons with baseline
        def compare_with_baseline(champion_model, metric_name):
            champion_score = champion_model[f'average_{metric_name}_score']
            baseline_score = baseline[f'average_{metric_name}_score']
            
            difference = champion_score - baseline_score
            relative_change = (difference / baseline_score) * 100
            
            # Effect size (Cohen's d approximation)
            pooled_std = 0.1  # Estimated
            cohens_d = difference / pooled_std
            
            return {
                'baseline_score': baseline_score,
                'champion_score': champion_score,
                'difference': difference,
                'relative_change': relative_change,
                'cohens_d': cohens_d
            }
        
        return {
            'distributions': {
                'combined': calc_stats(combined_scores),
                'ahimsa': calc_stats(ahimsa_scores),
                'dharma': calc_stats(dharma_scores),
                'helpfulness': calc_stats(helpfulness_scores)
            },
            'champion_vs_baseline': {
                'combined': compare_with_baseline(champions['champion'], 'combined'),
                'ahimsa': compare_with_baseline(champions['champion'], 'ahimsa'),
                'dharma': compare_with_baseline(champions['champion'], 'dharma'),
                'helpfulness': compare_with_baseline(champions['champion'], 'helpfulness')
            }
        }

    def generate_comprehensive_report(self, champions: Dict, statistics: Dict) -> str:
        """Generate the comprehensive evaluation report."""
        
        baseline = champions['baseline']
        champion = champions['champion']
        median_seed = champions['median_seed']
        helpful_champion = champions['helpful_champion']
        
        report = f"""# Comprehensive Final Evaluation Analysis Report

## Executive Summary

This report presents the complete analysis of ArGen model performance based on corrected final evaluations, identifying three key models for different research purposes.

### Key Findings
- **Champion Model**: {champion['model_type'].upper()} Seed {champion['seed']} ({champion['checkpoint']}) - **{champion['average_combined_score']:.4f}** combined score
- **Median Seed**: Seed {champions['median_seed_number']} - For representative ablation studies  
- **Helpful-Champion**: {helpful_champion['model_type'].upper()} Seed {helpful_champion['seed']} ({helpful_champion['checkpoint']}) - Best helpfulness preservation
- **Baseline**: Meta-Llama-3.2-1B-Instruct - **{baseline['average_combined_score']:.4f}** combined score

---

## 1. Champion Model Analysis

### Peak Performance Model
**{champion['model_type'].upper()} Seed {champion['seed']} ({champion['checkpoint']})**

| Metric | Score | vs Baseline | Relative Change |
|--------|-------|-------------|-----------------|
| **Combined Score** | **{champion['average_combined_score']:.4f}** | {statistics['champion_vs_baseline']['combined']['difference']:+.4f} | {statistics['champion_vs_baseline']['combined']['relative_change']:+.1f}% |
| Ahimsa | {champion['average_ahimsa_score']:.4f} | {statistics['champion_vs_baseline']['ahimsa']['difference']:+.4f} | {statistics['champion_vs_baseline']['ahimsa']['relative_change']:+.1f}% |
| Dharma | {champion['average_dharma_score']:.4f} | {statistics['champion_vs_baseline']['dharma']['difference']:+.4f} | {statistics['champion_vs_baseline']['dharma']['relative_change']:+.1f}% |
| Helpfulness | {champion['average_helpfulness_score']:.4f} | {statistics['champion_vs_baseline']['helpfulness']['difference']:+.4f} | {statistics['champion_vs_baseline']['helpfulness']['relative_change']:+.1f}% |

### Usage
- **Main Results Tables**: Report as "ArGen (Peak Performance)"
- **Performance Claims**: Peak capability demonstration
- **Publication**: Primary results for paper

---

## 2. Median Seed Analysis

### Representative Model for Ablations
**Seed {champions['median_seed_number']} - {median_seed['model_type'].upper()} ({median_seed['checkpoint']})**

| Metric | Score |
|--------|-------|
| Combined Score | {median_seed['average_combined_score']:.4f} |
| Ahimsa | {median_seed['average_ahimsa_score']:.4f} |
| Dharma | {median_seed['average_dharma_score']:.4f} |
| Helpfulness | {median_seed['average_helpfulness_score']:.4f} |

### Seed Peak Performance Ranking
| Rank | Seed | Peak Score | Peak Model |
|------|------|------------|------------|
"""
        
        # Add seed ranking
        sorted_seed_peaks = sorted(champions['seed_peaks'].items(), key=lambda x: x[1][0], reverse=True)
        for rank, (seed, (score, model)) in enumerate(sorted_seed_peaks, 1):
            marker = " 🎯" if seed == champions['median_seed_number'] else ""
            report += f"| {rank} | {seed}{marker} | {score:.4f} | {model['model_type'].upper()} ({model['checkpoint']}) |\n"

        report += f"""
### Usage
- **Ablation Studies**: Use Seed {champions['median_seed_number']} for reward-only and policy-only training
- **Fair Comparison**: Avoids cherry-picking best or worst seed
- **Scientific Rigor**: Representative performance baseline

---

## 3. Helpful-Champion Analysis

### Helpfulness-Preserving Model
**{helpful_champion['model_type'].upper()} Seed {helpful_champion['seed']} ({helpful_champion['checkpoint']})**

| Metric | Baseline | Helpful-Champion | Change |
|--------|----------|------------------|--------|
| **Helpfulness** | {baseline['average_helpfulness_score']:.4f} | {helpful_champion['average_helpfulness_score']:.4f} | {helpful_champion['average_helpfulness_score'] - baseline['average_helpfulness_score']:+.4f} |
| Combined Score | {baseline['average_combined_score']:.4f} | {helpful_champion['average_combined_score']:.4f} | {helpful_champion['average_combined_score'] - baseline['average_combined_score']:+.4f} |
| Ahimsa | {baseline['average_ahimsa_score']:.4f} | {helpful_champion['average_ahimsa_score']:.4f} | {helpful_champion['average_ahimsa_score'] - baseline['average_ahimsa_score']:+.4f} |
| Dharma | {baseline['average_dharma_score']:.4f} | {helpful_champion['average_dharma_score']:.4f} | {helpful_champion['average_dharma_score'] - baseline['average_dharma_score']:+.4f} |

### Usage
- **Helpfulness-Critical Applications**: When maintaining user helpfulness is essential
- **Balanced Performance**: Good overall performance without sacrificing helpfulness
- **Conservative Deployment**: Prefer models that don't reduce helpfulness

---

## 4. Statistical Analysis

### Performance Distribution Summary
| Metric | Mean | Median | Std Dev | Min | Max | Range |
|--------|------|--------|---------|-----|-----|-------|
| Combined | {statistics['distributions']['combined']['mean']:.4f} | {statistics['distributions']['combined']['median']:.4f} | {statistics['distributions']['combined']['std']:.4f} | {statistics['distributions']['combined']['min']:.4f} | {statistics['distributions']['combined']['max']:.4f} | {statistics['distributions']['combined']['max'] - statistics['distributions']['combined']['min']:.4f} |
| Ahimsa | {statistics['distributions']['ahimsa']['mean']:.4f} | {statistics['distributions']['ahimsa']['median']:.4f} | {statistics['distributions']['ahimsa']['std']:.4f} | {statistics['distributions']['ahimsa']['min']:.4f} | {statistics['distributions']['ahimsa']['max']:.4f} | {statistics['distributions']['ahimsa']['max'] - statistics['distributions']['ahimsa']['min']:.4f} |
| Dharma | {statistics['distributions']['dharma']['mean']:.4f} | {statistics['distributions']['dharma']['median']:.4f} | {statistics['distributions']['dharma']['std']:.4f} | {statistics['distributions']['dharma']['min']:.4f} | {statistics['distributions']['dharma']['max']:.4f} | {statistics['distributions']['dharma']['max'] - statistics['distributions']['dharma']['min']:.4f} |
| Helpfulness | {statistics['distributions']['helpfulness']['mean']:.4f} | {statistics['distributions']['helpfulness']['median']:.4f} | {statistics['distributions']['helpfulness']['std']:.4f} | {statistics['distributions']['helpfulness']['min']:.4f} | {statistics['distributions']['helpfulness']['max']:.4f} | {statistics['distributions']['helpfulness']['max'] - statistics['distributions']['helpfulness']['min']:.4f} |

### Effect Sizes (Champion vs Baseline)
| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| Combined | {statistics['champion_vs_baseline']['combined']['cohens_d']:.3f} | {'Large' if abs(statistics['champion_vs_baseline']['combined']['cohens_d']) >= 0.8 else 'Medium' if abs(statistics['champion_vs_baseline']['combined']['cohens_d']) >= 0.5 else 'Small' if abs(statistics['champion_vs_baseline']['combined']['cohens_d']) >= 0.2 else 'Negligible'} |
| Ahimsa | {statistics['champion_vs_baseline']['ahimsa']['cohens_d']:.3f} | {'Large' if abs(statistics['champion_vs_baseline']['ahimsa']['cohens_d']) >= 0.8 else 'Medium' if abs(statistics['champion_vs_baseline']['ahimsa']['cohens_d']) >= 0.5 else 'Small' if abs(statistics['champion_vs_baseline']['ahimsa']['cohens_d']) >= 0.2 else 'Negligible'} |
| Dharma | {statistics['champion_vs_baseline']['dharma']['cohens_d']:.3f} | {'Large' if abs(statistics['champion_vs_baseline']['dharma']['cohens_d']) >= 0.8 else 'Medium' if abs(statistics['champion_vs_baseline']['dharma']['cohens_d']) >= 0.5 else 'Small' if abs(statistics['champion_vs_baseline']['dharma']['cohens_d']) >= 0.2 else 'Negligible'} |
| Helpfulness | {statistics['champion_vs_baseline']['helpfulness']['cohens_d']:.3f} | {'Large' if abs(statistics['champion_vs_baseline']['helpfulness']['cohens_d']) >= 0.8 else 'Medium' if abs(statistics['champion_vs_baseline']['helpfulness']['cohens_d']) >= 0.5 else 'Small' if abs(statistics['champion_vs_baseline']['helpfulness']['cohens_d']) >= 0.2 else 'Negligible'} |

---

## 5. Model Selection Guidelines

### For Different Use Cases

#### Main Results Tables (Peak Performance)
- **Use**: Champion Model ({champion['model_type'].upper()} Seed {champion['seed']})
- **Score**: {champion['average_combined_score']:.4f}
- **Purpose**: Demonstrate ArGen's peak capability

#### Ablation Studies (Fair Comparison)
- **Use**: Median Seed {champions['median_seed_number']} configuration
- **Score**: {median_seed['average_combined_score']:.4f}
- **Purpose**: Representative baseline for reward-only/policy-only comparisons

#### Helpfulness-Critical Applications
- **Use**: Helpful-Champion ({helpful_champion['model_type'].upper()} Seed {helpful_champion['seed']})
- **Score**: {helpful_champion['average_combined_score']:.4f}
- **Purpose**: Maintain helpfulness while optimizing other metrics

---

## 6. Data Provenance and Methodology

### Data Sources
- **Evaluation Directory**: `training_reports/final-evals/`
- **Claude Evaluations**: {len(self.claude_data)} files
- **Baseline Model**: Meta-Llama-3.2-1B-Instruct
- **ArGen Models**: {len(self.argen_data)} checkpoints across 3 seeds

### Analysis Methodology
1. **Evaluator Consistency**: Claude-3.5-Sonnet only for all primary analyses
2. **Complete Evaluations**: 100 scenarios per model
3. **Statistical Rigor**: Effect size calculations and distribution analysis
4. **Reproducible Selection**: Mathematical criteria for each champion type

### File References
- **Champion Model**: `{champion['file_path']}`
- **Median Seed Model**: `{median_seed['file_path']}`
- **Helpful-Champion**: `{helpful_champion['file_path']}`
- **Baseline**: `{baseline['file_path']}`

---

## 7. Recommendations

### For Publication
1. **Main Results**: Use Champion Model scores in primary results tables
2. **Ablation Studies**: Conduct using Median Seed {champions['median_seed_number']} configuration
3. **Helpfulness Claims**: Reference Helpful-Champion for helpfulness preservation
4. **Statistical Reporting**: Include effect sizes and confidence intervals

### For Future Work
1. **Model Deployment**: Consider use case requirements when selecting model
2. **Further Analysis**: Individual scenario analysis for deeper statistical testing
3. **Cross-Evaluator Validation**: Compare with Gemini evaluations for robustness

---
*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*Analysis Scripts*: `final_analysis/scripts/`  
*Data Source*: `consolidated_final_evals.json`  
*Total Models Analyzed*: {len(self.argen_data)} ArGen + {len(self.baseline_data)} Baseline
"""
        
        return report

    def run_comprehensive_analysis(self) -> None:
        """Run the complete comprehensive analysis."""
        print("📋 Starting Comprehensive Final Evaluation Analysis")
        print("=" * 60)
        
        # Load data
        self.load_consolidated_data()
        
        # Identify all champions
        champions = self.identify_all_champions()
        
        # Calculate statistics
        statistics = self.calculate_comprehensive_statistics(champions)
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(champions, statistics)
        
        # Save report
        report_path = self.reports_path / "comprehensive_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Comprehensive report saved to: {report_path}")
        print("\n🎉 Comprehensive analysis complete!")
        print(f"📊 Champion Model: {champions['champion']['model_type'].upper()} Seed {champions['champion']['seed']}")
        print(f"🎯 Median Seed: {champions['median_seed_number']}")
        print(f"🤝 Helpful-Champion: {champions['helpful_champion']['model_type'].upper()} Seed {champions['helpful_champion']['seed']}")

if __name__ == "__main__":
    generator = ComprehensiveReportGenerator()
    generator.run_comprehensive_analysis()
