#!/usr/bin/env python3
"""
Helpful-Champion Analysis - Final Evaluations

This script identifies the Helpful-Champion Model.
The Helpful-Champion is the model with the least drop (or maximum gain) in helpfulness
compared to baseline while maximizing combined reward.

Usage: python helpful_champion_analysis_final.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class HelpfulChampionAnalyzer:
    def __init__(self):
        """Initialize the Helpful-Champion analyzer."""
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
        
        print(f"✓ Loaded {len(self.consolidated_data['evaluations'])} evaluations")

    def filter_claude_evaluations(self) -> None:
        """Filter to Claude-only evaluations."""
        print("🔍 Filtering Claude-only evaluations...")
        
        all_evals = self.consolidated_data['evaluations']
        self.claude_data = [eval for eval in all_evals if eval['evaluator_type'] == 'claude']
        
        # Separate baseline and ArGen models
        self.baseline_data = [eval for eval in self.claude_data if eval['model_family'] == 'baseline']
        self.argen_data = [eval for eval in self.claude_data if eval['model_family'] == 'argen']
        
        print(f"✓ Claude evaluations: {len(self.claude_data)}")
        print(f"✓ Baseline models: {len(self.baseline_data)}")
        print(f"✓ ArGen models: {len(self.argen_data)}")

    def get_best_baseline(self) -> Optional[Dict]:
        """Get the best baseline model for comparison."""
        if not self.baseline_data:
            return None
        
        # Use the baseline with highest combined score
        best_baseline = max(self.baseline_data, key=lambda x: x['average_combined_score'])
        return best_baseline

    def calculate_helpfulness_preservation(self, baseline: Dict) -> List[Dict]:
        """Calculate helpfulness preservation metrics for all ArGen models."""
        print("🤝 Calculating helpfulness preservation metrics...")
        
        baseline_helpfulness = baseline['average_helpfulness_score']
        
        helpfulness_analysis = []
        
        for model in self.argen_data:
            model_helpfulness = model['average_helpfulness_score']
            
            # Calculate helpfulness change
            helpfulness_change = model_helpfulness - baseline_helpfulness
            helpfulness_change_pct = (helpfulness_change / baseline_helpfulness) * 100
            
            # Calculate preservation score (higher is better)
            # Models that maintain or improve helpfulness get positive scores
            preservation_score = helpfulness_change
            
            analysis = {
                'model': model,
                'helpfulness_score': model_helpfulness,
                'baseline_helpfulness': baseline_helpfulness,
                'helpfulness_change': helpfulness_change,
                'helpfulness_change_pct': helpfulness_change_pct,
                'preservation_score': preservation_score,
                'combined_score': model['average_combined_score'],
                'is_helpful_preserving': helpfulness_change >= 0  # Maintains or improves helpfulness
            }
            
            helpfulness_analysis.append(analysis)
        
        return helpfulness_analysis

    def identify_helpful_champion(self, helpfulness_analysis: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Identify the Helpful-Champion model."""
        print("🏆 Identifying Helpful-Champion model...")
        
        # Strategy: Among models that preserve helpfulness (change >= 0), 
        # select the one with highest combined score.
        # If no models preserve helpfulness, select the one with smallest drop
        # that still has competitive combined score.
        
        # First, try to find models that preserve or improve helpfulness
        helpful_preserving = [a for a in helpfulness_analysis if a['is_helpful_preserving']]
        
        if helpful_preserving:
            # Among helpful-preserving models, pick the one with highest combined score
            helpful_champion_analysis = max(helpful_preserving, key=lambda x: x['combined_score'])
            strategy = "helpfulness_preserving"
            print(f"✓ Found {len(helpful_preserving)} helpfulness-preserving models")
        else:
            # No models preserve helpfulness, so pick the one with smallest drop
            # but still reasonable combined performance
            
            # Sort by helpfulness preservation (least negative change) first,
            # then by combined score as tiebreaker
            sorted_by_preservation = sorted(helpfulness_analysis, 
                                          key=lambda x: (-x['helpfulness_change'], x['combined_score']), 
                                          reverse=True)
            
            helpful_champion_analysis = sorted_by_preservation[0]
            strategy = "minimal_helpfulness_drop"
            print("⚠️ No models preserve helpfulness, selecting minimal drop model")
        
        # Sort all models by preservation score for ranking
        ranked_models = sorted(helpfulness_analysis, 
                              key=lambda x: (x['preservation_score'], x['combined_score']), 
                              reverse=True)
        
        helpful_champion = helpful_champion_analysis['model']
        
        print(f"✓ Helpful-Champion: {helpful_champion['model_type']} Seed {helpful_champion['seed']} ({helpful_champion['checkpoint']})")
        print(f"✓ Helpfulness change: {helpful_champion_analysis['helpfulness_change']:+.4f} ({helpful_champion_analysis['helpfulness_change_pct']:+.1f}%)")
        print(f"✓ Combined score: {helpful_champion_analysis['combined_score']:.4f}")
        print(f"✓ Selection strategy: {strategy}")
        
        return helpful_champion_analysis, ranked_models

    def analyze_helpfulness_trade_offs(self, helpfulness_analysis: List[Dict]) -> Dict:
        """Analyze trade-offs between helpfulness and other metrics."""
        
        # Calculate correlations
        helpfulness_scores = [a['helpfulness_score'] for a in helpfulness_analysis]
        combined_scores = [a['combined_score'] for a in helpfulness_analysis]
        ahimsa_scores = [a['model']['average_ahimsa_score'] for a in helpfulness_analysis]
        dharma_scores = [a['model']['average_dharma_score'] for a in helpfulness_analysis]
        
        # Simple correlation calculations
        def correlation(x, y):
            return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        
        return {
            'helpfulness_combined_corr': correlation(helpfulness_scores, combined_scores),
            'helpfulness_ahimsa_corr': correlation(helpfulness_scores, ahimsa_scores),
            'helpfulness_dharma_corr': correlation(helpfulness_scores, dharma_scores),
            'models_preserving_helpfulness': sum(1 for a in helpfulness_analysis if a['is_helpful_preserving']),
            'total_models': len(helpfulness_analysis),
            'preservation_rate': sum(1 for a in helpfulness_analysis if a['is_helpful_preserving']) / len(helpfulness_analysis)
        }

    def generate_helpful_champion_report(self, helpful_champion_analysis: Dict, ranked_models: List[Dict], 
                                       baseline: Dict, trade_off_analysis: Dict) -> str:
        """Generate comprehensive Helpful-Champion analysis report."""
        
        champion = helpful_champion_analysis['model']
        
        report = f"""# Helpful-Champion Analysis Report - Final Evaluations

## Executive Summary

**Helpful-Champion Model**: {champion['model_type'].upper()} Seed {champion['seed']} ({champion['checkpoint']})
- **Helpfulness Score**: {helpful_champion_analysis['helpfulness_score']:.4f}
- **Helpfulness Change**: {helpful_champion_analysis['helpfulness_change']:+.4f} ({helpful_champion_analysis['helpfulness_change_pct']:+.1f}%)
- **Combined Score**: {helpful_champion_analysis['combined_score']:.4f}
- **Selection Rationale**: {'Preserves helpfulness while maximizing combined performance' if helpful_champion_analysis['is_helpful_preserving'] else 'Minimal helpfulness drop with competitive performance'}

## Methodology

The Helpful-Champion model is selected to demonstrate ArGen's ability to maintain helpfulness:

1. **Calculate Helpfulness Change**: Compare each ArGen model's helpfulness vs. baseline
2. **Identify Preserving Models**: Find models that maintain or improve helpfulness (change ≥ 0)
3. **Select Champion**: Among preserving models, choose highest combined score
4. **Fallback Strategy**: If no models preserve helpfulness, select minimal drop with good performance

## Helpful-Champion Model Details

### Model Identification
- **Model Type**: {champion['model_type'].upper()}
- **Seed**: {champion['seed']}
- **Checkpoint**: {champion['checkpoint']}
- **Full Model Name**: `{champion['model_name']}`

### Performance Comparison with Baseline
| Metric | Baseline | Helpful-Champion | Change | Relative Change |
|--------|----------|------------------|--------|-----------------|
| **Helpfulness** | {baseline['average_helpfulness_score']:.4f} | {helpful_champion_analysis['helpfulness_score']:.4f} | {helpful_champion_analysis['helpfulness_change']:+.4f} | {helpful_champion_analysis['helpfulness_change_pct']:+.1f}% |
| **Combined Score** | {baseline['average_combined_score']:.4f} | {helpful_champion_analysis['combined_score']:.4f} | {helpful_champion_analysis['combined_score'] - baseline['average_combined_score']:+.4f} | {((helpful_champion_analysis['combined_score'] - baseline['average_combined_score']) / baseline['average_combined_score']) * 100:+.1f}% |
| Ahimsa | {baseline['average_ahimsa_score']:.4f} | {champion['average_ahimsa_score']:.4f} | {champion['average_ahimsa_score'] - baseline['average_ahimsa_score']:+.4f} | {((champion['average_ahimsa_score'] - baseline['average_ahimsa_score']) / baseline['average_ahimsa_score']) * 100:+.1f}% |
| Dharma | {baseline['average_dharma_score']:.4f} | {champion['average_dharma_score']:.4f} | {champion['average_dharma_score'] - baseline['average_dharma_score']:+.4f} | {((champion['average_dharma_score'] - baseline['average_dharma_score']) / baseline['average_dharma_score']) * 100:+.1f}% |

### Detailed Helpfulness Breakdown
| Component | Baseline | Helpful-Champion | Change |
|-----------|----------|------------------|--------|
| Clarity | {baseline['average_clarity_score']:.4f} | {champion['average_clarity_score']:.4f} | {champion['average_clarity_score'] - baseline['average_clarity_score']:+.4f} |
| Relevance | {baseline['average_relevance_score']:.4f} | {champion['average_relevance_score']:.4f} | {champion['average_relevance_score'] - baseline['average_relevance_score']:+.4f} |
| Completeness | {baseline['average_completeness_score']:.4f} | {champion['average_completeness_score']:.4f} | {champion['average_completeness_score'] - baseline['average_completeness_score']:+.4f} |

## Helpfulness Preservation Analysis

### Overall Statistics
- **Models Preserving Helpfulness**: {trade_off_analysis['models_preserving_helpfulness']}/{trade_off_analysis['total_models']} ({trade_off_analysis['preservation_rate']:.1%})
- **Baseline Helpfulness**: {baseline['average_helpfulness_score']:.4f}

### Top 10 Models by Helpfulness Preservation
| Rank | Model | Seed | Checkpoint | Helpfulness | Change | Change % | Combined Score | Status |
|------|-------|------|------------|-------------|--------|----------|----------------|--------|
"""
        
        for i, analysis in enumerate(ranked_models[:10], 1):
            model = analysis['model']
            status = "✅ Preserving" if analysis['is_helpful_preserving'] else "⚠️ Declining"
            marker = " 🏆" if model == champion else ""
            
            report += f"| {i}{marker} | {model['model_type'].upper()} | {model['seed']} | {model['checkpoint']} | {analysis['helpfulness_score']:.4f} | {analysis['helpfulness_change']:+.4f} | {analysis['helpfulness_change_pct']:+.1f}% | {analysis['combined_score']:.4f} | {status} |\n"

        report += f"""
## Trade-off Analysis

### Metric Correlations
- **Helpfulness vs Combined Score**: {trade_off_analysis['helpfulness_combined_corr']:.3f}
- **Helpfulness vs Ahimsa**: {trade_off_analysis['helpfulness_ahimsa_corr']:.3f}
- **Helpfulness vs Dharma**: {trade_off_analysis['helpfulness_dharma_corr']:.3f}

### Key Insights
"""
        
        if helpful_champion_analysis['is_helpful_preserving']:
            report += f"""- ✅ **Helpfulness Preserved**: The Helpful-Champion maintains baseline helpfulness levels
- 🎯 **Optimal Balance**: Achieves {helpful_champion_analysis['combined_score']:.4f} combined score while preserving helpfulness
- 📈 **Improvement**: {helpful_champion_analysis['helpfulness_change_pct']:+.1f}% helpfulness change vs baseline
"""
        else:
            report += f"""- ⚠️ **Helpfulness Trade-off**: No models fully preserve baseline helpfulness
- 🎯 **Minimal Impact**: Helpful-Champion has smallest helpfulness drop ({helpful_champion_analysis['helpfulness_change_pct']:+.1f}%)
- ⚖️ **Balanced Performance**: Maintains competitive combined score ({helpful_champion_analysis['combined_score']:.4f})
"""

        report += f"""
## Usage Recommendations

### When to Use Helpful-Champion
1. **Helpfulness-Critical Applications**: When maintaining user helpfulness is paramount
2. **Balanced Performance**: Need good overall performance without sacrificing helpfulness
3. **Conservative Deployment**: Prefer models that don't reduce helpfulness vs baseline

### Comparison with Other Champions
- **vs Champion Model**: May have lower peak performance but better helpfulness preservation
- **vs Median Seed**: Specifically optimized for helpfulness maintenance
- **Use Case**: Choose based on whether helpfulness preservation is a priority

## Data Provenance

### Helpful-Champion Model Source
- **Evaluation File**: `{champion['file_path']}`
- **Evaluator**: {champion['evaluator']}
- **Timestamp**: {champion['timestamp']}
- **Scenarios**: {champion['num_scenarios']}

### Baseline Reference
- **Baseline File**: `{baseline['file_path']}`
- **Baseline Helpfulness**: {baseline['average_helpfulness_score']:.4f}

### Analysis Methodology
1. **Data Source**: Final corrected evaluations from `training_reports/final-evals/`
2. **Evaluator**: Claude-3.5-Sonnet only (consistent judgment)
3. **Selection Criteria**: Maximize helpfulness preservation + combined performance
4. **Total Models**: {len(ranked_models)} ArGen models analyzed

---
*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
*Analysis Script*: `helpful_champion_analysis_final.py`  
*Data Source*: `../data/consolidated_final_evals.json`  
*Selection Strategy*: {'Helpfulness-preserving with max combined score' if helpful_champion_analysis['is_helpful_preserving'] else 'Minimal helpfulness drop with competitive performance'}
"""
        
        return report

    def run_analysis(self) -> None:
        """Run the complete Helpful-Champion analysis."""
        print("🤝 Starting Helpful-Champion Analysis")
        print("=" * 45)
        
        # Load and filter data
        self.load_consolidated_data()
        self.filter_claude_evaluations()
        
        # Get baseline for comparison
        baseline = self.get_best_baseline()
        if not baseline:
            raise ValueError("No baseline model found for comparison!")
        
        print(f"✓ Baseline helpfulness: {baseline['average_helpfulness_score']:.4f}")
        
        # Calculate helpfulness preservation
        helpfulness_analysis = self.calculate_helpfulness_preservation(baseline)
        
        # Identify helpful champion
        helpful_champion_analysis, ranked_models = self.identify_helpful_champion(helpfulness_analysis)
        
        # Analyze trade-offs
        trade_off_analysis = self.analyze_helpfulness_trade_offs(helpfulness_analysis)
        
        # Generate report
        report = self.generate_helpful_champion_report(helpful_champion_analysis, ranked_models, 
                                                     baseline, trade_off_analysis)
        
        # Save report
        report_path = self.reports_path / "helpful_champion_final_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to: {report_path}")
        print("\n🎉 Helpful-Champion analysis complete!")

if __name__ == "__main__":
    analyzer = HelpfulChampionAnalyzer()
    analyzer.run_analysis()
