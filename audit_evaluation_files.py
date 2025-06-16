#!/usr/bin/env python3
"""
Evaluation Files Audit Script

This script audits all evaluation files to identify:
1. Which evaluator was used (Claude, Gemini, etc.)
2. File timestamps and organization
3. Missing or corrupted data
4. Proper categorization for Champion/Median/Helpful analysis

Usage: python audit_evaluation_files.py
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

def extract_evaluator_info(file_path: str) -> Tuple[str, str, str]:
    """Extract evaluator information from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        config = data.get('evaluation_config', {})
        evaluator = config.get('evaluator', 'Unknown')
        timestamp = config.get('timestamp', 'Unknown')
        model_name = config.get('model_name', 'Unknown')
        
        # Categorize evaluator
        evaluator_lower = evaluator.lower()
        if 'claude' in evaluator_lower or 'anthropic' in evaluator_lower:
            evaluator_type = 'claude'
        elif 'gemini' in evaluator_lower:
            evaluator_type = 'gemini'
        elif 'gpt' in evaluator_lower or 'openai' in evaluator_lower:
            evaluator_type = 'openai'
        else:
            evaluator_type = 'other'
            
        return evaluator_type, evaluator, timestamp
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 'error', 'error', 'error'

def parse_model_info(model_name: str) -> Dict[str, str]:
    """Parse model information from model name."""
    info = {
        'seed': 'unknown',
        'checkpoint': 'unknown',
        'model_type': 'unknown'
    }
    
    if 'grpo_5_seed_1' in model_name:
        info['seed'] = '1'
        info['model_type'] = 'grpo5'
    elif 'grpo_6_seed_2_4' in model_name:
        info['seed'] = '2'
        info['model_type'] = 'grpo6'
    elif 'grpo_7_seed_3_3' in model_name:
        info['seed'] = '3'
        info['model_type'] = 'grpo7'
    elif 'Llama-3.2-1B-Instruct' in model_name:
        info['model_type'] = 'baseline'
        info['seed'] = 'baseline'
    
    if 'checkpoint-' in model_name:
        try:
            checkpoint = model_name.split('checkpoint-')[1].split('/')[0]
            info['checkpoint'] = checkpoint
        except:
            info['checkpoint'] = 'unknown'
    else:
        info['checkpoint'] = 'final'
    
    return info

def audit_all_evaluation_files() -> List[Dict]:
    """Audit all evaluation JSON files in training_reports."""
    
    # Find all JSON files in training_reports
    json_files = []
    for pattern in [
        'training_reports/**/*.json',
        'training_reports/**/eval_*.json'
    ]:
        json_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    json_files = list(set(json_files))
    
    audit_results = []
    
    print(f"Found {len(json_files)} JSON files to audit...")
    
    for file_path in sorted(json_files):
        print(f"Auditing: {file_path}")
        
        # Extract file info
        file_stat = os.stat(file_path)
        file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
        file_size = file_stat.st_size
        
        # Extract evaluator info
        evaluator_type, evaluator_full, timestamp = extract_evaluator_info(file_path)
        
        # Extract model info
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            model_name = data.get('evaluation_config', {}).get('model_name', 'Unknown')
            num_scenarios = data.get('evaluation_config', {}).get('num_scenarios', 0)
            combined_score = data.get('summary_metrics', {}).get('average_combined_score', 0)
            helpfulness_score = data.get('summary_metrics', {}).get('average_helpfulness_score', 0)
        except:
            model_name = 'Error'
            num_scenarios = 0
            combined_score = 0
            helpfulness_score = 0
        
        model_info = parse_model_info(model_name)
        
        # Determine folder category
        folder_path = os.path.dirname(file_path)
        if 'grpo5-evals' in folder_path:
            folder_category = 'organized_grpo5'
        elif 'grpo6-grpo7-evals' in folder_path:
            folder_category = 'organized_grpo6_grpo7'
        elif 'baseline-eval' in folder_path:
            folder_category = 'organized_baseline'
        elif 'evaluate_baseline_' in folder_path:
            folder_category = 'timestamped'
        else:
            folder_category = 'other'
        
        audit_results.append({
            'file_path': file_path,
            'folder_category': folder_category,
            'file_mtime': file_mtime,
            'file_size': file_size,
            'evaluator_type': evaluator_type,
            'evaluator_full': evaluator_full,
            'eval_timestamp': timestamp,
            'model_name': model_name,
            'seed': model_info['seed'],
            'checkpoint': model_info['checkpoint'],
            'model_type': model_info['model_type'],
            'num_scenarios': num_scenarios,
            'combined_score': combined_score,
            'helpfulness_score': helpfulness_score
        })
    
    return audit_results

def generate_audit_report(audit_results: List[Dict]) -> str:
    """Generate comprehensive audit report."""
    
    df = pd.DataFrame(audit_results)
    
    report = f"""# Evaluation Files Audit Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total files audited: {len(audit_results)}

## Summary by Evaluator Type

"""
    
    # Evaluator summary
    evaluator_summary = df.groupby('evaluator_type').agg({
        'file_path': 'count',
        'combined_score': 'mean',
        'helpfulness_score': 'mean'
    }).round(4)
    
    report += evaluator_summary.to_string() + "\n\n"
    
    # Folder category summary
    report += "## Summary by Folder Category\n\n"
    folder_summary = df.groupby(['folder_category', 'evaluator_type']).size().unstack(fill_value=0)
    report += folder_summary.to_string() + "\n\n"
    
    # Claude evaluations (for Champion/Median/Helpful selection)
    claude_evals = df[df['evaluator_type'] == 'claude']
    report += f"## Claude Evaluations (For Champion/Median/Helpful Selection)\n\n"
    report += f"Total Claude evaluations: {len(claude_evals)}\n\n"
    
    if len(claude_evals) > 0:
        claude_by_model = claude_evals.groupby(['model_type', 'seed', 'checkpoint']).agg({
            'combined_score': 'first',
            'helpfulness_score': 'first',
            'file_path': 'first'
        }).round(4)
        report += claude_by_model.to_string() + "\n\n"
    
    # Gemini evaluations (for tabulation)
    gemini_evals = df[df['evaluator_type'] == 'gemini']
    report += f"## Gemini Evaluations (For Tabulation Only)\n\n"
    report += f"Total Gemini evaluations: {len(gemini_evals)}\n\n"
    
    if len(gemini_evals) > 0:
        gemini_by_model = gemini_evals.groupby(['model_type', 'seed', 'checkpoint']).agg({
            'combined_score': 'first',
            'helpfulness_score': 'first',
            'file_path': 'first'
        }).round(4)
        report += gemini_by_model.to_string() + "\n\n"
    
    # Missing evaluations
    report += "## Missing Evaluations Analysis\n\n"
    
    # Expected models
    expected_models = [
        ('grpo5', '1', 'final'),
        ('grpo5', '1', '1000'),
        ('grpo5', '1', '2000'),
        ('grpo5', '1', '2600'),
        ('grpo5', '1', '3000'),
        ('grpo6', '2', 'final'),
        ('grpo6', '2', '1000'),
        ('grpo6', '2', '2000'),
        ('grpo6', '2', '3000'),
        ('grpo6', '2', '4000'),
        ('grpo7', '3', 'final'),
        ('grpo7', '3', '1000'),
        ('grpo7', '3', '2000'),
        ('grpo7', '3', '3000'),
        ('grpo7', '3', '4000'),
        ('baseline', 'baseline', 'baseline')
    ]
    
    claude_models = set((row['model_type'], row['seed'], row['checkpoint']) 
                       for _, row in claude_evals.iterrows())
    
    missing_claude = []
    for expected in expected_models:
        if expected not in claude_models:
            missing_claude.append(expected)
    
    if missing_claude:
        report += "### Missing Claude Evaluations:\n"
        for model_type, seed, checkpoint in missing_claude:
            report += f"- {model_type} seed {seed} checkpoint {checkpoint}\n"
    else:
        report += "### ✅ All expected Claude evaluations found!\n"
    
    report += "\n"
    
    return report

if __name__ == "__main__":
    print("🔍 Starting evaluation files audit...")
    
    # Audit all files
    audit_results = audit_all_evaluation_files()
    
    # Generate report
    print("\n📝 Generating audit report...")
    report = generate_audit_report(audit_results)
    
    # Save report
    with open('evaluation_files_audit_report.md', 'w') as f:
        f.write(report)
    
    # Save detailed CSV
    df = pd.DataFrame(audit_results)
    df.to_csv('evaluation_files_audit_detailed.csv', index=False)
    
    print("✅ Audit complete!")
    print("📄 Report saved to: evaluation_files_audit_report.md")
    print("📊 Detailed data saved to: evaluation_files_audit_detailed.csv")
    print("\n🔍 Key findings:")
    
    # Quick summary
    evaluator_counts = df['evaluator_type'].value_counts()
    for evaluator, count in evaluator_counts.items():
        print(f"  - {evaluator}: {count} files")
