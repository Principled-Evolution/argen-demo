#!/usr/bin/env python3
"""
Organize Evaluation Files Script

This script creates a clean folder structure and organizes evaluation files
by evaluator type to prevent future corruption and enable proper analysis.

Usage: python organize_evaluation_files.py
"""

import json
import os
import shutil
from pathlib import Path
import pandas as pd

def create_clean_folder_structure():
    """Create clean folder structure for organized evaluations."""
    
    base_dir = "training_reports_clean"
    
    # Create main directories
    directories = [
        f"{base_dir}/claude_evaluations/grpo5_seed1",
        f"{base_dir}/claude_evaluations/grpo6_seed2", 
        f"{base_dir}/claude_evaluations/grpo7_seed3",
        f"{base_dir}/claude_evaluations/baseline",
        f"{base_dir}/gemini_evaluations/grpo5_seed1",
        f"{base_dir}/gemini_evaluations/grpo6_seed2",
        f"{base_dir}/gemini_evaluations/grpo7_seed3", 
        f"{base_dir}/gemini_evaluations/baseline",
        f"{base_dir}/openai_evaluations/grpo5_seed1",
        f"{base_dir}/openai_evaluations/grpo6_seed2",
        f"{base_dir}/openai_evaluations/grpo7_seed3",
        f"{base_dir}/openai_evaluations/baseline",
        f"{base_dir}/other_evaluations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return base_dir

def get_target_directory(base_dir: str, evaluator_type: str, model_type: str, seed: str) -> str:
    """Get target directory for a file based on evaluator and model info."""
    
    if evaluator_type == 'claude':
        evaluator_dir = 'claude_evaluations'
    elif evaluator_type == 'gemini':
        evaluator_dir = 'gemini_evaluations'
    elif evaluator_type == 'openai':
        evaluator_dir = 'openai_evaluations'
    else:
        return f"{base_dir}/other_evaluations"
    
    if model_type == 'baseline':
        model_dir = 'baseline'
    elif model_type == 'grpo5':
        model_dir = 'grpo5_seed1'
    elif model_type == 'grpo6':
        model_dir = 'grpo6_seed2'
    elif model_type == 'grpo7':
        model_dir = 'grpo7_seed3'
    else:
        return f"{base_dir}/other_evaluations"
    
    return f"{base_dir}/{evaluator_dir}/{model_dir}"

def generate_clean_filename(original_path: str, evaluator_type: str, model_type: str, 
                          seed: str, checkpoint: str, timestamp: str) -> str:
    """Generate clean, unique filename that prevents overwrites."""
    
    # Extract original filename parts
    original_name = os.path.basename(original_path)
    
    # Create clean filename with evaluator and timestamp
    if model_type == 'baseline':
        model_part = 'baseline'
    else:
        model_part = f"{model_type}_seed{seed}"
    
    if checkpoint == 'final':
        checkpoint_part = 'final'
    else:
        checkpoint_part = f"checkpoint-{checkpoint}"
    
    # Clean timestamp for filename
    clean_timestamp = timestamp.replace(':', '').replace('-', '').replace('T', '_')[:15]
    
    # Generate filename
    filename = f"eval_{model_part}_{checkpoint_part}_{evaluator_type}_{clean_timestamp}.json"
    
    return filename

def organize_evaluation_files():
    """Organize all evaluation files into clean structure."""
    
    # Load audit results
    if not os.path.exists('evaluation_files_audit_detailed.csv'):
        print("Error: Please run audit_evaluation_files.py first!")
        return
    
    df = pd.read_csv('evaluation_files_audit_detailed.csv')
    
    # Create clean folder structure
    print("Creating clean folder structure...")
    base_dir = create_clean_folder_structure()
    
    # Track file operations
    operations = []
    
    print("\nOrganizing files...")
    
    for _, row in df.iterrows():
        original_path = row['file_path']
        evaluator_type = row['evaluator_type']
        model_type = row['model_type']
        seed = str(row['seed'])
        checkpoint = str(row['checkpoint'])
        timestamp = row['eval_timestamp']
        
        # Skip error files
        if evaluator_type == 'error':
            print(f"Skipping error file: {original_path}")
            continue
        
        # Get target directory
        target_dir = get_target_directory(base_dir, evaluator_type, model_type, seed)
        
        # Generate clean filename
        clean_filename = generate_clean_filename(
            original_path, evaluator_type, model_type, seed, checkpoint, timestamp
        )
        
        target_path = os.path.join(target_dir, clean_filename)
        
        # Copy file
        try:
            shutil.copy2(original_path, target_path)
            print(f"✓ Copied: {original_path} -> {target_path}")
            
            operations.append({
                'original_path': original_path,
                'target_path': target_path,
                'evaluator_type': evaluator_type,
                'model_type': model_type,
                'seed': seed,
                'checkpoint': checkpoint,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"✗ Error copying {original_path}: {e}")
            operations.append({
                'original_path': original_path,
                'target_path': target_path,
                'evaluator_type': evaluator_type,
                'model_type': model_type,
                'seed': seed,
                'checkpoint': checkpoint,
                'status': f'error: {e}'
            })
    
    # Save operations log
    operations_df = pd.DataFrame(operations)
    operations_df.to_csv(f'{base_dir}/file_organization_log.csv', index=False)
    
    # Generate summary report
    generate_organization_summary(base_dir, operations_df)
    
    print(f"\n✅ File organization complete!")
    print(f"📁 Clean files organized in: {base_dir}/")
    print(f"📊 Organization log: {base_dir}/file_organization_log.csv")

def generate_organization_summary(base_dir: str, operations_df: pd.DataFrame):
    """Generate summary of file organization."""
    
    successful_ops = operations_df[operations_df['status'] == 'success']
    
    summary = f"""# File Organization Summary

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Base directory: {base_dir}

## Summary Statistics

Total files processed: {len(operations_df)}
Successfully organized: {len(successful_ops)}
Errors: {len(operations_df) - len(successful_ops)}

## Files by Evaluator Type

"""
    
    evaluator_summary = successful_ops.groupby('evaluator_type').size()
    for evaluator, count in evaluator_summary.items():
        summary += f"- {evaluator}: {count} files\n"
    
    summary += "\n## Files by Model Type\n\n"
    
    model_summary = successful_ops.groupby(['evaluator_type', 'model_type']).size().unstack(fill_value=0)
    summary += model_summary.to_string() + "\n\n"
    
    # Claude evaluations (for Champion/Median/Helpful selection)
    claude_files = successful_ops[successful_ops['evaluator_type'] == 'claude']
    summary += f"## Claude Evaluations (For Analysis)\n\n"
    summary += f"Total Claude files: {len(claude_files)}\n\n"
    
    if len(claude_files) > 0:
        claude_by_model = claude_files.groupby(['model_type', 'seed', 'checkpoint']).size()
        summary += "### Claude Files by Model:\n"
        for (model_type, seed, checkpoint), count in claude_by_model.items():
            summary += f"- {model_type} seed {seed} checkpoint {checkpoint}: {count} file(s)\n"
    
    summary += f"""
## Directory Structure

```
{base_dir}/
├── claude_evaluations/          # For Champion/Median/Helpful selection
│   ├── grpo5_seed1/
│   ├── grpo6_seed2/
│   ├── grpo7_seed3/
│   └── baseline/
├── gemini_evaluations/          # For tabulation only
│   ├── grpo5_seed1/
│   ├── grpo6_seed2/
│   ├── grpo7_seed3/
│   └── baseline/
├── openai_evaluations/          # For tabulation only
│   ├── grpo5_seed1/
│   ├── grpo6_seed2/
│   ├── grpo7_seed3/
│   └── baseline/
└── other_evaluations/           # Other evaluators
```

## Next Steps

1. Update analysis scripts to use claude_evaluations/ for Champion/Median/Helpful selection
2. Update tabulation scripts to include all evaluator types
3. Re-run Champion/Median/Helpful analysis with clean Claude data
4. Generate comprehensive comparison tables with all evaluators

---
*File organization completed successfully*
"""
    
    with open(f'{base_dir}/organization_summary.md', 'w') as f:
        f.write(summary)
    
    print(f"📄 Summary saved to: {base_dir}/organization_summary.md")

if __name__ == "__main__":
    print("🗂️  Starting file organization...")
    organize_evaluation_files()
    print("\n🎉 File organization complete!")
