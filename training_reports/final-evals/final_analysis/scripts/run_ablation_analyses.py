#!/usr/bin/env python3
"""
Run All Ablation Analyses - Master Script

This script runs all ablation analysis scripts in the correct sequence
and generates a comprehensive execution summary.

Usage: python run_ablation_analyses.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class AblationAnalysisRunner:
    def __init__(self):
        """Initialize the ablation analysis runner."""
        self.scripts_path = Path(".")
        self.reports_path = Path("../reports")
        self.reports_path.mkdir(exist_ok=True)
        
        # Define analysis scripts in execution order
        self.analysis_scripts = [
            {
                'name': 'Data Consolidation',
                'script': 'ablation_data_consolidator.py',
                'description': 'Consolidate Claude and Gemini ablation evaluation data',
                'required': True
            },
            {
                'name': 'Statistical Analysis',
                'script': 'ablation_statistical_analysis.py',
                'description': 'Perform statistical comparisons and effect size calculations',
                'required': True
            },
            {
                'name': 'Evaluator Consistency',
                'script': 'ablation_evaluator_consistency.py',
                'description': 'Analyze consistency between Claude and Gemini evaluations',
                'required': False
            },
            {
                'name': 'Comprehensive Report',
                'script': 'ablation_comprehensive_report.py',
                'description': 'Generate comprehensive ablation analysis report',
                'required': True
            }
        ]
        
        self.execution_results = []

    def run_script(self, script_info: Dict) -> Tuple[bool, str, float]:
        """Run a single analysis script."""
        script_name = script_info['script']
        script_path = self.scripts_path / script_name
        
        if not script_path.exists():
            return False, f"Script not found: {script_path}", 0.0
        
        print(f"🚀 Running {script_info['name']}...")
        print(f"   Script: {script_name}")
        print(f"   Description: {script_info['description']}")
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=self.scripts_path
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {script_info['name']} completed successfully ({execution_time:.1f}s)")
                return True, result.stdout, execution_time
            else:
                print(f"❌ {script_info['name']} failed (exit code: {result.returncode})")
                print(f"   Error: {result.stderr}")
                return False, result.stderr, execution_time
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Exception running {script_name}: {str(e)}"
            print(f"❌ {script_info['name']} failed with exception")
            print(f"   Error: {error_msg}")
            return False, error_msg, execution_time

    def run_all_analyses(self) -> None:
        """Run all ablation analysis scripts in sequence."""
        print("🎯 Starting Comprehensive Ablation Analysis")
        print("=" * 60)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        total_start_time = time.time()
        successful_runs = 0
        failed_runs = 0
        
        for i, script_info in enumerate(self.analysis_scripts, 1):
            print(f"[{i}/{len(self.analysis_scripts)}] {script_info['name']}")
            print("-" * 40)
            
            success, output, exec_time = self.run_script(script_info)
            
            result = {
                'script_info': script_info,
                'success': success,
                'output': output,
                'execution_time': exec_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.execution_results.append(result)
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
                
                # Stop execution if required script fails
                if script_info['required']:
                    print(f"\n❌ Required script failed: {script_info['name']}")
                    print("   Stopping execution to prevent cascading failures.")
                    break
            
            print()
        
        total_execution_time = time.time() - total_start_time
        
        print("=" * 60)
        print("🏁 Ablation Analysis Execution Complete")
        print(f"Total Time: {total_execution_time:.1f} seconds")
        print(f"Successful: {successful_runs}/{len(self.execution_results)}")
        print(f"Failed: {failed_runs}/{len(self.execution_results)}")
        
        if failed_runs == 0:
            print("🎉 All analyses completed successfully!")
        else:
            print("⚠️  Some analyses failed. Check the execution summary for details.")

    def generate_execution_summary(self) -> str:
        """Generate execution summary report."""
        total_time = sum(result['execution_time'] for result in self.execution_results)
        successful_count = sum(1 for result in self.execution_results if result['success'])
        
        summary = f"""# Ablation Analysis Execution Summary

## Overview

**Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Scripts**: {len(self.execution_results)}
**Successful**: {successful_count}
**Failed**: {len(self.execution_results) - successful_count}
**Total Execution Time**: {total_time:.1f} seconds

## Script Execution Details

| # | Script | Status | Time (s) | Description |
|---|--------|--------|----------|-------------|
"""
        
        for i, result in enumerate(self.execution_results, 1):
            script_info = result['script_info']
            status = "✅ Success" if result['success'] else "❌ Failed"
            
            summary += f"| {i} | {script_info['name']} | {status} | {result['execution_time']:.1f} | {script_info['description']} |\n"
        
        summary += """
## Detailed Results

"""
        
        for i, result in enumerate(self.execution_results, 1):
            script_info = result['script_info']
            status_icon = "✅" if result['success'] else "❌"
            
            summary += f"""### {i}. {script_info['name']} {status_icon}

**Script**: `{script_info['script']}`
**Status**: {'Success' if result['success'] else 'Failed'}
**Execution Time**: {result['execution_time']:.1f} seconds
**Timestamp**: {result['timestamp']}

"""
            
            if result['success']:
                summary += "**Output Summary**: Script completed successfully\n\n"
            else:
                summary += f"""**Error Details**:
```
{result['output'][:500]}{'...' if len(result['output']) > 500 else ''}
```

"""
        
        # Add generated files summary
        summary += """## Generated Files

### Data Files
- `consolidated_ablation_evals.json` - Unified ablation dataset
- `ablation_statistical_results.json` - Statistical analysis results
- `ablation_consistency_results.json` - Evaluator consistency analysis
- `ablation_performance_matrix.csv` - Performance comparison matrix

### Reports
- `ablation_data_consolidation_report.md` - Data consolidation validation
- `ablation_comprehensive_report.md` - Main analysis report
- `ablation_evaluator_consistency_report.md` - Cross-evaluator validation
- `ablation_execution_summary.md` - This execution summary

## Next Steps

"""
        
        if successful_count == len(self.execution_results):
            summary += """✅ **All analyses completed successfully**

1. Review the comprehensive ablation report for key findings
2. Use the statistical results for publication and further analysis
3. Reference the evaluator consistency report for robustness validation
4. Consider the performance matrix for additional comparisons

"""
        else:
            summary += """⚠️ **Some analyses failed**

1. Check the error details above for failed scripts
2. Resolve any data or dependency issues
3. Re-run failed scripts individually or use this master script again
4. Ensure all required input files are available

"""
        
        summary += f"""## Analysis Framework

This execution used the comprehensive ablation analysis framework consisting of:

1. **Data Consolidation**: Unified Claude and Gemini ablation evaluations
2. **Statistical Analysis**: Effect sizes, performance comparisons, and degradation analysis
3. **Evaluator Consistency**: Cross-evaluator validation and bias detection
4. **Report Generation**: Comprehensive findings and recommendations

For questions or issues, refer to individual script documentation or the main analysis README.

---
*Summary Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Master Script*: `run_ablation_analyses.py`
*Framework Version*: 1.0
"""
        
        return summary

    def save_execution_summary(self) -> None:
        """Save the execution summary report."""
        print("📝 Generating execution summary...")
        
        summary = self.generate_execution_summary()
        summary_path = self.reports_path / "ablation_execution_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"✅ Execution summary saved to: {summary_path}")

    def run(self) -> None:
        """Run the complete ablation analysis suite."""
        # Run all analyses
        self.run_all_analyses()
        
        # Generate execution summary
        self.save_execution_summary()
        
        print(f"\n📊 Ablation analysis suite complete!")
        print(f"📁 Check the reports directory for all generated files")

if __name__ == "__main__":
    runner = AblationAnalysisRunner()
    runner.run()
