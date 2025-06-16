#!/usr/bin/env python3
"""
Master Script - Run All Final Evaluation Analyses

This script runs all analysis scripts in the correct order:
1. Data consolidation
2. Champion model analysis
3. Median seed analysis
4. Helpful-champion analysis
5. Comprehensive report generation
6. Evaluator consistency analysis

Usage: python run_all_analyses.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

class MasterAnalysisRunner:
    def __init__(self):
        """Initialize the master analysis runner."""
        self.scripts_path = Path(".")
        self.reports_path = Path("../reports")
        
        # Define analysis scripts in execution order
        self.analysis_scripts = [
            {
                'name': 'Data Consolidation',
                'script': 'final_evals_consolidator.py',
                'description': 'Consolidate and validate all evaluation data'
            },
            {
                'name': 'Champion Model Analysis',
                'script': 'champion_model_analysis_final.py',
                'description': 'Identify peak performance model'
            },
            {
                'name': 'Median Seed Analysis',
                'script': 'median_seed_analysis_final.py',
                'description': 'Identify representative seed for ablations'
            },
            {
                'name': 'Helpful-Champion Analysis',
                'script': 'helpful_champion_analysis_final.py',
                'description': 'Identify helpfulness-preserving model'
            },
            {
                'name': 'Comprehensive Report',
                'script': 'final_evaluation_comprehensive_report.py',
                'description': 'Generate unified analysis report'
            },
            {
                'name': 'Evaluator Consistency',
                'script': 'evaluator_consistency_analysis.py',
                'description': 'Validate Claude vs Gemini consistency'
            }
        ]

    def run_script(self, script_info: dict) -> bool:
        """Run a single analysis script."""
        script_name = script_info['script']
        script_path = self.scripts_path / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f"🚀 Running: {script_info['name']}")
        print(f"📝 Description: {script_info['description']}")
        print(f"📄 Script: {script_name}")
        print(f"{'='*60}")
        
        try:
            # Run the script
            result = subprocess.run([
                sys.executable, script_name
            ], cwd=self.scripts_path, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {script_info['name']} completed successfully")
                if result.stdout:
                    print("📤 Output:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ {script_info['name']} failed with return code {result.returncode}")
                if result.stderr:
                    print("📤 Error output:")
                    print(result.stderr)
                if result.stdout:
                    print("📤 Standard output:")
                    print(result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {script_info['name']} timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"💥 {script_info['name']} failed with exception: {e}")
            return False

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        print("🔍 Checking dependencies...")
        
        required_modules = ['json', 'pandas', 'numpy', 'scipy', 'pathlib', 'datetime']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module}")
            except ImportError:
                missing_modules.append(module)
                print(f"❌ {module}")
        
        if missing_modules:
            print(f"\n❌ Missing required modules: {missing_modules}")
            print("Please install missing dependencies and try again.")
            return False
        
        print("✅ All dependencies available")
        return True

    def generate_execution_summary(self, results: list) -> str:
        """Generate a summary of the execution results."""
        
        successful = sum(1 for success in results if success)
        total = len(results)
        
        summary = f"""# Final Evaluation Analysis Execution Summary

## Overview
- **Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Scripts**: {total}
- **Successful**: {successful}
- **Failed**: {total - successful}
- **Success Rate**: {(successful/total)*100:.1f}%

## Script Execution Results

| # | Script | Status | Description |
|---|--------|--------|-------------|
"""
        
        for i, (script_info, success) in enumerate(zip(self.analysis_scripts, results), 1):
            status = "✅ Success" if success else "❌ Failed"
            summary += f"| {i} | {script_info['name']} | {status} | {script_info['description']} |\n"
        
        summary += f"""
## Generated Reports

The following reports should be available in `../reports/`:

"""
        
        expected_reports = [
            "data_consolidation_report.md",
            "champion_model_final_report.md", 
            "median_seed_final_report.md",
            "helpful_champion_final_report.md",
            "comprehensive_evaluation_report.md",
            "evaluator_consistency_report.md"
        ]
        
        for report in expected_reports:
            report_path = self.reports_path / report
            if report_path.exists():
                summary += f"- ✅ `{report}`\n"
            else:
                summary += f"- ❌ `{report}` (missing)\n"
        
        summary += f"""
## Data Files

The following data files should be available in `../data/`:

- `consolidated_final_evals.json` - Unified evaluation data
- `model_performance_matrix.csv` - Performance matrix for analysis

## Next Steps

1. **Review Reports**: Check all generated reports for insights
2. **Validate Results**: Ensure champion selections align with expectations
3. **Use for Publication**: Apply findings to research paper and ablation studies

## Troubleshooting

If any scripts failed:
1. Check error messages in the execution log above
2. Ensure all evaluation files are present in `../../claude-runs/` and `../../gemini-runs/`
3. Verify data file integrity
4. Re-run individual scripts if needed

---
*Summary generated by*: `run_all_analyses.py`  
*Execution completed*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return summary

    def run_all_analyses(self) -> None:
        """Run all analysis scripts in sequence."""
        print("🎯 Final Evaluation Analysis - Master Execution")
        print("=" * 60)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Working directory: {Path.cwd()}")
        print(f"📊 Total scripts to run: {len(self.analysis_scripts)}")
        
        # Check dependencies
        if not self.check_dependencies():
            return
        
        # Ensure reports directory exists
        self.reports_path.mkdir(exist_ok=True)
        
        # Run all scripts
        results = []
        failed_scripts = []
        
        for i, script_info in enumerate(self.analysis_scripts, 1):
            print(f"\n📋 Step {i}/{len(self.analysis_scripts)}")
            success = self.run_script(script_info)
            results.append(success)
            
            if not success:
                failed_scripts.append(script_info['name'])
                print(f"⚠️ {script_info['name']} failed - continuing with remaining scripts...")
        
        # Generate execution summary
        summary = self.generate_execution_summary(results)
        summary_path = self.reports_path / "execution_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        # Final status
        print(f"\n{'='*60}")
        print("🏁 FINAL EVALUATION ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        successful = sum(results)
        total = len(results)
        
        print(f"📊 Results: {successful}/{total} scripts successful ({(successful/total)*100:.1f}%)")
        
        if failed_scripts:
            print(f"❌ Failed scripts: {', '.join(failed_scripts)}")
        else:
            print("✅ All scripts completed successfully!")
        
        print(f"📄 Execution summary: {summary_path}")
        print(f"📁 Reports directory: {self.reports_path.absolute()}")
        print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if successful == total:
            print("\n🎉 SUCCESS: All analyses completed successfully!")
            print("📋 Next steps:")
            print("   1. Review generated reports")
            print("   2. Validate champion model selections")
            print("   3. Use findings for publication and ablation studies")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: {total - successful} scripts failed")
            print("🔧 Check error messages and re-run failed scripts individually")

if __name__ == "__main__":
    runner = MasterAnalysisRunner()
    runner.run_all_analyses()
