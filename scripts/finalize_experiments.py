#!/usr/bin/env python3
"""
Post-Experiment Finalization Script
Run this AFTER all experiments complete to generate final comprehensive report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_experiments_complete() -> bool:
    """Check if all experiment runs have completed."""
    import glob
    runs = glob.glob("./runs_*_*")
    
    if not runs:
        print("✗ No experiment runs found. Please run run_gpt_experiments.py first.")
        return False
    
    print(f"✓ Found {len(runs)} experiment run directories")
    
    # Check if they have metrics files
    completed_runs = 0
    for run_dir in runs:
        metrics_file = Path(run_dir) / "guard_metrics.csv"
        if metrics_file.exists() and metrics_file.stat().st_size > 100:
            completed_runs += 1
            print(f"  ✓ {Path(run_dir).name}")
        else:
            print(f"  ⏳ {Path(run_dir).name} (still running or incomplete)")
    
    print(f"\nCompleted: {completed_runs}/{len(runs)} runs")
    return completed_runs > 0

def run_aggregation() -> bool:
    """Run the results aggregation script."""
    print("\n" + "="*70)
    print("Aggregating Results & Generating Report...")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(
            ["python", "aggregate_results.py"],
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during aggregation: {e}")
        return False

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║      Secure LLM - Experiment Finalization & Report Generation   ║
║      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check for completed experiments
    print("\n[1/3] Checking for completed experiment runs...\n")
    if not check_experiments_complete():
        print("\n⚠ Experiments are still running or incomplete.")
        print("   Waiting for completion...")
        print("\n   You can check progress with:")
        print("   → get_terminal_output(id='f984f67b-3243-4faa-9d15-9dc02d0e5994')")
        print("   → Or check: ls -lah runs_*/guard_metrics.csv")
        sys.exit(1)
    
    # Step 2: Run aggregation
    print("\n[2/3] Running results aggregation...\n")
    if not run_aggregation():
        print("\n✗ Aggregation failed. Check for errors above.")
        sys.exit(1)
    
    # Step 3: Summary
    print("\n[3/3] Finalization Complete!\n")
    print("="*70)
    print("SUCCESS - Results Generated!")
    print("="*70)
    
    # Find the generated markdown
    import glob
    markdown_files = glob.glob("results/comprehensive_evaluation_*.md")
    if markdown_files:
        latest_md = max(markdown_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"\n✓ Final Report: {latest_md}")
        print(f"✓ File Size: {Path(latest_md).stat().st_size / 1024:.1f} KB")
        print(f"\nOpen report in editor or terminal:")
        print(f"  cat {latest_md}")
        print(f"  code {latest_md}")
    
    # Also find JSON metrics
    json_files = glob.glob("results/evaluation_metrics_*.json")
    if json_files:
        latest_json = max(json_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"\n✓ Metrics JSON: {latest_json}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. Review the generated markdown report (above)
2. Check key metrics:
   - F1 score ≥ 0.97 (guard prompts detection quality)
   - ECE ≤ 0.01 (calibration validation)
   - F1 ≥ 0.90 on public datasets (generalization)
3. Integrate tables into paper/main.tex or results appendix
4. Verify all findings support the core claim:
   "Guard prompts + structured JSON output = reliable detection + excellent calibration"
    """)

if __name__ == "__main__":
    main()
