#!/usr/bin/env python3
"""
Workflow Runner for SCM MgH2 Hydrogen Absorption Simulation
This script orchestrates the complete computational workflow.
"""

import sys
import os
import subprocess
from typing import Optional

def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Completed successfully!")
        if result.stdout:
            print(f"  Output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED! Return code: {e.returncode}")
        if e.stderr:
            print(f"  Error: {e.stderr[:200]}...")
        return False
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False

def main():
    """Main workflow runner."""
    print("=" * 60)
    print("SCM MgH2 Hydrogen Absorption - Workflow Runner")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required!")
        sys.exit(1)
    
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check for required packages
    print("\nChecking required packages...")
    packages = ['numpy', 'pandas']
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  {pkg}: OK")
        except ImportError:
            print(f"  {pkg}: MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\nError: Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    # Optional: Check matplotlib
    try:
        __import__('matplotlib')
        print("  matplotlib: OK (optional)")
    except ImportError:
        print("  matplotlib: NOT FOUND (visualizations will be skipped)")
    
    # Create directories
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    
    # Step 1: Run main simulation
    print("\n" + "=" * 60)
    print("Step 1: Running SCM Simulation")
    print("=" * 60)
    
    if not run_command([sys.executable, 'scm_mgh2_simulation.py'], 
                       "Running main simulation..."):
        print("Error: Failed to run simulation!")
        sys.exit(1)
    
    # Step 2: Generate visualizations (optional)
    print("\n" + "=" * 60)
    print("Step 2: Generating Visualizations")
    print("=" * 60)
    
    try:
        run_command([sys.executable, 'scm_visualization.py'], 
                    "Generating visualizations...")
    except:
        print("Visualizations skipped (matplotlib may not be available)")
    
    # Step 3: Run analysis
    print("\n" + "=" * 60)
    print("Step 3: Performing Analysis")
    print("=" * 60)
    
    if not run_command([sys.executable, 'scm_analysis.py'], 
                       "Running comprehensive analysis..."):
        print("Error: Failed to run analysis!")
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Workflow Completed Successfully!")
    print("=" * 60)
    print("\nOutput files:")
    results_dir = './results'
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        for f in sorted(files):
            fpath = os.path.join(results_dir, f)
            size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
            print(f"  {f} ({size:,} bytes)")
    
    print("\nView analysis report:")
    print("  cat ./results/analysis_report.txt")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()