#!/bin/bash
# SCM MgH2 Hydrogen Absorption Workflow Script
# This script runs the complete computational workflow

echo "========================================"
echo "SCM MgH2 Hydrogen Absorption Workflow"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found!"
    exit 1
fi

echo "Python3: $(python3 --version)"
echo ""

# Check required packages
echo "Checking required packages..."
python3 -c "import numpy; import pandas; print('numpy: numpy.__version__')" && echo "numpy: OK" || echo "numpy: NOT FOUND"
python3 -c "import pandas; print('pandas: pandas.__version__')" && echo "pandas: OK" || echo "pandas: NOT FOUND"
python3 -c "import matplotlib; print('matplotlib: matplotlib.__version__')" && echo "matplotlib: OK" || echo "matplotlib: OK (optional)"
echo ""

# Create output directories
mkdir -p ./results ./data
echo "Directories created."
echo ""

# Step 1: Run main simulation
echo "Step 1: Running main simulation..."
python3 scm_mgh2_simulation.py
if [ $? -ne 0 ]; then
    echo "Error: Simulation failed!"
    exit 1
fi
echo ""

# Step 2: Generate visualizations
echo "Step 2: Generating visualizations..."
python3 scm_visualization.py
if [ $? -ne 0 ]; then
    echo "Error: Visualization failed!"
    exit 1
fi
echo ""

# Step 3: Perform analysis
echo "Step 3: Performing comprehensive analysis..."
python3 scm_analysis.py
if [ $? -ne 0 ]; then
    echo "Error: Analysis failed!"
    exit 1
fi
echo ""

echo "========================================"
echo "Workflow completed successfully!"
echo "========================================"
echo ""
echo "Output files:"
ls -la ./results/ 2>/dev/null || echo "Results directory not found"
echo ""
echo "View the analysis report:"
cat ./results/analysis_report.txt 2>/dev/null || echo "Report not found"