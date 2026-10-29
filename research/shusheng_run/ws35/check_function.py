#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import importlib
# Reload the module to get the updated version
if 'comprehensive_analysis' in sys.modules:
    del sys.modules['comprehensive_analysis']
import comprehensive_analysis as ca
import inspect
source = inspect.getsource(ca.aggregate_bootstrap_results)
print(source)