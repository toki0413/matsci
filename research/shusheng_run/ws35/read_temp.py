import csv
import sys

with open('/workspace/research/shusheng_run/ws35/analysis.py', 'r') as f:
    content = f.read()
print(content, file=sys.stderr)