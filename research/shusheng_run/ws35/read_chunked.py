import sys
with open('/workspace/research/shusheng_run/ws35/comprehensive_analysis.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    sys.stdout.write(f'{i+1:4}: {line}')