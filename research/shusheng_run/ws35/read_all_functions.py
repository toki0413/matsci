import sys

# Read the entire file
with open('/workspace/research/shusheng_run/ws35/analysis.py', 'r', encoding='utf-8') as f:
    lines = f.read()

# Write to a temp file for inspection
with open('/workspace/research/shusheng_run/ws35/analysis_full.txt', 'w', encoding='utf-8') as f:
    f.write(lines)

print("Full analysis.py written to analysis_full.txt")
print(f"Total characters: {len(lines)}")

# Now let's count lines
line_count = lines.count('\n')
print(f"Total lines: {line_count}")