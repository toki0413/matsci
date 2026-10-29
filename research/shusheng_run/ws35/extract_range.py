import sys

def extract_lines(filepath, start_line, end_line):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Convert to 0-based index
    start = start_line - 1
    end = end_line  # exclusive
    return ''.join(lines[start:end])

# Extract h1_analysis (lines 82-197)
h1 = extract_lines('/workspace/research/shusheng_run/ws35/analysis.py', 82, 197)
print("=== h1_analysis ===")
print(h1)

print("\n" + "="*80 + "\n")

# Extract h2_analysis (lines 198-288)
h2 = extract_lines('/workspace/research/shusheng_run/ws35/analysis.py', 198, 288)
print("=== h2_analysis ===")
print(h2)

print("\n" + "="*80 + "\n")

# Extract h3_analysis (lines 289-327)
h3 = extract_lines('/workspace/research/shusheng_run/ws35/analysis.py', 289, 327)
print("=== h3_analysis ===")
print(h3)

print("\n" + "="*80 + "\n")

# Extract h4_analysis (lines 328-386)
h4 = extract_lines('/workspace/research/shusheng_run/ws35/analysis.py', 328, 386)
print("=== h4_analysis ===")
print(h4)

print("\n" + "="*80 + "\n")

# Extract h5_prediction (lines 387-423)
h5 = extract_lines('/workspace/research/shusheng_run/ws35/analysis.py', 387, 423)
print("=== h5_prediction ===")
print(h5)