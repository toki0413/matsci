import sys

def extract_lines(filepath, start_line, end_line):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Convert to 0-based index
    start = start_line - 1
    end = end_line  # exclusive
    return ''.join(lines[start:end])

# Extract each function and write to a file
functions = [
    ('h1_analysis', 82, 197),
    ('h2_analysis', 198, 288),
    ('h3_analysis', 289, 327),
    ('h4_analysis', 328, 386),
    ('h5_prediction', 387, 423),
]

for name, start, end in functions:
    content = extract_lines('/workspace/research/shusheng_run/ws35/analysis.py', start, end)
    filename = f'{name}.py'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote {name} to {filename} ({len(content.splitlines())} lines)")

print("\nAll functions extracted to separate files.")