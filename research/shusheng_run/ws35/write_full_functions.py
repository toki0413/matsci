import ast
import sys

# Read the original file
with open('/workspace/research/shusheng_run/ws35/analysis.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Parse the AST
tree = ast.parse(source)

# Find all function definitions
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        # Get the source code for this function
        # We need to find the line range
        start_line = node.lineno - 1  # 0-based
        # Find the end line - look for the next function or class at the same indentation level
        # This is tricky with AST, so let's use a simpler approach: read the file and extract by line numbers
        pass

# Simpler approach: read the file line by line and extract functions based on def lines
with open('/workspace/research/shusheng_run/ws35/analysis.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find function definitions
functions = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('def '):
        func_name = line.split('(')[0][4:].strip()  # Extract function name
        start = i + 1  # Start after the def line
        # Find the end of the function (next def or end of file with same indentation)
        # Simple approach: find the next line that starts with 'def ' or 'class ' at column 0
        j = start
        while j < len(lines) and not (lines[j].startswith('def ') and j != start):
            j += 1
        end = j
        functions.append((func_name, start, end, lines[start:end]))
    i += 1

# Write each function to a file
for func_name, start, end, func_lines in functions:
    filename = f'{func_name}.py'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(''.join(func_lines))
    print(f"Wrote {func_name} to {filename} ({len(func_lines)} lines)")