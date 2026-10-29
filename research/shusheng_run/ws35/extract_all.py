import re

# Read the entire file
with open('/workspace/research/shusheng_run/ws35/analysis_full.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all function definitions using regex
# This pattern matches def function_name followed by parameters and then the function body
# We'll use a simpler approach: split by 'def ' and process each section

sections = content.split('def ')
for section in sections:
    if not section.strip():
        continue
    # Extract function name
    first_line = section.split('\n')[0]
    match = re.match(r'(\w+)\s*\(', first_line)
    if match:
        func_name = match.group(1)
        print(f"\n=== {func_name} ===")
        # Print the first 200 chars of the function
        func_body = section[:2000]  # Just print first 2000 chars for now
        print(func_body)
        print("...")