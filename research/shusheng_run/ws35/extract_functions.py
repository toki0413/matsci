import sys

# Read the entire file
with open('/workspace/research/shusheng_run/ws35/analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and extract h1_analysis function
def extract_function(func_name):
    start = content.find(f'def {func_name}(')
    if start == -1:
        return None
    # Find the end of the function (next def or end of file)
    end = content.find('def ', start + 1)
    if end == -1:
        end = len(content)
    # Also look for top-level function boundaries by counting indentation
    # This is a simple approach - just return from start to next def
    return content[start:end]

h1 = extract_function('h1_analysis')
h2 = extract_function('h2_analysis')
h3 = extract_function('h3_analysis')
h4 = extract_function('h4_analysis')
h5 = extract_function('h5_prediction')

print("=== h1_analysis ===")
print(h1 if h1 else "Not found")
print("\n=== h2_analysis ===")
print(h2 if h2 else "Not found")
print("\n=== h3_analysis ===")
print(h3 if h3 else "Not found")
print("\n=== h4_analysis ===")
print(h4 if h4 else "Not found")
print("\n=== h5_prediction ===")
print(h5 if h5 else "Not found")