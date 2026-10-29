#!/usr/bin/env python3
# Fix the aggregate_bootstrap_results function in comprehensive_analysis.py

import re

with open('comprehensive_analysis.py', 'r') as f:
    content = f.read()

# Find the aggregate_bootstrap_results function and replace it
new_func = '''
def aggregate_bootstrap_results(loop_results):
    agg = {
        'h1': {'mean_delta_ea_boot': [], 'ci_lower_boot': [], 'ci_upper_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h2': {'eta_squared_boot': [], 'p_value_perm_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h3': {'ratio_boot': [], 'p_value_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h4': {'rmse_boot': [], 'within_10_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h5': {}
    }
    for lr in loop_results:
        if lr['h1']:
            agg['h1']['mean_delta_ea_boot'].append(lr['h1']['mean_delta_ea'])
            agg['h1']['ci_lower_boot'].append(lr['h1']['ci_95_lower'])
            agg['h1']['ci_upper_boot'].append(lr['h1']['ci_95_upper'])
            if lr['h1']['supported']:
                agg['h1']['supported_count'] += 1
        if lr['h2']:
            agg['h2']['eta_squared_boot'].append(lr['h2']['eta_squared'])
            agg['h2']['p_value_perm_boot'].append(lr['h2']['p_value_permutation'])
            if lr['h2']['supported']:
                agg['h2']['supported_count'] += 1
        if lr['h3']:
            agg['h3']['ratio_boot'].append(lr['h3']['ratio_method_to_ea'])
            agg['h3']['p_value_boot'].append(lr['h3']['p_value'])
            if lr['h3']['supported']:
                agg['h3']['supported_count'] += 1
        if lr['h4']:
            agg['h4']['rmse_boot'].append(lr['h4']['rmse'])
            agg['h4']['within_10_boot'].append(lr['h4']['within_10_kJmol'])
            if lr['h4']['supported']:
                agg['h4']['supported_count'] += 1
        if lr['h5']:
            agg['h5']['prediction'] = lr['h5']['predicted_ea_kJ_per_mol']
            agg['h5']['rmse'] = lr['h5']['rmse_all_samples']
    # Compute bootstrap confidence intervals for aggregated results (only for value keys)
    value_keys = ['mean_delta_ea_boot', 'eta_squared_boot', 'ratio_boot']
    for key in value_keys:
        if key in agg and agg[key] and len(agg[key]) > 0:
            arr = np.array(agg[key])
            agg[f'{key}_mean'] = float(np.mean(arr))
            agg[f'{key}_ci_lower'] = float(np.percentile(arr, 2.5))
            agg[f'{key}_ci_upper'] = float(np.percentile(arr, 97.5))
    return agg
'''

# Replace the function in the content
pattern = r'def aggregate_bootstrap_results\(loop_results\):.*?(?=\n\s*def|\n\s*@|\n\s*if __name__ == \'__main__\':|\n\s*$)'
# Use a simpler approach - replace from the function definition to the return statement
lines = content.split('\n')
in_func = False
func_start = -1
func_end = -1
brace_count = 0
for i, line in enumerate(lines):
    if line.strip().startswith('def aggregate_bootstrap_results'):
        in_func = True
        func_start = i
        brace_count = line.count('{') - line.count('}')
    if in_func and brace_count <= 0 and i > func_start:
        # Check if this line contains 'return agg'
        if 'return agg' in line:
            func_end = i + 1
            break
        # Count braces in this line
        brace_count += line.count('{') - line.count('}')

if func_start >= 0 and func_end >= 0:
    # Replace the function
    new_lines = lines[:func_start] + [''] + lines[func_end:]
    # Actually, let's just do a string replacement
    pass

# Simpler approach: use regex with DOTALL to match the entire function
pattern = re.compile(r'(def aggregate_bootstrap_results\(loop_results\):.*?)(\n\s*return agg)', re.DOTALL)
match = pattern.search(content)
if match:
    old_func = match.group(0)
    print("Found function:")
    print(old_func[:500])
    print("\n---")
    # Replace with new function
    new_content = content[:match.start()] + new_func + content[match.end():]
    with open('comprehensive_analysis.py', 'w') as f:
        f.write(new_content)
    print("Function replaced successfully!")
else:
    print("Could not find function pattern")