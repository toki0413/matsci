#!/usr/bin/env python3
"""Comprehensive analysis of the MgH2 activation energy dataset."""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = '/workspace/research/shusheng_run/ws35/data'
original = pd.read_csv(data_dir + '/ea_dataset.csv')
cleaned = pd.read_csv(data_dir + '/cleaned_ea_dataset.csv')

# ============================================================
# 1. Basic Dataset Information
# ============================================================
print('=' * 70)
print('COMPREHENSIVE ANALYSIS OF MgH2 ACTIVATION ENERGY DATASET')
print('=' * 70)

print('\nOriginal dataset: ' + str(original.shape[0]) + ' rows, ' + str(original.shape[1]) + ' columns')
print('Cleaned dataset: ' + str(cleaned.shape[0]) + ' rows, ' + str(cleaned.shape[1]) + ' columns')
print('Rows removed: ' + str(len(original) - len(cleaned)))

# ============================================================
# 2. Activation Energy Statistics
# ============================================================
print('\n' + '=' * 70)
print('ACTIVATION ENERGY STATISTICS (Ea_parsed)')
print('=' * 70)

ea_stats = cleaned['Ea_parsed'].describe()
print('\nMean: ' + str(round(ea_stats['mean'], 2)) + ' kJ/mol')
print('Median: ' + str(round(ea_stats['50%'], 2)) + ' kJ/mol')
print('Std: ' + str(round(ea_stats['std'], 2)) + ' kJ/mol')
print('Min: ' + str(round(ea_stats['min'], 2)) + ' kJ/mol')
print('Max: ' + str(round(ea_stats['max'], 2)) + ' kJ/mol')
print('Q1 (25%): ' + str(round(ea_stats['25%'], 2)) + ' kJ/mol')
print('Q3 (75%): ' + str(round(ea_stats['75%'], 2)) + ' kJ/mol')
print('IQR: ' + str(round(ea_stats['75%'] - ea_stats['25%'], 2)) + ' kJ/mol')

# ============================================================
# 3. Distribution by Catalyst Family
# ============================================================
print('\n' + '=' * 70)
print('ACTIVATION ENERGY BY CATALYST FAMILY')
print('=' * 70)

for family in cleaned['catalyst_family'].unique():
    subset = cleaned[cleaned['catalyst_family'] == family]
    print('\n' + family + ':')
    print('  Count: ' + str(len(subset)))
    print('  Ea mean: ' + str(round(subset['Ea_parsed'].mean(), 2)) + ' kJ/mol')
    print('  Ea median: ' + str(round(subset['Ea_parsed'].median(), 2)) + ' kJ/mol')
    print('  Ea range: [' + str(round(subset['Ea_parsed'].min(), 2)) + ', ' + str(round(subset['Ea_parsed'].max(), 2)) + ']')

# ============================================================
# 4. Distribution by Material Class
# ============================================================
print('\n' + '=' * 70)
print('ACTIVATION ENERGY BY MATERIAL CLASS')
print('=' * 70)

for mclass in cleaned['material_class'].unique():
    subset = cleaned[cleaned['material_class'] == mclass]
    print('\n' + mclass + ':')
    print('  Count: ' + str(len(subset)))
    print('  Ea mean: ' + str(round(subset['Ea_parsed'].mean(), 2)) + ' kJ/mol')
    print('  Ea median: ' + str(round(subset['Ea_parsed'].median(), 2)) + ' kJ/mol')

# ============================================================
# 5. Direction Analysis (Desorption only)
# ============================================================
print('\n' + '=' * 70)
print('DIRECTION ANALYSIS')
print('=' * 70)

desorption = cleaned[cleaned['direction'] == 'desorption']
absorption = cleaned[cleaned['direction'] == 'absorption']

print('\nDesorption rows: ' + str(len(desorption)))
print('Absorption rows: ' + str(len(absorption)))
print('\nDesorption Ea stats:')
print('  Mean: ' + str(round(desorption['Ea_parsed'].mean(), 2)) + ' kJ/mol')
print('  Median: ' + str(round(desorption['Ea_parsed'].median(), 2)) + ' kJ/mol')

# ============================================================
# 6. Method Analysis
# ============================================================
print('\n' + '=' * 70)
print('ANALYSIS METHOD DISTRIBUTION')
print('=' * 70)

method_counts = cleaned['method'].value_counts()
print(method_counts)

for method in cleaned['method'].unique():
    subset = cleaned[cleaned['method'] == method]
    print('\n' + method + ':')
    print('  Count: ' + str(len(subset)))
    print('  Ea mean: ' + str(round(subset['Ea_parsed'].mean(), 2)) + ' kJ/mol')

# ============================================================
# 7. Year Distribution
# ============================================================
print('\n' + '=' * 70)
print('YEAR DISTRIBUTION')
print('=' * 70)

year_counts = cleaned['year'].value_counts().sort_index()
print('Years covered: ' + str(int(cleaned['year'].min())) + ' - ' + str(int(cleaned['year'].max())))
print('Total unique years: ' + str(len(year_counts)))
print('Most common year: ' + str(int(year_counts.idxmax())) + ' (' + str(int(year_counts.max())) + ' papers)')

# ============================================================
# 8. Save summary
# ============================================================
summary = {
    "original_rows": len(original),
    "cleaned_rows": len(cleaned),
    "original_columns": list(original.columns),
    "cleaned_columns": list(cleaned.columns),
    "ea_mean_cleaned": float(cleaned['Ea_parsed'].mean()),
    "ea_median_cleaned": float(cleaned['Ea_parsed'].median()),
    "ea_std_cleaned": float(cleaned['Ea_parsed'].std()),
    "ea_min_cleaned": float(cleaned['Ea_parsed'].min()),
    "ea_max_cleaned": float(cleaned['Ea_parsed'].max()),
    "desorption_rows": len(desorption),
    "absorption_rows": len(absorption),
    "catalyst_family_counts": {k: int(v) for k, v in cleaned['catalyst_family'].value_counts().items()},
    "material_class_counts": {k: int(v) for k, v in cleaned['material_class'].value_counts().items()},
    "method_counts": {k: int(v) for k, v in cleaned['method'].value_counts().items()},
    "year_range": [int(cleaned['year'].min()), int(cleaned['year'].max())],
}

with open(data_dir + '/dataset_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('\nSummary saved to ' + data_dir + '/dataset_summary.json')

# ============================================================
# 9. Create Visualizations
# ============================================================
print('\n' + '=' * 70)
print('GENERATING VISUALIZATIONS')
print('=' * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MgH2 Activation Energy Dataset Analysis', fontsize=16, fontweight='bold')

# Plot 1: Distribution of Ea values
sns.histplot(cleaned['Ea_parsed'], ax=axes[0, 0], kde=True, bins=15, color='steelblue')
axes[0, 0].set_xlabel('Activation Energy (kJ/mol)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Activation Energies')
axes[0, 0].axvline(cleaned['Ea_parsed'].mean(), color='red', linestyle='--', label='Mean: ' + str(round(cleaned['Ea_parsed'].mean(), 1)))
axes[0, 0].axvline(cleaned['Ea_parsed'].median(), color='green', linestyle='--', label='Median: ' + str(round(cleaned['Ea_parsed'].median(), 1)))
axes[0, 0].legend()

# Plot 2: Ea by Catalyst Family
family_order = ['none', 'oxide', 'metal_nanoparticle', 'mxene_carbon', 'other']
family_valid = [f for f in family_order if f in cleaned['catalyst_family'].values]
family_data = [cleaned[cleaned['catalyst_family'] == f]['Ea_parsed'] for f in family_valid]
bp = axes[0, 1].boxplot(family_data, patch_artist=True)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
for patch, color in zip(bp['boxes'], colors[:len(family_valid)]):
    patch.set_facecolor(color)
axes[0, 1].set_xticklabels(family_valid, rotation=45)
axes[0, 1].set_ylabel('Activation Energy (kJ/mol)')
axes[0, 1].set_title('Activation Energy by Catalyst Family')

# Plot 3: Ea by Material Class
mclass_order = ['pure_MgH2', 'catalyzed_MgH2', 'alloy', 'complex_hydride']
mclass_valid = [m for m in mclass_order if m in cleaned['material_class'].values]
mclass_data = [cleaned[cleaned['material_class'] == m]['Ea_parsed'] for m in mclass_valid]
bp2 = axes[1, 0].boxplot(mclass_data, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors[:len(mclass_valid)]):
    patch.set_facecolor(color)
axes[1, 0].set_xticklabels(mclass_valid, rotation=45)
axes[1, 0].set_ylabel('Activation Energy (kJ/mol)')
axes[1, 0].set_title('Activation Energy by Material Class')

# Plot 4: Ea by Method
method_order = ['Kissinger', 'isothermal_Arrhenius', 'DSC']
method_valid = [m for m in method_order if m in cleaned['method'].values]
method_data = [cleaned[cleaned['method'] == m]['Ea_parsed'] for m in method_valid]
bp3 = axes[1, 1].boxplot(method_data, patch_artist=True)
for patch, color in zip(bp3['boxes'], colors[:len(method_valid)]):
    patch.set_facecolor(color)
axes[1, 1].set_xticklabels(method_valid, rotation=45)
axes[1, 1].set_ylabel('Activation Energy (kJ/mol)')
axes[1, 1].set_title('Activation Energy by Analysis Method')

plt.tight_layout()
plt.savefig(data_dir + '/ea_analysis.png', dpi=150, bbox_inches='tight')
print('Saved visualization to ' + data_dir + '/ea_analysis.png')

# Additional plot: Ea distribution by direction
fig2, ax = plt.subplots(figsize=(10, 6))
des_ea = desorption['Ea_parsed']
abs_ea = absorption['Ea_parsed']

bp_dir = ax.boxplot([des_ea, abs_ea], patch_artist=True)
colors2 = ['#ff9999', '#66b3ff']
for patch, color in zip(bp_dir['boxes'], colors2):
    patch.set_facecolor(color)
ax.set_xticklabels(['Desorption', 'Absorption'])
ax.set_ylabel('Activation Energy (kJ/mol)')
ax.set_title('Activation Energy by Direction')
ax.axhline(des_ea.mean(), color='red', linestyle='--', label='Desorption mean: ' + str(round(des_ea.mean(), 1)))
ax.legend()
plt.tight_layout()
plt.savefig(data_dir + '/ea_by_direction.png', dpi=150, bbox_inches='tight')
print('Saved visualization to ' + data_dir + '/ea_by_direction.png')

print('\nAnalysis complete!')