import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Mechanical properties
E_graphene = 1.0
t_graphene = 0.335
sigma_intrinsic = 130
epsilon_failure = 0.25

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Panel 1: Key properties bar chart ---
ax = axes[0]
props = ['Youngs\nModulus\n(x100 GPa)', 'Tensile\nStrength\n(x10 GPa)', 'Elongation\n(%)', 'Conductivity\n(x1e4 S/cm)']
colors = ['#1a9850', '#d73027', '#4575b4', '#f46d43', '#999999']
x = np.arange(len(props))
width = 0.15

vals = {
    'Graphene': [1000/100, 130/10, 25, 10],
    'ITO': [120/100, 0.3/10, 0.5, 1],
    'AgNW': [80/100, 0.5/10, 5, 5],
    'PEDOT:PSS': [2/100, 0.05/10, 10, 0.1],
    'Cu film': [120/100, 0.3/10, 3, 60],
}

for i, (name, v) in enumerate(vals.items()):
    offset = (i - 2) * width
    bars = ax.bar(x + offset, v, width, label=name, color=colors[i], alpha=0.85)
    for bar, val in zip(bars, v):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(props, fontsize=9)
ax.set_ylabel('Normalized Value', fontsize=11)
ax.set_title('Key Properties Comparison\n(Flexible Electronics Materials)', fontsize=13)
ax.legend(fontsize=8, loc='upper right')
ax.set_yscale('log')
ax.grid(alpha=0.3, axis='y')

# --- Panel 2: Electrical stability under strain ---
ax2 = axes[1]
strain_range = np.linspace(0, 0.30, 100)

R_graphene = 1 + 0.2 * strain_range + 2 * strain_range**2
R_ITO = 1 + 50 * strain_range + 500 * strain_range**4
R_AgNW = 1 + 5 * strain_range + 30 * strain_range**3
R_PEDOT = 1 + 3 * strain_range + 10 * strain_range**2

ax2.plot(strain_range*100, R_graphene, 'g-', lw=3, label='Graphene')
ax2.plot(strain_range*100, R_ITO, 'r-', lw=2, label='ITO')
ax2.plot(strain_range*100, R_AgNW, 'b-', lw=2, label='AgNW')
ax2.plot(strain_range*100, R_PEDOT, 'orange', lw=2, label='PEDOT:PSS')
ax2.axvline(0.5, color='red', ls='--', alpha=0.5, label='ITO failure limit')
ax2.axvline(25, color='green', ls='--', alpha=0.5, label='Graphene intrinsic limit')
ax2.set_xlabel('Tensile Strain (%)', fontsize=12)
ax2.set_ylabel('R/R0 (Normalized Resistance)', fontsize=12)
ax2.set_title('Electrical Stability Under Strain', fontsize=13)
ax2.set_yscale('log')
ax2.set_xlim(0, 30)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# --- Panel 3: Bending radius ---
ax3 = axes[2]
mat_names = ['Graphene', 'PEDOT:PSS', 'AgNW', 'Cu film', 'ITO']
bend_radii = [0.1, 0.5, 2, 5, 10]
bar_colors = ['#1a9850', '#f46d43', '#4575b4', '#999999', '#d73027']

bars = ax3.barh(range(len(mat_names)), bend_radii, color=bar_colors, alpha=0.85)
for bar, val in zip(bars, bend_radii):
    ax3.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             f'{val} mm', va='center', fontsize=10)
ax3.set_yticks(range(len(mat_names)))
ax3.set_yticklabels(mat_names, fontsize=11)
ax3.set_xlabel('Minimum Bending Radius (mm)', fontsize=12)
ax3.set_title('Mechanical Flexibility\n(smaller = more flexible)', fontsize=13)
ax3.set_xlim(0, 12)
ax3.grid(alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig('graphene_flexible_electronics.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure saved.')

# --- Quantitative analysis ---
print('='*70)
print('  Graphene Flexible Electronics: Quantitative Analysis')
print('='*70)

print('\n--- 1. Mechanical Properties ---')
print(f'  Youngs modulus:          {E_graphene:.1f} TPa')
print(f'  Monolayer thickness:     {t_graphene:.3f} nm')
print(f'  Intrinsic strength:      {sigma_intrinsic} GPa')
print(f'  Failure strain:          {epsilon_failure*100:.0f}%')
print(f'  Bending rigidity:        D = Et^3/12(1-nu^2) ~ {(E_graphene*1e9*(t_graphene*1e-9)**3/12/(1-0.165**2)):.2e} N*m')

print('\n--- 2. Flexibility Figure of Merit ---')
print(f'  FOM = sigma x epsilon_fail / R_sheet')
print(f'  Graphene:    {1e5*25/30:.1e}')
print(f'  ITO:         {1e4*0.5/10:.1e}')
print(f'  AgNW:        {5e4*5/15:.1e}')
print(f'  PEDOT:PSS:   {1e3*10/100:.1e}')
print(f'  -> Graphene FOM is ~{1e5*25/30/(1e4*0.5/10):.0f}x higher than ITO')

print('\n--- 3. Production Methods ---')
print('  [Mechanical exfoliation]')
print('    Quality: Highest (single-crystal, defect-free)')
print('    Size: < 100 um (lab scale)')
print('    Cost: Very high (manual)')
print('    Suitability: Research only')
print('  [CVD on Cu foil]')
print('    Quality: High (large-area, polycrystalline)')
print('    Size: Up to 30-inch (roll-to-roll)')
print('    Cost: Moderate ($50-100/m2)')
print('    Suitability: Industrial (touch screens, sensors)')
print('  [Epitaxial on SiC]')
print('    Quality: High (wafer-scale)')
print('    Size: Up to 8-inch wafers')
print('    Cost: High (SiC substrate)')
print('    Suitability: RF electronics, quantum')
print('  [Reduction of GO]')
print('    Quality: Low (defects, residual O)')
print('    Size: Large-area (solution)')
print('    Cost: Low ($10-30/m2)')
print('    Suitability: Coatings, printed electronics')
print('  [Liquid-phase exfoliation]')
print('    Quality: Low-moderate (small flakes)')
print('    Size: 100 nm - 10 um flakes')
print('    Cost: Low ($10-50/m2)')
print('    Suitability: Inks, composites')

print('\n--- 4. Key Challenges ---')
print('  1. Large-area uniformity: grain boundaries, wrinkles, multilayer patches')
print('  2. Transfer-induced damage: polymer residues, cracks, doping inhomogeneity')
print('  3. Contact resistance: Rc ~ 100-1000 Ohm*um limits device performance')
print('  4. Doping stability: chemical doping degrades in ambient')
print('  5. Zero bandgap: FET on/off ratio ~10; use graphene as electrode not channel')
print('  6. Environmental stability: dopants/contacts degrade; encapsulation needed')
print('  7. Scalable manufacturing: lab->fab with reproducible quality')
print('  8. CMOS integration: contamination control, standard processes')

print('\n--- 5. Application Readiness ---')
print('  [Touch screens & displays] TRL 7-8, timeline 1-3yr')
print('    Samsung/Huawei foldable phones with graphene screens')
print('  [Flexible OLED lighting] TRL 5-6, timeline 3-5yr')
print('    Graphene Flagship, Cambridge Display Technology')
print('  [Wearable sensors] TRL 4-5, timeline 2-4yr')
print('    High piezoresistive sensitivity, biocompatible')
print('  [Flexible batteries/supercaps] TRL 4-5, timeline 5-7yr')
print('    Samsung SDI, LG Chem')
print('  [E-textiles] TRL 3-4, timeline 5-10yr')
print('    Conductive fibers, washable, stretchable')
print('  [RFID & NFC antennas] TRL 6-7, timeline 2-4yr')
print('    Printed graphene antennas, low cost')
print('  [Implantable bioelectronics] TRL 2-3, timeline 10+yr')
print('    Flexible, biocompatible, high charge injection')

print('\n--- 6. Economic Projections ---')
print('  Global graphene market (2024): ~$1.5B')
print('  Projected (2030): ~$5-8B')
print('  Flexible electronics share: ~30-40%')
print('  Current CVD graphene: ~$100-500/m2')
print('  Target for mass adoption: <$20/m2')
print('  ITO replacement market: ~$5B/year')
