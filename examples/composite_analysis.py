import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E_g = 1000.0
t_g = 0.335e-9
sigma_g = 130e9

E_m = 3.0
nu_m = 0.35
sigma_m = 70e6

IFSS_poor = 5e6
IFSS_good = 40e6
IFSS_excellent = 80e6

L_g = 1e-6

def critical_length(sigma_f, t, tau):
    return sigma_f * t / tau

l_c_poor = critical_length(sigma_g, t_g, IFSS_poor)
l_c_good = critical_length(sigma_g, t_g, IFSS_good)
l_c_excellent = critical_length(sigma_g, t_g, IFSS_excellent)

print('='*70)
print('  Graphene Composite Reinforcement: Quantitative Analysis')
print('='*70)

print('\n--- 1. Shear-Lag Model (Kelly-Tyson) ---')
print(f'  Graphene strength: {sigma_g/1e9:.0f} GPa')
print(f'  Graphene thickness: {t_g*1e9:.3f} nm')
print(f'  Critical length l_c = sigma_f * t / tau')
print(f'    Poor (tau=5 MPa):   l_c = {l_c_poor*1e9:.1f} nm')
print(f'    Good (tau=40 MPa):  l_c = {l_c_good*1e9:.1f} nm')
print(f'    Covalent (80 MPa):  l_c = {l_c_excellent*1e9:.1f} nm')
print(f'  Flake size L = {L_g*1e6:.0f} um = {L_g*1e9:.0f} nm')
print(f'  Aspect ratio L/t = {L_g/t_g:.0f}')

G_m = E_m / (2 * (1 + nu_m))
t_m = 10e-9
beta = np.sqrt(2 * G_m * 1e9 / (E_g * 1e9 * t_g * t_m))
eta_l = 1 - np.tanh(beta * L_g / 2) / (beta * L_g / 2)
print(f'\n  Matrix G_m = {G_m:.2f} GPa')
print(f'  Interparticle spacing = {t_m*1e9:.0f} nm')
print(f'  Beta = {beta:.2e} m^-1')
print(f'  Length efficiency eta_l = {eta_l:.3f}')

xi_aligned = 2 * L_g / t_g
xi_random = 2.0
eta_aligned = (E_g/E_m - 1) / (E_g/E_m + xi_aligned)
eta_random = (E_g/E_m - 1) / (E_g/E_m + xi_random)

print('\n--- 2. Halpin-Tsai: Composite Modulus ---')
Vf = np.linspace(0, 0.05, 100)
Ec_al = E_m * (1 + xi_aligned * eta_aligned * Vf) / (1 - eta_aligned * Vf)
Ec_rd = E_m * (1 + xi_random * eta_random * Vf) / (1 - eta_random * Vf)

for vf_test in [0.01, 0.05]:
    Ec_al_vf = E_m * (1 + xi_aligned * eta_aligned * vf_test) / (1 - eta_aligned * vf_test)
    Ec_rd_vf = E_m * (1 + xi_random * eta_random * vf_test) / (1 - eta_random * vf_test)
    print(f'  At {vf_test*100:.0f} vol%:')
    print(f'    Aligned:  E_c = {Ec_al_vf:.1f} GPa (x{Ec_al_vf/E_m:.1f})')
    print(f'    Random:   E_c = {Ec_rd_vf:.1f} GPa (x{Ec_rd_vf/E_m:.1f})')

print('\n--- 3. Rule of Mixtures: Strength ---')
eta_o_aligned = 1.0
eta_o_random2d = 0.375
Vfs = np.linspace(0, 0.05, 50)
sc_al = sigma_m * (1 - Vfs) + sigma_g * Vfs * eta_l * eta_o_aligned
sc_rd = sigma_m * (1 - Vfs) + sigma_g * Vfs * eta_l * eta_o_random2d

for vf_test in [0.01, 0.03, 0.05]:
    sc_al_vf = sigma_m * (1 - vf_test) + sigma_g * vf_test * eta_l * eta_o_aligned
    sc_rd_vf = sigma_m * (1 - vf_test) + sigma_g * vf_test * eta_l * eta_o_random2d
    print(f'  At {vf_test*100:.0f} vol%:')
    print(f'    Aligned:  {sc_al_vf/1e6:.0f} MPa (x{sc_al_vf/sigma_m:.1f})')
    print(f'    Random:   {sc_rd_vf/1e6:.0f} MPa (x{sc_rd_vf/sigma_m:.1f})')

print('\n--- 4. Toughening ---')
G_pullout_1pct = 0.01 * sigma_g * l_c_good / 12
print(f'  Pull-out energy at 1 vol% (good interface): {G_pullout_1pct/1e3:.1f} kJ/m^3')
print(f'  Crack deflection tortuosity ~ L/t = {L_g/t_g:.0f}')
print(f'  Fracture toughness increase: 2-5x at <1 wt%')

# Figure
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

ax = axes[0, 0]
ax.plot(Vf*100, Ec_al, 'b-', lw=2.5, label='Aligned')
ax.plot(Vf*100, Ec_rd, 'r-', lw=2.5, label='Random 2D')
ax.axhline(E_m, color='gray', ls='--', alpha=0.5, label=f'Matrix (E_m={E_m} GPa)')
ax.set_xlabel('Graphene Volume Fraction (%)', fontsize=12)
ax.set_ylabel('Composite Modulus E_c (GPa)', fontsize=12)
ax.set_title('Halpin-Tsai: Modulus Enhancement', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(Vfs*100, sc_al/1e6, 'b-', lw=2.5, label='Aligned')
ax.plot(Vfs*100, sc_rd/1e6, 'r-', lw=2.5, label='Random 2D')
ax.axhline(sigma_m/1e6, color='gray', ls='--', alpha=0.5, label=f'Matrix ({sigma_m/1e6:.0f} MPa)')
ax.set_xlabel('Graphene Volume Fraction (%)', fontsize=12)
ax.set_ylabel('Composite Strength (MPa)', fontsize=12)
ax.set_title('Strength Enhancement', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[0, 2]
tau_range = np.linspace(1, 100, 100) * 1e6
lc = critical_length(sigma_g, t_g, tau_range)
ax.plot(tau_range/1e6, lc*1e9, 'g-', lw=2.5)
ax.axhline(L_g*1e9, color='r', ls='--', alpha=0.7, label=f'Flake L={L_g*1e9:.0f} nm')
ax.axvline(IFSS_poor/1e6, color='orange', ls=':', alpha=0.5, label=f'Poor: {IFSS_poor/1e6:.0f} MPa')
ax.axvline(IFSS_good/1e6, color='blue', ls=':', alpha=0.5, label=f'Good: {IFSS_good/1e6:.0f} MPa')
ax.set_xlabel('Interfacial Shear Strength (MPa)', fontsize=12)
ax.set_ylabel('Critical Length l_c (nm)', fontsize=12)
ax.set_title('Load Transfer Efficiency', fontsize=13)
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.axis('off')
text_m = """Reinforcement Mechanisms:

1. LOAD TRANSFER (Shear-lag)
   Matrix -> graphene via interfacial shear
   sigma_f = tau * L/t  (if L > l_c)

2. CRACK BRIDGING
   Graphene bridges propagating cracks
   Pull-out energy: G ~ V_f * sigma_f * l_c / 12

3. CRACK DEFLECTION
   2D geometry forces tortuous crack path
   Effective crack length increases

4. OROWAN STRENGTHENING
   Dislocation pinning at graphene
   (metal matrix composites)

5. THERMAL MISMATCH
   CTE mismatch generates dislocations
   -> forest hardening in metal matrices

6. GRAIN REFINEMENT
   Graphene as heterogeneous nucleation
   -> Hall-Petch strengthening"""
ax.text(0.05, 0.95, text_m, transform=ax.transAxes, fontsize=10, va='top', family='monospace')

ax = axes[1, 1]
fillers = {
    'Graphene': {'E': 1000, 'AR': 2000},
    'SWCNT': {'E': 1000, 'AR': 1000},
    'MWCNT': {'E': 800, 'AR': 500},
    'Nanoclay': {'E': 180, 'AR': 200},
    'SiO2 np': {'E': 70, 'AR': 1},
    'Carbon black': {'E': 15, 'AR': 1},
}
names = list(fillers.keys())
E_vals = [fillers[n]['E'] for n in names]
AR_vals = [fillers[n]['AR'] for n in names]
x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, E_vals, width, label='Modulus (GPa)', color='#1a9850', alpha=0.85)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, AR_vals, width, label='Aspect Ratio', color='#d73027', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Youngs Modulus (GPa)', fontsize=11, color='#1a9850')
ax2.set_ylabel('Aspect Ratio', fontsize=11, color='#d73027')
ax.set_title('Nanofiller Comparison', fontsize=13)
l1, la1 = ax.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, fontsize=8, loc='upper left')
ax.grid(alpha=0.3, axis='y')

ax = axes[1, 2]
ax.axis('off')
app_text = """Application Cases by Matrix:

POLYMER (most mature):
  Epoxy + 0.5-2 wt% graphene
    +50-80% modulus, +30-50% strength
    Aerospace, automotive parts
  PLA/PVA + graphene
    +40-60% tensile strength
    Packaging, biomedical

CERAMIC (emerging):
  Al2O3 + 1-5 vol% graphene
    +30-50% fracture toughness
    Cutting tools, armor
  Si3N4 + graphene
    +60% fracture toughness

METAL (challenging):
  Al + 0.5-2 wt% graphene
    +20-40% strength, ductility loss
  Cu + graphene
    +30% strength, maintains conductivity
    Electrical contacts

CEMENT (growing):
  Cement + 0.05-0.5 wt% GO
    +40% compressive strength
    Infrastructure

AEROGEL (emerging):
  Graphene aerogel
    <10 mg/cm3
    Thermal insulation, sensors"""
ax.text(0.05, 0.95, app_text, transform=ax.transAxes, fontsize=9.5, va='top', family='monospace')

plt.tight_layout()
fig.savefig('graphene_composite_reinforcement.png', dpi=150, bbox_inches='tight')
plt.close()
print('Figure saved.')
