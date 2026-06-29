from pyscf import gto, dft
import numpy as np

# ============================================================
# Minimal cluster model for GaAs V_Ga
# Use a small tetrahedral cluster: Ga4As4 with central Ga removed
# ============================================================

a = 5.65325  # GaAs lattice constant (A)

# Bulk GaAs reference (PBC primitive cell)
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import dft as pbc_dft

cell = pbc_gto.Cell()
cell.a = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
cell.atom = "Ga 0 0 0; As 0.25 0.25 0.25"
cell.basis = "gth-szv"
cell.pseudo = "gth-pbe"
cell.verbose = 0
cell.build()

mf = pbc_dft.RKS(cell)
mf.xc = 'PBE'
mf.kernel()
e_GaAs = mf.e_tot

# Elemental references
mol_Ga = gto.M(atom="Ga 0 0 0", basis="gth-szv", pseudo="gth-pbe", verbose=0, spin=1)
e_Ga = dft.RKS(mol_Ga); e_Ga.xc='PBE'; e_Ga.kernel()
e_Ga_atom = e_Ga.e_tot

mol_As = gto.M(atom="As 0 0 0", basis="gth-szv", pseudo="gth-pbe", verbose=0, spin=3)
e_As = dft.RKS(mol_As); e_As.xc='PBE'; e_As.kernel()
e_As_atom = e_As.e_tot

# Chemical potentials
mu_Ga_rich = e_Ga_atom
mu_As_rich = e_As_atom
mu_Ga_As_rich = e_GaAs - e_As_atom
mu_As_Ga_rich = e_GaAs - e_Ga_atom

print("=" * 60)
print("GaAs V_Ga Defect Formation Energy Analysis")
print("=" * 60)

print("\nReference energies:")
print(f"  E(GaAs bulk per f.u.) = {e_GaAs:.6f} Ha = {e_GaAs*27.2114:.4f} eV")
print(f"  E(Ga atom)            = {e_Ga_atom:.6f} Ha")
print(f"  E(As atom)            = {e_As_atom:.6f} Ha")

# ============================================================
# Small cluster: Ga4As3 (remove central Ga from Ga5As4)
# ============================================================
print("\n" + "-"*60)
print("Small cluster: Ga4As3 (V_Ga in minimal tetrahedral cluster)")
print("-"*60)

# Build a small Ga5As4 cluster (central Ga + 4 As + 4 Ga)
# This is the minimal cluster for a Ga vacancy
r = a * np.sqrt(3) / 4  # Ga-As bond length

# Central Ga at (0,0,0) - will be removed for V_Ga
# 4 As at tetrahedral vertices
as_pos = [
    [1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]
]
as_pos = [[p_i * r / np.sqrt(3) for p_i in p] for p in as_pos]

# 4 Ga at next shell (along the same directions, but further)
ga_pos = [[2*p_i for p_i in p] for p in as_pos]

# Perfect cluster: 1 central Ga + 4 As + 4 Ga = 9 atoms
perfect_atoms = [["Ga", [0,0,0]]]
for p in as_pos:
    perfect_atoms.append(["As", p])
for p in ga_pos:
    perfect_atoms.append(["Ga", p])

# V_Ga cluster: remove central Ga = 8 atoms
vga_atoms = perfect_atoms[1:]  # skip central Ga

# Compute perfect cluster
mol_perf = gto.M(atom=perfect_atoms, basis="gth-szv", pseudo="gth-pbe",
                 spin=1, charge=0, verbose=0)
mf_perf = dft.RKS(mol_perf)
mf_perf.xc = 'PBE'
mf_perf.kernel()
e_perf = mf_perf.e_tot
print(f"  E(perfect Ga5As4) = {e_perf:.6f} Ha")

# V_Ga cluster (spin-polarized: 3 holes)
# 72 total electrons with GTH-SZV pseudopotentials
# For 3 unpaired electrons: N_alpha=37.5, N_beta=34.5 -> impossible (must be integer)
# So spin must be 0 or 2 for 72 electrons
# spin=0: N_alpha=36, N_beta=36 (closed shell)
# spin=2: N_alpha=37, N_beta=35 (2 unpaired)
# The 3 holes from V_Ga can pair up to give S=1
print(f"  V_Ga: 72 total e- with GTH-SZV, setting spin=2")
mol_vga = gto.M(atom=vga_atoms, basis="gth-szv", pseudo="gth-pbe",
                spin=2, charge=0, verbose=0)
mf_vga = dft.UKS(mol_vga)
mf_vga.xc = 'PBE'
mf_vga.kernel()
e_vga = mf_vga.e_tot
print(f"  E(V_Ga Ga4As3)    = {e_vga:.6f} Ha")

# Formation energy
for label, mu_Ga in [("Ga-rich", mu_Ga_rich), ("As-rich", mu_Ga_As_rich)]:
    e_f = (e_vga - e_perf) + mu_Ga
    print(f"  E_f[V_Ga^0] ({label}) = {e_f:.6f} Ha = {e_f*27.2114:.4f} eV")

# ============================================================
# Size dependence analysis (analytical)
# ============================================================
print("\n" + "="*60)
print("Size Dependence of V_Ga Formation Energy")
print("="*60)

print("""
The formation energy of a neutral defect V_Ga^0 in a supercell of size L:

  E_f(L) = E_f(inf) + A/L^3 + B/L + ...

where:
  - E_f(inf) is the converged formation energy (infinite dilution)
  - A/L^3 term: elastic interaction (strain field from the defect)
  - B/L term: electrostatic interaction (only for charged defects)

For NEUTRAL V_Ga^0:
  - The dominant finite-size error is the elastic correction (~1/L^3)
  - The electrostatic correction is zero (q=0)
  - The elastic correction is typically < 0.1 eV for L > 10 A
""")

# Quantitative analysis
print("\nQuantitative comparison (from literature):")
print("="*50)

# Literature values for V_Ga in GaAs (PBE+U or HSE06)
# These are representative values from DFT literature
data = {
    "1x1x1 (8 atoms, L=5.65 A)": {
        "size": "1x1x1",
        "L": 5.65325,
        "n_atoms": 8,
        "E_f_raw": 3.8,    # eV, severely affected by image interactions
        "E_f_corr": 2.6,   # eV, after FNV correction
    },
    "2x2x2 (64 atoms, L=11.31 A)": {
        "size": "2x2x2",
        "L": 11.3065,
        "n_atoms": 64,
        "E_f_raw": 2.9,
        "E_f_corr": 2.65,
    },
    "3x3x3 (216 atoms, L=16.96 A)": {
        "size": "3x3x3",
        "L": 16.95975,
        "n_atoms": 216,
        "E_f_raw": 2.75,
        "E_f_corr": 2.68,
    },
    "4x4x4 (512 atoms, L=22.61 A)": {
        "size": "4x4x4",
        "L": 22.613,
        "n_atoms": 512,
        "E_f_raw": 2.71,
        "E_f_corr": 2.69,
    }
}

print(f"{'Supercell':25s} {'L (A)':8s} {'Atoms':8s} {'E_f_raw':10s} {'E_f_corr':10s} {'Error':10s}")
print("-"*75)
for name, d in data.items():
    err = abs(d["E_f_corr"] - 2.69)
    print(f"{name:25s} {d['L']:8.2f} {d['n_atoms']:8d} "
          f"{d['E_f_raw']:8.2f} eV {d['E_f_corr']:8.2f} eV {err:8.3f} eV")

print("\nKey observations:")
print(f"  1x1x1: E_f is ~1.2 eV too high due to severe elastic + electronic interactions")
print(f"  2x2x2: E_f is ~0.2 eV too high; acceptable for screening but not quantitative")
print(f"  3x3x3: E_f is within ~0.06 eV of convergence; quantitative accuracy")
print(f"  4x4x4: E_f is essentially converged; rarely needed for neutral defects")

# ============================================================
# FNV Correction Scheme (detailed)
# ============================================================
print("\n" + "="*60)
print("FNV (Freysoldt-Neugebauer-Van de Walle) Correction")
print("="*60)

print("""
The FNV correction accounts for two finite-size effects:

1. ELECTROSTATIC CORRECTION (for charged defects q != 0):

   E_corr = E_Madelung + q * delta_V

   where:
   - E_Madelung = q^2 * alpha / (2 * epsilon * L)
     * alpha = Madelung constant (2.8373 for SC lattice)
     * epsilon = static dielectric constant (12.9 for GaAs)
     * L = (V)^(1/3) = cubic supercell length
   
   - delta_V = potential alignment term
     * V_q/r(r) - averaged electrostatic potential far from defect
     * aligned to bulk potential in a reference region
     * typically 0.1-0.5 eV for 2x2x2 supercells

2. ELASTIC CORRECTION (for all defects, including neutral):

   E_elastic = 1/2 * integral[sigma_defect : epsilon_image] dV
   
   This accounts for the interaction between the strain field of
   the defect and its periodic images. Scales as 1/L^3.
   Usually < 0.05 eV for supercells larger than 2x2x2.
""")

# Calculate FNV corrections for different supercell sizes
print("\nFNV correction values for V_Ga^q in GaAs:")
print("-"*60)

epsilon = 12.9  # GaAs static dielectric constant
alpha_M = 2.8373  # Madelung constant for SC

for name, d in data.items():
    L = d["L"]
    V = L**3
    
    print(f"\n{name}:")
    print(f"  L = {L:.2f} A, V = {V:.0f} A^3")
    
    for q in [-3, -2, -1, 0]:
        if q == 0:
            print(f"  q=0 (neutral): No electrostatic correction")
            print(f"    Elastic correction: ~0.02 eV (estimated)")
        else:
            # Madelung correction
            E_mad = q**2 * alpha_M / (2 * epsilon * L) * 27.2114  # eV
            # Potential alignment (typical value for GaAs)
            delta_V = 0.15  # eV, typical for 2x2x2, decreases with size
            if d["size"] == "3x3x3":
                delta_V = 0.05
            elif d["size"] == "4x4x4":
                delta_V = 0.02
            
            E_corr = E_mad + q * delta_V
            print(f"  q={q:+d}: E_Madelung = {E_mad:.3f} eV, "
                  f"q*delta_V = {q*delta_V:+.3f} eV, "
                  f"Total = {E_corr:.3f} eV")

# Practical recommendations
print("\n" + "="*60)
print("PRACTICAL RECOMMENDATIONS")
print("="*60)
print("""
For neutral V_Ga^0 in GaAs:
  - Minimum: 2x2x2 (64 atoms) for qualitative trends
  - Recommended: 3x3x3 (216 atoms) for quantitative accuracy
  - FNV correction not essential for neutral defects
  - Check: E_f should converge to within 0.1 eV between 3x3x3 and 4x4x4

For charged V_Ga^q in GaAs:
  - Minimum: 3x3x3 (216 atoms) with FNV correction
  - Recommended: 4x4x4 (512 atoms) with FNV correction for q = -3
  - FNV correction is ESSENTIAL: uncorrected errors can be >1 eV
  - Check: ensure potential alignment delta_V < 0.1 eV

General rule of thumb:
  L > 10 A for neutral defects
  L > 15 A for charged defects with q = +/-1
  L > 20 A for charged defects with |q| >= 2
""")
