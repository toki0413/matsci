# Skill: Band Structure Calculation

## Description
Calculate electronic band structure along high-symmetry k-point paths.

## Trigger Conditions
- User asks for "band structure", "electronic bands", "band gap"
- User wants to understand electronic properties

## Workflow

### Step 1: Prerequisites Check
- Verify relaxed CONTCAR exists
- Check if CHGCAR from SCF calculation is available
- If not, warn user that self-consistent calculation is needed first

### Step 2: K-point Path Generation
- Detect space group from structure
- Generate high-symmetry path using SeeK-path or pymatgen
- Common paths:
  - Cubic: Γ-X-M-Γ-R-X|M-R
  - Hexagonal: Γ-M-K-Γ-A-L-H-A|L-M|K-H
  - Orthorhombic: Γ-X-S-Y-Γ-Z-U-R-T-Z|Y-T|U-X|S-R

### Step 3: INCAR Setup
- ICHARG=11 (read CHGCAR, non-selfconsistent)
- LORBIT=11 (projected DOS)
- ISMEAR=0 (Gaussian) with small SIGMA=0.05
- NBANDS should be >= default + 20%

### Step 4: Execution
- Use `vasp_tool` with action="band"
- Ensure sufficient NBANDS for empty states

### Step 5: Post-processing
- Extract band energies
- Calculate band gap (direct and indirect)
- Identify VBM and CBM positions
- Plot band structure with proper labels

## Parameters Template
```python
{
    "action": "band",
    "structure_file": "CONTCAR",
    "charge_density": "CHGCAR",
    "kpoints_path": "auto",
    "params": {
        "ICHARG": 11,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LORBIT": 11,
        "NEDOS": 3000,
    }
}
```

## Validation Checks
- Band gap should be positive for insulators/semiconductors
- Compare with experimental/literature values
- Check for band crossing at Fermi level in metals
