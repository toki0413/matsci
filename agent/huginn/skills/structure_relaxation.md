# Skill: Structure Relaxation

## Description
Perform geometry optimization of a crystal structure to find the equilibrium configuration.

## Trigger Conditions
- User mentions "relax", "optimize structure", "geometry optimization"
- User provides a structure file and asks about stable configuration

## Workflow

### Step 1: Structure Analysis
- Use `structure_tool` to read and validate the input structure
- Check space group, lattice parameters, and atomic positions
- Warn if structure has obvious issues (overlapping atoms, unusual bond lengths)

### Step 2: Parameter Selection
- Determine appropriate ISIF:
  - ISIF=2: Relax ions only (fixed cell) — for known experimental lattice
  - ISIF=3: Relax ions + cell shape + volume — for theoretical prediction
  - ISIF=4: Relax ions + cell shape (fixed volume) — for anisotropic stress
- Set IBRION=2 (CG) for robustness, or IBRION=1 (quasi-Newton) for speed
- EDIFFG = -0.01 (tight) or -0.05 (loose) depending on purpose

### Step 3: Execution
- Use `vasp_tool` with action="relax"
- Submit via `job_tool` if running on HPC
- Monitor convergence

### Step 4: Validation
- Check force convergence (all forces < |EDIFFG|)
- Verify energy decreased from initial
- Check lattice parameters against experimental values if available
- Use `validate_tool` to run physical validation

### Step 5: Reporting
- Report final energy, lattice parameters, volume change
- Compare with initial structure
- Warn if symmetry changed unexpectedly

## Parameters Template
```python
{
    "action": "relax",
    "structure_file": "<user_provided>",
    "params": {
        "ISIF": 3,
        "IBRION": 2,
        "EDIFFG": -0.01,
        "NSW": 200,
        "POTIM": 0.5,
    }
}
```

## Common Issues
- **Convergence failure**: See convergence diagnosis skill
- **Symmetry breaking**: May be physical (Jahn-Teller) or numerical (check EDIFFG)
- **Volume explosion**: Check initial structure for close contacts
