"""RDKit cheminformatics tool — molecular manipulation for drug discovery.

Provides SMILES parsing, descriptor computation, fingerprinting, similarity
search, substructure matching, 2D depiction, and 3D conformer generation.
RDKit is imported lazily so the tool loads even without rdkit installed.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class RDKitInput(BaseModel):
    action: Literal[
        "smiles_to_mol",
        "descriptors",
        "fingerprint",
        "similarity",
        "substructure_search",
        "draw",
        "conformers",
        "smiles_to_sdf",
    ] = Field(...)
    smiles: str | None = Field(default=None, description="SMILES string")
    smiles_list: list[str] | None = Field(
        default=None, description="Multiple SMILES for batch operations"
    )
    substructure: str | None = Field(
        default=None, description="Substructure SMILES for substructure_search"
    )
    reference_smiles: str | None = Field(
        default=None, description="Reference molecule SMILES for similarity"
    )
    query_smiles: str | None = Field(
        default=None, description="Query molecule SMILES for similarity"
    )
    output_file: str | None = Field(
        default=None, description="Output path for SDF/PNG files"
    )
    fingerprint_type: Literal["morgan", "maccs", "atom_pair", "topological"] = Field(
        default="morgan", description="Fingerprint type"
    )
    radius: int = Field(default=2, ge=1, le=4, description="Morgan fingerprint radius")
    n_bits: int = Field(default=2048, ge=128, le=8192, description="Fingerprint bit count")
    n_conformers: int = Field(default=1, ge=1, le=50, description="Number of 3D conformers")
    optimize: bool = Field(default=True, description="MMFF94 force-field optimization for conformers")
    image_size: tuple[int, int] = Field(default=(500, 500), description="2D draw image size")


class RDKitTool(HuginnTool):
    """Cheminformatics toolkit for molecular drug discovery."""

    name = "rdkit_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.PLANNING}),
    )
    description = (
        "Cheminformatics: parse SMILES, compute molecular descriptors (MW, "
        "LogP, TPSA, HBD/HBA), generate fingerprints, compute Tanimoto "
        "similarity, substructure search, 2D depiction, and 3D conformers."
    )
    input_schema = RDKitInput
    read_only = True

    def is_read_only(self, args: RDKitInput) -> bool:
        return args.action not in ("draw", "conformers", "smiles_to_sdf")

    async def call(self, args: RDKitInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "smiles_to_mol":
                return self._smiles_to_mol(args)
            if args.action == "descriptors":
                return self._descriptors(args)
            if args.action == "fingerprint":
                return self._fingerprint(args)
            if args.action == "similarity":
                return self._similarity(args)
            if args.action == "substructure_search":
                return self._substructure(args)
            if args.action == "draw":
                return self._draw(args)
            if args.action == "conformers":
                return self._conformers(args)
            if args.action == "smiles_to_sdf":
                return self._smiles_to_sdf(args)
            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")
        except ImportError as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"{exc}. Install with: pip install rdkit",
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _mol_from_smiles(smiles: str):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    # ── actions ──────────────────────────────────────────────

    def _smiles_to_mol(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem

        smiles = args.smiles
        if not smiles:
            return ToolResult(data=None, success=False, error="smiles is required")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToolResult(data=None, success=False, error=f"Invalid SMILES: {smiles}")

        info = {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_rings": Chem.rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic_rings": Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_rotatable_bonds": Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
        }
        return ToolResult(data=info)

    def _descriptors(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        smiles_list = args.smiles_list or ([args.smiles] if args.smiles else [])
        if not smiles_list:
            return ToolResult(data=None, success=False, error="Provide smiles or smiles_list")

        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append({"smiles": smi, "error": "invalid SMILES"})
                continue

            results.append({
                "smiles": smi,
                "canonical_smiles": Chem.MolToSmiles(mol),
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "logp": round(Crippen.MolLogP(mol), 2),
                "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
                "h_bond_donors": Lipinski.NumHDonors(mol),
                "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
                "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
                "num_valence_electrons": rdMolDescriptors.CalcNumValenceElectrons(mol),
                "lipinski_violations": sum([
                    Descriptors.MolWt(mol) > 500,
                    Crippen.MolLogP(mol) > 5,
                    Lipinski.NumHDonors(mol) > 5,
                    Lipinski.NumHAcceptors(mol) > 10,
                ]),
            })

        return ToolResult(data={"molecules": results, "count": len(results)})

    def _fingerprint(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem

        smiles = args.smiles
        if not smiles:
            return ToolResult(data=None, success=False, error="smiles is required")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToolResult(data=None, success=False, error=f"Invalid SMILES: {smiles}")

        fp, n_bits = self._compute_fingerprint(mol, args.fingerprint_type, args.radius, args.n_bits)

        # Return as bit-vector string for compactness; callers can convert back
        from rdkit.DataStructs import ConvertToExplicitBitVect
        bv = ConvertToExplicitBitVect(fp) if hasattr(fp, 'GetNumBits') else fp
        on_bits = list(bv.GetOnBits())

        return ToolResult(data={
            "smiles": smiles,
            "fingerprint_type": args.fingerprint_type,
            "n_bits": n_bits,
            "n_on_bits": len(on_bits),
            "density": round(len(on_bits) / n_bits, 4),
            "on_bits": on_bits[:200],  # cap for large fingerprints
        })

    def _similarity(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem
        from rdkit.DataStructs import TanimotoSimilarity

        ref = args.reference_smiles
        query = args.query_smiles
        if not ref or not query:
            return ToolResult(data=None, success=False, error="reference_smiles and query_smiles required")

        mol_ref = Chem.MolFromSmiles(ref)
        mol_q = Chem.MolFromSmiles(query)
        if mol_ref is None or mol_q is None:
            return ToolResult(data=None, success=False, error="Invalid SMILES in reference or query")

        fp_ref, n_bits = self._compute_fingerprint(mol_ref, args.fingerprint_type, args.radius, args.n_bits)
        fp_q, _ = self._compute_fingerprint(mol_q, args.fingerprint_type, args.radius, args.n_bits)

        sim = TanimotoSimilarity(fp_ref, fp_q)
        # Dice coefficient as secondary metric
        from rdkit.DataStructs import DiceSimilarity
        dice = DiceSimilarity(fp_ref, fp_q)

        return ToolResult(data={
            "reference_smiles": ref,
            "query_smiles": query,
            "fingerprint_type": args.fingerprint_type,
            "tanimoto": round(sim, 4),
            "dice": round(dice, 4),
            "interpretation": (
                "highly similar" if sim > 0.85
                else "moderately similar" if sim > 0.6
                else "weakly similar" if sim > 0.3
                else "dissimilar"
            ),
        })

    def _substructure(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem

        sub = args.substructure
        if not sub:
            return ToolResult(data=None, success=False, error="substructure SMILES is required")

        patt = Chem.MolFromSmiles(sub)
        if patt is None:
            return ToolResult(data=None, success=False, error=f"Invalid substructure: {sub}")

        targets = args.smiles_list or ([args.smiles] if args.smiles else [])
        if not targets:
            return ToolResult(data=None, success=False, error="Provide smiles or smiles_list")

        matches = []
        for smi in targets:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                matches.append({"smiles": smi, "match": False, "error": "invalid SMILES"})
                continue
            has = mol.HasSubstructMatch(patt)
            entry = {"smiles": smi, "match": has}
            if has:
                entry["match_count"] = len(mol.GetSubstructMatches(patt))
            matches.append(entry)

        n_hits = sum(1 for m in matches if m.get("match"))
        return ToolResult(data={
            "substructure": sub,
            "results": matches,
            "n_total": len(matches),
            "n_hits": n_hits,
        })

    def _draw(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem
        from rdkit.Chem import Draw

        smiles = args.smiles
        if not smiles:
            return ToolResult(data=None, success=False, error="smiles is required")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToolResult(data=None, success=False, error=f"Invalid SMILES: {smiles}")

        w, h = args.image_size
        png = Draw.MolDraw2DCairo(w, h)
        png.DrawMolecule(mol)
        png.FinishDrawing()
        img_bytes = png.GetDrawingText()

        out = Path(args.output_file) if args.output_file else Path("molecule.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(img_bytes)

        return ToolResult(data={
            "smiles": smiles,
            "image_file": str(out),
            "image_base64": base64.b64encode(img_bytes).decode()[:500] + "...",
            "size": [w, h],
        })

    def _conformers(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        smiles = args.smiles
        if not smiles:
            return ToolResult(data=None, success=False, error="smiles is required")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ToolResult(data=None, success=False, error=f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=args.n_conformers, params=params)

        if args.optimize:
            for cid in conf_ids:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid)

        out = Path(args.output_file) if args.output_file else Path("conformers.sdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(out))
        for cid in conf_ids:
            writer.write(mol, confId=cid)
        writer.close()

        # RMSD between first and last conformer as a spread indicator
        rmsds = []
        if len(conf_ids) > 1:
            for cid in conf_ids[1:]:
                rmsd = Chem.rdMolAlign.AlignMol(mol, mol, prbCid=cid, refCid=conf_ids[0])
                rmsds.append(round(rmsd, 3))

        return ToolResult(data={
            "smiles": smiles,
            "n_conformers": len(conf_ids),
            "sdf_file": str(out),
            "optimized": args.optimize,
            "rmsd_to_first": rmsds,
        })

    def _smiles_to_sdf(self, args: RDKitInput) -> ToolResult:
        from rdkit import Chem

        smiles_list = args.smiles_list or ([args.smiles] if args.smiles else [])
        if not smiles_list:
            return ToolResult(data=None, success=False, error="Provide smiles or smiles_list")

        out = Path(args.output_file) if args.output_file else Path("molecules.sdf")
        out.parent.mkdir(parents=True, exist_ok=True)

        writer = Chem.SDWriter(str(out))
        n_written = 0
        errors = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                errors.append({"index": i, "smiles": smi, "error": "invalid SMILES"})
                continue
            mol.SetProp("_Name", f"mol_{i}")
            writer.write(mol)
            n_written += 1
        writer.close()

        return ToolResult(data={
            "sdf_file": str(out),
            "n_written": n_written,
            "n_errors": len(errors),
            "errors": errors,
        })

    @staticmethod
    def _compute_fingerprint(mol, fp_type: str, radius: int, n_bits: int):
        """Return (fingerprint, n_bits) for the requested type."""
        from rdkit.Chem import AllChem, MACCSkeys

        if fp_type == "morgan":
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), n_bits
        if fp_type == "maccs":
            return MACCSkeys.GenMACCSKeys(mol), 167
        if fp_type == "atom_pair":
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
            return gen.GetFingerprint(mol), n_bits
        if fp_type == "topological":
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetRDKitGenerator(fpSize=n_bits)
            return gen.GetFingerprint(mol), n_bits
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
