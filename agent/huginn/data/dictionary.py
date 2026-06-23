"""Data dictionary — registry of known data schemas."""
from __future__ import annotations

from typing import Any

from huginn.data.types import DataField, DataSchema, DataType


class DataDictionary:
    _schemas: dict[DataType, DataSchema] = {}

    @classmethod
    def register(cls, schema: DataSchema) -> None:
        cls._schemas[schema.type_name] = schema

    @classmethod
    def get(cls, data_type: DataType) -> DataSchema | None:
        return cls._schemas.get(data_type)

    @classmethod
    def list_types(cls) -> list[str]:
        return [dt.value for dt in cls._schemas]

    @classmethod
    def validate(cls, data_type: DataType, data: dict[str, Any]) -> list[str]:
        """Validate data against its schema. Returns list of error messages."""
        schema = cls._schemas.get(data_type)
        if not schema:
            return [f"Unknown data type: {data_type.value}"]
        errors = []
        for f in schema.fields:
            if f.required and f.name not in data:
                errors.append(f"Missing required field: {f.name}")
        return errors


# Register built-in schemas
def _register_builtins() -> None:
    DataDictionary.register(
        DataSchema(
            type_name=DataType.CRYSTAL_STRUCTURE,
            description="Crystal structure data (POSCAR, CIF, XYZ formats)",
            fields=[
                DataField("formula", "str", required=True, description="Chemical formula"),
                DataField(
                    "lattice_params", "dict", description="a, b, c, alpha, beta, gamma"
                ),
                DataField("spacegroup", "str", description="Space group symbol"),
                DataField("num_atoms", "int", description="Number of atoms"),
            ],
            tags=["structure", "crystallography"],
        )
    )
    DataDictionary.register(
        DataSchema(
            type_name=DataType.DFT_RESULT,
            description="DFT calculation results (VASP, QE, CP2K)",
            fields=[
                DataField(
                    "energy", "float", required=True, description="Total energy", unit="eV"
                ),
                DataField(
                    "converged",
                    "int",
                    required=True,
                    description="Convergence flag (0/1)",
                ),
                DataField("forces", "array", description="Atomic forces"),
                DataField(
                    "band_gap", "float", description="Electronic band gap", unit="eV"
                ),
            ],
            tags=["dft", "electronic_structure"],
        )
    )
    DataDictionary.register(
        DataSchema(
            type_name=DataType.MOLECULAR_DYNAMICS,
            description="Molecular dynamics trajectory and thermodynamic data",
            fields=[
                DataField(
                    "n_frames",
                    "int",
                    required=True,
                    description="Number of trajectory frames",
                ),
                DataField(
                    "n_atoms", "int", required=True, description="Number of atoms"
                ),
                DataField(
                    "temperature",
                    "array",
                    description="Temperature time series",
                    unit="K",
                ),
                DataField(
                    "pressure",
                    "array",
                    description="Pressure time series",
                    unit="bar",
                ),
            ],
            tags=["md", "trajectory"],
        )
    )
    DataDictionary.register(
        DataSchema(
            type_name=DataType.POTENTIAL,
            description="Machine learning potential data",
            fields=[
                DataField(
                    "potential_type",
                    "str",
                    required=True,
                    description="NEP, SNAP, GAP, ACE",
                ),
                DataField(
                    "training_rmse_energy",
                    "float",
                    description="Training RMSE for energy",
                    unit="eV/atom",
                ),
                DataField(
                    "training_rmse_force",
                    "float",
                    description="Training RMSE for forces",
                    unit="eV/A",
                ),
            ],
            tags=["ml", "potential"],
        )
    )
    DataDictionary.register(
        DataSchema(
            type_name=DataType.JOB_RECORD,
            description="HPC job record",
            fields=[
                DataField(
                    "job_id", "str", required=True, description="Job identifier"
                ),
                DataField(
                    "status", "str", required=True, description="Job status"
                ),
                DataField(
                    "walltime_hours", "float", description="Walltime in hours"
                ),
            ],
            tags=["hpc", "job"],
        )
    )
    DataDictionary.register(
        DataSchema(
            type_name=DataType.EXPERIMENTAL_DATA,
            description="Experimental measurement data",
            fields=[
                DataField(
                    "measurement_type",
                    "str",
                    required=True,
                    description="XRD, SEM, TEM, etc.",
                ),
                DataField("sample_id", "str", description="Sample identifier"),
                DataField("data_file", "str", description="Path to data file"),
            ],
            tags=["experimental"],
        )
    )
    DataDictionary.register(
        DataSchema(
            type_name=DataType.DESCRIPTOR,
            description="Material descriptors for ML models",
            fields=[
                DataField(
                    "descriptor_name",
                    "str",
                    required=True,
                    description="Descriptor name",
                ),
                DataField(
                    "values", "array", required=True, description="Descriptor values"
                ),
                DataField(
                    "dimension", "int", description="Descriptor dimensionality"
                ),
            ],
            tags=["ml", "descriptor"],
        )
    )


_register_builtins()
