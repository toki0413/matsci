"""Data dictionary endpoints — list, inspect, and validate data schemas."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from huginn.data.dictionary import DataDictionary
from huginn.data.types import DataType

router = APIRouter(tags=["data"])


class ValidateRequest(BaseModel):
    type_name: str
    data: dict[str, Any]


@router.get("/data/dictionary")
async def list_data_types() -> list[dict[str, Any]]:
    """List all registered data types with their descriptions."""
    result = []
    for dt in DataType:
        schema = DataDictionary.get(dt)
        if schema is not None:
            result.append(
                {
                    "type_name": dt.value,
                    "description": schema.description,
                    "version": schema.version,
                    "tags": schema.tags,
                    "num_fields": len(schema.fields),
                }
            )
    return result


@router.get("/data/dictionary/{type_name}")
async def get_data_schema(type_name: str) -> dict[str, Any]:
    """Get full schema details for a specific data type."""
    try:
        dt = DataType(type_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown data type: {type_name}")

    schema = DataDictionary.get(dt)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"Schema not found: {type_name}")

    return {
        "type_name": schema.type_name.value,
        "description": schema.description,
        "version": schema.version,
        "tags": schema.tags,
        "fields": [
            {
                "name": f.name,
                "dtype": f.dtype,
                "required": f.required,
                "description": f.description,
                "unit": f.unit,
            }
            for f in schema.fields
        ],
    }


@router.post("/data/validate")
async def validate_data(req: ValidateRequest) -> dict[str, Any]:
    """Validate data against a registered schema."""
    try:
        dt = DataType(req.type_name)
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Unknown data type: {req.type_name}"
        )

    errors = DataDictionary.validate(dt, req.data)
    return {
        "type_name": req.type_name,
        "valid": len(errors) == 0,
        "errors": errors,
    }
