"""Desktop pet endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.pet import (
    configure_pet,
    feed_pet,
    get_pet_bus,
    pet_stroke,
    reset_pet_progress,
    toggle_pet_accessory,
)

router = APIRouter(tags=["pet"])


@router.post("/pet/feed")
async def pet_feed(amount: int = 25) -> dict[str, Any]:
    """Feed the desktop pet."""
    feed_pet(amount)
    bus = get_pet_bus()
    return {"ok": True, "hunger": bus.state.hunger}


@router.post("/pet/pet")
async def pet_pet(amount: int = 15) -> dict[str, Any]:
    """Stroke/pet the desktop pet."""
    pet_stroke(amount)
    bus = get_pet_bus()
    return {"ok": True, "happiness": bus.state.happiness}


@router.post("/pet/accessory")
async def pet_accessory(params: dict[str, Any]) -> dict[str, Any]:
    """Toggle an accessory on the desktop pet."""
    accessory_id = params.get("accessory_id", "")
    toggle_pet_accessory(accessory_id)
    bus = get_pet_bus()
    return {"ok": True, "accessories": bus.state.accessories}


@router.post("/pet/reset")
async def pet_reset() -> dict[str, Any]:
    """Reset pet gamification progress."""
    reset_pet_progress()
    bus = get_pet_bus()
    return {"ok": True, "level": bus.state.level, "experience": bus.state.experience}


@router.post("/pet/configure")
async def pet_configure(params: dict[str, Any]) -> dict[str, Any]:
    """Configure pet name, personality, and avatar."""
    name = params.get("name")
    personality = params.get("personality")
    avatar = params.get("avatar")
    configure_pet(name=name, personality=personality, avatar=avatar)
    bus = get_pet_bus()
    return {"ok": True, "name": bus.state.name, "personality": bus.state.personality}


@router.get("/pet/status")
async def pet_status() -> dict[str, Any]:
    """Get current pet status including gamification state."""
    bus = get_pet_bus()
    return bus.state.to_dict()
