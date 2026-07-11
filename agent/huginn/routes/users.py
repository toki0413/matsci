"""User management endpoints (admin-only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from huginn.security.auth import get_user_store, require_admin_key
from huginn.security.rbac import Role, User

router = APIRouter(
    prefix="/users",
    tags=["users"],
    dependencies=[Depends(require_admin_key)],
)


def _mask_key(api_key_hash: str) -> str:
    """Mask a key hash for display: first 8 + **** + last 4."""
    if not api_key_hash or len(api_key_hash) < 12:
        return "****"
    return api_key_hash[:8] + "****" + api_key_hash[-4:]


def _user_public(user: User) -> dict[str, Any]:
    """Serialize a User for API responses — never exposes the raw hash."""
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role.value,
        "active": user.active,
        "created_at": user.created_at,
        "metadata": dict(user.metadata),
        "api_key_masked": _mask_key(user.api_key_hash),
    }


def _parse_role(role_str: str | None) -> Role | None:
    """Map a string like 'admin' to a Role. Returns None on bad input."""
    if role_str is None:
        return None
    try:
        return Role(role_str)
    except ValueError:
        return None


@router.get("")
async def list_users(active_only: bool = False) -> dict[str, Any]:
    """List all registered users."""
    store = get_user_store()
    users = store.list_users(active_only=active_only)
    return {"users": [_user_public(u) for u in users], "count": len(users)}


@router.post("")
async def create_user(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new user. The plaintext API key is returned once."""
    username = params.get("username")
    if not username:
        return {"success": False, "error": "username is required"}

    role = _parse_role(params.get("role", "viewer"))
    if role is None:
        return {
            "success": False,
            "error": "invalid role; must be one of: viewer, operator, admin",
        }

    metadata = params.get("metadata") or {}
    store = get_user_store()
    try:
        user, api_key = store.create_user(
            username=username,
            role=role,
            metadata=metadata,
        )
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    data = _user_public(user)
    data["api_key"] = api_key  # plaintext — only shown on create/rotate
    data["success"] = True
    return data


@router.get("/{user_id}")
async def get_user(user_id: str) -> dict[str, Any]:
    """Get details for a single user."""
    store = get_user_store()
    user = store.get_user(user_id)
    if user is None:
        return {"success": False, "error": "user not found"}
    return _user_public(user)


@router.patch("/{user_id}")
async def update_user_role(user_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Update a user's role."""
    role = _parse_role(params.get("role"))
    if role is None:
        return {
            "success": False,
            "error": "invalid role; must be one of: viewer, operator, admin",
        }
    store = get_user_store()
    try:
        user = store.update_role(user_id, role)
    except KeyError:
        return {"success": False, "error": "user not found"}
    return {"success": True, **_user_public(user)}


@router.post("/{user_id}/rotate-key")
async def rotate_api_key(user_id: str) -> dict[str, Any]:
    """Rotate a user's API key. Returns the new plaintext key once."""
    store = get_user_store()
    try:
        new_key = store.rotate_api_key(user_id)
    except KeyError:
        return {"success": False, "error": "user not found"}
    user = store.get_user(user_id)
    return {
        "success": True,
        "user_id": user_id,
        "api_key": new_key,
        "api_key_masked": _mask_key(user.api_key_hash) if user else "****",
    }


@router.post("/{user_id}/deactivate")
async def deactivate_user(user_id: str) -> dict[str, Any]:
    """Deactivate a user without deleting."""
    store = get_user_store()
    try:
        store.deactivate_user(user_id)
    except KeyError:
        return {"success": False, "error": "user not found"}
    return {"success": True, "user_id": user_id, "active": False}


@router.delete("/{user_id}")
async def delete_user(user_id: str) -> dict[str, Any]:
    """Permanently delete a user."""
    store = get_user_store()
    try:
        store.delete_user(user_id)
    except KeyError:
        return {"success": False, "error": "user not found"}
    return {"success": True, "user_id": user_id}


@router.get("/{user_id}/test")
async def test_user_key(user_id: str) -> dict[str, Any]:
    """Check whether a user's stored credentials are usable."""
    store = get_user_store()
    user = store.get_user(user_id)
    if user is None:
        return {"success": False, "error": "user not found"}

    # A valid SHA-256 hex digest is 64 chars. Shorter means the key was
    # never set or got corrupted.
    hash_ok = len(user.api_key_hash) == 64 and all(
        c in "0123456789abcdef" for c in user.api_key_hash
    )
    return {
        "success": True,
        "user_id": user_id,
        "username": user.username,
        "active": user.active,
        "api_key_valid": hash_ok,
        "api_key_masked": _mask_key(user.api_key_hash),
    }
