"""Persistent user store for RBAC.

Stores users in a JSON file (one entry per user) with SHA-256 hashed API keys.
Thread-safe; suitable for small-to-medium deployments.  For larger deployments
swap in a database-backed implementation that satisfies the same interface.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from huginn.security.rbac import (
    Role,
    User,
    generate_api_key,
    hash_api_key,
)


class UserStore:
    """JSON-file-backed user store with in-memory cache."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = Path.home() / ".huginn" / "users.json"
        self.store_path = Path(store_path)
        self._lock = threading.Lock()
        self._users: dict[str, User] = {}          # user_id → User
        self._key_index: dict[str, str] = {}        # api_key_hash → user_id
        self._username_index: dict[str, str] = {}   # username → user_id
        self._load()

    # -- persistence ----------------------------------------------------

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, encoding="utf-8") as f:
                data = json.load(f)
            for uid, udata in data.get("users", {}).items():
                user = User.from_dict(udata)
                self._users[uid] = user
                if user.api_key_hash:
                    self._key_index[user.api_key_hash] = uid
                self._username_index[user.username] = uid
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupted file — start fresh but keep backup
            backup = self.store_path.with_suffix(".json.bak")
            if self.store_path.exists():
                self.store_path.rename(backup)
            self._users.clear()
            self._key_index.clear()
            self._username_index.clear()

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": time.time(),
            "users": {uid: u.to_dict() for uid, u in self._users.items()},
        }
        tmp = self.store_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.store_path)

    # -- CRUD -----------------------------------------------------------

    def create_user(
        self,
        username: str,
        role: Role = Role.VIEWER,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[User, str]:
        """Create a user and return ``(user, plaintext_api_key)``.

        The plaintext key is shown **once**; only the hash is stored.
        """
        with self._lock:
            if username in self._username_index:
                raise ValueError(f"Username '{username}' already exists")

            import uuid
            uid = user_id or uuid.uuid4().hex[:16]
            api_key = generate_api_key()
            key_hash = hash_api_key(api_key)

            user = User(
                user_id=uid,
                username=username,
                role=role,
                api_key_hash=key_hash,
                metadata=metadata or {},
            )
            self._users[uid] = user
            self._key_index[key_hash] = uid
            self._username_index[username] = uid
            self._save()
            return user, api_key

    def get_user(self, user_id: str) -> User | None:
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        with self._lock:
            uid = self._username_index.get(username)
            return self._users.get(uid) if uid else None

    def get_user_by_api_key(self, api_key: str) -> User | None:
        with self._lock:
            key_hash = hash_api_key(api_key)
            uid = self._key_index.get(key_hash)
            return self._users.get(uid) if uid else None

    def update_role(self, user_id: str, role: Role) -> User:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                raise KeyError(f"User '{user_id}' not found")
            user.role = role
            self._save()
            return user

    def deactivate_user(self, user_id: str) -> None:
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                raise KeyError(f"User '{user_id}' not found")
            user.active = False
            self._save()

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            user = self._users.pop(user_id, None)
            if user is None:
                raise KeyError(f"User '{user_id}' not found")
            self._key_index.pop(user.api_key_hash, None)
            self._username_index.pop(user.username, None)
            self._save()

    def list_users(self, *, active_only: bool = False) -> list[User]:
        users = list(self._users.values())
        if active_only:
            users = [u for u in users if u.active]
        return users

    def rotate_api_key(self, user_id: str) -> str:
        """Generate a new API key for the user.  Returns plaintext key."""
        with self._lock:
            user = self._users.get(user_id)
            if user is None:
                raise KeyError(f"User '{user_id}' not found")
            # Remove old key from index
            self._key_index.pop(user.api_key_hash, None)
            new_key = generate_api_key()
            user.api_key_hash = hash_api_key(new_key)
            self._key_index[user.api_key_hash] = user_id
            self._save()
            return new_key
