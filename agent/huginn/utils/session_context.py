"""Per-request session context backed by contextvars.

Solves the singleton race condition: when the HuginnAgent is shared across
concurrent WebSocket connections, instance attributes like ``thread_id`` and
``_current_user_message`` get overwritten by whichever coroutine runs last.

contextvars.ContextVar gives each async task its own isolated copy, so
concurrent ``chat()`` calls on the same agent instance no longer stomp on
each other's state.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

# Each var defaults to a sentinel so we can tell "not set" apart from
# legitimately falsy values like "" or "default".
_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "huginn_thread_id", default=None,
)
_user_message: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "huginn_user_message", default=None,
)


def set_thread_id(thread_id: str) -> contextvars.Token[None]:
    """Set the current thread_id for this async context.

    Returns a token that can be passed to :func:`reset_thread_id` to restore
    the previous value.  In most cases you want the context manager instead.
    """
    return _thread_id.set(thread_id)


def get_thread_id() -> str | None:
    """Read the thread_id for the current async context, or None."""
    return _thread_id.get()


def reset_thread_id(token: contextvars.Token) -> None:
    _thread_id.reset(token)


def set_user_message(message: str) -> contextvars.Token[None]:
    return _user_message.set(message)


def get_user_message() -> str | None:
    return _user_message.get()


def reset_user_message(token: contextvars.Token) -> None:
    _user_message.reset(token)


@contextmanager
def session_scope(thread_id: str, user_message: str = "") -> Iterator[None]:
    """Context manager that sets thread_id + user_message for the duration
    of a single agent turn, then restores the previous values.

    Usage::

        with session_scope(thread_id, message):
            # all code inside here sees the correct per-request values
            await agent._invoke_with_hooks(...)
    """
    tok_tid = _thread_id.set(thread_id)
    tok_msg = _user_message.set(user_message)
    try:
        yield
    finally:
        _thread_id.reset(tok_tid)
        _user_message.reset(tok_msg)
