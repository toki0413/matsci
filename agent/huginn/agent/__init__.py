"""HuginnAgent package — re-exports from split modules for backward compat.

All existing imports `from huginn.agent import HuginnAgent` continue to work.
The implementation is split across:
  core.py       — HuginnAgent class (init, factory, graph, dispatch)
  context.py    — system prompt assembly, tool filtering, cache stats
  streaming.py  — chat() async generator, compaction, phase transitions
  reflection.py — evolution engine, summarizer, post-turn reflection
  session.py    — session state properties, cross-session continuity
  callbacks.py  — hook wrapping, approval callbacks, scheduler admission
  middlewares.py — FixDanglingToolCallsMiddleware, RateLimitMiddleware
"""

from huginn.agent.core import Agent, HuginnAgent

__all__ = ["HuginnAgent", "Agent"]
