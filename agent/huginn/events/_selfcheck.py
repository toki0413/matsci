"""Self-check for the events package. No frameworks, just asserts.

Run: python -m huginn.events._selfcheck
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Make sure we can import huginn when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


async def test_basic_publish_subscribe():
    """Publish an event, verify subscriber receives it."""
    from huginn.events.event_bus import AgentEvent, EventBus
    import time

    bus = EventBus(history_size=100)
    received = []

    def handler(event: AgentEvent):
        received.append(event)

    unsub = bus.subscribe("tool.call", handler)

    evt = AgentEvent(
        type="tool.call",
        timestamp=time.time(),
        data={"tool": "vasp", "input": {"structure": "Si.cif"}},
        thread_id="t1",
        source="test",
    )
    await bus.publish(evt)

    assert len(received) == 1, f"expected 1 event, got {len(received)}"
    assert received[0].type == "tool.call"
    assert received[0].data["tool"] == "vasp"
    print("  [OK] basic publish/subscribe")

    unsub()
    await bus.publish(AgentEvent(type="tool.call", timestamp=time.time()))
    assert len(received) == 1, "unsubscribed handler should not fire"
    print("  [OK] unsubscribe works")
    bus.shutdown()


async def test_wildcard_subscriber():
    """Wildcard subscriber gets all events."""
    from huginn.events.event_bus import AgentEvent, EventBus
    import time

    bus = EventBus()
    all_events = []
    bus.subscribe("*", lambda e: all_events.append(e))

    await bus.publish(AgentEvent(type="tool.call", timestamp=time.time()))
    await bus.publish(AgentEvent(type="compact.start", timestamp=time.time()))
    await bus.publish(AgentEvent(type="session.end", timestamp=time.time()))

    assert len(all_events) == 3, f"expected 3, got {len(all_events)}"
    print("  [OK] wildcard subscriber receives all")
    bus.shutdown()


async def test_history_and_recent():
    """recent_events returns filtered history."""
    from huginn.events.event_bus import AgentEvent, EventBus
    import time

    bus = EventBus(history_size=50)
    for i in range(10):
        await bus.publish(AgentEvent(
            type="tool.call" if i < 5 else "tool.result",
            timestamp=time.time(),
            data={"i": i},
        ))

    recent_all = bus.recent_events(n=100)
    assert len(recent_all) == 10, f"expected 10, got {len(recent_all)}"
    # most-recent-first
    assert recent_all[0].data["i"] == 9

    recent_calls = bus.recent_events(n=100, event_type="tool.call")
    assert len(recent_calls) == 5
    print("  [OK] recent_events with filter")
    bus.shutdown()


async def test_sse_stream():
    """SSE stream yields formatted event strings."""
    from huginn.events.event_bus import AgentEvent, EventBus
    import time

    bus = EventBus()

    async def consumer():
        async for sse in bus.sse_stream():
            return sse

    # Start consumer, then publish
    task = asyncio.ensure_future(consumer())
    await asyncio.sleep(0.01)  # let consumer register its queue

    await bus.publish(AgentEvent(
        type="tool.call",
        timestamp=time.time(),
        data={"tool": "bash"},
        thread_id="t1",
        source="test",
    ))

    sse = await asyncio.wait_for(task, timeout=1.0)
    assert sse.startswith("event: tool.call\n")
    assert '"tool": "bash"' in sse
    assert sse.endswith("\n\n")
    print("  [OK] SSE stream format correct")
    bus.shutdown()


async def test_integration_helpers():
    """Integration helpers publish events without crashing."""
    from huginn.events import integration
    from huginn.events.event_bus import EventBus

    # Reset the singleton so we get a clean bus
    EventBus._instance = None
    bus = EventBus.shared()
    received = []
    bus.subscribe("*", lambda e: received.append(e))

    await integration.publish_tool_event(
        "vasp_tool", {"structure": "Si.cif"}, {"energy": -0.5}, "t1"
    )
    assert len(received) == 2  # tool.call + tool.result
    assert received[0].type == "tool.call"
    assert received[1].type == "tool.result"

    await integration.publish_compact_event(85.0, 40.0, "t1")
    assert any(e.type == "compact.start" for e in received)
    assert any(e.type == "compact.end" for e in received)

    # Overflow threshold triggers context.overflow
    received.clear()
    await integration.publish_compact_event(95.0, 40.0, "t1")
    assert any(e.type == "context.overflow" for e in received)

    await integration.publish_pipeline_event("try DFT first", "t1", stage="scf")
    assert any(e.type == "pipeline.stage_change" for e in received)

    await integration.publish_pipeline_event("consider MD", "t1")
    assert any(e.type == "pipeline.suggest" for e in received)

    print("  [OK] integration helpers work")
    bus.shutdown()


async def test_audit_log():
    """Audit subscriber writes JSONL lines to file."""
    from huginn.events.event_bus import AgentEvent, EventBus
    from huginn.events import audit_log

    with tempfile.TemporaryDirectory() as tmp:
        audit_path = Path(tmp) / "audit.jsonl"
        EventBus._instance = None
        bus = EventBus.shared()

        unsub = audit_log.install_audit_subscriber(
            bus=bus, path=audit_path
        )

        await bus.publish(AgentEvent(
            type="tool.call",
            timestamp=1234567890.0,
            data={"tool": "bash"},
            thread_id="t1",
            source="test",
        ))
        await bus.publish(AgentEvent(
            type="session.end",
            timestamp=1234567891.0,
            thread_id="t1",
            source="test",
        ))

        # Give the sync file write a moment
        await asyncio.sleep(0.01)

        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2, f"expected 2 lines, got {len(lines)}"

        first = json.loads(lines[0])
        assert first["type"] == "tool.call"
        assert first["data"]["tool"] == "bash"
        assert first["thread_id"] == "t1"

        second = json.loads(lines[1])
        assert second["type"] == "session.end"

        print("  [OK] audit log writes JSONL correctly")
        unsub()
        bus.shutdown()


async def test_safety_noop():
    """Integration helpers don't crash when events are disabled."""
    from huginn.events import integration

    integration.disable_events()
    # Should be no-ops, no exceptions
    await integration.publish_tool_event("x", {}, {}, "t1")
    await integration.publish_compact_event(50, 30, "t1")
    await integration.publish_pipeline_event("x", "t1")
    integration.enable_events()
    print("  [OK] disabled events are no-ops")


def test_sse_format():
    """AgentEvent.to_sse produces valid SSE frame."""
    from huginn.events.event_bus import AgentEvent

    evt = AgentEvent(
        type="tool.call",
        timestamp=1234567890.0,
        data={"tool": "bash"},
        thread_id="t1",
        source="test",
    )
    sse = evt.to_sse()
    assert sse.startswith("event: tool.call\n")
    assert "data: " in sse
    assert sse.endswith("\n\n")
    # Data should be valid JSON
    data_line = [l for l in sse.split("\n") if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["type"] == "tool.call"
    assert payload["data"]["tool"] == "bash"
    print("  [OK] SSE format is valid")


async def main():
    print("Running events package self-checks...")
    test_sse_format()
    await test_basic_publish_subscribe()
    await test_wildcard_subscriber()
    await test_history_and_recent()
    await test_sse_stream()
    await test_integration_helpers()
    await test_audit_log()
    await test_safety_noop()
    print("\nAll self-checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
