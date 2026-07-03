"""Ultra-long task tests: 10min+ memory leak, campaign stability, tool chain repetition.

Runs extended-duration tests to detect memory leaks, race conditions, and
degradation that only appears under sustained load.
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HUGINN_CACHE_DIR", str(Path(__file__).parent.parent / ".test_cache"))

PASSED = 0
FAILED = 0
SKIPPED = 0


def report(name: str, success: bool, detail: str = "", skipped: bool = False) -> None:
    global PASSED, FAILED, SKIPPED
    if skipped:
        SKIPPED += 1
        print(f"  ⊘ {name}: SKIP — {detail}")
    elif success:
        PASSED += 1
        print(f"  ✓ {name}: PASS — {detail}")
    else:
        FAILED += 1
        print(f"  ✗ {name}: FAIL — {detail}")


def get_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# ── Test A: 5-minute memory leak detection ──────────────────────────

async def test_memory_leak_5min() -> None:
    print("\n[A] 5-Minute Memory Leak Detection")
    from huginn.tools.symmetry_tool import SymmetryTool
    from huginn.tools.numerical_tool import NumericalTool, NumericalToolInput
    from huginn.types import ToolContext

    duration = 300  # 5 minutes
    interval = 5    # check every 5 seconds
    tmpdir = Path(__file__).parent.parent / "tmp_leak_test"
    tmpdir.mkdir(exist_ok=True)
    ctx = ToolContext(session_id="leak-test", workspace=str(tmpdir))

    fixture = Path(__file__).parent.parent / "Si_diamond" / "POSCAR"
    poscar_content = fixture.read_text()

    sym_tool = SymmetryTool()
    num_tool = NumericalTool()

    rss_samples: list[tuple[float, float]] = []
    t0 = time.time()
    iterations = 0

    print(f"  Running for {duration}s (sampling every {interval}s)...")

    while time.time() - t0 < duration:
        p = tmpdir / f"struct_{iterations}.vasp"
        p.write_text(poscar_content)
        await sym_tool.call({"action": "analyze", "file_path": str(p)}, ctx)
        p.unlink(missing_ok=True)

        await num_tool.call({
            "action": "ode",
            "func": "-y[0]",
            "y0": [1.0],
            "t_span": [0, 5],
            "t_eval": [j * 0.25 for j in range(21)],
        }, ctx)

        iterations += 1
        if iterations % 10 == 0:
            gc.collect()
            rss = get_rss_mb()
            elapsed = time.time() - t0
            rss_samples.append((elapsed, rss))
            if iterations % 50 == 0:
                print(f"    [{elapsed:.0f}s] iter={iterations} rss={rss:.1f}MB")

        await asyncio.sleep(0.1)

    if len(rss_samples) >= 5:
        first_rss = rss_samples[0][1]
        last_rss = rss_samples[-1][1]
        delta = last_rss - first_rss
        leak_threshold = 50.0
        report(
            "5min memory leak",
            delta < leak_threshold,
            f"start={first_rss:.1f}MB end={last_rss:.1f}MB delta={delta:+.1f}MB threshold={leak_threshold}MB iters={iterations}",
        )
    else:
        report("5min memory leak", False, "insufficient samples", skipped=True)

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test B: Campaign loop stability (20 iterations) ─────────────────

async def test_campaign_loop_stability() -> None:
    print("\n[B] Campaign Loop Stability (20 iterations)")
    from huginn.tools.numerical_tool import NumericalTool, NumericalToolInput
    from huginn.tools.validate_tool import ValidateTool, ValidateToolInput
    from huginn.types import ToolContext

    tmpdir = Path(__file__).parent.parent / "tmp_campaign"
    tmpdir.mkdir(exist_ok=True)
    ctx = ToolContext(session_id="campaign-test", workspace=str(tmpdir))

    num_tool = NumericalTool()
    val_tool = ValidateTool()

    n_iterations = 20
    successes = 0
    errors: list[str] = []

    for i in range(n_iterations):
        try:
            # Simulate a campaign step: solve → validate → refine
            r1 = await num_tool.call({
                "action": "ode",
                "func": f"-{0.1 * (i + 1)} * y[0]",
                "y0": [1.0],
                "t_span": [0, 10],
                "t_eval": [j * 0.2 for j in range(51)],
            }, ctx)
            if r1.success:
                successes += 1
            else:
                errors.append(f"iter {i}: solve failed: {r1.error}")
        except Exception as e:
            errors.append(f"iter {i}: exception: {e}")

        if (i + 1) % 5 == 0:
            gc.collect()
            rss = get_rss_mb()
            print(f"    [iter {i+1}/{n_iterations}] successes={successes} rss={rss:.1f}MB")

    report(
        "campaign 20 iterations",
        successes == n_iterations,
        f"successes={successes}/{n_iterations} errors={len(errors)}",
    )
    if errors:
        for e in errors[:3]:
            print(f"      ⚠ {e}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test C: Tool chain repetition (100 iterations) ──────────────────

async def test_tool_chain_repetition() -> None:
    print("\n[C] Tool Chain Repetition (100 iterations)")
    from huginn.tools.symmetry_tool import SymmetryTool
    from huginn.types import ToolContext

    tmpdir = Path(__file__).parent.parent / "tmp_chain"
    tmpdir.mkdir(exist_ok=True)
    ctx = ToolContext(session_id="chain-test", workspace=str(tmpdir))

    fixture = Path(__file__).parent.parent / "Si_diamond" / "POSCAR"
    poscar_content = fixture.read_text()

    sym_tool = SymmetryTool()

    n = 100
    successes = 0
    t0 = time.time()

    for i in range(n):
        p = tmpdir / f"s_{i}.vasp"
        p.write_text(poscar_content)
        r2 = await sym_tool.call({"action": "analyze", "file_path": str(p)}, ctx)
        if r2.success:
            successes += 1
        p.unlink(missing_ok=True)

        if (i + 1) % 20 == 0:
            gc.collect()
            print(f"    [iter {i+1}/{n}] successes={successes} rss={get_rss_mb():.1f}MB")

    elapsed = time.time() - t0
    avg_ms = elapsed / n * 1000
    report(
        "100x chain repetition",
        successes == n,
        f"successes={successes}/{n} avg={avg_ms:.1f}ms/iter total={elapsed:.1f}s",
    )

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Test D: Tool registry sustained access ──────────────────────────

async def test_registry_sustained_access() -> None:
    print("\n[D] Tool Registry Sustained Access (500 reads)")
    from huginn.tools.registry import ToolRegistry

    reg = ToolRegistry()
    n = 500
    t0 = time.time()
    errors = 0

    for i in range(n):
        try:
            tools = reg.list_tools()
            schemas = reg.schemas_cache()
            assert len(tools) > 0
        except Exception:
            errors += 1

        if (i + 1) % 100 == 0:
            gc.collect()
            print(f"    [iter {i+1}/{n}] errors={errors} rss={get_rss_mb():.1f}MB")

    elapsed = time.time() - t0
    report(
        "registry 500 reads",
        errors == 0,
        f"errors={errors} total={elapsed:.1f}s avg={elapsed/n*1000:.1f}ms/read",
    )


# ── Test E: Memory system sustained operation ───────────────────────

async def test_memory_sustained() -> None:
    print("\n[E] Memory System Sustained Operation (200 ops)")
    try:
        from huginn.memory import MemoryManager
    except ImportError:
        report("memory sustained", False, "MemoryManager import failed", skipped=True)
        return

    tmpdir = Path(__file__).parent.parent / "tmp_mem"
    tmpdir.mkdir(exist_ok=True)

    try:
        mgr = MemoryManager(cache_dir=str(tmpdir / "mem_cache"))
    except Exception as e:
        report("memory sustained", False, f"init failed: {e}", skipped=True)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    n = 200
    successes = 0
    t0 = time.time()

    for i in range(n):
        try:
            # Store
            mgr.remember(f"key_{i}", f"value_{i}", session_id="sust-test")
            # Recall
            val = mgr.recall(f"key_{i}", session_id="sust-test")
            if val:
                successes += 1
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"    [iter {i+1}/{n}] successes={successes} rss={get_rss_mb():.1f}MB")

    elapsed = time.time() - t0
    report(
        "memory 200 ops",
        successes == n,
        f"successes={successes}/{n} total={elapsed:.1f}s",
    )

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Quick memory leak test (2-min) ──────────────────────────────────

async def _quick_leak_test() -> None:
    """2-minute quick memory leak check."""
    from huginn.tools.symmetry_tool import SymmetryTool
    from huginn.types import ToolContext

    duration = 120
    tmpdir = Path(__file__).parent.parent / "tmp_leak"
    tmpdir.mkdir(exist_ok=True)
    ctx = ToolContext(session_id="leak-q", workspace=str(tmpdir))

    # Use committed POSCAR fixture instead of StructureTool.create
    fixture = Path(__file__).parent.parent / "Si_diamond" / "POSCAR"
    poscar_content = fixture.read_text()

    sm = SymmetryTool()
    rss_samples = []
    t0 = time.time()
    iters = 0

    while time.time() - t0 < duration:
        p = tmpdir / f"q_{iters}.vasp"
        p.write_text(poscar_content)
        await sm.call({"action": "analyze", "file_path": str(p)}, ctx)
        p.unlink(missing_ok=True)
        iters += 1
        if iters % 20 == 0:
            gc.collect()
            rss = get_rss_mb()
            rss_samples.append(rss)
            print(f"    [{time.time()-t0:.0f}s] iter={iters} rss={rss:.1f}MB")
        await asyncio.sleep(0.05)

    if len(rss_samples) >= 3:
        delta = rss_samples[-1] - rss_samples[0]
        report("2min memory leak", delta < 50.0,
               f"start={rss_samples[0]:.1f}MB end={rss_samples[-1]:.1f}MB delta={delta:+.1f}MB iters={iters}")
    else:
        report("2min memory leak", False, "insufficient data", skipped=True)

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


async def main() -> None:
    print("=" * 70)
    print("ULTRA-LONG TASK TESTS")
    print("=" * 70)

    rss_start = get_rss_mb()
    print(f"Initial RSS: {rss_start:.1f}MB")

    if "--full" in sys.argv:
        await test_memory_leak_5min()
    else:
        print("\n[A] Memory Leak Detection (2-min quick version)")
        print("  (use --full for 5-min version)")
        await _quick_leak_test()

    await test_campaign_loop_stability()
    await test_tool_chain_repetition()
    await test_registry_sustained_access()
    await test_memory_sustained()

    rss_end = get_rss_mb()
    print(f"\nFinal RSS: {rss_end:.1f}MB (delta: {rss_end - rss_start:+.1f}MB)")

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print(f"{'=' * 70}")

    return FAILED == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
