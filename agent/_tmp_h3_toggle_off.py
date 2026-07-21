"""1 loop, 1 iteration, toggle OFF — 验证是否 autoloop 本身崩."""
from __future__ import annotations
import asyncio
import time
from pathlib import Path

from huginn.autoloop.engine import AutoloopEngine
print("[import] OK", flush=True)

async def run_one():
    t0 = time.time()
    engine = AutoloopEngine(workspace=Path("."))
    print("[run] engine ready, calling run_cognitive...", flush=True)
    result = await engine.run_cognitive(
        objective="Compare LDA vs GGA exchange-correlation functionals for silicon band gap: summarize literature consensus without running DFT",
        max_iterations=1,
        progressive_budget=False,
    )
    print(f"[run] done in {time.time()-t0:.1f}s, success={getattr(result,'success',False)}", flush=True)

asyncio.run(run_one())
print("[done] OK", flush=True)
