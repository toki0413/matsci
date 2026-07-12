"""Async multi-track research pipeline integration test.

Runs 3 research tracks in parallel (perovskite, electrolyte, battery),
each with 4 stages (survey -> topic -> research -> paper), then a
cross-track synthesis stage. Requires a running backend.

Usage:
    python tests/test_async_pipeline.py
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from websockets.asyncio.client import connect

# Allow running standalone from tests/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WS_URL = "ws://127.0.0.1:8000/ws/agent?token=dev"
MAX_SIZE = 2**24

# Relaxed timeouts — research is long-horizon
STAGE_TIMEOUT = 1800       # 30 min hard ceiling per stage
RECV_TIMEOUT = 180          # 3 min between messages
PROGRESS_EXTENSION = 600    # extend if still producing output

TRACKS = [
    {
        "id": "perovskite",
        "topic": "钙钛矿太阳能电池稳定性",
        "stages": [
            ("调研", "research",
             "我正在进行钙钛矿太阳能电池稳定性的系统性文献调研。"
             "请概述基本原理，分析主要降解机制（湿度/热/光/离子迁移），"
             "总结稳定性提升策略，并查询几种典型钙钛矿材料的带隙数据。"),
            ("选题", "research",
             "基于前面的调研，确定一个具体的研究方向。"
             "分析未解决的关键问题，选出最具价值且可行的方向，"
             "提出研究假设，设计初步的实验/计算验证方案。"),
            ("研究", "research",
             "深入研究选定方向。用符号计算工具建立带隙与容忍因子的数学模型，"
             "分析不同A位阳离子对结构稳定性的影响，"
             "查询材料数据库验证模型，讨论界面工程的定量影响。"),
            ("论文", "research",
             "将所有研究成果整合为论文框架：摘要、章节结构、各章概述、"
             "需要进一步验证的部分、创新点和预期贡献。"),
        ],
    },
    {
        "id": "electrolyte",
        "topic": "固态电解质高通量筛选",
        "stages": [
            ("调研", "research",
             "调研固态锂离子电解质的最新进展。"
             "概述主要材料体系（LLZO、LGPS、argyrodite），"
             "分析离子导电性的微观机制，总结高通量计算筛选的方法学。"),
            ("选题", "research",
             "确定一个具体的固态电解质研究方向。"
             "分析当前瓶颈（室温离子电导率、界面稳定性），"
             "提出研究假设，设计描述符和筛选标准。"),
            ("研究", "research",
             "深入研究选定方向。用符号计算工具分析离子迁移激活能的表达式，"
             "讨论掺杂策略对导电性的影响，"
             "查询材料数据库获取候选材料数据，提出高通量筛选流程。"),
            ("论文", "research",
             "整合研究成果为论文框架：摘要、方法、结果讨论、结论。"
             "标注需要DFT验证的材料体系，提出实验验证方案。"),
        ],
    },
    {
        "id": "battery",
        "topic": "锂离子电池正极材料降解",
        "stages": [
            ("调研", "research",
             "调研锂离子电池正极材料（NCM/NCA/LFP）的降解机制。"
             "概述主要降解路径（结构相变、过渡金属溶解、电解液分解），"
             "总结多尺度建模方法在电池研究中的应用。"),
            ("选题", "research",
             "确定一个具体的正极材料降解研究方向。"
             "分析多尺度模拟的局限性，"
             "提出研究假设，设计跨尺度验证方案。"),
            ("研究", "research",
             "深入研究选定方向。用符号计算工具分析相变热力学条件，"
             "讨论循环次数对微观结构的影响，"
             "查询材料数据库获取正极材料的热力学数据。"),
            ("论文", "research",
             "整合研究成果为论文框架：摘要、多尺度方法、结果分析、结论。"
             "提出工程化改进建议，标注需要实验验证的关键假设。"),
        ],
    },
]


@dataclass
class StageResult:
    track_id: str
    stage_name: str
    elapsed_s: float = 0.0
    reasoning_chunks: int = 0
    text_chunks: int = 0
    text_length: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    errors: list = field(default_factory=list)
    completed: bool = False
    text_preview: str = ""


@dataclass
class TrackResult:
    track_id: str
    topic: str
    stages: list[StageResult] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    completed: bool = False


async def run_stage(track_id, stage_name, persona, prompt, thread_id):
    result = StageResult(track_id=track_id, stage_name=stage_name)
    start = time.time()
    full_text = []
    last_progress = start

    try:
        async with connect(
            WS_URL,
            open_timeout=60,
            ping_interval=None,
            ping_timeout=None,
            max_size=MAX_SIZE,
        ) as ws:
            await ws.send(json.dumps({
                "type": "user_input",
                "content": prompt,
                "thread_id": thread_id,
                "persona": persona,
            }))

            deadline = start + STAGE_TIMEOUT

            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    idle = time.time() - last_progress
                    if idle > RECV_TIMEOUT * 3:
                        result.errors.append(f"stalled: no output for {idle:.0f}s")
                        break
                    continue

                last_progress = time.time()
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "reasoning_delta":
                    result.reasoning_chunks += 1
                    deadline = max(deadline, time.time() + PROGRESS_EXTENSION)
                elif msg_type == "text_delta":
                    result.text_chunks += 1
                    full_text.append(data.get("text", ""))
                    deadline = max(deadline, time.time() + PROGRESS_EXTENSION)
                elif msg_type == "tool_call":
                    result.tool_calls += 1
                    deadline = max(deadline, time.time() + PROGRESS_EXTENSION)
                elif msg_type == "tool_result":
                    result.tool_results += 1
                    deadline = max(deadline, time.time() + PROGRESS_EXTENSION)
                elif msg_type in ("done", "message_complete"):
                    result.completed = True
                    break
                elif msg_type == "error":
                    result.errors.append(data.get("error", "unknown"))
                elif msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                elif msg_type == "plan_proposed":
                    plan_id = data.get("plan_id", "")
                    if plan_id:
                        await ws.send(json.dumps({
                            "type": "plan_confirm",
                            "plan_id": plan_id,
                            "approved": True,
                            "thread_id": thread_id,
                        }))
                elif msg_type == "approval_request":
                    req_id = data.get("id", data.get("request_id", ""))
                    if req_id:
                        await ws.send(json.dumps({
                            "type": "approval_response",
                            "request_id": req_id,
                            "approved": True,
                            "thread_id": thread_id,
                        }))
                elif msg_type == "clarification":
                    q_id = data.get("question_id", "")
                    await ws.send(json.dumps({
                        "type": "clarification_response",
                        "question_id": q_id,
                        "answer": "请使用你的最佳判断继续。",
                        "thread_id": thread_id,
                    }))
    except Exception as e:
        result.errors.append(f"{type(e).__name__}: {e}")

    result.elapsed_s = round(time.time() - start, 1)
    text = "".join(full_text)
    result.text_length = len(text)
    result.text_preview = text[:500] if text else ""
    return result


async def run_track(track, base_time):
    tr = TrackResult(track_id=track["id"], topic=track["topic"])
    track_start = time.time()

    for idx, (name, persona, prompt) in enumerate(track["stages"]):
        tid = f"{base_time}-{track['id']}-s{idx+1}"
        print(f"  [{track['id']}] Stage {idx+1}/4: {name}")
        sr = await run_stage(track["id"], name, persona, prompt, tid)
        tr.stages.append(sr)
        status = "OK" if sr.completed else "FAIL"
        print(f"  [{track['id']}] {name} {status}: {sr.elapsed_s}s, {sr.tool_calls} tools")
        if idx < len(track["stages"]) - 1:
            await asyncio.sleep(3)

    tr.total_elapsed_s = round(time.time() - track_start, 1)
    tr.completed = all(s.completed for s in tr.stages)
    return tr


async def cross_track_synthesis(tracks, base_time):
    print("\n  [synthesis] Starting cross-track synthesis")
    parts = []
    for tr in tracks:
        previews = [s.text_preview[:200] for s in tr.stages if s.text_preview]
        parts.append(f"## {tr.topic} ({tr.track_id})\n" + "\n".join(previews))
    context = "\n\n".join(parts)
    prompt = (
        f"以下是三个并行研究方向的关键发现：\n\n{context}\n\n"
        "请进行交叉分析：(1) 三个方向之间有什么共性科学问题？"
        "(2) 一个方向的发现能否启发另一个方向的解决方案？"
        "(3) 是否存在跨领域的方法论迁移机会？"
        "(4) 提出一个基于交叉分析的新研究设想。"
    )
    return await run_stage("synthesis", "交叉综合", "research", prompt, f"{base_time}-synthesis")


async def main():
    print("=" * 60)
    print("Async Multi-Track Research Pipeline")
    print(f"Tracks: {len(TRACKS)} | Stages/track: 4 | Timeout: {STAGE_TIMEOUT}s/stage")
    print("=" * 60)

    base_time = int(time.time())
    total_start = time.time()

    tasks = []
    for track in TRACKS:
        task = asyncio.create_task(run_track(track, base_time))
        tasks.append(task)
        await asyncio.sleep(2)  # stagger starts

    print(f"\n  {len(tasks)} tracks launched in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  [{TRACKS[i]['id']}] FATAL: {r}")
            clean.append(TrackResult(track_id=TRACKS[i]["id"], topic=TRACKS[i]["topic"]))
        else:
            clean.append(r)

    synthesis = None
    if any(t.stages for t in clean):
        synthesis = await cross_track_synthesis(clean, base_time)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    all_stages = completed_stages = 0
    for tr in clean:
        status = "DONE" if tr.completed else "PARTIAL"
        print(f"  {tr.topic} [{status}]")
        for s in tr.stages:
            all_stages += 1
            if s.completed:
                completed_stages += 1
            mark = "+" if s.completed else "-"
            print(f"    {mark} {s.stage_name}: {s.elapsed_s}s, {s.tool_calls} tools, {s.text_length} chars")

    if synthesis:
        print(f"  Synthesis: {synthesis.elapsed_s}s, {synthesis.text_length} chars")

    total_tools = sum(s.tool_calls for tr in clean for s in tr.stages)
    total_text = sum(s.text_length for tr in clean for s in tr.stages)
    total_errors = sum(len(s.errors) for tr in clean for s in tr.stages)
    print(f"\n  Totals: {completed_stages}/{all_stages} stages, {total_tools} tools, {total_text} chars, {total_errors} errors")
    seq = sum(tr.total_elapsed_s for tr in clean) / 60
    print(f"  Wall: {total_elapsed/60:.1f} min (vs {seq:.1f} min sequential)")

    report = {
        "total_duration_s": round(total_elapsed, 1),
        "total_duration_min": round(total_elapsed / 60, 1),
        "tracks_total": len(TRACKS),
        "stages_completed": completed_stages,
        "stages_total": all_stages,
        "total_tool_calls": total_tools,
        "total_text_chars": total_text,
        "total_errors": total_errors,
        "parallelism_factor": round(seq * 60 / max(total_elapsed, 1), 2),
        "tracks": [
            {
                "id": tr.track_id, "topic": tr.topic, "completed": tr.completed,
                "duration_s": tr.total_elapsed_s,
                "stages": [
                    {"name": s.stage_name, "elapsed_s": s.elapsed_s, "completed": s.completed,
                     "tool_calls": s.tool_calls, "text_length": s.text_length,
                     "errors": s.errors, "preview": s.text_preview}
                    for s in tr.stages
                ],
            }
            for tr in clean
        ],
        "synthesis": {
            "completed": synthesis.completed if synthesis else False,
            "elapsed_s": synthesis.elapsed_s if synthesis else 0,
            "text_length": synthesis.text_length if synthesis else 0,
            "preview": synthesis.text_preview if synthesis else "",
        } if synthesis else None,
    }

    out = os.path.join(os.path.dirname(__file__), "async_pipeline_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Results: {out}")


if __name__ == "__main__":
    asyncio.run(main())
