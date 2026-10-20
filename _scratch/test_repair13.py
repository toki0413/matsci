import asyncio, os, sys
sys.path.insert(0, "/workspace/agent")

from huginn.llm import get_model
from huginn.research.code_lab import build_author_prompt, extract_code, sandbox_run

OBJ = open("/workspace/research_outputs/shusheng_rsi_run13/objective.txt", encoding="utf-8").read()
MAX_REPAIRS = int(os.environ.get("HUGINN_CODELAB_REPAIR_ATTEMPTS", "2"))

async def gen(model, goal, repair_hint=""):
    prompt = build_author_prompt(goal, repair_hint=repair_hint)
    resp = await model.ainvoke(prompt) if hasattr(model, "ainvoke") else model.invoke(prompt)
    text = str(getattr(resp, "content", resp))
    return extract_code(text)

async def main():
    model = get_model()
    code = await gen(model, OBJ)
    print("=== initial code len", len(code), "===")
    last_err = ""
    for attempt in range(MAX_REPAIRS + 1):
        res, reason = sandbox_run(code, {"seed": 0},
                                  timeout=float(os.environ.get("HUGINN_CODELAB_TIMEOUT_S", "600")))
        print(f"--- attempt {attempt}: reason={reason!r} ok={res is not None}")
        if res is not None:
            print("SUCCESS objectives:", str(res.get("objectives"))[:400])
            print("SUCCESS summary:", str(res.get("summary"))[:400])
            return
        last_err = reason
        if attempt < MAX_REPAIRS:
            code = await gen(model, OBJ, repair_hint=last_err)
            print(f"--- repaired code len {len(code)}")

asyncio.run(main())