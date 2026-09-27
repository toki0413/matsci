import asyncio, os, sys
sys.path.insert(0, "/workspace/agent")

from huginn.llm import get_model
from huginn.research.code_lab import build_author_prompt, extract_code, sandbox_run

OBJ = open("/workspace/research_outputs/shusheng_rsi_run6/objective.txt", encoding="utf-8").read()

async def main():
    model = get_model()
    code = ""
    last = ""
    for attempt in range(3):
        prompt = build_author_prompt(OBJ, repair_hint=last)
        resp = await model.ainvoke(prompt)
        text = str(getattr(resp, "content", resp))
        code = extract_code(text)
        print(f"--- attempt {attempt}: code_len={len(code)}")
        if not code:
            last = "未产出可解析代码"; continue
        res, reason = sandbox_run(code, {"seed": 0},
                                  timeout=float(os.environ.get("HUGINN_CODELAB_TIMEOUT_S", "600")))
        if res is not None:
            print("SUCCESS objectives=", res.get("objectives"))
            print("summary=", res.get("summary"))
            return
        last = reason or "unknown"
        print("   fail:", reason)

asyncio.run(main())