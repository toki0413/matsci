import asyncio, os, sys
sys.path.insert(0, "/workspace/agent")

from huginn.llm import get_model
from huginn.research.code_lab import build_author_prompt, extract_code, sandbox_run

OBJ = open("/workspace/research_outputs/shusheng_rsi_run13/objective.txt", encoding="utf-8").read()

async def main():
    print("provider=", os.environ.get("HUGINN_PROVIDER"), "model=", os.environ.get("HUGINN_MODEL"),
          "key?", bool(os.environ.get("INTERNLM_API_KEY")))
    model = get_model()
    prompt = build_author_prompt(OBJ)
    print("=== prompt len", len(prompt), "===")
    resp = await model.ainvoke(prompt) if hasattr(model, "ainvoke") else model.invoke(prompt)
    text = str(getattr(resp, "content", resp))
    print("=== raw len", len(text), "===")
    code = extract_code(text)
    print("=== extracted code len", len(code), "===")
    print(code[:3000])
    if not code:
        print("!! no code extracted")
        return
    res, reason = sandbox_run(code, {"seed": 0}, timeout=float(os.environ.get("HUGINN_CODELAB_TIMEOUT_S", "600")))
    print("=== sandbox reason:", reason)
    print("=== sandbox res:", res)

asyncio.run(main())