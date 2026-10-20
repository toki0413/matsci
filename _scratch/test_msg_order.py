"""验证: 消息顺序 (Human/System 交错) 是否触发 internlm 'in prompt processing error'."""
import os
import sys

sys.path.insert(0, "/workspace/agent")

os.environ.setdefault("HUGGIN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")


def probe(model, label, msgs):
    try:
        r = model.invoke(msgs)
        print(f"[{label}] {r.content!r}")
    except Exception as exc:
        print(f"[{label}] EXC {type(exc).__name__}: {exc}")


def main() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from huginn.server_core import get_agent_factory

    factory = get_agent_factory()
    model = factory.model_registry.resolve(factory.model_registry.default_alias())

    S1 = SystemMessage(content="## 通信风格定制\n- 语言: 英文为主\n- 格式: markdown")
    H0 = HumanMessage(content="Say exactly: HELLO_RIGIDITY. Nothing else.")
    H2 = HumanMessage(content="[Current inner state] You feel slightly down. ...\n---\n\nSay exactly: HELLO_RIGIDITY. Nothing else.")
    S3 = SystemMessage(content="## Tool Call Budget\n- This turn: 0/15 tool calls used/allowed.")

    probe(model, "clean [S,H]", [SystemMessage(content="You are concise."), H0])
    probe(model, "agent order [H,S,H,S]", [H0, S1, H2, S3])
    probe(model, "harmonized [S,H,H]", [S1, H0, H2])
    probe(model, "double system [S,S,H]", [S1, S3, H2])
    probe(model, "three system [S,S,S,H]", [S1, S3, SystemMessage(content="x"), H2])
    probe(model, "with prior AI [H,S,A,H,S]", [H0, S1, AIMessage(content="ok"), H2, S3])


if __name__ == "__main__":
    main()