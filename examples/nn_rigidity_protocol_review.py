"""A · 用书生 Intern-S2 把 NN 刚性探针协议评审写成正式文档.

输入: 协议(TraeWork 共享文档)要点 + B 轮三条实测发现。
输出: examples/out/protocol_review/ nn-rigidity-probe-protocol-review.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except Exception:
    sys.exit("需要 pip install openai")

_PATH = Path(__file__).resolve()
OUT = _PATH.parent / "out" / "protocol_review"
_OUT_MD = OUT / "nn-rigidity-probe-protocol-review.md"


_PROTOCOL_TLDR = """
[待评审协议] 数值协议草案 0.1: 以神经网络容量为探针测量 bootstrap 解空间刚性
核心转译: 把"bootstrap 约束是否唯一固定解"转译为"须多大 NN 容量 C* 才能在跨结构
留出约束上达到零违规"。刚性端点=弦振幅(Cheung-Remmen-Sciotti-Tarquini 唯一性定理),
胖流形端点=已知族维数的势散射族。旋钮: Kpol(极点谱) / Kzero(零点自由度) /
Ksoft(高能软性)。测量量: 临界容量 C*(ε;N)、重训散布 σH、约束谱有效秩 rankeff。
协议判读(§4饱和检验): 刚性端 C* 随样本量 N 饱和; 胖端持续增长/发散。§5重训散布:
刚性 σH 在 p>C* 塌缩; 可数歧义族 σH 停在 O(1)。§6 守卫: C1 平滑替身 / C2 伪结构 /
C3 优化器审计 / C4 精度表示。
"""

_B_EMPIRICS = """
【B 轮实测发现——评审者须纳入】:
(1) N-判据歧义: 对单目标函数(解析 cos vs 非光滑 |x|)做偶多项式探针, 扫描样本量 N:
    C* 对 N 两者都走平(128→8192 不变), 无法单靠"N 饱和"区分刚性与胖目标。
    原因: 对固定精度截止 ε, 可截断的目标函数 C* 本就对 N 不敏感。
(2) 修正判据: 固定 N、扫精度截止 ε∈{1e-2..1e-12}: 解析目标 C*(ε) 收敛到有限平台;
    非光滑目标 C*(ε) 单调冲顶发散。即"饱和"应读 C*(ε;N) 对 ε→0 的极限,
    而非对 N 的依赖。协议 §4 写 N 会漏掉这一层。
(3) 真实 NN(协议原样宽度扫描 w∈{8,16,32,64,128}, 3层 Tanh, PINN u''=-cos(x)+点值约束):
    原样超参(Adam, 2500 步)下刚性端也不饱和 Vtr≈5e-2, 各宽度 Vho≈0.147 恒定。
    这正是协议 §4/§6 C3/C4 守卫要拦截的"优化器财政→管道失真"——若不先修优化管道,
    会把"优化不足"误读为"解族太胖"(假阳性刚性的反面即假胖). 实证了"先修管道再谈物理"。
"""

_PROMPT = f"""你是科学方法评审与数值协议审计者。请对下面这份"NN 容量探针测量 bootstrap
解空间刚性"的协议草案做批判性评审, 并纳入我方实测证据, 输出一份**正式技术评审文档**.
写作语言: 中文。结构:

## 0. 执行摘要(3-5 条结论)
## 1. 协议目标与方法转译是否自洽
## 2. 判读逻辑的漏洞与修正建议(必须处理"饱和判据应看 ε 还是 N"这一实证问题)
## 3. 旋钮标尺与守卫设计的合理性(Kpol/Kzero/Ksoft; C1-C4)
## 4. 本评审给出的可执行验证清单(优先序; 含如何用 ε-截止 + C3 优化器审计跑通 P0)
## 5. 开放问题与风险

{_PROTOCOL_TLDR}
{_B_EMPIRICS}
请给出有洞察、可操作、克制的判断; 不确定处明确标注为推断, 不要编造数值定理。
"""


def main() -> int:
    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 未设置 INTERNLM_API_KEY", file=sys.stderr)
        return 2
    model = os.environ.get("INTERNLM_MODEL", "intern-s2-preview")
    base = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
    client = OpenAI(api_key=key, base_url=base)

    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _PROMPT}],
        max_tokens=3000,
        temperature=0.3,
        extra_body={"thinking_mode": False},
    )
    text = r.choices[0].message.content or ""

    OUT.mkdir(parents=True, exist_ok=True)
    _OUT_MD.write_text(
        f"# 协议评审: NN 容量探针测 bootstrap 解空间刚性\n\n"
        f"> 生成: 书生 `{model}` × Huginn 本地实测(B 轮) · {_OUT_MD.stem}\n\n"
        + text.strip() + "\n",
        encoding="utf-8",
    )
    print("== 评审文档已写入 ==")
    print(_OUT_MD.resolve())
    print("\n---- 正文(前 1500 字) ----\n")
    print(text[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())