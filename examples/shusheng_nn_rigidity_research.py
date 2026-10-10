"""书生 × Huginn · NN 泛化行为作为 bootstrap 解空间刚性探针的可行性研究.

两轮:
  1. 主研究: 把"唯一性是否成立"转译为"容量-零违规"可测量量的严格性评估;
  2. 红队复核: 对抗性审稿, 攻击等价性假设并给最小决定性实验.

输入: 用户原始问题 + 本地已有实测 (B 轮三项发现).
输出: research_outputs/shusheng_nn_rigidity_probe/research_report.md

用法:
    export INTERNLM_API_KEY=<书生 token>
    python examples/shusheng_nn_rigidity_research.py
可用 INTERNLM_MODEL / INTERNLM_BASE_URL 覆盖模型与端点 (默认 intern-s2).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except Exception:  # noqa: BLE001
    sys.exit("需要 pip install openai")

_OUT = (
    Path(__file__).resolve().parents[1]
    / "research_outputs"
    / "shusheng_nn_rigidity_probe"
)
_OUT_MD = _OUT / "research_report.md"

_QUESTION = """
[待研究问题] 神经网络的"泛化行为"能否作为 bootstrap 解空间刚性的探针?
具体提法: 若解空间是一维的 (如 Veneziano 情形, 唯一性定理成立), 那么一个很小的模型见过
有限样本后, 应该能泛化到没见过的输入 —— 因为解空间太小, 模型实质上是在"记住"这个一维族.
若解空间是高维或连续的 (如隧穿振幅), 同样的小模型应表现出泛化失败 —— 约束不足, 数据流形
太"胖", 小容量模型装不下. 于是问: "唯一性是否成立"这个纯数学问题, 是否可以变成"多大容量的
模型能在留出集上零违规"这个可测量的量?
"""

_LOCAL_EMPIRICS = """
【本地已有实测——必须纳入, 不得与之矛盾】
(1) N-判据歧义: 对单目标函数 (解析 cos vs 非光滑 |x|) 做偶多项式探针并扫样本量 N,
    C* 对 N 两者都走平 (128→8192 不变), 单靠"N 饱和"无法区分刚性与胖目标.
(2) 修正判据: 固定 N、扫精度截止 ε∈{1e-2..1e-12}, 解析目标 C*(ε) 收敛到有限平台,
    非光滑目标 C*(ε) 单调冲顶发散. 即"饱和"应读作 C*(ε;N) 对 ε→0 的极限.
(3) 真实 NN (宽度 w∈{8,16,32,64,128}, 3 层 Tanh, PINN u''=-cos(x) 加点值约束):
    原样超参 (Adam, 2500 步) 下刚性端也不饱和 (V_tr≈5e-2, 各宽度 V_ho≈0.147 恒定).
    即"优化不足"会伪装成"解族太胖"(假胖)——先修优化管道再谈物理.
"""

_STAGE1 = f"""你是理论物理与机器学习交叉方向的研究分析者, 负责一份正式可行性研究.
{_QUESTION}
{_LOCAL_EMPIRICS}

请给出**严格、克制、可操作**的分析. 语言: 中文. 必须按以下结构作答:

## 1. 命题精化 (把直觉拆成可证伪命题)
先指出"小模型能泛化 ⇒ 解空间小"这一推理的逻辑缺口: 至少区分
(a) 解空间(可行性集)自身的维度; (b) 模型类的归纳偏置/表达能力; (c) 优化可达性;
(d) 插值/记忆 ≠ 学到正确解. 说明哪些是外生混淆, 哪些是问题本身.

## 2. 正确的判别量 (哪个量才是"解空间维度"的代理)
逐一评估三个候选观测量并排序, 说明各自测的是什么、在什么条件下失效:
(a) 临界容量对该判据的饱和结构 C*(ε,N); (b) 固定容量下留出约束违规 V_ho 随约束样本数 N
的衰减律; (c) 可行性集上约束泛函 Jacobian/Hessian 的有效秩 rank_eff (或零模个数).
明确指出: "最小容量能零违规"与"唯一性"之间是等式、单向不等式, 还是仅在特定极限下同构.

## 3. 两端点的可证伪预测 (Veneziano 唯一 vs 隧穿/连续族)
分别给出 V_ho(C,N) 与 C*(ε,N) 的定性预期, 并写成"若观察到 X 则支持刚性探针, 若观察到 Y 则
证伪"的形式. 若你认为不可分, 明确说不分, 不要硬编差别.

## 4. 会使结论翻转的混淆与反例 (至少 5 条)
每条给"机制 + 如何检测/扣除". 至少覆盖: 架构表达瓶颈造成的"假刚性"、参数化冗余/规范自由度
造成的"假胖"、大容量下的良性过拟合 (零训练违规却解错)、有限 ε 与浮点精度、约束采样偏差、
正则化/隐式偏置对解的选择.

## 5. 最小决定性实验设计
给具体观测量、扫描轴、判据阈值、以及"实验失败即证伪"的明确条件. 优先给能在一台机器上跑完的
设计, 并说明它如何与已有的 (1)(2)(3) 三项实测衔接.

## 6. 文献锚点
只列你有把握的; 对不确定的条目标注"需核验". **禁止编造**标题/作者/年份/结论.
特别处理: Veneziano 唯一性定理的成立条件, 以及"隧穿振幅"解空间是正维还是仅参数未定的确切表述.

## 7. 结论
回答标题问题: 唯一性能否(在什么意义上)变成"容量-零违规"这个可测量量. 给一句话结论 + 一句话
最强反方理由.

红线: 不确定处标注为推断; 不编造数值/定理; 不堆砌术语掩盖不确定性.
"""

def _stage2_prompt(stage1: str) -> str:
    # 用拼接而非 str.format: 提示词正文含 {1e-2..1e-12} 等花括号, format 会误当占位符.
    return f"""你是对抗性审稿人 (red team), 任务是**攻击**下面这份可行性研究, 不是复述它.
{_LOCAL_EMPIRICS}

请按以下结构作答 (中文):
## A. 致命问题 (按严重度排序)
逐一指出被评审文本中错误的等价性假设、循环论证、把数值伪影当物理结论之处.
## B. 更简单的替代解释
对每一个"支持刚性探针"的观测, 给出一个**更平凡**的假设 (如: 模型类太窄/优化器不到位/
约束采样不足) 也能解释同一观测, 并说明如何判决二者.
## C. 可构造的反例
至少给 2 个**具体的小系统** (可写成解析式或极简数值设置), 使得"唯一性 ⇔ 容量-零违规"的
对应**必然失效**; 说明构造机制. 一个偏向"唯一但探针报胖", 一个偏向"非唯一但探针报刚".
## D. 唯一最小决定性实验
若只允许做一次实验, 做哪一个最能区分"刚性探针成立"与"模型归纳偏置假象"? 给出可执行的设置与
判据. 要具体到可观测量与阈值.
## E. 对原研究结论的裁决
明确给: 接受 / 有条件接受(条件是什么) / 驳回, 并用一句话说明.

评审对象如下:
---
{stage1}
---
"""


def _compat_kwargs(client: OpenAI) -> dict:
    """端点感知: 仅 Intern/书生端点注入 thinking_mode 字段, 通用端点省略以免 400."""
    base = str(getattr(client, "base_url", None) or "")
    if "intern-ai.org.cn" in base or "/intern" in base:
        return {"extra_body": {"thinking_mode": False}}
    return {}


def _ask(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
        timeout=600,
        **_compat_kwargs(client),
    )
    return (r.choices[0].message.content or "").strip()


def main() -> int:
    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 未设置 INTERNLM_API_KEY", file=sys.stderr)
        return 2
    model = os.environ.get("INTERNLM_MODEL", "intern-s2")
    base = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
    client = OpenAI(api_key=key, base_url=base)

    print(f"== 书生 {model} @ {base} ==")
    print("[1/2] 主研究分析 ...")
    stage1 = _ask(client, model, _STAGE1, max_tokens=6000)
    print(f"      -> {len(stage1)} 字")

    print("[2/2] 红队复核 ...")
    try:
        stage2 = _ask(client, model, _stage2_prompt(stage1), max_tokens=4500)
    except Exception as exc:  # noqa: BLE001
        # 第二轮失败不丢第一轮: 落盘时标注缺失, 便于事后单独补跑.
        print(f"      !! 第二轮失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        stage2 = f"(红队复核未完成: {type(exc).__name__})"
    print(f"      -> {len(stage2)} 字")

    _OUT.mkdir(parents=True, exist_ok=True)
    _OUT_MD.write_text(
        "# NN 泛化行为作为 bootstrap 解空间刚性探针的可行性研究\n\n"
        f"> 生成: 书生 `{model}` × Huginn (两轮: 主分析 + 红队复核)\n"
        f"> 问题: 唯一性是否可转译为「多大容量模型能在留出集上零违规」\n\n"
        "---\n\n"
        "## 第一轮 · 主研究分析\n\n"
        f"{stage1}\n\n"
        "---\n\n"
        "## 第二轮 · 红队复核\n\n"
        f"{stage2}\n",
        encoding="utf-8",
    )
    print(f"\n== 报告已写入 ==\n{_OUT_MD.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())