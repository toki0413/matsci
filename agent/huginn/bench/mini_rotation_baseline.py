"""Mini rotation baseline — cognitive map 收益验证 (不依赖 atomworld 包).

用 StructureCognitiveMap 生成几个结构 + 旋转任务, 对比 flag on/off 下 agent
表现. atomworld 包没装时用这个快速验证 cognitive map 路线是否有效.

跑法:
    # flag off baseline (text-centric, 无 cognitive map 工具)
    $env:HUGINN_USE_COGNITIVE_MAP=0
    python -m huginn.bench.mini_rotation_baseline --n-questions 5

    # flag on (CodeAct 注入 cognitive map 工具)
    $env:HUGINN_USE_COGNITIVE_MAP=1
    python -m huginn.bench.mini_rotation_baseline --n-questions 5

输出: per-question correct/total + 汇总 success_rate.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# 必须在 import huginn 之前设, 否则 ResearchLog singleton 用 ~/.huginn
# ponytail: sandbox 不让写 ~/.huginn db-wal, 重定向到 tempdir
_CACHE_DIR = Path(tempfile.mkdtemp(prefix="mini_rot_huginn_"))
os.environ["HUGINN_CACHE_DIR"] = str(_CACHE_DIR)

import numpy as np

logger = logging.getLogger(__name__)

# 几个简短测试结构: NaCl / Si (diamond) / 单原子. 都是 mock CIF 不依赖 atomworld.
# ponytail: 3 个结构够看趋势, 不追求统计严格. 升级路径接 atomworld AtomMotor-2K.
_MOCK_CIFS = {
    "NaCl": """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
""",
    "Si": """data_Si
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.0 0.0 0.0
Si2 Si 0.25 0.25 0.25
""",
    "C_diamond": """data_C
_cell_length_a 3.57
_cell_length_b 3.57
_cell_length_c 3.57
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
C2 C 0.25 0.25 0.25
""",
    # 难题结构: 多原子 (5-10), 放在 10x10x10 box 里, 非整数旋转让 text-centric 心算失败
    "CH4": """data_CH4
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C0 C 0.5 0.5 0.5
H1 H 0.553 0.553 0.553
H2 H 0.553 0.447 0.447
H3 H 0.447 0.553 0.447
H4 H 0.447 0.447 0.553
""",
    "C8_cube": """data_C8
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C0 C 0.4 0.4 0.4
C1 C 0.6 0.4 0.4
C2 C 0.4 0.6 0.4
C3 C 0.6 0.6 0.4
C4 C 0.4 0.4 0.6
C5 C 0.6 0.4 0.6
C6 C 0.4 0.6 0.6
C7 C 0.6 0.6 0.6
""",
    "water_dimer": """data_H4O2
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.4 0.5 0.5
H1a H 0.45 0.55 0.5
H1b H 0.45 0.45 0.5
O2 O 0.6 0.5 0.5
H2a H 0.65 0.55 0.5
H2b H 0.65 0.45 0.5
""",
    "C10": """data_C10
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C0 C 0.3 0.3 0.3
C1 C 0.7 0.3 0.3
C2 C 0.3 0.7 0.3
C3 C 0.7 0.7 0.3
C4 C 0.3 0.3 0.7
C5 C 0.7 0.3 0.7
C6 C 0.3 0.7 0.7
C7 C 0.7 0.7 0.7
C8 C 0.5 0.5 0.5
C9 C 0.5 0.5 0.7
""",
}


# 5 个 rotation 问题. agent 要回答旋转后的坐标 / 距离是否变化.
# ponytail: 都是 verifiable (有 ground truth), 不引入 LLM judge.
_QUESTIONS = [
    {
        "id": "q1_distance_invariance",
        "structure": "NaCl",
        "question": (
            "Given this CIF, atom Na is at (0,0,0) and Cl at (0.5,0.5,0.5) in fractional coords. "
            "If we rotate the entire crystal 90 degrees around the z-axis, "
            "what is the new distance between Na and Cl? "
            "Answer with a single number in Angstroms, e.g. '4.88'."
        ),
        "verify": lambda ans: _check_distance_invariance(ans, "NaCl"),
    },
    {
        "id": "q2_rotated_coords",
        "structure": "Si",
        "question": (
            "Given Si CIF with Si1 at (0,0,0) and Si2 at (0.25,0.25,0.25) fractional. "
            "After rotating 180 degrees around the z-axis, "
            "what are the new fractional coordinates of Si2? "
            "Answer with 3 numbers separated by spaces, e.g. '0.25 0.25 0.25'."
        ),
        "verify": lambda ans: _check_rotated_coords(ans, "Si", axis="z", angle=180, target_idx=1),
    },
    {
        "id": "q3_distance_after_rot",
        "structure": "C_diamond",
        "question": (
            "Given diamond C CIF with C1 at (0,0,0) and C2 at (0.25,0.25,0.25) fractional. "
            "After rotating 90 degrees around x-axis, "
            "what is the new distance between the two C atoms in Angstroms? "
            "Answer with a single number."
        ),
        "verify": lambda ans: _check_distance_invariance(ans, "C_diamond"),
    },
    {
        "id": "q4_angle_invariance",
        "structure": "NaCl",
        "question": (
            "Given NaCl CIF, Na at (0,0,0), Cl at (0.5,0.5,0.5). "
            "Imagine a third atom at (1,0,0). "
            "After rotating the entire crystal 45 degrees around y-axis, "
            "what is the angle Na-Cl-ThirdAtom in degrees? "
            "Answer with a single number."
        ),
        # 三点角度 SE(3) 等变: 旋转前后角度不变
        "verify": lambda ans: _check_angle_invariance(ans, "NaCl"),
    },
    {
        "id": "q5_supercell_distance",
        "structure": "Si",
        "question": (
            "Given Si CIF (a=5.43A), if we make a 2x2x2 supercell, "
            "what is the distance between the original Si1 at (0,0,0) and "
            "the nearest periodic image of Si2 (originally at 0.25,0.25,0.25)? "
            "Answer with a single number in Angstroms."
        ),
        # 2x2x2 supercell: Si2 在新 cell 的 (0.125, 0.125, 0.125) frac, 距离不变
        "verify": lambda ans: _check_supercell_distance(ans, "Si"),
    },
]


def _check_distance_invariance(answer: str, struct_name: str) -> bool:
    """距离 SE(3) 等变: 旋转前后距离不变."""
    try:
        predicted = float(answer.strip().split()[0])
    except (ValueError, IndexError):
        return False
    from huginn.metacog.structure_cognitive_map import StructureCognitiveMap
    m = StructureCognitiveMap.from_cif(_MOCK_CIFS[struct_name])
    expected = m.query_distance(0, 1)
    return abs(predicted - expected) < 0.5  # 0.5Å 容差


def _check_rotated_coords(answer: str, struct_name: str, axis: str,
                           angle: float, target_idx: int) -> bool:
    """检查旋转后坐标."""
    try:
        nums = [float(x) for x in answer.strip().split()]
        if len(nums) != 3:
            return False
    except ValueError:
        return False
    from huginn.metacog.structure_cognitive_map import StructureCognitiveMap
    m = StructureCognitiveMap.from_cif(_MOCK_CIFS[struct_name])
    rotated = m.query_after_rotation([target_idx], axis=axis, angle=angle, degrees=True)
    expected = rotated[0]
    # 比较 frac coords (需要转 cart → frac)
    if m.lattice is not None:
        inv = np.linalg.inv(m.lattice)
        cart = np.array(expected)
        frac = cart @ inv
        # wrap to [0, 1)
        frac = frac % 1.0
        return all(abs(frac[i] - nums[i]) < 0.05 for i in range(3))
    return False


def _check_angle_invariance(answer: str, struct_name: str) -> bool:
    """角度 SE(3) 等变."""
    try:
        predicted = float(answer.strip().split()[0])
    except (ValueError, IndexError):
        return False
    # 原始角度 (Na-Cl-ThirdAtom at (1,0,0) frac)
    from huginn.metacog.structure_cognitive_map import StructureCognitiveMap
    m = StructureCognitiveMap.from_cif(_MOCK_CIFS[struct_name])
    # 加一个虚拟第三原子
    inv = np.linalg.inv(m.lattice)
    third_cart = np.array([1.0, 0.0, 0.0]) @ m.lattice
    m2 = m.add_atom("X", third_cart)
    expected = m2.query_angle(0, 1, 2)  # Na-Cl-X
    return abs(predicted - expected) < 10.0  # 10度容差


def _check_supercell_distance(answer: str, struct_name: str) -> bool:
    """2x2x2 supercell 后最近周期镜像距离."""
    try:
        predicted = float(answer.strip().split()[0])
    except (ValueError, IndexError):
        return False
    from huginn.metacog.structure_cognitive_map import StructureCognitiveMap
    m = StructureCognitiveMap.from_cif(_MOCK_CIFS[struct_name])
    # 原 cell 中 Si2 距离 Si1
    d_orig = m.query_distance(0, 1)
    # 2x2x2 supercell 不改变最近邻距离
    return abs(predicted - d_orig) < 0.5


# ── 难题 verify: 非整数旋转后坐标 / 距离 ─────────────────────────

def _hard_map(struct_name: str):
    """用 from_coords 构造 hard structure, atom index 确定.
    ponytail: 不用 from_cif 避免 pymatgen atom 重排导致 index 错位.
    """
    from huginn.metacog.structure_cognitive_map import StructureCognitiveMap
    import numpy as np
    lat10 = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    if struct_name == "CH4":
        return StructureCognitiveMap.from_coords(
            ["C", "H", "H", "H", "H"],
            np.array([[5, 5, 5], [5.53, 5.53, 5.53], [5.53, 4.47, 4.47],
                      [4.47, 5.53, 4.47], [4.47, 4.47, 5.53]]),
            lat10,
        )
    if struct_name == "C8_cube":
        v = [(0.4, 0.4, 0.4), (0.6, 0.4, 0.4), (0.4, 0.6, 0.4), (0.6, 0.6, 0.4),
             (0.4, 0.4, 0.6), (0.6, 0.4, 0.6), (0.4, 0.6, 0.6), (0.6, 0.6, 0.6)]
        return StructureCognitiveMap.from_coords(
            ["C"] * 8,
            np.array([[x * 10 for x in c] for c in v]),
            lat10,
        )
    if struct_name == "water_dimer":
        return StructureCognitiveMap.from_coords(
            ["O", "H", "H", "O", "H", "H"],
            np.array([[4, 5, 5], [4.5, 5.5, 5], [4.5, 4.5, 5],
                      [6, 5, 5], [6.5, 5.5, 5], [6.5, 4.5, 5]]),
            lat10,
        )
    if struct_name == "C10":
        v = [(0.3, 0.3, 0.3), (0.7, 0.3, 0.3), (0.3, 0.7, 0.3), (0.7, 0.7, 0.3),
             (0.3, 0.3, 0.7), (0.7, 0.3, 0.7), (0.3, 0.7, 0.7), (0.7, 0.7, 0.7),
             (0.5, 0.5, 0.5), (0.5, 0.5, 0.7)]
        return StructureCognitiveMap.from_coords(
            ["C"] * 10,
            np.array([[x * 10 for x in c] for c in v]),
            lat10,
        )
    raise KeyError(f"unknown hard structure: {struct_name}")


def _check_rotated_cart_coords(answer: str, struct_name: str, atom_idx: int,
                                axis: str, angle: float) -> bool:
    """验证旋转后原子 cart 坐标. 容差 0.3Å (非整数旋转容差大)."""
    try:
        nums = [float(x) for x in answer.strip().split()]
        if len(nums) != 3:
            return False
    except ValueError:
        return False
    m = _hard_map(struct_name)
    expected = m.query_after_rotation([atom_idx], axis=axis, angle=angle, degrees=True)[0]
    return all(abs(nums[i] - expected[i]) < 0.3 for i in range(3))


def _check_multistep_distance(answer: str, struct_name: str, axis: str,
                                angle: float, translation: tuple[float, float, float],
                                atom_i: int, atom_j: int) -> bool:
    """多步: 旋转 → 平移 → 查距离. 距离不受平移影响 (SE(3) 等变)."""
    try:
        predicted = float(answer.strip().split()[0])
    except (ValueError, IndexError):
        return False
    m = _hard_map(struct_name)
    m_rot = m.rotate(axis=axis, angle=angle, degrees=True)
    m_trans = m_rot.translate(translation)
    expected = m_trans.query_distance(atom_i, atom_j)
    return abs(predicted - expected) < 0.3


# 5 道难题: 多原子 (5-10) + 非整数旋转 (30/45/60/37°) + 多步变换.
# text-centric 心算 45° 旋转矩阵容易错, cognitive map 直接 query_after_rotation.
_HARD_QUESTIONS = [
    {
        "id": "h1_ch4_rot45_coords",
        "structure": "CH4",
        "question": (
            "Given this CIF (5 atoms: C at center, 4 H in tetrahedron), "
            "rotate the entire structure 45 degrees around the z-axis. "
            "What are the new Cartesian coordinates of atom H1 (originally at frac 0.553,0.553,0.553)? "
            "Answer with 3 numbers in Angstroms separated by spaces, e.g. '5.53 5.53 5.53'."
        ),
        "verify": lambda ans: _check_rotated_cart_coords(ans, "CH4", 1, "z", 45),
    },
    {
        "id": "h2_c8_rot30_coords",
        "structure": "C8_cube",
        "question": (
            "Given this CIF (8 C atoms at cube vertices, a=10A), "
            "rotate the entire structure 30 degrees around the y-axis. "
            "What are the new Cartesian coordinates of atom C0 (originally at frac 0.4,0.4,0.4)? "
            "Answer with 3 numbers in Angstroms separated by spaces."
        ),
        "verify": lambda ans: _check_rotated_cart_coords(ans, "C8_cube", 0, "y", 30),
    },
    {
        "id": "h3_water_rot60_coords",
        "structure": "water_dimer",
        "question": (
            "Given this CIF (6 atoms: 2 water molecules, O1/O2 + 4H), "
            "rotate the entire structure 60 degrees around the x-axis. "
            "What are the new Cartesian coordinates of atom O2 (atom index 3, originally at frac 0.6,0.5,0.5)? "
            "Answer with 3 numbers in Angstroms separated by spaces."
        ),
        "verify": lambda ans: _check_rotated_cart_coords(ans, "water_dimer", 3, "x", 60),
    },
    {
        "id": "h4_c10_rot37_coords",
        "structure": "C10",
        "question": (
            "Given this CIF (10 C atoms), "
            "rotate the entire structure 37 degrees around the z-axis. "
            "What are the new Cartesian coordinates of atom C8 (index 8, originally at frac 0.5,0.5,0.5)? "
            "Answer with 3 numbers in Angstroms separated by spaces."
        ),
        "verify": lambda ans: _check_rotated_cart_coords(ans, "C10", 8, "z", 37),
    },
    {
        "id": "h5_ch4_multistep",
        "structure": "CH4",
        "question": (
            "Given this CIF (5 atoms: CH4), perform these operations in order: "
            "1) rotate 45 degrees around z-axis, "
            "2) translate by (1.0, 2.0, 3.0) Angstroms. "
            "After both operations, what is the distance between atom C0 (index 0) and H4 (index 4)? "
            "Answer with a single number in Angstroms."
        ),
        "verify": lambda ans: _check_multistep_distance(
            ans, "CH4", "z", 45, (1.0, 2.0, 3.0), 0, 4
        ),
    },
]


async def _run_agent(question: str, cif_str: str) -> str:
    """用 HuginnAgent (CodeAct + DeepSeek) 跑一道题."""
    from huginn.config import HuginnConfig
    from huginn.models.registry import ModelRegistry
    from huginn.agent import Agent
    from huginn.memory.manager import MemoryManager
    from huginn.memory.longterm import LongTermMemory

    hcfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(hcfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    elif hcfg.provider and hcfg.provider != "default":
        model = registry.resolve(f"{hcfg.provider}/{hcfg.model or 'auto'}")
    else:
        raise RuntimeError("No model configured. Set HUGINN_PROVIDER and HUGINN_API_KEY.")

    mm = MemoryManager(longterm=LongTermMemory(db_path=_CACHE_DIR / "memory.db"))
    # ponytail: kb_enabled=False 避免 ChromaDB rust backend segfault (sandbox 不能写)
    agent = Agent(model=model, memory_manager=mm, kb_enabled=False)
    agent.register_tools_from_registry()

    prompt = f"""You are given a crystal structure CIF and a 3D reasoning question.

CIF:
```
{cif_str}
```

Question: {question}

Think step by step. If you have access to cognitive_map_* tools, use them to query
3D coordinates and verify your reasoning. Otherwise reason from the CIF text.

End your response with a line that starts with 'ANSWER: ' followed by just the
numeric answer (or 3 numbers separated by spaces for coordinates).
"""
    final = ""
    try:
        async def _collect() -> None:
            nonlocal final
            async for chunk in agent.chat(prompt):
                msgs = chunk.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    content = getattr(last, "content", "")
                    if content:
                        final = str(content)
        await asyncio.wait_for(_collect(), timeout=120)
    except Exception as e:
        logger.warning("agent.chat failed: %s", e)
        return ""
    return final


def _extract_answer(agent_response: str) -> str:
    """从 agent 响应里提取 'ANSWER: xxx'."""
    if not agent_response:
        return ""
    for line in agent_response.splitlines():
        line = line.strip()
        if line.startswith("ANSWER:"):
            return line[len("ANSWER:"):].strip()
    # 没找到 ANSWER 标记, 取最后一行非空
    lines = [l.strip() for l in agent_response.splitlines() if l.strip()]
    return lines[-1] if lines else ""


async def run_baseline(n_questions: int = 5, hard: bool = False) -> dict[str, Any]:
    """跑 mini rotation baseline."""
    flag_state = os.environ.get("HUGINN_USE_COGNITIVE_MAP", "0")
    mode = "HARD" if hard else "EASY"
    print(f"\n=== Mini Rotation Baseline ({mode}) ===")
    print(f"HUGINN_USE_COGNITIVE_MAP = {flag_state}")
    if hard:
        print(f"n_questions = {len(_HARD_QUESTIONS)} (hard set)")
        questions = _HARD_QUESTIONS
    else:
        print(f"n_questions = {n_questions}")
        questions = _QUESTIONS[:n_questions]
    print()

    results: list[dict[str, Any]] = []

    for q in questions:
        cif = _MOCK_CIFS[q["structure"]]
        print(f"[{q['id']}] structure={q['structure']}")
        try:
            response = await _run_agent(q["question"], cif)
            answer = _extract_answer(response)
            correct = q["verify"](answer)
            print(f"  agent_answer: {answer!r}")
            print(f"  correct: {correct}")
            print()
            results.append({
                "id": q["id"], "structure": q["structure"],
                "answer": answer, "correct": correct,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": q["id"], "structure": q["structure"],
                            "answer": "", "correct": False, "error": str(e)})

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    rate = n_correct / n if n > 0 else 0.0
    print(f"=== Summary ({mode}) ===")
    print(f"HUGINN_USE_COGNITIVE_MAP={flag_state}")
    print(f"correct: {n_correct}/{n} ({rate:.1%})")
    for r in results:
        status = "OK" if r.get("correct") else "FAIL"
        print(f"  [{status}] {r['id']}: ans={r.get('answer','')!r}")
    return {"flag": flag_state, "mode": mode, "results": results,
            "n_correct": n_correct, "n_total": n, "success_rate": rate}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-questions", type=int, default=5,
                   help="number of easy questions (1-5)")
    p.add_argument("--hard", action="store_true",
                   help="run hard questions (5 multi-atom non-integer rotation)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()
    result = asyncio.run(run_baseline(args.n_questions, hard=args.hard))
    sys.exit(0 if result["n_correct"] > 0 else 1)
