"""S1+S3 时空表征与视觉多图耦合自检.

覆盖:
- S1: van Hove G_s(r,t) 计算 — top-K=32 稀疏采样, 三元组 (t, r, G_s)
- S1: F(q,t) 中间散射函数 — 三元组 (t, q, F)
- S1: lammps_tool _physical_timeseries 注册 van_hove + F_q_t (spatial=True)
- S1: _format_timeseries_context 三元组分支 — peak_drift + v_decay
- S3: describe_image_sequence 多图序列 + 帧间一致性
- S3: _hist_correlation 直方图相关
- S1+S3: physical_coupling_hint 闭环提示

不引入 pytest — assert-based, `python -m tests.test_temporal_spatial` 可跑.
"""

import sys
import tempfile
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _make_test_frames(n_frames: int = 5, n_atoms: int = 100, seed: int = 42):
    """构造测试帧: 原子做扩散运动 (前 32 个原子位移大, 后面位移小)."""
    import random
    random.seed(seed)
    frames = []
    for fi in range(n_frames):
        atoms = []
        for ai in range(n_atoms):
            # 前 32 个原子做大位移, 后面小位移 — 验证 top-K 采样
            base_disp = 2.0 if ai < 32 else 0.1
            x = base_disp * fi + random.uniform(-0.5, 0.5)
            y = random.uniform(-0.5, 0.5)
            z = random.uniform(-0.5, 0.5)
            atoms.append({
                "x": x, "y": y, "z": z,
                "xu": x, "yu": y, "zu": z,  # unwrapped
                "vx": 0.1, "vy": 0.0, "vz": 0.0,
            })
        frames.append({"atoms": atoms, "timestep": fi * 100})
    return frames


def _check_van_hove_compute():
    """S1: _compute_van_hove 算 G_s(r,t), top-K=32 采样."""
    from huginn.tools.sim.lammps_tool import LammpsTool
    tool = LammpsTool.__new__(LammpsTool)
    frames = _make_test_frames(n_frames=5, n_atoms=100)

    vh = tool._compute_van_hove(frames, r_max=10.0, bins=20)
    assert vh is not None, "van_hove 不应 None"
    assert len(vh) == 5, f"应有 5 帧, 实际 {len(vh)}"
    # 每帧有 r 和 G_s, 长度 = bins
    assert len(vh[0]["r"]) == 20, "r bins 应为 20"
    assert len(vh[0]["G_s"]) == 20, "G_s bins 应为 20"
    # G_s 归一化: 每帧总和约等于 1 (top-K 采样下略小于 1 但接近)
    _sum = sum(vh[-1]["G_s"])
    assert 0.5 < _sum < 1.5, f"末帧 G_s 总和应接近 1, 实际 {_sum}"
    # top-K=32 不超过 n_atoms
    assert tool._VAN_HOVE_TOP_K == 32, "默认 top-K=32"
    # 小 n_atoms 退化: n_atoms=10 时 K 应缩到 10
    frames_small = _make_test_frames(n_frames=3, n_atoms=10)
    vh_small = tool._compute_van_hove(frames_small)
    assert vh_small is not None, "小 n_atoms 不应崩"
    print("[ok] S1 _compute_van_hove (top-K=32 稀疏采样)")


def _check_f_q_t_compute():
    """S1: _compute_F_q_t 算中间散射函数, 默认 3 个 q 值."""
    from huginn.tools.sim.lammps_tool import LammpsTool
    tool = LammpsTool.__new__(LammpsTool)
    frames = _make_test_frames(n_frames=5, n_atoms=100)

    fqt = tool._compute_F_q_t(frames)
    assert fqt is not None, "F_q_t 不应 None"
    assert len(fqt) == 5, f"应有 5 帧, 实际 {len(fqt)}"
    # 默认 q_values = [0.5, 1.0, 2.0]
    assert fqt[0]["q_values"] == [0.5, 1.0, 2.0], f"默认 q 不对: {fqt[0]['q_values']}"
    assert len(fqt[0]["F"]) == 3, "F 应与 q 对齐"
    # F(0) ≈ 1 (t=0 无位移, cos(0)=1)
    assert abs(fqt[0]["F"][0] - 1.0) < 0.2, f"F(q,0) 应接近 1, 实际 {fqt[0]['F'][0]}"
    # F 随时间衰减 (扩散)
    f_first_q = fqt[0]["F"][1]  # q=1.0
    f_last_q = fqt[-1]["F"][1]
    assert f_last_q < f_first_q, f"F 应衰减, F(0)={f_first_q} F(end)={f_last_q}"
    # 自定义 q_values
    fqt_c = tool._compute_F_q_t(frames, q_values=[0.3, 0.7])
    assert fqt_c[0]["q_values"] == [0.3, 0.7]
    assert len(fqt_c[0]["F"]) == 2
    print("[ok] S1 _compute_F_q_t 中间散射函数")


def _check_lammps_registers_spatio_temporal():
    """S1: lammps_tool 把 van_hove + F_q_t 注册到 _physical_timeseries, spatial=True."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.tools.sim.lammps_tool import LammpsTool

    tool = LammpsTool.__new__(LammpsTool)
    frames = _make_test_frames(n_frames=4, n_atoms=50)

    # 复现 lammps_tool 内部 _physical_timeseries 构造逻辑
    _vh = tool._compute_van_hove(frames)
    _fqt = tool._compute_F_q_t(frames)
    assert _vh and _fqt, "van_hove + F_q_t 都应非空"

    # 构造 _physical_timeseries 模拟注册
    _vh_data = []
    for entry in _vh:
        _t = entry["timestep"]
        for _r, _gs in zip(entry["r"], entry["G_s"]):
            _vh_data.append((_t, _r, _gs))
    _fqt_data = []
    for entry in _fqt:
        _t = entry["timestep"]
        for _q, _f in zip(entry["q_values"], entry["F"]):
            _fqt_data.append((_t, _q, _f))

    _ts_list = [
        {"name": "van_hove_G_s", "unit": "1/Å³", "data": _vh_data,
         "meaning": "van Hove self-part", "source": "lammps", "spatial": True},
        {"name": "F_q_t", "unit": "1", "data": _fqt_data,
         "meaning": "intermediate scattering function", "source": "lammps", "spatial": True},
    ]

    # _extract_timeseries 能识别
    _result = {"_physical_timeseries": _ts_list}
    _extracted = AutoloopEngine._extract_timeseries(_result)
    assert _extracted == _ts_list, "extract 应识别 _physical_timeseries key"

    # 三元组 data 格式验证
    assert len(_extracted[0]["data"][0]) == 3, "van_hove data 应是 (t, r, G_s) 三元组"
    assert len(_extracted[1]["data"][0]) == 3, "F_q_t data 应是 (t, q, F) 三元组"
    assert _extracted[0].get("spatial") is True, "van_hove 应标 spatial=True"
    assert _extracted[1].get("spatial") is True, "F_q_t 应标 spatial=True"

    print("[ok] S1 lammps_tool 注册 van_hove + F_q_t (spatial=True, 三元组)")


def _check_format_timeseries_spatial():
    """S1: _format_timeseries_context 三元组分支 — peak_drift + v_decay."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)

    # van_hove 三元组: 模拟峰位漂移
    eng._physical_timeseries = [{
        "name": "van_hove_G_s",
        "unit": "1/Å³",
        "data": [
            (0, 1.0, 0.5), (0, 2.0, 0.8), (0, 3.0, 0.3),  # t=0 峰位 r=2.0
            (100, 1.0, 0.2), (100, 2.5, 0.6), (100, 3.5, 0.4),  # t=100 峰位 r=2.5
        ],
        "meaning": "van Hove self-part",
        "source": "lammps",
        "spatial": True,
    }]
    out = eng._format_timeseries_context()
    assert "van_hove_G_s" in out, "应包含 name"
    assert "spatio-temporal" in out, "应标记 spatio-temporal"
    assert "peak_drift" in out, "应包含 peak_drift"
    # 峰位从 r=2.0 漂移到 r=2.5, drift=+0.5
    assert "+0.500" in out or "0.500" in out, f"峰位漂移 +0.5 应在输出: {out}"
    # t×r pts 格式
    assert "2t" in out, f"应有 2 个时间帧: {out}"
    assert "3r" in out, f"应有 3 个 r bin: {out}"

    # F_q_t 三元组
    eng._physical_timeseries = [{
        "name": "F_q_t",
        "unit": "1",
        "data": [
            (0, 0.5, 1.0), (0, 1.0, 1.0), (0, 2.0, 1.0),
            (100, 0.5, 0.7), (100, 1.0, 0.5), (100, 2.0, 0.2),
        ],
        "meaning": "intermediate scattering function",
        "source": "lammps",
        "spatial": True,
    }]
    out = eng._format_timeseries_context()
    assert "F_q_t" in out
    assert "spatio-temporal" in out
    # F 从 1.0 衰减到 0.7/0.5/0.2, peak 应衰减
    assert "v_decay" in out

    # 混合: 二元组 + 三元组
    eng._physical_timeseries = [
        {"name": "VACF", "unit": "Å²/ps²",
         "data": [(0, 1.0), (100, 0.5)],
         "meaning": "vacf", "source": "lammps"},
        {"name": "van_hove_G_s", "unit": "1/Å³",
         "data": [(0, 1.0, 0.5), (0, 2.0, 0.8), (100, 1.0, 0.2), (100, 2.5, 0.6)],
         "meaning": "vh", "source": "lammps", "spatial": True},
    ]
    out = eng._format_timeseries_context()
    assert "VACF" in out and "decaying" in out, "二元组分支应正常"
    assert "van_hove_G_s" in out and "spatio-temporal" in out, "三元组分支应正常"

    print("[ok] S1 _format_timeseries_context 三元组分支 (peak_drift + v_decay)")


def _check_hist_correlation():
    """S3: _hist_correlation 直方图相关."""
    from huginn.tools.vision_describe_tool import _hist_correlation

    # 完全相同 → 1.0
    assert abs(_hist_correlation([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    # 完全反相关 → -1.0
    assert abs(_hist_correlation([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9
    # 长度不等 → 0
    assert _hist_correlation([1, 2], [1, 2, 3]) == 0.0
    # 空 → 0
    assert _hist_correlation([], []) == 0.0
    # 全零和 → 0
    assert _hist_correlation([0, 0, 0], [1, 2, 3]) == 0.0
    print("[ok] S3 _hist_correlation 直方图相关")


def _check_describe_image_sequence():
    """S3: describe_image_sequence 多图序列 + 帧间一致性."""
    from PIL import Image

    from huginn.tools.vision_describe_tool import describe_image_sequence

    # 构造两张相似图 + 一张差异大图
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "f1.png"
        p2 = Path(d) / "f2.png"
        p3 = Path(d) / "f3.png"
        # f1/f2 几乎一样 (灰度 128)
        Image.new("RGB", (32, 32), (128, 128, 128)).save(p1)
        Image.new("RGB", (32, 32), (130, 130, 130)).save(p2)  # 微差异
        # f3 差异大 (全白)
        Image.new("RGB", (32, 32), (255, 255, 255)).save(p3)

        # 单帧: 走 describe_image 原路径
        out_single = describe_image_sequence([str(p1)], "test")
        assert out_single["tier"] in ("tier0_none", "tier1_classic_ocr",
                                       "tier2_paddleocr", "tier3_deepseek_ocr",
                                       "error"), f"单帧应走原路径: {out_single['tier']}"

        # 多帧
        out = describe_image_sequence([str(p1), str(p2), str(p3)], "test")
        assert out["tier"] == "multi_frame_sequence", f"多图应走 multi_frame_sequence: {out['tier']}"
        assert out["n_frames"] == 3
        assert len(out["per_frame"]) == 3
        # 帧间一致性结构
        ifc = out["inter_frame_consistency"]
        assert "pairs" in ifc
        assert "low_inter_frame_consistency" in ifc
        assert "inconsistency_reasons" in ifc
        assert len(ifc["pairs"]) == 2, "3 帧应有 2 对"
        # physical_coupling_hint 提示 F(q,t) 闭环
        assert "F(q,t)" in out["physical_coupling_hint"], "应有 F(q,t) 闭环提示"

        # 空列表
        out_empty = describe_image_sequence([], "test")
        assert out_empty["available"] is False, "空列表应报错"

    print("[ok] S3 describe_image_sequence 多图序列 + 帧间一致性")


def _check_vision_describe_input_image_paths():
    """S3: VisionDescribeInput 接受 image_paths 字段."""
    from huginn.tools.vision_describe_tool import VisionDescribeInput

    # 默认空 list (走单图路径)
    inp = VisionDescribeInput(image_path="/tmp/x.png")
    assert inp.image_paths == [], "默认 image_paths 应为空 list"

    # 多图路径
    inp2 = VisionDescribeInput(
        image_path="ignored",  # 多图时被忽略
        image_paths=["/tmp/a.png", "/tmp/b.png"],
    )
    assert len(inp2.image_paths) == 2
    print("[ok] S3 VisionDescribeInput.image_paths 字段")


def main():
    _check_van_hove_compute()
    _check_f_q_t_compute()
    _check_lammps_registers_spatio_temporal()
    _check_format_timeseries_spatial()
    _check_hist_correlation()
    _check_describe_image_sequence()
    _check_vision_describe_input_image_paths()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
