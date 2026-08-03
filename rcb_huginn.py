"""RCBench HuginnAgent 适配器.

直接调 HuginnAgent Python API, 绕过 RCBench 的 subprocess + shell 机制 (Windows 不兼容).
复用 RCBench TaskRunner 的 workspace setup, 跑完调 score.py 评分.

用法:
  python rcb_huginn.py --task Material_000
  python rcb_huginn.py --task Material_000 --score
  python rcb_huginn.py --task Material_000 --timeout 3600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# RCBench 路径
RCB_ROOT = Path(__file__).parent / "ResearchClawBench"
sys.path.insert(0, str(RCB_ROOT))

# Huginn 路径
AGENT_ROOT = Path(__file__).parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))

# 根目录 .env 是 agent 配置权威源 (HUGINN_MODEL/HUGINN_PROVIDER/DEEPSEEK_API_KEY).
# 系统/用户环境变量可能是过时值 (如 HUGINN_MODEL=deepseek-chat 已废弃),
# override=True 让 .env 覆盖过时系统变量. 用户临时测试改 shell env 不生效时,
# 改 .env 即可. ponytail: 不逐个 setdefault, 一次性 load 最简.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass  # python-dotenv 未装, 走系统环境变量 (CI/无依赖场景)

# RCBench 指令长 (7K+ chars), 默认 5000 token/s 限制会误杀, 必须在 import huginn 前设
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "50000")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")

# 关掉熔断器: code_tool 前几次试错失败后会被 circuit-open 锁死 60s,
# RCBench 是无人工场景, agent 没法等. code_tool 试错是正常科研流程.
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")

# 关掉 trajectory fork: RCBench 是复现任务, 多视角分叉不提升分数,
# 反而每 fork 跑完整 sub-agent (k=3 时 3x agent 调用), iter 1 光分叉就 22 分钟.
# ponytail: 复现任务确定性高, 单轨迹够用. ceiling: 创意探索任务可开.
os.environ.setdefault("HUGINN_RCB_FORK_ENABLED", "0")

# block subagent_tool: subagent 内部想调 execute 但被 SandboxBackendProtocol
# 拒绝, 转而嵌套调 sub-sub-agent 死等. Physics_000 iter 1 卡死根因.
# agent 改用 file_read_tool + write_file_tool 直接做 PDF 提取, 不损失能力.
# ponytail: 复现任务工具齐全, 不需要 subagent 并行. ceiling: 多视角探索任务可开.
os.environ.setdefault("HUGINN_RCB_BLOCKED_TOOLS", "subagent_tool")
# file_read_tool 默认限制在 cwd 下, 但 agent 可能传绝对路径读 data/
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
# code_tool/bash_tool 需要本地执行后端, 否则 get_executor() 直接拒绝
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")

# Huginn 内部组件 (audit/snapshots/reflections/logs/completions) 默认写
# ~/.huginn/, TRAE 沙箱拦截 → sqlite3/文件写入失败. 重定向到 workspace 内.
# ponytail: 每个组件单独改路径要改 10+ 处, 用 HUGINN_CACHE_DIR 一刀切.
# 升级路径: 给每个组件加 workspace 相对路径参数 (YAGNI, 当前一刀切够用).
os.environ.setdefault("HUGINN_CACHE_DIR", str(Path(__file__).parent / "ResearchClawBench" / "workspaces" / "_cache"))

# TRAE 沙箱拦截 ~/.huginn/ 和 AppData/ 写入. 禁用全局 stable_principles 继承,
# 只用 workspace 内的 .huginn/stable_principles.jsonl.
os.environ.setdefault("HUGINN_RCB_INHERIT_PRINCIPLES", "0")

# agent code_tool 里的 Python 可能用 tempfile.gettempdir() 写 C:\tmp\,
# TRAE 沙箱拦截. 重定向到 workspace 内的 tmp 目录.
_tmpdir = str(Path(__file__).parent / "ResearchClawBench" / "workspaces" / "_tmp")
Path(_tmpdir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TEMP", _tmpdir)
os.environ.setdefault("TMP", _tmpdir)

# arviz 默认写 ~/.arviz/ 或 AppData/Local/arviz/, TRAE 沙箱拦截 → 进程退出.
# 重定向到 workspace _cache 内. ponytail: arviz 无 disable-cache 选项, 只能改路径.
os.environ.setdefault("ARVIZ_CACHE_DIR", str(Path(_tmpdir) / "arviz"))

# torch/transformers 默认下载模型到 ~/.cache/torch 和 ~/.cache/huggingface,
# TRAE 沙箱拦截 → torch.save('C:\\tmp\\test.pt') 也挂. 重定向所有 ML 缓存路径
# 到 workspace _cache 内. ponytail: 一刀切所有 ML 库的缓存路径.
_ml_cache = str(Path(_tmpdir) / "ml_cache")
Path(_ml_cache).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", _ml_cache)
os.environ.setdefault("HF_HOME", _ml_cache)
os.environ.setdefault("XDG_CACHE_HOME", _ml_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", _ml_cache)
os.environ.setdefault("HF_HUB_CACHE", _ml_cache)

# onnxruntime 每次新建 session 都尝试 TensorRT/CUDA EP, 失败后 fallback CPU.
# 噪音刷爆日志掩盖 agent 实际输出, 还浪费时间. 更严重的是 paddlepaddle 加载
# cudnn_ops_infer64_8.dll 失败时触发栈缓冲区溢出 (exit 0xC0000409), agent 在
# verify figures 阶段调用 vision_describe/image_analysis_tool 即崩溃.
# GPU 策略: 检测 torch + cudnn 完整性, 可用则放开 CUDA_VISIBLE_DEVICES 让 torch
# 工具用 GPU; onnxruntime 仍强制 CPU EP (独立于 torch, vision 工具 CPU 够用).
def _detect_gpu_safe() -> bool:
    """检测 GPU 是否可用且 cudnn 不崩溃. 跟 rcb_runner.py:84 一致."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        x = torch.randn(8, 8, device="cuda")
        y = x @ x.T
        z = torch.nn.functional.conv2d(
            torch.randn(1, 1, 8, 8, device="cuda"),
            torch.randn(1, 1, 3, 3, device="cuda"),
        )
        del x, y, z
        return True
    except Exception:
        return False


if _detect_gpu_safe():
    print("[RCB] GPU detected and verified, enabling CUDA for torch tools", flush=True)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["HUGINN_TORCH_DEVICE"] = "cuda"
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["HUGINN_TORCH_DEVICE"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "4")  # CPU 推理限线程数避免吃满
os.environ.setdefault("ORT_DISABLE_TENSORRT_EP", "1")
os.environ.setdefault("ORT_DISABLE_CUDA_EP", "1")

# monkey-patch onnxruntime InferenceSession 强制 CPU-only providers, 避免每次
# session 创建都尝试加载 tensorrt_providers.dll (刷屏 + 浪费时间 + 偶发栈溢出).
# ponytail: onnxruntime 跟 torch 独立, 即使 GPU 可用也强制 CPU EP.
try:
    import onnxruntime as _ort
    _orig_sess_init = _ort.InferenceSession.__init__
    def _cpu_sess_init(self, *a, **kw):
        kw['providers'] = ['CPUExecutionProvider']
        _orig_sess_init(self, *a, **kw)
    _ort.InferenceSession.__init__ = _cpu_sess_init
except ImportError:
    pass

# paddlepaddle 装在 site-packages 但 huginn 不直接用. transformers/chroma 可能
# try import paddle 触发 cudnn_ops_infer64_8.dll 加载, 栈溢出崩溃 (0xC0000409).
# sys.modules 放 None 让 import paddle 抛 ImportError, transformers 走 torch.
# ponytail: 升级路径 pip uninstall paddlepaddle, 但可能影响其他任务环境.
sys.modules.setdefault("paddle", None)
sys.modules.setdefault("paddlepaddle", None)

# RestrictedPython 禁了 os/pathlib/open/pickle/eval — 科学计算全要用.
# 在 import huginn 前 monkey-patch validate_code 为空操作.
# ponytail: RCBench workspace 是隔离的临时目录, 风险可控. 升级: 加白名单而非全禁.
try:
    import huginn.security.restricted_python as _rp
    _rp.validate_code = lambda code: None  # type: ignore
except ImportError:
    pass  # security 模块不存在, 说明 RestrictedPython 已移除或重构, 不需要 patch

# RCBench 只需要这几个工具, 其他 80+ 工具只会分散注意力 + 占 context tokens
RCB_TOOL_FILTER = [
    "code_tool",             # Python 执行 — 数据分析、画图、建模
    "bash_tool",             # pip install 缺包
    "file_read_tool",        # 读 related_work/ 里的 PDF/文本
    "file_write_tool",       # 写 report.md
    "file_edit_tool",        # 改代码
    "glob",                  # 找文件
    "grep",                  # 搜内容
    "web_search_tool",       # 查常数/定义
    "subagent_tool",         # Layer 3: explore/coder/analyst 并行
    "plot_tool",             # 画图 (Arial 20pt+ 加粗)
    "image_analysis_tool",   # 反向 CV 分析自己生成的 PNG, 闭环视觉验证
    "vision_describe",       # 分层视觉描述 (OCR/CV), image criterion 闭环
]


class _AsyncTee:
    """异步 stdout tee: print() 入队即返回, 文件线程和控制台线程独立.

    PowerShell `| Tee-Object` 是同步管道, 控制台渲染慢会反压到 Python stdout,
    卡住 agent.chat 的 async generator. AsyncTee 用两个独立队列解耦:
    - write(): 同时入文件队列 + 控制台队列 (O(1), 非阻塞), 立即返回
    - 文件线程: 从文件队列取数据写文件 (落盘保证, 不受控制台阻塞影响)
    - 控制台线程: 从控制台队列取数据写原始 stdout (阻塞只阻塞自己)
    - 控制台队列满时丢弃旧数据 (控制台是 best-effort, 文件是 source of truth)
    """

    def __init__(self, log_path: str):
        import queue
        import threading
        self._file_q: queue.Queue = queue.Queue(maxsize=8192)
        self._console_q: queue.Queue = queue.Queue(maxsize=512)
        self._file_thread = threading.Thread(target=self._file_writer, daemon=True)
        self._console_thread = threading.Thread(target=self._console_writer, daemon=True)
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._original = sys.stdout
        self._closed = False
        self._queue = queue
        self._log_path = log_path

    def _file_writer(self):
        while True:
            item = self._file_q.get()
            if item is None:
                break
            try:
                self._file.write(item)
                self._file.flush()
            except Exception:
                pass
            self._file_q.task_done()

    def _console_writer(self):
        while True:
            item = self._console_q.get()
            if item is None:
                break
            try:
                self._original.write(item)
                self._original.flush()
            except Exception:
                pass
            self._console_q.task_done()

    def write(self, text: str):
        if self._closed:
            return
        # 文件队列: 优先保证落盘, timeout 30s
        try:
            self._file_q.put(text, timeout=30)
        except Exception:
            pass
        # 控制台队列: best-effort, 满了就丢 (不阻塞 agent)
        try:
            self._console_q.put_nowait(text)
        except Exception:
            pass

    def flush(self):
        try:
            self._file_q.join()
        except Exception:
            pass

    def install(self):
        sys.stdout = self
        self._file_thread.start()
        self._console_thread.start()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._file_q.put(None)
        self._console_q.put(None)
        self._file_thread.join(timeout=5)
        self._console_thread.join(timeout=3)
        self._file.close()
        sys.stdout = self._original


def setup_workspace(task_id: str) -> tuple[Path, str]:
    """复用 RCBench TaskRunner 建 workspace, 返回 (workspace_path, instructions)."""
    from evaluation.run_task import TaskRunner

    runner = TaskRunner(task_id, agent_cmd="huginn", agent_name="Huginn")
    runner.setup_workspace()
    instructions = runner.instructions_path.read_text(encoding="utf-8")
    return runner.workspace, instructions


def score_run(workspace: Path) -> dict:
    """调 RCBench score.py 对 workspace 评分."""
    # score.py 直接 os.environ.get 不 load .env, rcb_huginn 入口得自己 load
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / "ResearchClawBench" / "evaluation" / ".env")
    # .env 里 JUDGE_API_KEY 可能过期/无效, 优先用环境变量里的 DEEPSEEK_API_KEY
    # (shell 里 $env:DEEPSEEK_API_KEY 是用户当前有效 key)
    if os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["JUDGE_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
        os.environ.setdefault("JUDGE_API_BASE", "https://api.deepseek.com/v1")
        os.environ.setdefault("JUDGE_MODEL_NAME", "deepseek-v4-flash")
    # config.JUDGE_MODEL_NAME 是模块级变量, import 时就固定. 如果 rcb_runner
    # 之前间接 import 了 config (此时 env 还没 load), JUDGE_MODEL_NAME 就是空.
    # monkey-patch 回去, 不依赖 reload.
    import evaluation.config as _cfg
    _cfg.JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "deepseek-v4-flash")

    # image 项走 score.py 自带的 3 档 fallback:
    #   档 1 vision LLM → v4-flash 不支持 image_url 抛异常被吞
    #   档 2 CV 算子 (SSIM/HOG/histogram) + text LLM 评分 ← 落到这里
    #   档 3 LLM 也挂 → 纯 CV 分
    # 之前的 monkey-patch 把 image 项强改 type=text, CV 信号 100% 丢失,
    # 与 system prompt 的 FIGURE SELF-VERIFICATION 自相矛盾. 删 patch 让
    # score.py 走自己的 fallback 链, CV 信号保留.
    # 升级路径: 换 vision-capable judge (gpt-4o/claude) 改 .env JUDGE_MODEL_NAME.
    from evaluation.score import score_workspace
    return score_workspace(workspace)


def main():
    parser = argparse.ArgumentParser(description="Run HuginnAgent on RCBench task")
    parser.add_argument("--task", required=True, help="RCBench task id (e.g. Material_000)")
    parser.add_argument("--score", action="store_true", help="Score after run")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout in seconds (default 7200 = 2h)")
    parser.add_argument("--max-tool-calls", type=int, default=100, help="Max tool calls (legacy path only, default 100)")
    parser.add_argument(
        "--extreme", action="store_true",
        help="v6 极限模式: thinking=high, max_tool_calls=300, context_budget=200K, "
             "autoloop 50/50/20/15. 异步委派 700 万步能力需要此 flag.",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="异步 tee 日志文件. print() 入队即返回, 后台线程写文件+控制台, "
             "避免 PowerShell 管道反压阻塞 async event loop.",
    )
    args = parser.parse_args()

    # 异步 stdout tee: 解耦 print() 和管道写入.
    # PowerShell `| Tee-Object` 是同步管道, 控制台渲染慢会反压到 Python stdout,
    # 卡住 agent.chat 的 async generator. AsyncTee 让 print 入队即返回,
    # 后台线程负责落盘 + 控制台, 控制台阻塞只阻塞后台线程.
    _tee = _AsyncTee(args.log_file) if args.log_file else None
    if _tee:
        _tee.install()

    print(f"[RCB] Task: {args.task}")
    workspace, instructions = setup_workspace(args.task)
    print(f"[RCB] Workspace: {workspace}")
    print(f"[RCB] Instructions: {len(instructions)} chars")

    # 在 workspace 目录下执行, 让 code_tool/bash_tool 的 cwd 指向 workspace
    os.chdir(workspace)

    start = time.time()
    # 方案 A: Time-Aware Agent — 把 deadline 暴露给 pre_model_hook, agent
    # 看到 remaining_time < 300s 就该停训写 report. ponytail: env var 传值,
    # 不改 rcb_runner / agent __init__ 签名. 升级: langgraph config 传.
    os.environ["HUGINN_RCB_DEADLINE"] = str(start + args.timeout)
    os.environ["HUGINN_RCB_TIMEOUT"] = str(args.timeout)
    # 极限模式 env — rcb_runner.run() 读此 flag 激活 300 max_tool_calls +
    # 200K context + thinking=high + 异步委派 DAG. 不开则 150 max_tool_calls.
    if args.extreme:
        os.environ["HUGINN_EXTREME_DISPATCH"] = "1"
        # P5: persistent goal mode — stagnation 不 early stop, 跑满 timeout.
        # rcb_runner.run() 会创建 active goal + wall_clock_budget_seconds,
        # 主循环每轮查 wall_clock_expired 守卫 break.
        os.environ.setdefault("HUGINN_PERSISTENT_GOAL_MODE", "1")
        # 跨任务 curiosity hint: 跨任务共享 db 后, self_model 积累历史弱 persona,
        # 注入 iter prompt. 单任务首次跑 self_model 空, 不触发.
        # skill_abstraction/self_goal_synthesis 在 rcb_runner 里没有读取代码,
        # 设了无效, 删掉避免误导.
        os.environ.setdefault("HUGINN_CURIOSITY_HINT", "1")
    # 跨任务 memory db 重定向 — TRAE 沙箱拦截 ~/.huginn/ 写入.
    # 落到 ResearchClawBench/workspaces/_cross_task/, 跨任务共享, 沙箱可写.
    # rcb_runner 读此变量决定 memory_dir, 默认开 (HUGINN_RCB_CROSS_TASK=1).
    os.environ.setdefault(
        "HUGINN_RCB_CROSS_TASK_DIR",
        str(Path(__file__).parent / "ResearchClawBench" / "workspaces" / "_cross_task"),
    )
    _mode_tag = "EXTREME" if args.extreme else "normal"
    print(f"[RCB] Starting agent (mode={_mode_tag}, timeout={args.timeout}s)")
    # 默认走 rcb_runner.run() (extreme 模式). legacy run_agent 已删除.
    try:
        from huginn.cli.rcb_runner import run as _rcb_run
        rc = asyncio.run(asyncio.wait_for(_rcb_run(str(workspace), extreme=args.extreme), timeout=args.timeout))
        final = "" if rc == 0 else f"rcb_runner.run exited rc={rc}"
    except asyncio.TimeoutError:
        final = f"[TIMEOUT after {args.timeout}s]"
        # 第三层反射防御: wait_for 取消 run() task, run() 内的 _stream_chat
        # finally 块虽会执行, 但 CancelledError 传播链上 streaming.py 的
        # async generator finally 不保证跑. 在入口拿全局 agent 再调一次,
        # 确保 evolution 把最后的 tool 失败写进 tool_calls.jsonl.
        try:
            from huginn.cli.rcb_runner import _last_agent_for_reflection as _agent_ref
            if _agent_ref is not None and hasattr(_agent_ref, "_run_post_turn_reflection"):
                _agent_ref._run_post_turn_reflection()
                print("[RCB-ENTRY] post-timeout reflection done", flush=True)
        except Exception as _re:
            print(f"[RCB-ENTRY] post-timeout reflection error: {_re}", flush=True)
    except Exception as _rcb_e:
        import traceback as _tb
        _tb_str = _tb.format_exc()
        final = f"[RCB ERROR: {type(_rcb_e).__name__}: {_rcb_e}]\n{_tb_str[-500:]}"
        print(final, file=sys.stderr, flush=True)
    finally:
        # meta.json 必须在 finally 写: asyncio.run 清理时可能挂起 (task 内
        # ainvoke 无法立即取消), 外部 kill 后 L420 没机会执行.
        elapsed = round(time.time() - start)
        report_path = workspace / "report" / "report.md"
        report_exists = report_path.exists()
        print(f"[RCB] Done in {elapsed}s. Report exists: {report_exists}", flush=True)
        if report_exists:
            print(f"[RCB] Report size: {report_path.stat().st_size} bytes", flush=True)
        meta = {
            "task_id": args.task,
            "agent_name": "Huginn",
            "duration_seconds": elapsed,
            "report_exists": report_exists,
            "final_output_preview": (final[:500] if final else ""),
            "agent_model": os.environ.get("HUGINN_MODEL", "unknown"),
            "agent_provider": os.environ.get("HUGINN_PROVIDER", "default"),
            "judge_model": os.environ.get("JUDGE_MODEL_NAME", "deepseek-chat"),
        }
        try:
            (workspace / "_huginn_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False)
            )
        except Exception:
            pass

    if args.score and report_exists:
        print("[RCB] Scoring...")
        try:
            result = score_run(workspace)
            if "error" in result:
                print(f"[RCB] Score error: {result['error']}")
            else:
                print(f"[RCB] Score: {result.get('total_score', 0)}/100")
                for item in result.get("items", []):
                    print(f"  [{item['type']}] w={item['weight']} score={item['score']} :: {item['content'][:80]}")
        except Exception as exc:
            print(f"[RCB] Scoring failed: {exc}")

    if _tee:
        _tee.close()
    return 0 if report_exists else 1


if __name__ == "__main__":
    sys.exit(main())
