"""v18 Bandit-Based Effort Controller — advisory effort allocation.

tabular UCB1 bandit, 双时间尺度 reward (Δoutputs_progress + Δdarwin_score).
跨任务持久化 Q table. 所有异常 fallback 到 "continue" 不阻断主流程.

ponytail: 不引依赖, 不做 function approximation. 天花板 = 4 维 state space
稀疏, 升级路径 = tile coding 或网络逼近. 跨任务持久化让稀疏问题部分缓解.

参考: MemCon (arXiv:2607.13591) — memory ops MDP + UCB + 跨任务学习.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ACTIONS = ("continue", "switch", "requery")
_C = 1.0
_ALPHA = 1.0
_BETA = 0.5
_GAMMA = 0.9  # MC discounted return 折扣 — 与 autoloop/bandit.py 的 0.99 同源,
# 取 0.9 更强调近期 reward, 信用分配更短视更稳 (短 episode 不衰减到 0).
_PERSIST_FLUSH_EVERY = 10


def _bucket(value: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


@dataclass(frozen=True)
class _BanditState:
    item_idx: int
    time_bucket: int
    calls_bucket: int
    progress_bucket: int

    def key(self) -> str:
        return f"{self.item_idx}|{self.time_bucket}|{self.calls_bucket}|{self.progress_bucket}"


@dataclass
class _ItemRuntime:
    item_idx: int
    start_ts: float
    tool_calls: int = 0
    last_progress_pct: float = 0.0
    last_advice: str = "continue"
    same_advice_streak: int = 0


class EffortBandit:
    """单例. thread-safe. advisory only — 所有公开方法 catch Exception."""

    _instance: "EffortBandit | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, persist_path: Path | None = None):
        self._lock = threading.RLock()
        self._Q: dict[str, dict[str, float]] = {}
        self._N: dict[str, dict[str, int]] = {}
        # DeLM shared verified context: {item_pattern: {action: {"avg": float, "n": int}}}
        # cross-task prior, bandit cold-start 时用. ponytail: 不引DB, JSON持久化够用.
        self._verified_lessons: dict[str, dict[str, dict]] = {}
        self._persist_path = persist_path
        self._update_count = 0
        self._runtime: _ItemRuntime | None = None
        self._items_count = 0
        self._items_names: list[str] = []
        self._items_labels: list[str] = []  # [EXACT]/[VARIANT]/... per item
        self._last_darwin_score = 0.5
        # MDP 升级: episode 轨迹缓冲 + 起点 darwin (终点奖励的参照).
        # HUGINN_BANDIT_MDP=0 时回退旧单步增量更新, 零行为变化 (回归逃生门).
        self._mdp_enabled = os.environ.get("HUGINN_BANDIT_MDP", "1") != "0"
        self._trajectory: list[tuple[str, str, float]] = []
        self._episode_start_darwin = 0.5
        self._load()

    @classmethod
    def get_instance(cls) -> "EffortBandit":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = EffortBandit(cls._default_persist_path())
        return cls._instance

    @staticmethod
    def _default_persist_path() -> Path | None:
        # env var 优先, 否则用文件位置定位项目根.
        # ponytail: 不能用 cwd — agent subprocess 的 cwd 是 workspace 子目录,
        # cwd-relative 找不到 _cross_task/, _save() 静默跳过, Q table 丢不学.
        try:
            _env = os.environ.get("HUGINN_BANDIT_Q_PATH")
            if _env:
                _p = Path(_env)
                _p.parent.mkdir(parents=True, exist_ok=True)
                return _p
            # bandit_controller.py 在 agent/huginn/agent/, parents[3] = 项目根
            # __file__ 在 staticmethod 里走 module globals, 不会是 caller 的 globals.
            import huginn.agent.bandit_controller as _self_mod
            _root = Path(_self_mod.__file__).resolve().parents[3]
            _cand = _root / "ResearchClawBench" / "workspaces" / "_cross_task" / "bandit_q.json"
            _cand.parent.mkdir(parents=True, exist_ok=True)
            return _cand
        except Exception as _e:
            logger.warning("[v19] _default_persist_path failed: %s", _e)
            return None

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            _data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._Q = _data.get("Q", {})
            self._N = _data.get("N", {})
            # 向后兼容: 旧 v18 文件没有 verified_lessons, 默认空 dict.
            self._verified_lessons = _data.get("verified_lessons", {})
            logger.info("[v19] bandit loaded %d states, %d patterns",
                        len(self._Q), len(self._verified_lessons))
        except Exception as _e:
            logger.warning("[v19] bandit load failed: %s", _e)

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            _data = {"version": 2, "C": _C, "alpha": _ALPHA, "beta": _BETA,
                     "Q": self._Q, "N": self._N,
                     "verified_lessons": self._verified_lessons}
            _tmp = self._persist_path.with_suffix(".tmp")
            _tmp.write_text(json.dumps(_data), encoding="utf-8")
            _tmp.replace(self._persist_path)
        except Exception as _e:
            logger.warning("[v19] bandit save failed: %s", _e)

    def set_items(self, items: list[Any]) -> None:
        with self._lock:
            self._items_count = len(items)
            self._items_names = [getattr(i, "name", f"item{i}") for i in items]
            # DeLM: item pattern 来自 checklist label, 用于 verified_lessons 聚合.
            # 无 label 的 item 归到 "UNLABELED" bucket, 不影响主流程.
            self._items_labels = [getattr(i, "label", "") or "UNLABELED" for i in items]
            self._runtime = None

    def _ensure_runtime(self, item_idx: int = 0) -> _ItemRuntime:
        if self._runtime is None or self._runtime.item_idx != item_idx:
            self._runtime = _ItemRuntime(item_idx=item_idx, start_ts=time.time())
        return self._runtime

    def _build_state(self, rt: _ItemRuntime, progress_pct: float) -> _BanditState:
        _elapsed = time.time() - rt.start_ts
        return _BanditState(
            rt.item_idx,
            _bucket(_elapsed, [30, 120, 300]),
            _bucket(float(rt.tool_calls), [5, 15, 30]),
            4 if progress_pct >= 100 else _bucket(progress_pct, [1, 25, 75, 100]),
        )

    def _scan_outputs_progress(self, item_idx: int) -> float:
        try:
            _cwd = Path.cwd()
            _out = _cwd / "outputs"
            if not _out.exists():
                return self._proxy_progress(item_idx)
            _status = _out / f"item_{item_idx}_status.md"
            if _status.exists():
                _txt = _status.read_text(encoding="utf-8", errors="ignore")[:2000]
                import re as _re
                _m = _re.search(r"progress[:\s]+(\d+)%", _txt, _re.IGNORECASE)
                if _m:
                    return float(_m.group(1))
                if "complete" in _txt.lower():
                    return 100.0
            _n_files = sum(1 for _ in _out.glob("*") if _.is_file() and _.stat().st_size > 100)
            _file_prog = min(100.0, (_n_files / max(self._items_count, 1) / 3) * 100)
            if _file_prog > 0:
                return _file_prog
            # agent 不写 outputs/item_N_status.md 时, 用 tool_calls 作 proxy.
            # ponytail: 单调递增, 不准但比恒 0 强, reward_slow (darwin) 兜底.
            # 天花板: agent 调工具不代表有产出, 升级路径 = 扫 chat 输出关键词.
            return self._proxy_progress(item_idx)
        except Exception:
            return self._runtime.last_progress_pct if self._runtime else 0.0

    def _proxy_progress(self, item_idx: int) -> float:
        """tool_calls based progress proxy — agent 不写 status 文件时的 fallback."""
        if self._runtime is None or self._runtime.item_idx != item_idx:
            return self._runtime.last_progress_pct if self._runtime else 0.0
        _calls = self._runtime.tool_calls
        if _calls <= 0:
            return 0.0
        # 分桶: 1-5→10%, 6-15→25%, 16-30→50%, 31+→75%
        # 不给 100% — 100% 只由 status 文件 "complete" 触发, 避免假完成.
        if _calls <= 5:
            return 10.0
        if _calls <= 15:
            return 25.0
        if _calls <= 30:
            return 50.0
        return 75.0

    def _policy_locked(self, _prog: float | None = None) -> str:
        """UCB1 策略 — 调用者已持锁. _prog 传入时复用, 省一次文件系统扫描."""
        if self._runtime is None or self._items_count == 0:
            return "continue"
        rt = self._runtime
        if rt.tool_calls < 10:
            return "continue"
        if _prog is None:
            _prog = self._scan_outputs_progress(rt.item_idx)
        if _prog >= 100.0:
            return "continue"
        st = self._build_state(rt, _prog)
        k = st.key()
        if k not in self._Q:
            return self._prior_from_lessons(rt.item_idx)
        N_s = sum(self._N[k].values())
        if N_s == 0:
            return self._prior_from_lessons(rt.item_idx)
        best_a, best_ucb = "continue", -float("inf")
        for a in _ACTIONS:
            N_sa = self._N[k].get(a, 0)
            if N_sa == 0:
                return a
            _ucb = self._Q[k][a] + _C * math.sqrt(math.log(N_s) / N_sa)
            if _ucb > best_ucb:
                best_ucb, best_a = _ucb, a
        if best_a == rt.last_advice and rt.same_advice_streak >= 3:
            return "continue"
        return best_a

    def policy(self) -> str:
        try:
            with self._lock:
                return self._policy_locked()
        except Exception as _e:
            logger.debug("[v19] policy fallback: %s", _e)
            return "continue"

    def _prior_from_lessons(self, item_idx: int) -> str:
        """DeLM shared verified context: 同 item_pattern 的 cross-task best action.

        ponytail: 强信号 (n>=3 且 avg>0.05) 才信, 否则默认 continue 让 UCB1 explore.
        天花板: pattern 聚合丢失 state 细节, 升级路径 = tile coding + pattern 混合.
        """
        if item_idx >= len(self._items_labels):
            return "continue"
        _pattern = self._items_labels[item_idx]
        _lessons = self._verified_lessons.get(_pattern, {})
        if not _lessons:
            return "continue"
        best_a, best_avg = "continue", 0.05  # 阈值: avg>0.05 才信
        for a in _ACTIONS:
            _entry = _lessons.get(a, {})
            if _entry.get("n", 0) >= 3 and _entry.get("avg", 0.0) > best_avg:
                best_avg = _entry["avg"]
                best_a = a
        return best_a

    def record_tool_call(self) -> None:
        try:
            with self._lock:
                if self._runtime is None:
                    return
                rt = self._ensure_runtime(self._runtime.item_idx)
                rt.tool_calls += 1
                _prog = self._scan_outputs_progress(rt.item_idx)
                _delta = _prog - rt.last_progress_pct
                rt.last_progress_pct = _prog
                _reward = _ALPHA * max(0.0, _delta) / 100.0
                _advice = self._policy_locked(_prog)
                if _advice == rt.last_advice:
                    rt.same_advice_streak += 1
                else:
                    rt.last_advice = _advice
                    rt.same_advice_streak = 1
                _st = self._build_state(rt, _prog)
                if self._mdp_enabled:
                    # MDP: fast reward 记入 episode 轨迹, episode 结束再 MC 回传.
                    self._record_step(_st, _advice, _reward)
                    # item 完成 → 立即 flush 本 episode (terminal reward=1.0).
                    if _prog >= 100.0:
                        self._flush_trajectory(1.0)
                else:
                    self._update_q(_st, _advice, _reward)
        except Exception as _e:
            logger.debug("[v18] record_tool_call fallback: %s", _e)

    def update_iter_end(self, darwin_score: float) -> None:
        try:
            with self._lock:
                if self._runtime is None:
                    self._last_darwin_score = darwin_score
                    return
                _delta = darwin_score - self._last_darwin_score
                self._last_darwin_score = darwin_score
                _reward_slow = _BETA * _delta
                rt = self._runtime
                _prog = self._scan_outputs_progress(rt.item_idx)
                st = self._build_state(rt, _prog)
                _a = rt.last_advice
                if self._mdp_enabled:
                    # MDP: slow reward 合并进当前步轨迹, 不直接改 Q.
                    self._record_step(st, _a, _reward_slow)
                else:
                    k = st.key()
                    if k in self._Q and _a in self._Q[k]:
                        _N_sa = self._N[k][_a]
                        self._Q[k][_a] += _reward_slow / max(_N_sa, 1)
                # DeLM: 同步更新 verified_lessons (cross-task shared context).
                # ponytail: incremental mean, 不存原始 reward 序列, JSON 够小.
                _pattern = (self._items_labels[rt.item_idx]
                            if rt.item_idx < len(self._items_labels) else "UNLABELED")
                _pl = self._verified_lessons.setdefault(_pattern, {})
                _entry = _pl.setdefault(_a, {"avg": 0.0, "n": 0})
                _n = _entry["n"]
                _entry["avg"] = _entry["avg"] + (_reward_slow - _entry["avg"]) / (_n + 1)
                _entry["n"] = _n + 1
                self._update_count += 1
                if self._update_count % _PERSIST_FLUSH_EVERY == 0:
                    self._save()
        except Exception as _e:
            logger.debug("[v19] update_iter_end fallback: %s", _e)

    def _update_q(self, st: _BanditState, action: str, reward: float) -> None:
        k = st.key()
        if k not in self._Q:
            self._Q[k] = {a: 0.0 for a in _ACTIONS}
            self._N[k] = {a: 0 for a in _ACTIONS}
        _N_sa = self._N[k][action]
        _Q_sa = self._Q[k][action]
        self._Q[k][action] = _Q_sa + (reward - _Q_sa) / (_N_sa + 1)
        self._N[k][action] = _N_sa + 1
        self._update_count += 1
        if self._update_count % _PERSIST_FLUSH_EVERY == 0:
            self._save()

    # ── MDP 升级: episode 轨迹 + MC 信用分配 ──────────────────────────
    # 从 contextual bandit (单步即时更新) 升级为 episode 级 Monte Carlo:
    # 每一步的 reward 先记入轨迹, episode 结束沿轨迹自后向前回传 discounted
    # return — 让"整条 item 的成败"落到前面每一步的 state-action, 而不只是
    # 最后一步的瞬时 Δprogress. 这是 RL 三要素里缺的时间信用分配.

    def _record_step(self, st: _BanditState, action: str, reward: float) -> None:
        """把一步 (state, action, reward) 记入当前 episode 轨迹.

        同一 (state, action) 连续注入时累加 reward — fast (record_tool_call)
        与 slow (update_iter_end) 两条 stream 合并进同一步, 便于 MC 回传.
        """
        k = st.key()
        if self._trajectory and \
                self._trajectory[-1][0] == k and self._trajectory[-1][1] == action:
            _sk, _a, _r = self._trajectory[-1]
            self._trajectory[-1] = (_sk, _a, _r + reward)
        else:
            self._trajectory.append((k, action, reward))

    def _current_terminal_reward(self) -> float:
        """episode 终点奖励: item 完成 +1, 否则按 darwin 相对起点增量.

        作为 MC return 的 r_T — 让"这条 item 是否做成"作为终点信号回传.
        """
        if self._runtime is None:
            return 0.0
        _prog = self._scan_outputs_progress(self._runtime.item_idx)
        if _prog >= 100.0:
            return 1.0
        return max(0.0, self._last_darwin_score - self._episode_start_darwin)

    def _flush_trajectory(self, terminal_reward: float) -> None:
        """episode 级 MC discounted return 回传.

        自后向前算 G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}, 其中 r_T = terminal_reward;
        对轨迹上每个 (s,a) 做 MC 更新 Q(s,a) ← Q(s,a) + α(G_t − Q(s,a)).
        这是把单步 bandit 升成 MDP 的信用分配: 整条 episode 的累计回报沿
        轨迹回传到每一步的 state-action.

        ponytail: 离线全量 MC, 无函数近似 (表格 Q 保持 empirical). 升级路径:
        TD(λ)/eligibility trace 做在线 credit assignment, 或 tile coding 逼近.
        """
        if not self._trajectory:
            return
        _g = terminal_reward
        for _k, _a, _r in reversed(self._trajectory):
            _g = _r + _GAMMA * _g
            if _k not in self._Q:
                self._Q[_k] = {ac: 0.0 for ac in _ACTIONS}
                self._N[_k] = {ac: 0 for ac in _ACTIONS}
            _Q_sa = self._Q[_k][_a]
            self._Q[_k][_a] = _Q_sa + _ALPHA * (_g - _Q_sa)
            self._N[_k][_a] += 1
            self._update_count += 1
        self._trajectory = []
        if self._update_count % _PERSIST_FLUSH_EVERY == 0:
            self._save()

    def end_episode(self) -> None:
        """显式结束当前 episode, flush 轨迹 (run 结束 / 无后续 item 时调用).

        terminal reward 按当前进度判定, 避免最后一条轨迹永不回传.
        """
        with self._lock:
            if self._mdp_enabled:
                self._flush_trajectory(self._current_terminal_reward())

    def switch_item(self, new_idx: int) -> None:
        with self._lock:
            # MDP: 切走旧 item 前 flush 其轨迹 (terminal 按进度判定).
            if self._mdp_enabled:
                self._flush_trajectory(self._current_terminal_reward())
            self._episode_start_darwin = self._last_darwin_score
            self._runtime = _ItemRuntime(item_idx=new_idx, start_ts=time.time())

    def build_hint(self) -> str:
        try:
            with self._lock:
                if self._runtime is None or self._items_count == 0:
                    return ""
                _advice = self.policy()
                if _advice == "continue":
                    return ""
                rt = self._runtime
                _prog = rt.last_progress_pct
                _elapsed = time.time() - rt.start_ts
                _cur = (self._items_names[rt.item_idx]
                        if rt.item_idx < len(self._items_names) else f"item{rt.item_idx}")
                if _advice == "switch":
                    # DeLM 去中心化 task queue: 给 candidate list, agent 自治选, 不强制指定.
                    _cands = []
                    for _i in range(rt.item_idx + 1, min(rt.item_idx + 4, self._items_count)):
                        _name = (self._items_names[_i]
                                 if _i < len(self._items_names) else f"item{_i}")
                        _pat = (self._items_labels[_i]
                                if _i < len(self._items_labels) else "UNLABELED")
                        _cands.append(f"  - item {_i + 1}: {_name} [{_pat}]")
                    _cand_str = "\n".join(_cands) if _cands else "  (no further items)"
                    return (
                        f"\n\n## Effort Controller Hint (advisory)\n"
                        f"Current: item {rt.item_idx + 1}/{self._items_count} ({_cur}), "
                        f"{_elapsed:.0f}s in, {rt.tool_calls} tool calls, progress {_prog:.0f}%.\n"
                        f"Suggestion: progress plateaued — consider claiming a different item:\n"
                        f"{_cand_str}\n"
                        f"Reason: bandit Q(switch) > Q(continue), cross-task lessons suggest pivot.\n"
                        f"You can ignore this, continue current, or pick any item (not limited above).\n"
                    )
                if _advice == "requery":
                    return (
                        f"\n\n## Effort Controller Hint (advisory)\n"
                        f"Current: item {rt.item_idx + 1}/{self._items_count} ({_cur}), "
                        f"{_elapsed:.0f}s in, {rt.tool_calls} tool calls, progress {_prog:.0f}%.\n"
                        f"Suggestion: appears stuck — try an alternative approach or re-query KB.\n"
                        f"You can ignore this and continue if you have a clear plan.\n"
                    )
                return ""
        except Exception as _e:
            logger.debug("[v19] build_hint fallback: %s", _e)
            return ""

    def force_save(self) -> None:
        with self._lock:
            self._save()
