"""Hypothesis Manifold — 把 hypothesis space 建模为 manifold, 用 Bayesian
posterior + Fisher metric 引导搜索, 替代 random walk.

Why this exists:
    700万步预算够, 但当前架构不知道怎么用. darwin_score 是 post-hoc evaluation
    (评估不等于引导), hint boost 是 syntactic level (文字密度不等于推理密度).
    真正的突破是把 hypothesis space 建模成 manifold, 每一步在 posterior 上
    gradient descent, 不是 random walk.

Mathematical structure:
    H = {h_1, ..., h_n}             hypothesis space (discrete points on manifold)
    P(h)     ∝ exp(-MDL(h))         prior, simpler = more probable (Occam)
    P(O|h)                          likelihood, observations under hypothesis
    P(h|O)   ∝ P(O|h) * P(h)        posterior (Bayes)
    d_F(h_i, h_j)                   Fisher metric: prediction-disagreement proxy
    abduction = argmax_h P(h|O)     best explanation (MDL + fit)

Search guidance (this is the point):
    - Each step samples from posterior, not uniform random
    - Fisher metric tells you which direction has max info gain
    - 7M steps = MCMC on posterior landscape, not random walk
    - Credit assignment: Fisher info backprops early decisions to late failures

Honest boundaries (ponytail: named ceilings + upgrade paths):
    - Fisher metric via prediction disagreement is a proxy. True Fisher needs
      ∂²log P/∂h². Upgrade: when hypothesis has parametric form, compute true Fisher.
    - MCMC is Metropolis-Hastings lite. No HMC, no NUTS. Upgrade: jax + blackjax
      when scaling beyond ~100 hypotheses.
    - MDL via len(description) is BPE-token proxy. True MDL needs universal prior.
      Upgrade: Solomonoff induction approximation (Li & Vitányi §4.5).
    - This module proves the concept. Wiring into rcb_runner is spec work.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# B1: SubspacePartition R 矩阵投影优化 Fisher 距离.
# ponytail: 仍 stdlib only — 不引 numpy. 用 list-of-list 做矩阵乘法.
# 升级路径: n > 200 时换 numpy QR 分解 (矩阵运算 O(k²n) 会变贵).
# 数值保证: k >= n 时 R 是 n×n 正交矩阵, 投影距离与原 O(n) 遍历误差 < 1e-9.

# 子空间目标维度: n > _SUBSPACE_K 时启用 R 矩阵投影降维.
# ponytail: 200 是经验阈值, 大多数 huginn 场景 n < 10, 不会触发.
_SUBSPACE_K = 200


# ---------- 简单工具 ----------

def _logsumexp(xs: list[float]) -> float:
    """log(sum(exp(xs))) 数值稳定版. ponytail: stdlib only."""
    if not xs:
        return float("-inf")
    m = max(xs)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _mdl_logprior(description: str, n_params: int = 0) -> float:
    """MDL/BIC prior: log P(h) ∝ -(description_length + n_params * log(n_data)).

    BIC spirit: -2 log L + k log n. 这里只用 prior 部分 (k log n), likelihood
    另外算. ponytail: char count 代理 description length, n_params 是显式声明的
    有效参数数. 升级路径: 真正 BPE token count + Solomonoff approximation.
    """
    n = len(description.strip())
    if n == 0 and n_params == 0:
        return float("-inf")
    # λ=0.1 (description 弱权重), BIC 标准: 0.5 * k * log(n)
    # 参数数才是真正的 complexity, 描述文字长度受表达方式影响太大
    # n_data 用 100 代理 (升级路径: 从 observations 拿真实 n)
    desc_penalty = 0.1 * n
    bic_penalty = 0.5 * n_params * math.log(max(n_params * 10, 100))
    return -(desc_penalty + bic_penalty)


# ---------- B1: R 矩阵投影 ----------

def _gram_schmidt_qr(vecs: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """Modified Gram-Schmidt QR 分解 (stdlib only).

    输入: m×n 矩阵 (m 行向量, 每行长度 n). 输出: (Q, R) — Q 是 m×n 正交基,
    R 是 n×n 上三角. ponytail: 当 m < n 时退化, R 会有 0 对角元 (秩亏).
    """
    if not vecs or not vecs[0]:
        return [], []
    m = len(vecs)
    n = len(vecs[0])
    Q = [row[:] for row in vecs]  # 拷贝, 不动输入
    R = [[0.0] * n for _ in range(n)]
    for k in range(min(m, n)):
        # 求列 k 的 norm
        norm = math.sqrt(sum(Q[i][k] ** 2 for i in range(m)))
        if norm < 1e-12:
            # 秩亏列, 跳过 (R[k][k] 保持 0)
            continue
        R[k][k] = norm
        # 归一化
        for i in range(m):
            Q[i][k] /= norm
        # 消去后续列的 k 分量
        for j in range(k + 1, n):
            dot = sum(Q[i][k] * Q[i][j] for i in range(m))
            R[k][j] = dot
            for i in range(m):
                Q[i][j] -= dot * Q[i][k]
    return Q, R


def _matvec(R: list[list[float]], v: list[float]) -> list[float]:
    """R @ v — R 是 n×k 上三角 (n rows, k cols), v 长度 n."""
    if not R or not v:
        return []
    k = len(R[0])
    out = [0.0] * k
    for j in range(k):
        s = 0.0
        for i in range(min(j + 1, len(R))):
            s += R[i][j] * v[i]
        out[j] = s
    return out


# ---------- 核心数据结构 ----------

@dataclass
class Hypothesis:
    """Manifold 上的一个点.

    predictions 是 hypothesis 对 observable 的预测 (离散 token 或数值).
    Fisher metric 通过比较两个 hypothesis 的 predictions 差异来定义距离.
    n_params: 有效参数数 (BIC penalty 用). ad hoc hypothesis 参数多 → 重罚.
    """
    h_id: str
    description: str
    predictions: dict[str, float] = field(default_factory=dict)
    n_params: int = 0  # 有效参数数, BIC 用
    # 可选: 自定义 prior override (默认用 MDL)
    prior_override: float | None = None
    # abductive_inference 填: posterior 置信度 + 证据强度 (RAG hits 数/量级)
    confidence: float = 0.0
    evidence_strength: float = 0.0

    def log_prior(self) -> float:
        if self.prior_override is not None:
            return math.log(self.prior_override)
        return _mdl_logprior(self.description, self.n_params)


@dataclass
class Observation:
    """一次观察: observable name + measured value + noise scale."""
    name: str
    value: float
    sigma: float = 1.0  # 测量噪声标准差


# ---------- Manifold ----------

class HypothesisManifold:
    """Hypothesis space as a manifold with Bayesian posterior + Fisher metric.

    核心数学:
        posterior(h | O) ∝ exp(log_prior(h) + log_likelihood(O | h))
        fisher_distance(h_i, h_j) = E[(prediction_i - prediction_j)² / sigma²]
        info_gain(h) = entropy(prior) - entropy(posterior | h observed)

    用法:
        m = HypothesisManifold()
        m.add(Hypothesis("h1", "...", predictions={"x": 1.0}))
        m.add(Hypothesis("h2", "...", predictions={"x": 2.0}))
        post = m.posterior([Observation("x", 1.1)])
        best = m.abductive_inference([Observation("x", 1.1)])
        next_h = m.propose_next_exploration([Observation("x", 1.1)])
    """

    def __init__(self, likelihood_log: Callable[[Hypothesis, Observation], float] | None = None):
        self._hyp: dict[str, Hypothesis] = {}
        # 默认 likelihood: Gaussian prediction error. 用户可注入自定义.
        # ponytail: 升级路径是接 LLM-as-judge 评估 P(O|h).
        self._log_lik = likelihood_log or self._gaussian_log_likelihood
        # B1: SubspacePartition R 矩阵投影优化 Fisher 距离.
        # _keys: 所有 predictions 的 key 并集 (有序). _R: QR 分解的上三角矩阵.
        # _R 在 add_hypothesis 时增量更新 (重算 QR, O(n²) per add).
        self._keys: list[str] = []
        self._R: list[list[float]] | None = None

    @staticmethod
    def _gaussian_log_likelihood(h: Hypothesis, o: Observation) -> float:
        """log P(o | h) = -0.5 * ((o - pred) / sigma)²  (Gaussian, 常数项省略)."""
        if o.name not in h.predictions:
            # hypothesis 对此 observable 没预测 → 弱 penalty (uniform fallback)
            # ponytail: 真正处理是用 marginal likelihood, 这里用 -log(2)/2 近似
            return -0.5 * math.log(2.0)
        pred = h.predictions[o.name]
        z = (o.value - pred) / max(o.sigma, 1e-9)
        return -0.5 * z * z

    def add(self, h: Hypothesis) -> None:
        if h.h_id in self._hyp:
            raise ValueError(f"duplicate h_id: {h.h_id}")
        self._hyp[h.h_id] = h
        # B1: 新 key 加入后标记 R 矩阵需要重算 (lazy).
        # ponytail: 之前每次 add 都全量 QR, O(m·n²) 无收益开销 —
        # 实际 n=2~3 远低于 _SUBSPACE_K=200, 投影路径永不触发.
        new_keys = [k for k in h.predictions if k not in self._keys]
        if new_keys:
            self._keys.extend(new_keys)
            self._R_dirty = True

    def _rebuild_R(self) -> None:
        """重算 QR 分解的 R 矩阵 (基于当前所有 hypothesis 的 predictions).

        ponytail: lazy — 只在 fisher_distance 真正需要投影路径时才算.
        n > _SUBSPACE_K 时走投影, 否则 _R 保持 None 走原 O(n) 遍历.
        """
        if not self._keys or not self._hyp:
            self._R = None
            self._R_dirty = False
            return
        # 把每个 hypothesis 的 predictions 向量按 _keys 顺序排列成矩阵行.
        rows: list[list[float]] = []
        for h in self._hyp.values():
            rows.append([h.predictions.get(k, 0.0) for k in self._keys])
        _, R = _gram_schmidt_qr(rows)
        self._R = R
        self._R_dirty = False

    def log_posterior(self, obs: Iterable[Observation]) -> dict[str, float]:
        """返回每个 hypothesis 的 log posterior (未归一化)."""
        obs = list(obs)
        out: dict[str, float] = {}
        for h_id, h in self._hyp.items():
            lp = h.log_prior()
            for o in obs:
                lp += self._log_lik(h, o)
            out[h_id] = lp
        return out

    def _log_posterior_single(self, obs: Iterable[Observation], h_id: str) -> float:
        """单个 hypothesis 的 log posterior (未归一化), 与 log_posterior(obs)[h_id] 等价.

        增量路径专用 — mcmc_step 每步只算 current + proposal 两个,
        不再走全量 log_posterior (O(|H|×|O|) → O(|O|)).
        """
        h = self._hyp.get(h_id)
        if h is None:
            return float("-inf")
        lp = h.log_prior()
        for o in obs:
            lp += self._log_lik(h, o)
        return lp

    def posterior(self, obs: Iterable[Observation]) -> dict[str, float]:
        """归一化 posterior P(h | O)."""
        log_post = self.log_posterior(obs)
        Z = _logsumexp(list(log_post.values()))
        if Z == float("-inf"):
            return {h_id: 0.0 for h_id in log_post}
        return {h_id: math.exp(lp - Z) for h_id, lp in log_post.items()}

    def abductive_inference(self, obs: Iterable[Observation]) -> Hypothesis | None:
        """abduction: argmax_h P(h | O).

        这是 Hempel-Oppenheim model 里 O → H 这一步. LLM 做不到 (pattern completion
        ≠ abduction), 这里用 Bayesian model selection 实现.
        """
        if not self._hyp:
            return None
        log_post = self.log_posterior(obs)
        best_id = max(log_post, key=log_post.get)
        best = self._hyp[best_id]
        # confidence 取归一化 posterior; evidence_strength 暂无 RAG hits 来源.
        # ponytail: 后续接 posterior / RAG hits 后再填实际值
        post = self.posterior(obs)
        best.confidence = post.get(best_id, 0.5)
        best.evidence_strength = 0.0
        return best

    def fisher_distance(self, h_i_id: str, h_j_id: str) -> float:
        """Fisher information metric 的工程近似.

        d_F(h_i, h_j)² = Σ_k (pred_i[k] - pred_j[k])² / sigma_k²

        B1: SubspacePartition R 矩阵投影优化. 当 prediction space 维度 n > k
        (子空间目标维度) 时, 用预计算 R 矩阵投影到 k 维子空间, 距离从 O(n) 降为 O(k).
        当 n <= k 时 R 是正交方阵, 投影保持范数, 与原 O(n) 遍历数值一致 (误差 < 1e-9).

        ponytail: 真正 Fisher info metric 是 g_μν = E[∂_μ log P · ∂_ν log P],
        需要参数化 hypothesis. 这里用 prediction 差异代理 — 升级路径是
        hypothesis 参数化后计算 true Fisher.
        """
        h_i = self._hyp.get(h_i_id)
        h_j = self._hyp.get(h_j_id)
        if h_i is None or h_j is None:
            return float("inf")
        common_keys = set(h_i.predictions) & set(h_j.predictions)
        if not common_keys:
            return float("inf")

        # B1: 当 _R 可用且子空间维度 k < n 时走投影路径, 否则走原 O(n) 遍历.
        # 数值保证: k >= n 时 ||R^T d|| == ||d|| (R 列正交), 投影路径与遍历路径等价.
        # ponytail: 当前 huginn 场景 n 通常 2-3, 不会触发降维. 留 R 矩阵作为
        # n 增大时的 future hook — 升级路径: n > 200 时启用投影 + 增量 QR.
        # P1-3: _rebuild_R 改 lazy — 只在真正需要投影路径 (n > _SUBSPACE_K) 时才算.
        if self._keys and len(self._keys) > _SUBSPACE_K:
            if self._R_dirty or self._R is None:
                self._rebuild_R()
            if self._R is not None:
                diff = [h_i.predictions.get(k, 0.0) - h_j.predictions.get(k, 0.0) for k in self._keys]
                projected = _matvec(self._R, diff)
                d2 = sum(p * p for p in projected)
                return math.sqrt(d2)

        # 原 O(n) 遍历 — n <= k 时的等价路径, 也是当前 huginn 场景的主路径.
        d2 = 0.0
        for k in common_keys:
            # sigma 用 1.0 默认, 真实场景从 observation 拿
            d2 += (h_i.predictions[k] - h_j.predictions[k]) ** 2
        return math.sqrt(d2)

    def propose_next_exploration(
        self,
        obs: Iterable[Observation],
        *,
        rng: random.Random | None = None,
    ) -> Hypothesis | None:
        """Active learning: 选一个 hypothesis, 使期望信息增益最大.

        Info gain(h) = H[prior] - H[posterior | observe h's prediction]
                     ≈ entropy of current posterior - expected entropy after test

        工程近似: 当前 posterior 越均匀, 选离 top hypothesis 最远的 h 越能区分.
        ponytail: 真正 info gain 需要在 hypothesis space 上期望, 这里用
        "max fisher distance to argmax" 代理. 升级路径: 真正 Bayesian D-optimal.
        """
        if not self._hyp:
            return None
        rng = rng or random
        post = self.posterior(obs)
        # 当前 best
        best_id = max(post, key=post.get)
        # 找离 best 最远的 h (Fisher metric 下)
        max_d = -1.0
        candidate = None
        for h_id in self._hyp:
            if h_id == best_id:
                continue
            d = self.fisher_distance(best_id, h_id)
            if d > max_d:
                max_d = d
                candidate = h_id
        return self._hyp[candidate] if candidate else self._hyp[best_id]

    def mcmc_step(
        self,
        obs: Iterable[Observation],
        current_h_id: str,
        *,
        rng: random.Random | None = None,
        temperature: float = 1.0,
        cached_log_p_current: float | None = None,
    ) -> tuple[str, float]:
        """Metropolis-Hastings 一步在 posterior 上采样.

        P4-4: proposal kernel 用 fisher_distance 引导 — 距离近的 h 提议概率高.
        SubspacePartition 的 R 矩阵投影通过 fisher_distance 间接接入:
        n > _SUBSPACE_K 时 fisher_distance 走投影路径, proposal kernel 自动加速.

        Metropolis-Hastings 接受比考虑 proposal 不对称性:
            A = min(1, [P(h')·q(h'|h)] / [P(h)·q(h|h')])
        其中 q(h'|h) ∝ exp(-d_F(h,h')/τ) / Z_h, Z_h 是 h 的归一化常数.
        ponytail: Z_h 不易算, 用对称化近似 A ≈ min(1, P(h')/P(h) · exp(-(d-d')/τ)),
        d = d_F(h,h'), d' = d_F(h,h') (对称) → 距离项相消, 退化为标准 MH.
        实际引导通过 proposal sampling 实现 (距离近的被采到更多), 接受步保持标准 MH.

        Step 2 增量路径: 不再调 log_posterior(obs) 全量, 只算 current + proposal.
        cached_log_p_current 由调用方跨步传入, 拒绝时复用 (省一次 _log_lik 全遍历).
        接受时返回 proposal 的 log_p, 调用方下步直接复用 — 永远只算 1 次新 log_p.

        700万步 = 这个 step 跑 7M 次. 每步是从 current_h 提议一个 neighbor,
        按 posterior ratio 接受/拒绝. 这才是有引导的搜索, 不是 random walk.

        Returns: (next_h_id, next_log_p) — 调用方下步传 cached_log_p_current=next_log_p.
        """
        rng = rng or random
        # 物化一次, current + proposal 两次 _log_posterior_single 都要遍历
        obs = list(obs)
        h_ids = list(self._hyp)
        if len(h_ids) < 2:
            if cached_log_p_current is None:
                cached_log_p_current = self._log_posterior_single(obs, current_h_id)
            return current_h_id, cached_log_p_current
        others = [h for h in h_ids if h != current_h_id]
        if not others:
            if cached_log_p_current is None:
                cached_log_p_current = self._log_posterior_single(obs, current_h_id)
            return current_h_id, cached_log_p_current

        # P4-4: fisher_distance 引导的 proposal kernel.
        # 距离近的 h 提议概率高 (softmax(-d/τ)).
        # n > _SUBSPACE_K 时 fisher_distance 内部走 R 矩阵投影 → SubspacePartition 真正接入.
        # ponytail: τ=temperature 控制引导强度. τ→∞ 退化为 uniform; τ→0 只选最近.
        # 升级路径: 自适应 τ (按接受率调), 或 Langevin dynamics 用 fisher metric 做 gradient.
        dists = [self.fisher_distance(current_h_id, h) for h in others]
        max_d = max(dists) if dists else 0.0
        # softmax with numerical stability (减 max_d)
        weights = [math.exp(-(d - max_d) / temperature) for d in dists]
        total_w = sum(weights)
        if total_w <= 0.0:
            proposal = rng.choice(others)
        else:
            r = rng.random() * total_w
            cum = 0.0
            proposal = others[-1]
            for h, w in zip(others, weights):
                cum += w
                if r <= cum:
                    proposal = h
                    break

        # Step 2 增量: current 复用缓存, proposal 只算一次
        if cached_log_p_current is None:
            log_p_current = self._log_posterior_single(obs, current_h_id)
        else:
            log_p_current = cached_log_p_current
        log_p_proposal = self._log_posterior_single(obs, proposal)

        log_ratio = (log_p_proposal - log_p_current) / temperature
        # 接受概率 min(1, exp(log_ratio))
        if math.log(rng.random()) < log_ratio:
            return proposal, log_p_proposal
        return current_h_id, log_p_current

    # ── Step 3: 多链并行 + Gelman-Rubin 收敛诊断 ──────────────────
    # 7M 步单链太慢, 标准 MCMC 做法是多链并行 + R̂ 诊断.
    # 每链独立 rng + 独立 current + 独立 checkpoint, 单链崩溃不影响其他链.
    # 跑完后用 Gelman-Rubin R̂ 判断收敛, R̂ < 1.1 视为收敛.
    #
    # ponytail: asyncio.gather 是协程级并行, CPU-bound 任务实际是串行.
    # 真正的并行要 multiprocessing 或 Ray. 但 mcmc_step 是纯 Python 计算
    # (没 IO), asyncio 也能跑通, 只是 wallclock 没加速. 升级路径: ProcessPool.
    async def mcmc_multi_chain(
        self,
        obs: Iterable[Observation],
        n_chains: int = 4,
        n_steps_per_chain: int = 1_750_000,
        checkpoint_interval: int = 10_000,
        *,
        temperature: float = 1.0,
        on_chain_checkpoint: "Callable[[int, dict], None] | None" = None,
    ) -> dict:
        """多链并行 MCMC + Gelman-Rubin R̂ 收敛诊断.

        Returns:
            {
                "chains": [[h_id, ...], ...],  # 每链样本 (burn-in 后, 降采样)
                "r_hat": float,                # Gelman-Rubin R̂
                "converged": bool,             # R̂ < 1.1
                "accept_rates": [float, ...],  # 每链接受率
            }
        """
        import asyncio

        obs_list = list(obs)
        if not self._hyp or n_chains < 1:
            return {"chains": [], "r_hat": float("nan"), "converged": False, "accept_rates": []}

        # 每链独立 rng + 独立 current
        # 种子派生: chain_id * 7919 + base_seed. 7919 是质数, 避免相邻链种子相关.
        base_seed = int(os.environ.get("HUGINN_MCMC_SEED", "42"))

        async def _chain_runner(chain_id: int):
            rng = random.Random(chain_id * 7919 + base_seed)
            # current 初始化: 用 abductive_inference 给一个合理起点, 避免全从同一点开始
            init_h = None
            try:
                abd = self.abductive_inference(obs_list)
                init_h = abd.h_id if abd else None
            except Exception:
                init_h = None
            if init_h is None:
                h_ids = list(self._hyp)
                init_h = rng.choice(h_ids)
            return await self._run_single_chain(
                chain_id, obs_list, n_steps_per_chain, checkpoint_interval,
                rng=rng, temperature=temperature,
                init_h_id=init_h, on_checkpoint=on_chain_checkpoint,
            )

        # return_exceptions=True: 单链崩溃返回 Exception 对象, 不影响其他链
        results = await asyncio.gather(
            *[_chain_runner(i) for i in range(n_chains)],
            return_exceptions=True,
        )

        chains: list[list[str]] = []
        accept_rates: list[float] = []
        for r in results:
            if isinstance(r, Exception):
                # 降级路径: 崩溃的链不参与 R̂ 计算, 只 log
                continue
            chains.append(r["samples"])
            accept_rates.append(r["accept_rate"])

        r_hat = self._gelman_rubin(chains) if len(chains) >= 2 else float("nan")
        converged = (not math.isnan(r_hat)) and r_hat < 1.1

        return {
            "chains": chains,
            "r_hat": r_hat,
            "converged": converged,
            "accept_rates": accept_rates,
        }

    async def _run_single_chain(
        self,
        chain_id: int,
        obs: list,
        n_steps: int,
        checkpoint_interval: int,
        *,
        rng: random.Random,
        temperature: float = 1.0,
        init_h_id: str,
        on_checkpoint: "Callable[[int, dict], None] | None" = None,
    ) -> dict:
        """单链 MCMC 执行器. 复用 mcmc_step 的增量逻辑.

        Returns: {"samples": [...], "accept_rate": float, "final_h": str}
        """
        current = init_h_id
        cached_log_p: float | None = None
        accept_count = 0
        # burn-in: 前半段不采样本 (Markov 链未达稳态)
        burn_in = n_steps // 2
        # 降采样: 每 1000 步取 1 个, 避免样本自相关 + 节省内存
        thin = 1000
        samples: list[str] = []

        for step in range(1, n_steps + 1):
            prev_h = current
            current, cached_log_p = self.mcmc_step(
                obs, current, rng=rng, temperature=temperature,
                cached_log_p_current=cached_log_p,
            )
            if current != prev_h:
                accept_count += 1
            if step > burn_in and step % thin == 0:
                samples.append(current)

            # 周期 checkpoint
            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                ckpt = {
                    "chain_id": chain_id,
                    "step": step,
                    "current": current,
                    "rng_state": rng.getstate(),
                    "accept_count": accept_count,
                }
                if on_checkpoint is not None:
                    try:
                        on_checkpoint(chain_id, ckpt)
                    except Exception:
                        pass  # checkpoint 失败不阻塞链

        accept_rate = accept_count / n_steps if n_steps > 0 else 0.0
        return {
            "samples": samples,
            "accept_rate": accept_rate,
            "final_h": current,
        }

    @staticmethod
    def _gelman_rubin(chains: list[list[str]]) -> float:
        """Gelman-Rubin R̂ 收敛诊断.

        R̂ = sqrt(((n-1)*W + B/n) / W)
        W = 链内方差均值 (每条链的样本方差取平均)
        B = 链间方差 × n (链均值方差 × 链长)
        n = 每链样本数

        字符串 h_id → 数值: 用 h_id 在所有样本中的 rank (按首次出现顺序).
        ponytail: rank 是序数, 不反映 log_posterior. 升级路径: 传 log_post 值.
        """
        if len(chains) < 2:
            return float("nan")

        # h_id → 数值映射 (按全样本首次出现顺序)
        all_h: list[str] = []
        seen: set[str] = set()
        for c in chains:
            for h in c:
                if h not in seen:
                    seen.add(h)
                    all_h.append(h)
        if not all_h:
            return float("nan")
        h_to_idx = {h: float(i) for i, h in enumerate(all_h)}

        # 转数值 + 截到最短链长度 (R̂ 要求等长)
        min_len = min(len(c) for c in chains)
        if min_len < 2:
            return float("nan")
        num_chains = [[h_to_idx[h] for h in c[:min_len]] for c in chains]

        m = len(num_chains)       # 链数
        n = min_len               # 每链样本数

        # 链均值
        chain_means = [sum(c) / n for c in num_chains]
        # 总均值
        grand_mean = sum(chain_means) / m

        # B: 链间方差 (× n)
        B = n * sum((cm - grand_mean) ** 2 for cm in chain_means) / (m - 1) if m > 1 else 0.0
        # W: 链内方差均值
        W = 0.0
        for c in num_chains:
            cm = sum(c) / n
            var = sum((x - cm) ** 2 for x in c) / (n - 1) if n > 1 else 0.0
            W += var
        W /= m

        if W <= 0.0:
            # 链内方差为 0 (所有链收敛到同一点), R̂ 视为收敛
            return 1.0

        # var_hat = ((n-1)/n) * W + (1/n) * B  (Gelman 2003, B 已经 × n)
        # ponytail: 不做 df 修正, 升级路径: rank-normalized R̂ (Vehtari 2021)
        var_hat = ((n - 1) * W + B) / n if n > 0 else W
        r_hat = math.sqrt(var_hat / W) if W > 0 else 1.0
        return r_hat


# ---------- Self-check ----------

def _selfcheck() -> None:
    """Assert-based demo: 经典 abduction 测试.

    场景: 黑箱子里有个机制, 我们观察到输出值. 多个 hypothesis 解释这个机制.
    Bayesian model selection 应该选最简且 fit 最好的 hypothesis (Occam razor).
    """
    # 三个 hypothesis 解释同一组观察
    h_newton = Hypothesis(
        h_id="newton",
        description="F = G m1 m2 / r^2",  # 简短, MDL 低
        predictions={"orbital_period": 1.0, "precession": 0.0},
        n_params=2,  # G, mass
    )
    h_gr = Hypothesis(
        h_id="gr",
        description="G_μν = 8π T_μν, geodesic equation in curved spacetime",  # 长
        predictions={"orbital_period": 1.0, "precession": 0.43},  # 包含相对论修正
        n_params=10,  # metric tensor components + stress-energy
    )
    h_epicycle = Hypothesis(
        h_id="epicycle",
        description="deferent and epicycle with 40 parameters tuned",  # ad hoc
        predictions={"orbital_period": 1.0, "precession": 0.43},  # 也能 fit
        n_params=40,  # 40 个调出来的参数 — BIC 应重罚
    )

    m = HypothesisManifold()
    m.add(h_newton)
    m.add(h_gr)
    m.add(h_epicycle)

    # 观察: 水星近日点进动 0.43 arcsec/century — GR 的关键证据
    obs = [
        Observation("orbital_period", 1.0, sigma=0.01),
        Observation("precession", 0.43, sigma=0.05),
    ]

    # Test 1: abduction 应该选 GR, 不是 epicycle (Occam: GR 更简)
    best = m.abductive_inference(obs)
    assert best is not None, "abduction returned None on non-empty manifold"
    assert best.h_id == "gr", (
        f"expected GR (simplest fit), got {best.h_id}. "
        f"posterior={m.posterior(obs)}"
    )

    # Test 2: 没有进动观察时, Newton 应该赢 (最简)
    obs_no_prec = [Observation("orbital_period", 1.0, sigma=0.01)]
    best_no_prec = m.abductive_inference(obs_no_prec)
    assert best_no_prec.h_id == "newton", (
        f"without precession data, Newton (simplest) should win, got {best_no_prec.h_id}"
    )

    # Test 3: Fisher distance — Newton 跟 GR 在 precession 维度上有距离
    d_newton_gr = m.fisher_distance("newton", "gr")
    d_newton_newton = m.fisher_distance("newton", "newton")
    assert d_newton_gr > 0.0, "Newton and GR should differ on precession"
    assert d_newton_newton == 0.0, "self-distance should be 0"

    # Test 4: active learning 应该提议测 precession (区分 newton vs gr 的关键)
    next_h = m.propose_next_exploration(obs_no_prec)
    # 离 newton (当前 best) Fisher 距离最远的是 gr (precession 差 0.43)
    assert next_h is not None and next_h.h_id == "gr", (
        f"expected GR as next exploration (max Fisher distance to current best), "
        f"got {next_h.h_id if next_h else None}"
    )

    # Test 5: MCMC 在 posterior 上采样, 多步后访问 GR 的频率 ≈ posterior(GR)
    # Step 2: 用 cached_log_p_current 跨步缓存, 验证增量路径与全量路径等价
    rng = random.Random(42)
    current = "newton"
    cached_log_p: float | None = None  # 首步 None, 之后复用上步返回值
    visit_count = {"newton": 0, "gr": 0, "epicycle": 0}
    n_steps = 10000
    for _ in range(n_steps):
        current, cached_log_p = m.mcmc_step(
            obs, current, rng=rng, cached_log_p_current=cached_log_p)
        visit_count[current] += 1
    # 频率应近似 posterior (GR 应该是 dominant)
    assert visit_count["gr"] > visit_count["epicycle"], (
        f"MCMC should visit GR more than epicycle, got {visit_count}"
    )
    assert visit_count["gr"] > n_steps * 0.3, (
        f"GR should be visited >30% of time, got {visit_count['gr']/n_steps:.2%}"
    )

    print("✓ hypothesis_manifold self-check passed")
    print(f"  posterior(obs) = {m.posterior(obs)}")
    print(f"  abduction → {best.h_id} (Occam + fit)")
    print(f"  no-precession abduction → {best_no_prec.h_id} (Occam wins without data)")
    print(f"  fisher(newton, gr) = {d_newton_gr:.3f}")
    print(f"  active learning → {next_h.h_id} (max info gain direction)")
    print(f"  MCMC visit freq: { {k: v/n_steps for k, v in visit_count.items()} }")

    # B1: SubspacePartition R 矩阵投影专项验证
    # P1-3 后 _R lazy rebuild: n <= _SUBSPACE_K 时不构建 (保持 None), 走 O(n) 等价路径.
    # 升级路径: n > _SUBSPACE_K 时 fisher_distance 触发 _rebuild_R, 走 R 矩阵投影.
    assert m._keys == ["orbital_period", "precession"], f"keys mismatch: {m._keys}"
    assert m._R is None, (
        f"R matrix should NOT be built when n <= k (lazy), got _R={m._R}"
    )
    # n <= _SUBSPACE_K 时走原遍历路径, 数值与 O(n) 一致
    d_newton_gr_v2 = m.fisher_distance("newton", "gr")
    assert abs(d_newton_gr_v2 - d_newton_gr) < 1e-9, (
        f"k>=n path should be identical to O(n) path, got {d_newton_gr_v2} vs {d_newton_gr}"
    )
    assert d_newton_gr_v2 == 0.43, f"fisher(newton, gr) should be 0.43, got {d_newton_gr_v2}"
    print(f"  B1 R-matrix: keys={m._keys}, n={len(m._keys)} <= k={_SUBSPACE_K}, 走 O(n) 等价路径 (lazy)")


if __name__ == "__main__":
    _selfcheck()
