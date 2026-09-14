# Direct self-source modification（轨道 A3）— 运行时函数级自改源码

> 状态：design（2026-09-14）。前置：M-R1（递归改进器）+ A1（strategist 可演化 + 复合护栏）已落地。本轨道补 RSI 第三道质变缺口：**让 agent 能直接改进自己的代码**，而不只是 prompt 模板。A1 的 failure-modes #5 曾把"直接自改源码"列为不做，现经安全化设计启动 v0。
> 关联：`docs/staging/specs/2026-09-13-recursive-compounding-design.md`、`docs/staging/specs/2026-09-13-recursive-improver.md`

## 动机与边界（诚实声明）

- 之前两层都只在"prompt 模板"空间演化（improver_prompt / strategist_prompt），从未触碰真实代码。第三道质变缺口 = agent 能**改自己的实现**而不只改提示。
- **v0 只做"运行时函数级自改"**：`apply` 把候选新函数体 **monkeypatch 进目标模块的 `__dict__`**（运行时代码对象替换），绝不直接改磁盘源码。这样全程可逆（`RevertibleContext` 补偿）、可验证（先 `compile` + 沙箱自测再应用）、不可能破坏仓库（不落盘）。
- 磁盘文件级自改写（直接改写 `.py` 并依赖 git 回滚）留作 v1，需文件级护栏 + CI 重跑，成本高，明确不在本 spec。
- 所有自改**默认关**（`harness_source_patch`），关闭时零行为变更，session 安全。

## contract: `huginn/harness/source_patch.py`（新增）

```
SourcePatch  dataclass                    # 一个运行时函数级源码补丁
    id: str
    module: str          # 目标模块, 如 "huginn.harness.prompt_patch"
    symbol: str          # 被替换符号, 如 "DEFAULT_IMPROV_TEMPLATE" 或某函数名
    new_code: str        # 新代码文本 (def <symbol>(...): ... 或 <symbol> = ...)
    op: str = "replace_symbol"
    active: bool = False
    created_at: float

SourcePatchStore                          # 持久化 store, .huginn/source_patches/<id>.json, LRU
    get_instance() -> SourcePatchStore
    add_patch(patch) ; list_patches() ; get(patch_id)
    clear_active(module, symbol)          # 停用同 (module,symbol) 的旧补丁

verify_source_patch(patch) -> dict        # 应用前验证
    # ① compile(new_code) 须合法 (SyntaxError → 拒)
    # ② 隔离 ns 里 exec 定义无异常 (ImportError/NameError → 拒)
    # ③ 目标符号在被换前的模块中真实存在 (找不到 anchor → 拒)
    # ④ 可选 probe: 若带自测文本则经 RestrictedPython/SandboxExecutor 跑断言
    # 返回 {passed, error, issues}

apply_source_patch(patch, ctx=None) -> bool   # 验证通过后才应用
    # 只改运行时 __dict__: 记 orig = module.__dict__[symbol], exec(new_code, module.__dict__)
    # ctx 存在时 compensate("source_patch_apply", orig snapshot) — 可逆回滚
    # 原子: verify 失败绝不 apply; apply 后 active 落盘 + meta_trace

revert_source_patch(payload) -> None      # 补偿器: 恢复 module.__dict__[symbol] = orig
```

## A3 v0 ring（并入 `MetaImprover` 第三环，同单例）

- `maybe_propose_source(llm_chat_fn) -> str | None`：给 LLM"目标模块某函数的当前源码 + self-directive（如 r_phys 缺口）"，让它产一段**替代同一符号的新函数源码**。产出必须是可 `compile` 的对象；非法/缺 anchor → 拒。
- `evaluate_source(patch_id, llm_chat_fn) -> dict`：`verify_source_patch` + 代理分（代码是否语法可用 + 是否命中 directive 方向），配对注册进 `SignificanceGate` + `OODHoldoutValidator`；返回 `{green, sig, ood, verified, n}`。
- `maybe_promote_source(patch_id, ctx=None) -> bool`：`verify` 通过 **且** sig+OOD GREEN **且**（`harness_rphys_gate` 开时真实 r_phys 未退化的 advisory/硬闸）→ 在 `RevertibleContext.transaction()` 内 `apply_source_patch`（可逆）。默认 advisory；`harness_source_patch` 显式开才可能真正应用。
- 驱动周期：`note_generation` 里每隔 `_SOURCE_EVERY_N_GENERATIONS`（默认略大于 strategist 周期）触发一次 source 环。
- 与 A1 同列时空可组合：换源码是**时间可逆**效应（revertible）；不新增空间依赖（源码自改不破坏现有组件 availability）。

## data shape / CLI

- `.huginn/source_patches/<id>.json`: `SourcePatch.to_dict()`。
- `meta_trace.jsonl` 追加 `source_propose / source_evaluate / source_promote / source_reject / source_revert`。
- toggle：`huginn.toml [feature_flags] harness_source_patch=true`（默认 off，`_enabled.py` 读）。

## 验收（test）

新增到 `tests/test_source_patch.py` + `python -m huginn.harness.source_patch` selfcheck：

1. **零回归**：`harness_source_patch` off → `maybe_promote_source` 不应用、无行为变更。
2. **验证门**：非法语法（`compile` 失败）→ `verify passed=False` 拒绝；合法但运行时 `NameError` → 拒；目标符号不存在 → 拒。
3. **应用 + 可逆**：`apply_source_patch` 后 `module.__dict__[symbol]` 换成新实现且行为改变；`revert_all()` 恢复原实现。
4. **已兑现**：好 source 候选（验证过 + sig GREEN + r_phys 未退化）→ promote 应用；差候选不应用。
5. **持久化**：`source_patches/` 落盘；`active` reload 保留。

## failure modes / deferred

1. 只 monkeypatch，不落盘 → 进程内会话级自改；重启后不保留磁盘（好事：不会污染仓库，代价是不跨 session 存活）。**明确接受**为 v0 边界。
2. monkeypatch 对 C 扩展/顶层类定义限制：仅函数/顶层赋值内置原生可换；涉及 `import` 语句重排的不支持（文档注明）。
3. 并发/线程安全：自改在单线程 harness 路径内、经单例串行。不做更严格的内存隔离。
4. v1（磁盘文件级自改写 + git 回滚 + CI）留后续。