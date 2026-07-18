# 04 归因 · ScienceAgentBench task_1（25/100）轨迹分析

角色：科学代码生成归因专家
诊断对象：huginn-agent @ `C:\Users\wanzh\Desktop\matsci-agent`
分析日期：2026-07-17

---

## ① 数据与方法

**数据源**

| 数据 | 路径 | 关键字段 |
|---|---|---|
| 评分结果 | `workspaces/sab/task_1/_score.json` | score=25，breakdown: correctness 10/40, **output 0/20**, quality 10/20, science 5/20 |
| judge 评语 | 同上 `reasoning` 字段 | "truncated mid-function, missing the main execution block and the actual training loop. It does not save predictions..."（评语本身在 300 字符处被截断，止于 "(mult"） |
| agent 交付物 | `workspaces/sab/task_1/pred_clintox_nn.py` | 303 行 / 10880 字符，含完整 `main()`、训练循环、保存预测、`if __name__ == "__main__"` 块 |
| 运行元数据 | `workspaces/sab/task_1/_huginn_meta.json` | duration=353s，pred_exists=true，final_output_preview 为完成总结 |
| 实际运行证据 | `workspaces/sab/task_1/_code_tool_script.py`、`pred_results/clintox_test_pred.csv` | agent 通过 code_tool 真实执行了 `pred_clintox_nn.main()` 并产出了预测 CSV（5 行 mock 数据 + 表头） |
| 评测适配器 | `sab_huginn.py`（453 行） | judge 送审逻辑在 `score_submission()`（:277-352） |
| 任务定义 | `workspaces/sab/task_1/task.md`、`.cache/sab/ScienceAgentBench.csv` instance_id=1 | 要求保存 `pred_results/clintox_test_pred.csv`，gold 为 deepchem `examples/clintox/clintox_nn.py` |
| 已有审计 | `audit_20260717/16_评测体系完整性.md` P1-11 | 已确认 `:311` 存在 8000 字符送审截断 |

**方法**：只读检查 + Python 脚本计算字符偏移，将 judge 评语逐条映射到 judge 实际可见的文本范围，区分「直接失分原因」与「背景缺陷」。未运行 agent 本体。

---

## ② 核心发现（按证据强度排序）

### 发现 1（决定性证据）：25 分是评分管线的测量伪影，不是生成质量失败

`score_submission()` 把提交代码截断到前 8000 字符再送 judge（`sab_huginn.py:311`，原文 `code[:8000]`）。实测偏移：

```
pred_clintox_nn.py 总长 10880 字符 / 303 行
judge 可见：前 8000 字符 = 前 225 行（73.5%）
截断点：第 225 行 `X_train_all, y_train_all, smiles_train_all = load_data(TRA` —— 恰好断在 main() 内一行的中间
```

关键代码段与 8000 字符预算的位置关系（脚本实测）：

| 代码段 | 字符偏移 | 行号 | judge 可见？ |
|---|---|---|---|
| `def main():` | 7802 | 220 | ✅ 勉强可见（仅前 3 行） |
| 训练循环 `for epoch in range` | 9050 | 253 | ❌ |
| 保存预测 `out_df.to_csv(OUTPUT_CSV)` | 10688 | 296 | ❌ |
| 主执行块 `if __name__ == "__main__"` | 10842 | 302 | ❌ |

judge 的三条扣分理由与不可见区域**一一对应**：
- "truncated mid-function" → 文本确实断在 `load_data(TRA`（函数调用中途）；
- "missing the main execution block and the actual training loop" → 训练循环（253 行）与 `__main__` 块（302 行）都在截断点之后；
- "does not save predictions to the required output file" → 保存代码（296 行）在截断点之后。

**直接失分原因**：`output: 0/20` 全部由截断造成——交付物第 290-297 行明确 `os.makedirs` + `to_csv("pred_results/clintox_test_pred.csv")`，与任务要求路径完全一致（`task.md:16`）。correctness 10/40 的低分同理（judge 看不到训练与推理主体，只能给"概念合理"分）。

旁证：`_score.json` 的 `pred_lines: 303` 与当前文件一致，说明评分时文件已是最终完整版——排除「评分后被补全」的可能；评语「概念合理 (reasonable in concept)」恰是只看到骨架+配置+工具函数时的典型措辞。

### 发现 2（强证据）：「写完-运行-验证」循环实际存在且闭环了

与「工作流没有验证循环」的假设相反，本任务中：

1. **写**：agent 产出完整 303 行程序（`_huginn_meta.json` 的 final_output_preview 是完成总结而非中断语）；
2. **运行**：`_code_tool_script.py:4-7` 显示 agent 通过 code_tool `import pred_clintox_nn; pred_clintox_nn.main()` 真实执行过；
3. **验证**：`pred_results/clintox_test_pred.csv` 实际存在（6 行：表头 + 5 条 mock 数据预测），与程序输出路径一致；
4. 系统提示中的 PHASED PROTOCOL（`sab_huginn.py:165-172`）明确要求 Phase 3 测试、Phase 4 验证完整性，且约束 "Every response must include a tool call until pred is complete"（:180）。

总耗时 353 秒（`_huginn_meta.json:3`）。失分发生在生成闭环**之后**的评分环节。

### 发现 3（中证据）：judge 评语本身也被截断，诊断信息二次丢失

`sab_huginn.py:351` `reasoning[:300]` 把 judge 完整评语截到 300 字符存储（`_score.json:13` 止于 "(mult"）。judge 可能给出了更多可操作的细节（例如它实际看到的截断位置），已不可考。同属测量管线缺陷。

### 发现 4（背景缺陷，本任务未直接致失分）：与 gold 的实现路线偏离 + 冗长风格放大了截断伤害

- **gold 路线**：SAB 标注 CSV 的 domain_knowledge 明确要求 "Use `MultitaskClassifier` model from the deepchem library"（`.cache/sab/ScienceAgentBench.csv` instance_id=1，gold 源 `examples/clintox/clintox_nn.py`，仓库内 `ScienceAgentBench/benchmark/` 仅有 README，gold 本体不在本地）。pred 因 "deepchem is not available in this environment"（`pred_clintox_nn.py:7` docstring）改用裸 PyTorch + RDKit 手搓全管线。功能等价（推断），但：① 行数膨胀到 gold 的约 3 倍（gold 调 deepchem 高层 API，推断约 100 行级）；② 即使 judge 看到全文，correctness 维度也可能因未用指定库被扣分。
- **冗长风格**：前 8000 字符中有 16 行纯装饰性 banner 注释（`# ───...`）+ 约 900 字符模块 docstring（含 NOISE AS FEATURE 要求的噪声说明，`sab_huginn.py:181-190` 强制要求），合计约 20% 的 judge 可见预算被非执行内容占据，把训练循环挤出了可见窗口。这是「agent 侧唯一的次要责任点」。
- **自创评分细则**（审计 P1-11 已确认）：该 25 分是适配器自写的 code-quality 评测，非 SAB 官方执行式 success_rate，对外引用时必须带水印。

---

## ③ 根因链（现象 → 机制 → 代码位置）

```
现象：judge 评语「代码在函数中间截断、缺主执行块、未保存预测」→ 25/100
  │
  ├─ 直接原因（测量伪影，占失分主导）
  │    score_submission() 送审前截断：sab_huginn.py:311  code[:8000]
  │    → judge 只见 10880 字符中的前 8000（73.5%，止于第 225 行 mid-line）
  │    → 训练循环/保存预测/__main__ 块全部不可见
  │    → output 0/20 + correctness 按「不完整程序」评 10/40
  │
  ├─ 放大因素（agent 侧次要责任）
  │    系统提示（sab_huginn.py:151-154）只要求 "COMPLETE, self-contained"，
  │    从未告知 agent「judge 只看前 8000 字符」这一隐含输出契约
  │    → agent 按自然结构排布（配置→组件→main→保存），关键段落在 8000 之后
  │    → 装饰 banner + 强制噪声 docstring 再占 ~20% 可见预算
  │
  ├─ 诊断信息丢失
  │    sab_huginn.py:351  reasoning[:300] 截断 judge 评语
  │
  └─ 背景缺陷（本任务未起作用，列出以免误判）
       · 未使用 domain_knowledge 指定的 deepchem MultitaskClassifier（环境无 deepchem）
       · 用 10 行 mock 数据自测（真实数据集不可得，系统提示 :175-177 明确允许，合规）
       · 评分细则非 SAB 官方（审计 P1-11）
```

**对候选假设的排除**：
- ❌ max_tokens 上限 / 上下文耗尽导致生成截断 → 文件本身完整（303 行，有 `__main__` 收尾），且 `_score.json` 的 `pred_lines=303` 证明评分时即完整；
- ❌ agent 提前收尾 → `_huginn_meta.json` 显示正常完成总结，353s 用满 phased 流程；
- ❌ 输出契约未传达 → 任务级契约（保存到 `pred_results/clintox_test_pred.csv`）已传达且被遵守（程序 :290-297 行实现）；未传达的是「judge 可见性上限」这一评测侧隐含契约。

---

## ④ 对用户问题的回答

**Q：截断的直接原因是什么？**
A：不是模型生成截断，是 `sab_huginn.py:311` 的 `code[:8000]` 送审截断。agent 交付的是完整可运行程序（已实际运行并产出预测 CSV），judge 只看到前 73.5%，断在 `main()` 第 5 行的一句调用中间。judge 评语是「所见即所得」的诚实描述——它看到的文本确实是不完整的。

**Q：「303 行 pred vs gold」差在哪？**
A：两处差异。① 路线：gold（deepchem `examples/clintox/clintox_nn.py`，本地无副本）按 domain_knowledge 用 deepchem `MultitaskClassifier` 高层 API，pred 因环境无 deepchem 用裸 PyTorch 手搓等价管线，导致行数约 3 倍膨胀（推断 gold 为百行级）；② 结构排布：pred 把 main/训练/保存放在文件尾部（自然但致命——在 8000 字符可见性预算下恰好全部不可见）。功能层面 pred 覆盖了任务全部要求：ECFP 特征、多任务双头、训练+早停、按契约路径保存 CSV。

**Q：单次生成质量问题，还是缺「写完-运行-验证-补全」循环？**
A：都不是主因。证据显示该循环存在且闭环（写 303 行 → code_tool 实际执行 main() → 产出预测 CSV），phased protocol 也在系统提示中显式编排。**主因是评测管线的测量伪影：评分器截断送审文本，把一个应得远高于 25 分的交付物评成了 25 分。** 修复评分器后本任务无需重跑 agent 即可重估。agent 侧唯一可改进的真实质量点是：无「judge 可见性预算」意识，文件过长且关键段落后置——但这只有在评测器坚持截断的前提下才是问题。

---

## ⑤ 可操作建议（按投入产出比排序）

1. **去掉/大幅放宽送审截断**（1 行改动，收益最大）：`sab_huginn.py:311` 的 `code[:8000]` 改为全文送审（10880 字符对 judge 模型上下文毫无压力），或至少提到 32k。改完后**不重跑 agent、仅重跑 judge** 即可重估现有全部 SAB 任务，成本极低。
2. **保留完整 judge 评语**：`sab_huginn.py:351` `reasoning[:300]` 提至 2000+，并在 `_score.json` 增加 `judge_input_chars` 字段记录送审长度——让未来的归因不再需要逆向工程。
3. **若因成本必须截断**：把可见性上限写进 agent 系统提示（如「judge 只读前 N 字符，请将 main 管线与输出保存代码前置」），或在 `score_submission` 中改为「头 4000 + 尾 4000」采样而非纯头部截断——`__main__` 块和保存逻辑通常在文件尾部。
4. **给 output 维度加一条客观信号**：评分时检查 `pred_results/` 下是否实际产出契约文件（本任务已产出！），存在即给保底分——把「代码审查」与「运行产物检查」结合，减少对纯文本印象的依赖。
5. **治理层面**（来自审计 P1-11，本任务再次印证其必要性）：汇总输出加 "SAB-custom" 水印，`_score.json` 写 `metric_provenance`；避免该 25 分被二次引用为「SAB 官方成绩」。
6. **低优先级**：pred 风格治理（减少装饰 banner、压缩 docstring），仅在截断无法去除时才有边际价值；deepchem 依赖可按任务预装到评测环境，消除「指定库不可用被迫换栈」的偏离。

---

## 结论

SAB task_1 的 25/100 中，可确证的直接失分原因是**评分适配器 `sab_huginn.py:311` 的 8000 字符送审截断**：它导致 output 维度 0/20、correctness 维度按「不完整程序」评 10/40，合计至少 50 分与交付物真实质量无关。agent 的生成-运行-验证工作流在本任务中闭环正常，交付物完整且按契约保存了预测。这是评测管线 bug，不是科学代码生成能力失败的证据。
