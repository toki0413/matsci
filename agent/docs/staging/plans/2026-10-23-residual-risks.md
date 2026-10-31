# Residual risk fixes (ws9/ws10 follow-up)

来源：书生实战 10 轮（`research/shusheng_run/ws*`）暴露的 6 条残余风险。
纪律：每条先写失败测试（RED），最小实现（GREEN），再重构与同步文档。

## Tasks

- [x] R1 审计门抓不到量纲/符号笔误（`1.93e-14` 写成 `1.93e14`）
- [x] R2 正文小整数（“10 步”）计入 citation 候选值
- [x] R3 四舍五入复述（`6061` vs `6061.85`）仍记为未溯源
- [x] R4 `provenance.jsonl` 只覆盖内建快路，adapter 工具调用 0 快照
- [x] R5 沙箱/工具缺省 work_dir 落 `cwd`，产物写进源码树
- [x] R5b 空 `allowed_work_dirs` 的 `SandboxExecutor()` 缺省收敛到 run workspace
- [x] R6 `code_lab` 超时不收紧、按原超时反复重试

## Verification

```bash
pytest tests/test_residual_risks.py tests/test_structural_defect_fixes.py \
       tests/test_security.py -o addopts="" -q
```

## 实战验收 (ws11, `loop_2aade16e`, 书生原样重跑)

| 编号 | 现场证据 | 结论 |
|---|---|---|
| R1/R2/R3 | `report_citation untraceable=0/9`，无 `magnitude_typos` | 9 个候选值全部可溯源，量词噪声未再计入 |
| R4 | `provenance.tool_chain` 56 条 == trajectory `tool_calls` 56 条（ws9 为 5 vs 147） | 快照覆盖全部工具 |
| R5/R5b | 产物全落 `ws11/`（`_code_tool_script.py`、`nlse/`），`/workspace/agent` 无新增文件；启动日志不再出现 `path scoping inactive` | 源码树零污染 |
| R6 | 3 次超时分散在 3 个独立 `code_lab` 循环（iteration 3 ×2、iteration 7），无同循环 600s×4 重试 | 未再空转；半预算分支本轮未触发，由单测覆盖 |

工具面：56 次调用 / 41 成功（73%），6 种工具（`code_tool` 32、`bash_tool` 12、`file_write_tool` 6、`code_lab` 4、`git_tool` 1、`file_edit_tool` 1）。
