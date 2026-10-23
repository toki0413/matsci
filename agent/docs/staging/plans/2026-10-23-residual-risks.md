# Residual risk fixes (ws9/ws10 follow-up)

来源：书生实战 10 轮（`research/shusheng_run/ws*`）暴露的 6 条残余风险。
纪律：每条先写失败测试（RED），最小实现（GREEN），再重构与同步文档。

## Tasks

- [x] R1 审计门抓不到量纲/符号笔误（`1.93e-14` 写成 `1.93e14`）
- [x] R2 正文小整数（“10 步”）计入 citation 候选值
- [x] R3 四舍五入复述（`6061` vs `6061.85`）仍记为未溯源
- [x] R4 `provenance.jsonl` 只覆盖内建快路，adapter 工具调用 0 快照
- [x] R5 沙箱/工具缺省 work_dir 落 `cwd`，产物写进源码树
- [x] R6 `code_lab` 超时不收紧、按原超时反复重试

## Verification

```bash
pytest tests/test_residual_risks.py tests/test_structural_defect_fixes.py \
       tests/test_security.py -o addopts="" -q
```
