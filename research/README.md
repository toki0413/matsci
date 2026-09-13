# research/ — 内部研究沙箱（ADR-0001 治理边界）

> 定位: **非外部消费者**目录。放一次性的实验/探针/benchmark 载体,
> 直接 import `huginn.*` 业务模块以研究其内部机制。
> 这是种子一层 ADR-0001(单网关)的**有意逃生舱**, 但有硬约束, 且要能被治理。

## 为什么存在
`EXTERNAL_CONSUMER_DIRS = {scripts, examples, servers, sidecar}`（见
`agent/tests/test_arch_single_gateway.py`），会强制外部消费者走 `huginn.server`
HTTP、禁止直接 import 业务模块。但**内部研究**（探针 `huginn.metacog` 真分类器、
测 `imagination` 真实想象末态、bench 四件套 token 节省）天然需要进程内直读业务对象,
走 HTTP 反而无意义——所以 `research/` 位于扫描目录之外。

## 铁律（必须保持）
1. 只放**实验/研究载体**（探针、bench、一次性分析）, 不是生产入口的长期宿主。
   凡被生产复用的能力, 必须迁回 `huginn/` 或经 `/v1` API 暴露并登记, 不留在本目录。
2. 本目录**默认不被** `test_arch_single_gateway` 扫描 → 依赖 ADR-0001 免责,
   因此**不得悄悄扩容**成旁路; 迁移目标仍是「生产代码走 API / huginn/」。
3. Research 脚本若演变成生产路径, 需移出本目录并改走 gateway, 重入扫描。

## 规范
- 运行需要 `PYTHONPATH=/workspace/agent`（`huginn` 业务包在 `agent/` 下）。
- 每个脚本自包含: 命令、统计口径、诚实边界写进模块 docstring。
- 收敛的字段/机制, 优先复用已有模块（如 token 口径统一走
  `huginn.utils.tokens.rough_token_count_for_text`）。

## 当前内容
- `probe_failure_basins.py` — 分形吸引域假说 agent 端探针（真 `CompletionAuditor` 分类）。
- `research_imagination_real.py` — 真实 LLM 想象末态盆地探测。
- `bench_solpi_mechanisms.py` — SoL-Pi 四件套 token 节省 benchmark（`--json` 可复现）。

## 何时不该在这
- 新**生产**工具/端点: → 放 `huginn/` 并在 `agent/huginn/tools/__init__.py` 登记。
- 新**外部消费**程序: → 走 `huginn.server` `/v1` API，重在 `examples/scripts` 且受扫。
- 长期数据的沉淀: → `.huginn/`、`research_outputs/`。