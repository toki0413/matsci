# ToolUniverse 集成说明

## 评估结论

[ToolUniverse](https://github.com/zitniklab/ToolUniverse) 是 Harvard Zitnik Lab
的项目, 收录 350+ 生物医学数据分析工具, 覆盖数据源:

- FDA (药物审批 / 不良反应)
- OpenTargets (靶点-疾病关联)
- UniProt (蛋白质序列)
- PubChem (化合物)
- ChEMBL (生物活性)
- GWAS Catalog (全基因组关联)
- RCSB PDB (蛋白结构)

**材料科学重叠极少**: 350+ 工具里只有 `CrystalStructure_validate` 直接服务材料
计算, 其余 PubChem / RCSB PDB 部分工具在材料-生物交叉场景 (MOF, 生物材料,
药物载体) 里有间接价值. 全量接入会灌入 340+ 个材料科学用不上的生物医学工具,
污染工具注册表, 所以采用 **白名单精选** 策略.

## 白名单

`huginn/tools/mcp_adapter.py` 里的 `MATERIAL_SCIENCE_TOOL_WHITELIST`:

| 工具名 | 保留理由 |
|--------|----------|
| `CrystalStructure_validate` | 晶体结构验证, 直接服务材料计算 |
| `PubChem_get_record` | 化合物记录查询, 覆盖无机/有机化合物 |
| `PubChem_search_compounds` | 化合物搜索 |
| `PubChem_get_properties` | 化合物物性查询 |
| `ChEMBL_get_compound` | 生物活性数据, 材料相关小分子活性测量 |
| `RCSB_PDB_search` | 蛋白结构搜索, MOF/生物材料交叉场景 |
| `RCSB_PDB_get_entry` | 蛋白结构条目, 同上 |

共 7 个. 后续如发现新材料相关工具, 在此 set 里加名字即可.

## 接入方式

ToolUniverse 默认 **关闭**, 避免未装包时启动报错. 启用步骤:

1. 安装包:
   ```bash
   pip install tooluniverse
   ```
   (或 `pip install huginn-agent[tooluniverse]` 走 optional 依赖组)

2. 设置环境变量:
   ```bash
   export HUGINN_TOOLUNIVERSE_ENABLED=1
   ```

3. 启动 huginn. `lifespan._init_mcp_tools()` 会:
   - 连接 `python -m tooluniverse.smcp_server` MCP server
   - 走白名单注册 7 个材料相关工具 (过滤掉 343 个生物医学工具)

启动日志应看到:
```
[MCP] Registered N tools (ToolUniverse curated by whitelist)
```

## 手动验证

```bash
# 1. 确认 tooluniverse 装好
python -c "import tooluniverse; print('OK')"

# 2. 启用 + 起服务
export HUGINN_TOOLUNIVERSE_ENABLED=1
# (正常启动 huginn)

# 3. 查注册表里有没有白名单工具
python -c "
from huginn.tools.registry import ToolRegistry
from huginn.tools import register_all_tools
register_all_tools()
names = ToolRegistry.list_tools()
for w in ['CrystalStructure_validate','PubChem_get_record','RCSB_PDB_search']:
    assert w in names, f'{w} missing'
print('whitelist tools registered:', [n for n in names if 'PubChem' in n or 'Crystal' in n or 'RCSB' in n])
"

# 4. 调一个工具验证连通
# (在 agent 对话里让 LLM 调 PubChem_get_record 查一个化合物)
```

## 不在范围

- **全量 350+ 工具**: 不做. 噪音太大, 340+ 生物医学工具对材料科学无直接价值.
- **白名单自动同步**: 不做. ToolUniverse 工具名会随版本变, 手动维护白名单更稳.
- **HTTP 直连**: 不做. 走 MCP 协议, 跟 mat-db / math-anything 一致.
