"""DocGraph 管线的子进程执行体.

路由把上传的 PDF 写临时文件后, 用 ``python -m huginn.perception.doc_parse_worker
<in.pdf> <out.json> <filename> <workspace>`` 拉起. 这样 M1-M6 那段重活(和它里面
可能的 C 扩展原生崩溃, 比如 VCRUNTIME140 的 0xc0000005)只影响这个 worker, 不会
连坐主后端进程. KB 自动入库也放进这里 —— embedding 那段同样吃原生栈, 一并隔离.

进度通过 stdout 的 ``PROGRESS <json>`` 行逐阶段上报, 主进程逐行转成 SSE 事件.
成功写最终结果 dict 到 out.json 并打印 DONE; 失败向 stderr 打 FAIL 并退出码 1.
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 上传占前 10%, 之后每阶段 +15%, 收尾到 100.
_STAGE_OFFSETS = [10, 25, 40, 55, 70, 85]
_STAGE_LABELS = [
    "解析 PDF 元素",
    "提取图表数据",
    "构建文档图",
    "预测引用关系",
    "跨模态校验",
    "组装信息包",
]


def run_pipeline(pdf_path: str, filename: str, workspace: str) -> dict[str, Any]:
    """跑完整 DocGraph 管线 (M1-M6) + KB 自动入库. 返回给前端/后端的结果 dict."""
    # 延迟 import: worker 只在被 `-m` 拉起时才真正 import 感知管线, 避免把
    # 重型依赖带进无关路径.
    from huginn.perception.cross_validator import CrossModalAdapter
    from huginn.perception.data_extractor import FigureDataExtractor
    from huginn.perception.document_graph import DocumentGraph
    from huginn.perception.info_pack import InfoPackAssembler
    from huginn.perception.pdf_parser import PDFElementExtractor
    from huginn.perception.relation_predictor import RelationPredictor

    def report(idx: int) -> None:
        sys.stdout.write(
            "PROGRESS "
            + json.dumps(
                {"type": "stage", "pct": _STAGE_OFFSETS[idx], "message": _STAGE_LABELS[idx]}
            )
            + "\n"
        )
        sys.stdout.flush()

    # M1: 解析 PDF 为元素
    extractor = PDFElementExtractor()
    elements = extractor.extract(Path(pdf_path))
    report(0)

    # M2: 图表数据提取 (best-effort, 失败不阻断)
    try:
        FigureDataExtractor().process(elements)
    except Exception as exc:
        logger.warning("M2 figure data extraction skipped: %s", exc)
    report(1)

    # M3: 文档图 (仅结构边)
    graph = DocumentGraph(elements)
    report(2)

    # M4: 预测 REFERENCES 边 (mention -> figure/table)
    RelationPredictor().predict(graph)
    report(3)

    # M5: 跨模态校验 (注入 CLAIM 节点 + 判定边)
    CrossModalAdapter().process(graph)
    report(4)

    # M6: 组装信息包
    packages = InfoPackAssembler().assemble(graph)
    report(5)

    document_id = uuid.uuid4().hex
    record: dict[str, Any] = {
        "document_id": document_id,
        "filename": filename,
        "graph": graph.to_dict(),
        "packages": [p.to_dict() for p in packages],
        "stats": graph.stats(),
    }

    # KB 自动入库 (best-effort 旁路): 失败只记日志, 不阻断解析结果.
    # 在子进程里做, 即便 embedding 原生崩, 也只挂 worker.
    auto_ingested = 0
    if workspace:
        try:
            from huginn.knowledge import get_knowledge_base
            from huginn.perception.rag_bridge import RAGBridge

            kb = get_knowledge_base(workspace)
            if kb is not None:
                auto_ingested = RAGBridge(kb=kb).ingest(
                    packages, document_id=document_id, filename=filename
                )
        except Exception as exc:
            logger.warning("worker KB ingest failed: %s", exc, exc_info=True)
    record["auto_ingested"] = auto_ingested
    return record


def main() -> int:
    """CLI 入口: argv = in.pdf, out.json, [filename], [workspace]."""
    if len(sys.argv) < 3:
        sys.stderr.write("usage: doc_parse_worker <in.pdf> <out.json> [filename] [workspace]\n")
        return 2
    pdf_path, out_json = sys.argv[1], sys.argv[2]
    filename = sys.argv[3] if len(sys.argv) > 3 else Path(pdf_path).name
    workspace = sys.argv[4] if len(sys.argv) > 4 else ""
    try:
        record = run_pipeline(pdf_path, filename, workspace)
        Path(out_json).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        sys.stdout.write("DONE\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:
        sys.stderr.write(f"FAIL {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())