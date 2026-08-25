"""把某个 workspace 的知识库升级到当前 EMBED_MODEL 并重建 collection.

用法 (在普通终端, 非沙箱; 先退出桌面 App 释放 ~/.huginn_kb 的占用):
    python rebuild_kb.py [workspace]

workspace 默认取当前用户主目录 (桌面后端默认工作区, 对应 ~/.huginn_kb)。
脚本会:
  1. 备份 chroma 目录到 <workspace>/.huginn_kb/chroma.<ts>.bak
  2. 读出旧 collection 的全部 (id, 源文本, metadata) —— 源文本都在 chroma 里
  3. 删掉 collection
  4. 用 store.EMBED_MODEL (默认 BAAI/bge-m3, 首次会下载 ~2GB) 重新编码重灌
  5. 校验新 collection 条目数 == 旧条目数

因为重灌用的是"读出→重编→回填"整条链, 不存在删了没备份的丢数据窗口。
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

def main() -> int:
    args = sys.argv[1:]
    force = "--yes" in args
    args = [a for a in args if a != "--yes"]
    workspace = args[0] if args else str(Path.home())
    kb_root = Path(workspace).resolve() / ".huginn_kb"
    chroma_dir = kb_root / "chroma"
    coll_name = "huginn_kb"
    if not chroma_dir.exists():
        print(f"找不到 KB 目录: {chroma_dir}")
        return 2

    import chromadb
    from huginn.knowledge.store import EMBED_MODEL

    print(f"workspace : {workspace}")
    print(f"embed model: {EMBED_MODEL}")
    print(f"chroma    : {chroma_dir}")
    if EMBED_MODEL.lower() == "all-minilm-l6-v2":
        print("提醒: 当前模型仍是 all-MiniLM (可能没走新默认)。请用源码/新打包跑本脚本,")
        print("或设置 HUGINN_EMBED_MODEL=BAAI/bge-m3。")

    # 0) 先加载 embedding 模型 (首次会下载权重, 可能很慢).
    #    放在任何破坏性操作之前: 下载失败则数据原封不动, 可安全重跑.
    from sentence_transformers import SentenceTransformer

    print("加载 embedding 模型 (首次会下载, 可能较慢) ...")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"模型维度: {dim}")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    if coll_name not in [c.name for c in client.list_collections()]:
        print(f"collection {coll_name} 不存在, 无需重建。")
        return 0

    coll = client.get_collection(coll_name)
    data = coll.get(include=["documents", "metadatas"])
    ids: list[str] = list(data["ids"])
    docs: list[str] = [d or "" for d in data["documents"] or []]
    metas: list[dict] = [m or {} for m in data["metadatas"] or []]
    n_old = len(ids)
    print(f"旧 collection: {n_old} 条")

    if force or not ids:
        confirm = "yes"
    else:
        confirm = input(f"将删除并重建 collection ({n_old} 条, 先已备份)。输入 yes 继续: ").strip()
    if confirm.lower() != "yes":
        print("已取消。")
        return 1

    # 1) 备份
    bak = chroma_dir.with_name(f"{chroma_dir.name}.{int(time.time())}.bak")
    print(f"备份 -> {bak}")
    shutil.copytree(chroma_dir, bak)

    # 2) 读出源文本 (已在 data) 3) 删 collection
    client.delete_collection(coll_name)
    print("旧 collection 已删除")

    # 4) 用前面已加载的新模型重灌
    new_coll = client.get_or_create_collection(coll_name)
    BATCH = 64
    for i in range(0, n_old, BATCH):
        chunk_ids = ids[i : i + BATCH]
        chunk_docs = docs[i : i + BATCH]
        chunk_metas = metas[i : i + BATCH]
        embs = model.encode(chunk_docs, normalize_embeddings=True).tolist()
        new_coll.add(
            ids=chunk_ids,
            documents=chunk_docs,
            metadatas=chunk_metas,
            embeddings=embs,
        )
        print(f"  重灌 {min(i + BATCH, n_old)}/{n_old}", flush=True)

    n_new = new_coll.count()
    print(f"新 collection: {n_new} 条, 维度 {dim}, 条目{'一致' if n_new == n_old else f'不一致(旧 {n_old})'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())