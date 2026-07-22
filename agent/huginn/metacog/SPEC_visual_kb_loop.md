# Spec: 视觉→知识库→PMK 闭环 (G4-G10)

## 状态

| Gap | 状态 | commit | 说明 |
|-----|------|--------|------|
| G4  | ✅ 完成 | 5c429b0 | visual_primitives → RAG KB ingest |
| G5  | ✅ 完成 | 5c429b0 | hippocampus 接 visual_inspect (record+recall) |
| G6  | ✅ 完成 | 5c429b0 | PMK 的 K 查询含 visual_primitives |
| G8  | ✅ 完成 | 5c429b0 | knowledge_distiller 加 distill_visual_lessons |
| G9  | ✅ 完成 | (本 commit) | SE(3) rotate 产物入 hippocampus (补 G5 盲点) |
| G7  | ❌ 不做 | - | 跟 G2+G5 双重重复, 输入语义不匹配 |
| G10 | ❌ 不做 | - | mental_imagery 是 dead code, 合成模板入 RAG 价值低 |

## 闭环架构

```
visual_hook.extract_visual_primitives
    ↓ (G4)
    ├→ RAG KB (add_text, content_type=visual_primitives)
    │       ↓ (G6)
    │       └→ PMK build_pmk_state (attempted 含 Visual 段)
    │               ↓
    │               └→ check_pause_decision (立场检查看到视觉)
    │
    ├→ hippocampus record (G5)
    │       ↓ (G5 recall)
    │       └→ visual_inspect (visual_memory_prior 注入)
    │
    ├→ hippocampus record (G9, SE(3) rotate 产物)
    │       ↓ (G5 recall)
    │       └→ visual_inspect (同上)
    │
    └→ knowledge_distiller (G8)
            ↓
            └→ visual_lesson (蒸馏后回写 RAG KB)
```

## G7 不做理由

1. `trajectory_match` 的输入是 `list[str]` (action 名序列), visual_inspect 的 `description` 是单个自然语言字符串, 语义不匹配
2. G2 已在 `_check_stuck` + `_build_pm_text` (engine.py) 用 trajectory_match, 这是正确用法
3. G5 hippocampus recall 已注入视觉 prior, 跟 G7 目标重叠

## G10 不做理由

1. `mental_imagery_loop` 全库无 caller (grep 验证), 是 dead code
2. sketch 生成的是合成模板图 (lattice/particles/spectrum), 不是真实观察
3. KB 是文本 RAG, 不存图; verify 结果价值低 (verified=True/False)
4. 即便接线 (~30 行), 合成模板入 RAG 对真实材料分析帮助有限

## G9 实现细节

rotate 分支在 `visual_inspect.py:123` 早返回, 绕过末尾的 record 调用.
补一段 record (3-5 行), 让 `point3d_primitives` 也能跨 session recall.

## 已知 ceiling

1. hippocampus recall 用 tfidf, `<point3d>` 和 `<point>` 是不同 token — species 名仍能命中
2. G8 distill_visual_lessons 用关键词频率, 不识别语义相似 (peak/max/min 当 3 个不同词)
3. G6 attempted 字段前 150 字, 长 visual_primitives 会被截断
4. 所有视觉→KB 路径默认 off (HUGINN_USE_HIPPOCAMPUS=1) 或非阻塞, 不影响主线
