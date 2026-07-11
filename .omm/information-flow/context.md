# 信息流 — 关键设计决策

## 1. 为什么 memory/KB/KG 是并列仓库不互流

Memory、KnowledgeBase、KnowledgeGraph 三个仓库各自独立存储，之间没有直接的数据流。唯一的交叉发生在 ContextBuilder.build_kb_text() 中：KB 检索到 chunk 后，取 top-2 chunk 的文本回查 memory，形成 KB↔Memory 的单向交叉引用。这不是仓库互流，而是检索时的交叉引用——数据本身没有从一个仓库搬到另一个。

### 朗兰兹函子性

三个仓库对应三种不同的数学结构：
- **Memory** 是时序+语义结构：按时间记录事件，按语义相似度检索。条目之间没有显式关系，只有隐式的语义距离。
- **KB** 是向量空间结构：文档被切分为 chunk，嵌入到高维向量空间，按余弦相似度检索。chunk 之间没有显式关系。
- **KG** 是图结构：实体是节点，关系是边，支持多跳查询和超边。实体之间有显式的拓扑连接。

如果让仓库互流（比如 memory 写入时自动同步到 KB 和 KG），等于强制三种结构之间做全局翻译。这个翻译不可逆——时序信息进 KG 会丢失时间维度，图结构进 KB 会丢失拓扑信息，向量空间进 memory 会丢失语义聚类。

朗兰兹函子性的原则是：不同数学域之间的翻译只在自然交界处发生，且翻译保持结构。这里的"自然交界处"就是 ContextBuilder 的 prompt 组装——三种结构各自用自己最自然的查询方式产出文本（memory 做语义召回、KB 做向量检索、KG 做图遍历），文本合并进同一个 prompt。翻译发生在文本层，不发生在数据层。

代码中的体现：`_learn()` 对同一次迭代结果分别用三种方式写入——memory 存 hypothesis+r_phys+persona 的时序条目，KB 存实验摘要的可检索文档，KG 存 experiment 实体+hyperedge 的图节点。写入是并行的，但数据格式各不相同，没有统一的中间表示。

KB↔Memory 交叉引用是唯一的例外，因为它发生在检索时而非写入时：KB chunk 文本作为 memory 查询的输入，输出是 memory 文本。这是一个文本到文本的函数调用，不涉及数据层迁移。

## 2. 为什么 surprise 用 worst-case 而非 point estimate

`_compute_surprise_robust()` 对预测文本和实际文本的语义距离做多种扰动估计（不同停用词集、不同最小词长、unigram vs bigram Jaccard），返回 `{point, mean, worst, std}`。决策时取 `worst` 而非 `point` 或 `mean`。

### 分布鲁棒自由能

Surprise 在这里充当 intrinsic motivation 信号：surprise 高意味着 agent 的心智模型（预测）与 reality 偏差大，值得继续探索这个方向。如果 surprise 被低估，agent 会误以为"已经理解了"而跳过值得深挖的方向。

point estimate（单一 Jaccard 距离）的问题是它依赖特定的关键词提取策略。换一组停用词、改一个最小词长阈值，surprise 值可能从 0.3 跳到 0.7。这个方差不是噪声——它反映了"我们不确定预测和实际到底差多少"这个认识论不确定性。

分布鲁棒自由能的思路是：不假设某个特定的扰动分布，而是考虑最坏情况下的自由能上界。在 surprise 的语境下，这意味着：只要存在一种合理的文本比较方式能让 surprise 显著，就认为这个方向值得探索。worst-case 是保守的——它可能把一些"其实不太意外"的方向标记为值得探索（false positive），但不会漏掉真正意外的方向（false negative）。

对 intrinsic motivation 来说，false positive 的代价（多探索一个不太意外的方向）远低于 false negative 的代价（错过一个真正意外的方向）。前者浪费一些计算预算，后者可能让 agent 停留在错误的心智模型里。

`std` 作为副产物提供置信度信号：如果四种估计的方差很大，说明 surprise 本身不确定，决策时可以降权。当前实现没有使用 std 做加权（`_last_surprise` 直接取 worst），但数据已写入 memory 供后续分析。

代码中的体现：`_compute_surprise()` 直接返回 `robust["worst"]`，`_compute_surprise_robust()` 返回完整分布信息。`_pick_hypothesis_persona()` 在 `worst > 0.6` 时切换到 reviewer persona——用最坏情况触发批判性审视，确保即使只有一种比较方式发现了偏差，agent 也会认真对待。

## 3. 为什么 goal 持久化跨 session

GoalStore 把 goal 存在 `$HUGINN_CACHE_DIR/goals.json`，跨 session 持久化。每次 `ContextBuilder.build_goal_text()` 从 GoalStore 读取 active goal 并注入 prompt。AutoloopEngine 每轮 learn 后 `increment_iteration()` 并通过 GoalJudge 检查是否达成。

### 开放流网络

如果 goal 只存在于内存中（session 结束就丢失），系统就是一个封闭流：信息在单次 session 内循环，session 结束后所有上下文归零。下一次 session 需要用户重新描述目标，agent 从头开始。

开放流网络的核心特征是：系统通过与环境的持续信息交换维持非平衡稳态。这里的"环境"不是物理环境，而是用户和跨 session 的持久化层。goal 持久化是系统与"时间环境"的交换通道——它让系统记住"昨天在做什么"，从而在今天继续。

非平衡稳态需要持续的信息交换，这意味着：
- goal 必须在 session 之间存活（持久化到文件）
- 每轮 prompt 必须重新注入 goal（维持 LLM 对目标的感知）
- goal 的状态必须可演化（active → paused → completed，iteration 递增）

三者缺一不可。只有持久化没有注入，LLM 不知道目标存在；只有注入没有持久化，session 切换后目标丢失；只有持久化和注入没有状态演化，系统无法判断何时停止。

pause/resume 是非平衡稳态的直接体现：pause 不是"关闭系统"，而是"暂停信息交换"。goal 状态从 active 变为 paused，但 goal 对象本身、已积累的 memory/KB/KG 全部保留。resume 时信息交换恢复，系统从上次的 iteration 继续。这与热力学中的准静态过程类似——足够慢的 pause/resume 不会破坏系统的序。

代码中的体现：`GoalStore` 的 `_save()` 用临时文件 + `os.replace` 做原子写入，确保 pause/resume 时不会因为写入中断而损坏 goals.json。`build_goal_text()` 注入 "Persistent Goal (iter N)" 让 LLM 看到迭代进度。`GoalJudge` 在每轮 learn 后评估 goal 达成度，形成 goal → execution → judge → goal 的闭环。

## 约束

- **三仓库查询容错** — memory/KB/KG 查询全部包裹在 try-except 中，失败返回空字符串而非中断流程。单个仓库不可用不影响其他仓库的信息回流。
- **surprise 降级** — 当 prediction 或 actual 文本为空时，surprise 直接返回 0，不触发 persona 切换。首轮无预测基线时自动降级。
- **GoalStore 单例** — 模块级 `_store` 单例 + double-checked locking，确保整个进程共享同一个 goal 状态。测试时通过临时路径隔离。
- **KB 写入限流** — 每 10 轮迭代触发一次 `cleanup_old_documents(max_docs=200)`，防止 autoloop 长时间运行导致 KB 无限膨胀。
- **goal iteration 单调递增** — `increment_iteration()` 只增不减，确保进度不会回退。pause/resume 不影响 iteration 值。
