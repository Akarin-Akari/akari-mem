# Akari-Mem-MCP 架构优化提案

## 概述
当前系统基于 `SQLite FTS5` (关键字搜索) + `ChromaDB` (语义向量搜索) 构建了双引擎记忆检索服务。通过 RRF (Reciprocal Rank Fusion) 和 BGE-M3 的 Cross-Encoder 实现了极其优秀的检索召回率。整体架构遵循了 KISS 原则，简单高效。

为应对未来可能的**长文本处理**、**高频并发写入**以及进一步**降低本地资源占用**，特提出以下五个维度的架构优化提案（优先级从高到低排列）。

---

### 1. 语义分块策略 (Chunking Strategy)
**【状态：🔴 待实现】**

**【现状】**
当前在 `store.py` 的 `save()` 方法中，将 `title` 和 `text` 直接拼接为一个整体文档（`document = f"{title}\n{text}"`）存入 ChromaDB 并提取向量。

**【瓶颈】**
对于长文，核心语义会被稀释。BGE-M3 虽然支持最大 8192 token，但通常在 512~1024 token 长度内表现出最佳语义聚合度。

**【优化方案】**
- **引入 Text Splitter**：在入库前采用 `RecursiveCharacterTextSplitter` 或基于 Markdown 标题段落的分块机制。
- **关联映射**：将长文切分为若干小块后存入 ChromaDB，在 metadata 中新增 `parent_sqlite_id` 和 `chunk_index`。检索时命中分块后，通过 `parent_sqlite_id` 返回完整的 SQLite 记录原文。

---

### 2. 本地推理引擎轻量化 (Transition to FastEmbed)
**【状态：🔴 待实现】**

**【现状】**
`embeddings.py` 中依赖 `sentence-transformers` 库，并强依赖完整的 PyTorch 运行时来拉起 BGE-M3 本地模型。

**【瓶颈】**
PyTorch 运行时极其臃肿，即使仅用于推理，也会占用大量内存（1~2GB 起步），这对于主打轻量化的本地 MCP 插件来说过重。

**【优化方案】**
- **迁移至 FastEmbed**：替换底层提供者。Qdrant 开源的 `fastembed` 底层基于 ONNX Runtime 且专门针对 CPU 做了量化优化。
- **收益**：体积大幅缩减，且同等硬件下 CPU 推理速度通常可提升 2~5 倍，完美支持 BGE-M3。

---

### 3. 异步写入机制 (Asynchronous Processing)
**【状态：✅ 已完成 — commit `78a8cc0` + `a40b1fc`，2026-03-26/27】**

**【实现概要】**
该优化已在两次 commit 中完整落地，且实现质量超出提案预期：

1. **SQLite 快速路径** (`_save_to_sqlite`)：纯 SQLite 写入 <10ms，调用方立即拿到 `mem_id`。
2. **后台索引队列** (`_index_queue` + `_index_worker`)：通过 `queue.Queue` + 专用消费者守护线程异步处理 ChromaDB 向量化与存储。比提案中提议的 `asyncio.create_task` 或 `ThreadPoolExecutor` 更适合生产环境。
3. **模型预热线程** (`_background_warmup`)：MCP handshake 完成后 0.5s 启动，在后台线程加载 sentence-transformers/ChromaDB，不阻塞 stdio 管道。
4. **stdio 管道防死锁**：在模块顶部预导入 `chromadb`/`sentence_transformers`/`torch`，避免后台线程的 tqdm/protobuf/CUDA 输出填满管道缓冲区。
5. **运维可观测**：`warmup.log` 文件级诊断日志 + `get_stats()` 暴露 `_warmup_state` 和 `_index_queue.qsize()`。

**【遗留小瑕疵 — 可选改进】**

#### 3a. 索引失败无重试机制
`_index_worker` 中如果 `collection.add()` 失败（如 ChromaDB 磁盘满），只打日志后丢弃。该条记忆将永远只存在于 SQLite 中，向量库缺失。
- **建议**：加入有限重试（最多 3 次，指数退避），或引入 `failed_queue` 做死信记录，后续可手动或定时重试。

#### 3b. `store.py` 中 `save()` 仍为同步双写
`server.py` 的 MCP 入口绕过了 `store.save()`，直接调用 `_save_to_sqlite` + `_index_queue`。但 `store.py:MemoryStore.save()` 本身仍是同步双写逻辑。如果有其他入口（如 `api_server.py`）调用 `store.save()`，仍会遇到同步阻塞。
- **建议**：如需多入口支持，考虑将异步逻辑下沉至 `MemoryStore` 层面统一处理。

#### 3c. warmup 与 get_store() 的竞态冗余
两者都可初始化 `_store`。理论上如果 warmup 线程还在 loading 时用户调用了 `search_memory`，`get_store()` 会在锁内再初始化一次。实践中因 0.5s 延迟 + `_store_lock` 互斥几乎不会触发，但可在 `get_store()` 中增加对 `_warmup_state == "loading"` 的等待逻辑来彻底消除。

---

### 4. 存储引擎极简统一 (Consolidate with sqlite-vec)
**【状态：🔴 待实现】**

**【现状】**
系统目前维护了两个异构的存储层：关系型/全文引擎 `SQLite` 和 向量引擎 `ChromaDB`（底层同样使用了 SQLite+Parquet）。

**【瓶颈】**
双写系统带来了分布式状态的一致性维护成本；ChromaDB 对本地小型化应用来说偏重。

**【优化方案】**
- **引入 `sqlite-vec` 插件**：利用原生 SQLite 生态直接支持高效向量存储与近邻检索（ANN）。
- **收益**：将元数据、全文检索倒排索引、高维向量全部收束到单一的 `akari-mem.db` 文件中。完全解决一致性事务问题，实现极致的存储架构 KISS 和便于备份。

---

### 5. 检索元数据前置过滤 (Metadata Pre-filtering)
**【状态：🔴 待实现】**

**【现状】**
执行 `search()` 混合检索时，主要逻辑是对全局向量库做全量 Top-K 召回，再做融合。

**【瓶颈】**
当数据量逐渐变大时，未经过滤的全局扫描不仅耗时，还可能引入语义无关噪音。

**【优化方案】**
- **下推过滤条件**：在有明确查询上下文的场景中（如用户指令涉及特定 `project` 或 `tags`），将过滤条件通过 ChromaDB 的 `where` 参数直接注入向量引擎，在 ANN 计算前缩小候选集，大幅提升速度及召回精度。

---

> 📝 **架构师备注**：建议按照 **分块(Chunking)** → **轻量化引擎(FastEmbed)** → **sqlite-vec 统一存储** → **元数据前置过滤** 的路径优先迭代。异步写入已完成，可转入小瑕疵打磨阶段。
