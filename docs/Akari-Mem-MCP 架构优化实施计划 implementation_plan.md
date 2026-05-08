# Akari-Mem-MCP 架构优化实施计划

## 背景
当前 akari-mem-mcp 已稳定运行。本次优化落地提案文档中剩余的 4 项待实现提案（#1 分块、#2 FastEmbed、#4 sqlite-vec、#5 元数据过滤），按优先级逐步实施。

> [!IMPORTANT]
> 本计划为渐进式改造，每一步都保持系统可运行。不做 breaking change，不破坏现有数据。

## Open Questions

1. **FastEmbed 对 bge-reranker-v2-m3 的支持**：`rerank.py` 中的 `LocalReranker` 使用了 `sentence-transformers.CrossEncoder`。如果迁移 embedding 到 fastembed，reranker 仍需保留 `sentence-transformers` 依赖，除非找到 fastembed/onnx 的 reranker 替代方案。是否接受 embedding 用 fastembed + reranker 保留 sentence-transformers 的混合方案？
2. **sqlite-vec 为大手术**：统一存储引擎意味着完全替换 ChromaDB，涉及数据迁移。建议放在最后阶段单独评估。主人是否同意先做 #1 和 #5（低风险），再做 #2（中风险），最后评估 #4（高风险）？
3. **分块阈值**：提案 #1 中，多长的文本才需要分块？建议 text 长度 > 500 字符时才执行分块，短文本直接整体入库。主人觉得这个阈值合适吗？

## Proposed Changes

### Phase 1: 提案 #1 — 语义分块策略 (Chunking)

> 低风险改造，向后兼容。短文本行为不变，长文本自动分块提升召回率。

#### [NEW] [chunker.py](file:///f:/claude-tools/akari-mem-mcp/chunker.py)
- 新建文本分块模块，零外部依赖（纯 Python 实现）
- 实现 `chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]`
- 分块策略：优先按 `\n\n`（段落）切分，超长段落再按句子切分，保留 overlap 以维持上下文连贯
- 返回 chunk 列表，短文本（≤ max_chars）直接返回 `[text]`

#### [MODIFY] [server.py](file:///f:/claude-tools/akari-mem-mcp/server.py)
- 修改 `_index_worker()`：在向 ChromaDB `collection.add()` 之前，调用 `chunk_text()` 对 `document` 进行分块
- 每个 chunk 用 `mem_{id}_chunk_{i}` 作为 ChromaDB ID，metadata 中添加 `chunk_index` 和 `parent_sqlite_id`
- 修改 `save_memory()` 无需改动（SQLite 仍存完整文本）

#### [MODIFY] [store.py](file:///f:/claude-tools/akari-mem-mcp/store.py)
- 修改 `search()` 方法中向量召回部分：命中 chunk 时通过 `parent_sqlite_id` 去重，同一条记忆的多个 chunk 命中只取最高分
- 修改 `rebuild_vectors()`：重建时也要走分块逻辑
- 修改 `delete()` 方法：删除时用 `where={"sqlite_id": memory_id}` 清理所有 chunks

---

### Phase 2: 提案 #5 — 元数据前置过滤

> 极低风险，向后兼容。无过滤条件时行为不变。

#### [MODIFY] [store.py](file:///f:/claude-tools/akari-mem-mcp/store.py)
- 为 `search()` 和 `keyword_search()` 增加可选参数 `project: Optional[str] = None`, `tags: Optional[str] = None`
- 向量召回时构建 ChromaDB `where` 过滤条件
- FTS5 召回时在 SQL WHERE 中追加 `AND project = ?` / `AND tags LIKE ?`

#### [MODIFY] [server.py](file:///f:/claude-tools/akari-mem-mcp/server.py)
- `search_memory` 和 `quick_search` 工具增加可选参数 `project` 和 `tags`，透传到 store 层

---

### Phase 3: 提案 #2 — FastEmbed 轻量化
> ✅ **已完成 (2026-05-08)**: 成功迁移至 FastEmbed，默认使用 `BAAI/bge-small-zh-v1.5` 与 `jinaai/jina-reranker-v2-base-multilingual`，体积与内存消耗大幅下降，本地推理速度显著提升。详见 Git 提交记录。

#### [MODIFY] [embeddings.py](file:///f:/claude-tools/akari-mem-mcp/embeddings.py)
- 新增 `FastEmbedProvider(EmbeddingProvider)` 类，基于 `fastembed` 库
- 懒加载 `fastembed.TextEmbedding`，model = `BAAI/bge-m3`
- `create_provider()` 工厂增加 `mode = "fastembed"` 分支

#### [MODIFY] [pyproject.toml](file:///f:/claude-tools/akari-mem-mcp/pyproject.toml)
- `[project.optional-dependencies]` 中新增 `fastembed = ["fastembed>=0.4.0"]`

#### [MODIFY] [config.json](file:///f:/claude-tools/akari-mem-mcp/config.json)
- embedding mode 从 `"local"` 切换为 `"fastembed"` 来测试

---

### Phase 4: 提案 #4 — sqlite-vec 统一存储 (独立评估)

> 🔍 **预研完成 (2026-05-08)**: 已输出可行性预研报告。详见 `docs/sqlite_vec_feasibility.md`。
> 高风险大手术。完全替换 ChromaDB，需要数据迁移脚本。建议作为独立版本迭代，不在本轮实施。

---

## Verification Plan

### Automated Tests
- 每个 Phase 完成后运行 `python server.py --test` 验证检索质量
- Phase 1 完成后：构造一条长文本记忆（>500 字符），验证分块存入 ChromaDB、搜索命中后正确去重返回完整记录
- Phase 2 完成后：保存带 project/tags 的记忆，验证过滤检索只返回匹配项
- Phase 3 完成后：对比 sentence-transformers 与 fastembed 的检索结果质量（相同 query 的 Top-5 是否一致）

### Manual Verification
- 通过 MCP 客户端实际调用 `save_memory` → `search_memory` → `get_memory` 全链路验证
