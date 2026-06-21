# 多模型观点记忆与知识图谱演进设计

## 背景

当前 `akari-mem-mcp` 的核心数据模型是“一条 `memories.id` 对应一条完整记忆”。这条记录同时承担 SQLite 主键、FTS5 `rowid`、Chroma metadata 里的 `sqlite_id`、chunk 去重依据等职责。

这套模型适合单一观点记忆，但不适合表达“同一件事情由 GPT、Claude、Gemini 分别给出不同看法”。如果继续让不同模型各自写入独立 ID，会出现两个问题：

- 记录之间缺少“同一件事”的显式归组关系。
- 检索结果只能看到分散条目，无法稳定汇总多模型观点。

已有 `source` 字段不应直接复用为模型身份。它当前语义更接近写入来源或导入来源，例如 `mcp`、`codex`、`claude-mem:*`。把它改造成 `gpt/claude/gemini` 会混淆“谁写入”和“哪个模型生成观点”。

## 设计目标

- 保持现有一条 memory row 作为最小检索、FTS、向量索引单位。
- 支持多条 memory row 归属于同一个“事件/主题/事实对象”。
- 明确记录模型身份，避免污染现有 `source` 字段。
- 为后续知识图谱预留关系表，而不是一开始引入复杂图数据库。
- 兼容现有 334 条记忆，避免强制迁移成错误的主题分组。

## 推荐模型

采用“三层结构”：

1. `memory_subjects`：一件事情、一个主题、一个实体或一个长期问题。
2. `memories`：某个模型对这个 subject 的一条具体观点，仍然保留当前 row-level 检索模型。
3. `memory_relations`：subject 之间的显式关系，用于后续知识图谱。

### 为什么不把多个模型观点塞进同一条 ID

不推荐把 `text` 改成 JSON 或多字段拼接，因为当前系统的关键路径都假设“一行一份可索引文档”：

- FTS5 的 `rowid` 绑定 `memories.id`。
- Chroma chunk metadata 用 `sqlite_id` 回查主表。
- 深度搜索用 `id` 做 chunk 去重。
- 删除和更新需要用旧 `title/text/tags` 精确同步 FTS5。

如果一条 row 内包含多个模型观点，FTS、Chroma、更新、按模型过滤都会变复杂，且不利于重排器判断单条候选的语义边界。

## 数据库草案

### 新增 `memory_subjects`

```sql
CREATE TABLE IF NOT EXISTS memory_subjects (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    summary     TEXT DEFAULT '',
    project     TEXT DEFAULT '',
    tags        TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
```

说明：

- `id` 建议使用稳定字符串，例如 `subj_...`，方便外部工具引用。
- `summary` 是对同一件事的聚合摘要，不替代具体模型观点。
- `project/tags` 用于 subject 级过滤，memory row 仍保留自身 `project/tags`。

### 扩展 `memories`

```sql
ALTER TABLE memories ADD COLUMN subject_id TEXT DEFAULT '';
ALTER TABLE memories ADD COLUMN model_provider TEXT DEFAULT '';
ALTER TABLE memories ADD COLUMN model_name TEXT DEFAULT '';
ALTER TABLE memories ADD COLUMN perspective_role TEXT DEFAULT 'opinion';

CREATE INDEX IF NOT EXISTS idx_memories_subject_id
ON memories(subject_id);

CREATE INDEX IF NOT EXISTS idx_memories_model
ON memories(model_provider, model_name);
```

字段语义：

- `subject_id`：归属的事情或主题。为空表示历史单条记忆或尚未归组。
- `model_provider`：模型家族，例如 `openai`、`anthropic`、`google`、`local`。
- `model_name`：具体模型，例如 `gpt-5`、`claude-sonnet-4.5`、`gemini-2.5-pro`。
- `perspective_role`：观点类型，先保留简单枚举，如 `opinion`、`summary`、`critique`、`fact`。

`source` 保持原语义：记录来自 MCP、导入器、脚本、手工录入还是某个 agent 通道。

### 新增 `memory_relations`

```sql
CREATE TABLE IF NOT EXISTS memory_relations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    from_subject_id    TEXT NOT NULL,
    to_subject_id      TEXT NOT NULL,
    relation_type      TEXT NOT NULL,
    evidence_memory_id INTEGER,
    weight             REAL DEFAULT 1.0,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    UNIQUE(from_subject_id, to_subject_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_memory_relations_from
ON memory_relations(from_subject_id);

CREATE INDEX IF NOT EXISTS idx_memory_relations_to
ON memory_relations(to_subject_id);
```

关系类型先保持字符串，不急着做过度枚举。第一批可用：

- `same_as`
- `depends_on`
- `contradicts`
- `supports`
- `causes`
- `part_of`
- `related_to`

## Chroma 与 FTS5 同步

Chroma metadata 应扩展：

```python
base_meta = {
    "sqlite_id": mem_id,
    "subject_id": subject_id,
    "model_provider": model_provider,
    "model_name": model_name,
    "title": title[:200],
    "tags": tags,
    "project": project,
    "source": source,
}
```

深度搜索新增可选过滤条件：

- `subject_id`
- `model_provider`
- `model_name`

FTS5 可以先不新增列。当前 `title/text/tags` 足够支持关键词召回；subject 聚合展示通过 SQLite 回查完成。后续如果需要搜索 subject 摘要，再单独增加 `memory_subjects_fts`。

注意：任何会影响 Chroma metadata 的字段更新，都必须重新排队刷新 Chroma chunks。至少包括：

- `title`
- `text`
- `tags`
- `project`
- `subject_id`
- `model_provider`
- `model_name`
- `perspective_role`

## API 与 MCP 工具演进

### 第一阶段：兼容式扩展

扩展现有 `save_memory`：

```python
save_memory(
    title: str,
    text: str,
    tags: str = "",
    project: str = "",
    subject_id: str = "",
    model_provider: str = "",
    model_name: str = "",
    perspective_role: str = "opinion",
)
```

扩展现有 `search_memory` / `quick_search`：

```python
search_memory(
    query: str,
    limit: int = 5,
    project: str = "",
    tags: str = "",
    subject_id: str = "",
    model_provider: str = "",
    model_name: str = "",
)
```

新增工具：

- `create_subject(title, summary="", project="", tags="")`
- `get_subject(subject_id)`：返回 subject 与所有模型观点。
- `list_subject_perspectives(subject_id)`：只列观点，不做摘要。

### 第二阶段：多模型观点聚合

新增工具：

- `save_model_perspective(subject_id, model_provider, model_name, title, text, ...)`
- `compare_subject_perspectives(subject_id)`：按模型分组展示观点差异。
- `summarize_subject(subject_id)`：生成或更新 `memory_subjects.summary`。

这一阶段不需要改变底层向量召回方式；仍然是 row-level 召回，召回后按 `subject_id` 聚合。

### 第三阶段：轻量知识图谱

新增工具：

- `link_subjects(from_subject_id, to_subject_id, relation_type, evidence_memory_id=0, weight=1.0)`
- `unlink_subjects(...)`
- `get_subject_graph(subject_id, depth=1, relation_type="")`

先用 SQLite 邻接表完成图遍历，避免一开始引入 Neo4j 或图数据库运行时。等关系规模、查询模式和可视化需求稳定后，再考虑同步到专用图引擎。

## 迁移策略

1. 对现有 `memories` 增加 nullable/default-empty 字段，不强制给历史数据生成 subject。
2. 新保存的多模型观点必须指定或自动创建 `subject_id`。
3. 提供 `group_memories_as_subject(memory_ids, title, summary="")` 工具，用人工确认方式把旧 ID 归组。
4. 提供 `rebuild_vectors`，在新增 Chroma metadata 后重建向量索引。
5. 提供 `rebuild_fts`，只在 FTS schema 变化时使用；本方案第一阶段不需要改 FTS schema。

## 实施顺序

1. 修复并验证 Chroma metadata 更新一致性。
2. 增加 SQLite schema migration helper，支持 `ALTER TABLE` 幂等补列。
3. 扩展 `MemoryStore.save/search/list/get` 返回和过滤字段。
4. 扩展 daemon HTTP endpoint 与 `daemon_client.py`。
5. 扩展 MCP `server.py` 工具参数和展示格式。
6. 增加 `memory_subjects` 的 CRUD 工具。
7. 增加 subject 聚合查询与多模型观点展示。
8. 增加 `memory_relations` 与轻量图遍历工具。
9. 最后考虑可视化、自动关系抽取、外部图数据库同步。

## 风险与约束

- 不要复用 `source` 表示模型身份，否则会破坏历史导入来源语义。
- 不要让 `subject_id` 替代 `memories.id`，否则 FTS5 和 Chroma chunk 去重会被迫大改。
- 不要第一阶段就做自动实体抽取和自动图谱构建；先让人工/工具显式建 subject 和 relation。
- 更新 metadata 字段后必须刷新 Chroma，否则过滤条件会出现 SQLite 与向量索引不一致。
- 迁移必须保留旧数据可读，不应要求用户一次性清洗 334 条历史记忆。

## 推荐结论

最稳的长期结构是：

- `memories.id`：一条具体记忆或一个模型观点。
- `memory_subjects.id`：同一件事情的聚合 ID。
- `memories.subject_id`：把多个模型观点串起来。
- `memories.model_provider/model_name`：记录观点由哪个模型生成。
- `memory_relations`：后续承载 subject 级知识图谱。

这样既不打乱现有检索架构，又能把“多模型观点”和“知识图谱”自然接上。
