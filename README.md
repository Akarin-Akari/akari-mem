# akari-mem-mcp

阿卡林的个人 MCP 记忆服务器 — SQLite + ChromaDB 双引擎，支持 BGE-M3 / API / 内置三种 Embedding 模式。

## 架构

```
MCP Client (Cherry Studio / Antigravity / Claude Code)
  │  stdio JSON-RPC
  ▼
┌─────────────────────────────┐
│  akari-mem-mcp server       │
│  ┌───────┐  ┌────────────┐  │
│  │SQLite │  │ ChromaDB   │  │
│  │+FTS5  │  │ 向量搜索   │  │
│  └───────┘  └────────────┘  │
└─────────────────────────────┘
```

## 快速开始

```bash
# 1. 启动 MCP server（stdio 模式）
python F:\claude-tools\akari-mem-mcp\server.py

# 2. 从 claude-mem 迁移数据
python F:\claude-tools\akari-mem-mcp\server.py --migrate

# 3. 测试搜索
python F:\claude-tools\akari-mem-mcp\server.py --test

# 4. 重建向量索引（更换 embedding 模型后）
python F:\claude-tools\akari-mem-mcp\server.py --rebuild
```

## MCP Tools

| Tool | 说明 |
|------|------|
| `search_memory(query, limit)` | 向量语义搜索 |
| `save_memory(title, text, tags, project)` | 保存记忆 |
| `list_memories(limit)` | 列出最近记忆 |
| `delete_memory(memory_id)` | 删除记忆 |
| `get_stats()` | 查看统计信息 |

## Cherry Studio 接入

在 Cherry Studio 的 MCP 设置中添加：

```json
{
  "akari-mem": {
    "command": "python",
    "args": ["F:\\claude-tools\\akari-mem-mcp\\server.py"],
    "env": {
      "PYTHONPATH": "F:\\python-libs"
    }
  }
}
```

## Embedding 模式

编辑 `config.json` 的 `embedding` 字段：

### 模式 1: 内置 (default)
```json
{"mode": "default"}
```
使用 ChromaDB 内置的 all-MiniLM-L6-v2 (384维)，零配置。

### 模式 2: 本地模型 (local)
```json
{"mode": "local", "model": "BAAI/bge-m3", "cache_dir": "F:/models"}
```
需要安装 `sentence-transformers`。BGE-M3 为 SOTA 多语言模型 (1024维)。

### 模式 3: 线上 API (api)
```json
{
  "mode": "api",
  "url": "https://api.openai.com/v1/embeddings",
  "key": "sk-xxx",
  "model": "text-embedding-3-small",
  "dim": 1536
}
```
支持任意 OpenAI 兼容 API (OpenAI/Azure/Ollama/vLLM 等)。

> ⚠️ 切换 embedding 模式后需要运行 `--rebuild` 重建向量索引。
