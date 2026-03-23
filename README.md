# akari-mem-mcp

阿卡林的个人记忆系统 — **BGE-M3 向量 + FTS5 关键字 + RRF 融合 + Rerank 重排** 四级搜索管线。

支持 MCP Server (Claude Code / Cherry Studio) 和 Antigravity Skill 两种接入方式。

## 架构

```
┌───────────────┐  stdio MCP    ┌──────────────────────────────────────┐
│  Claude Code  ├──────────────►│  server.py (MCP Server)              │
│  Cherry Studio│               │                                      │
└───────────────┘               │  ┌──────────┐    ┌────────────────┐  │
                                │  │ SQLite    │    │ ChromaDB       │  │
┌───────────────┐  Python CLI   │  │ + FTS5    │    │ BGE-M3 (1024d) │  │
│  Antigravity  ├──────────────►│  └──────────┘    └────────────────┘  │
│  (Skill)      │  search.py    │                                      │
└───────────────┘               │  ┌──────────────────────────────┐    │
                                │  │  Hybrid Pipeline              │    │
┌───────────────┐  HTTP :37800  │  │  Vector → FTS5 → RRF → Rerank│    │
│  任意 HTTP    ├──────────────►│  └──────────────────────────────┘    │
│  客户端       │  api_server   └──────────────────────────────────────┘
└───────────────┘
```

## 双搜索策略

| 模式 | 引擎 | 速度 | 精度 | 适用场景 |
|:----:|------|:----:|:----:|----------|
| **Quick** | SQLite FTS5 | ~5ms | ★★★ | 精确关键字、名称 |
| **Deep** | BGE-M3 + FTS5 + RRF + Rerank | ~3s | ★★★★★ | 语义理解、模糊查询 |

**Quick Search** 直接查询 SQLite FTS5 索引，毫秒级返回，无需加载任何模型。

**Deep Search** 走完整管线：
1. **Vector Recall** — BGE-M3 向量召回 (1024维)
2. **Keyword Recall** — FTS5 关键字召回
3. **RRF Fusion** — Reciprocal Rank Fusion 两路结果融合
4. **Rerank** — BGE-Reranker-v2-M3 精排重排序

## 快速开始

### 方式一：MCP Server (Claude Code / Cherry Studio)

```bash
# 启动 MCP server（stdio 模式，模型懒加载）
python server.py

# 从 claude-mem 迁移数据
python server.py --migrate

# 重建向量索引（更换 embedding 模型后）
python server.py --rebuild

# 测试搜索
python server.py --test
```

#### Claude Code 配置

在 `~/.claude/settings.local.json` 的 `mcpServers` 中添加：

```json
{
  "akari-mem": {
    "command": "C:\\Users\\Akari\\scoop\\apps\\pyenv\\current\\pyenv-win\\versions\\3.10.5\\python.exe",
    "args": ["F:\\claude-tools\\akari-mem-mcp\\server.py"],
    "env": {
      "HF_HOME": "F:\\models",
      "HF_ENDPOINT": "https://hf-mirror.com"
    }
  }
}
```

#### Cherry Studio 配置

```json
{
  "akari-mem": {
    "command": "python",
    "args": ["F:\\claude-tools\\akari-mem-mcp\\server.py"],
    "env": { "PYTHONPATH": "F:\\python-libs" }
  }
}
```

### 方式二：Antigravity Skill（推荐）

无需常驻服务，纯按需启动。脚本位于 `~/.gemini/antigravity/skills/akari-mem/scripts/`

```bash
# 快速关键字搜索（毫秒级）
python search_memory.py "关键词" 5

# 语义深搜（首次~20s加载模型，后续~3s）
python search_memory.py "复杂查询" 3 --deep

# 列出最近记忆
python search_memory.py "*" 10

# 保存记忆
python save_memory.py "标题" "内容" "标签" "项目"
```

### 方式三：HTTP API

```bash
# 启动 API server
python api_server.py  # 端口 37800

# 调用
curl http://127.0.0.1:37800/search?q=关键词&limit=5    # 快搜
curl http://127.0.0.1:37800/deep?q=语义查询&limit=3     # 深搜
curl http://127.0.0.1:37800/list?limit=10               # 列表
curl http://127.0.0.1:37800/stats                       # 统计
curl -X POST http://127.0.0.1:37800/save \
  -d '{"title":"标题","text":"内容","tags":"tag1","project":"proj"}'
```

## MCP Tools

| Tool | 说明 | 模型加载 |
|------|------|:--------:|
| `quick_search(query, limit)` | FTS5 关键字搜索 | ❌ 不需要 |
| `search_memory(query, limit)` | 混合语义搜索 | ✅ 首次懒加载 |
| `save_memory(title, text, tags, project)` | 保存记忆 | ✅ 首次懒加载 |
| `list_memories(limit)` | 列出最近记忆 | ❌ 不需要 |
| `delete_memory(memory_id)` | 删除记忆 | ❌ 不需要 |
| `get_stats()` | 查看统计信息 | ❌ 不需要 |

> 标记为"不需要"的工具直接操作 SQLite，**秒响应**，不触发模型加载。

## 项目文件

```
akari-mem-mcp/
├── server.py          # MCP Server（stdio模式，模型懒加载）
├── api_server.py      # HTTP API Server（端口37800）
├── store.py           # MemoryStore 核心（SQLite + ChromaDB + 混合搜索管线）
├── embeddings.py      # Embedding Provider（local/api/default 三模式）
├── rerank.py          # Reranker Provider（BGE-Reranker-v2-M3）
├── migrate.py         # claude-mem → akari-mem 数据迁移
├── config.json        # 配置文件
├── hooks/             # Claude Code 生命周期 Hook
│   ├── session-start.py   # 会话启动时自动加载最近记忆
│   └── user-prompt.py     # 用户输入时自动关键字检索
├── benchmark.py       # 搜索质量基准测试
├── test_mcp.py        # MCP 协议握手测试
└── data/              # 数据目录
    ├── akari-mem.db       # SQLite 主数据库 + FTS5 索引
    └── chroma/            # ChromaDB 向量数据库
```

## Embedding 模式

编辑 `config.json` 的 `embedding` 字段切换模式：

| 模式 | 配置 | 维度 | 说明 |
|------|------|:----:|------|
| **Local** | `{"mode":"local","model":"BAAI/bge-m3"}` | 1024 | SOTA 多语言，推荐 |
| **API** | `{"mode":"api","url":"...","key":"sk-..."}` | 可配 | OpenAI 兼容 API |
| **Default** | `{"mode":"default"}` | 384 | ChromaDB 内置 MiniLM |

> ⚠️ 切换模型后需运行 `python server.py --rebuild` 重建向量索引。

## Reranker

```json
{
  "rerank": {
    "mode": "local",
    "model": "BAAI/bge-reranker-v2-m3"
  }
}
```

重排器在 Deep Search 的最后一步对召回结果精排，显著提升搜索质量。基准测试命中率 **100%**。

## 环境要求

- **Python 3.10+**（3.10.5 推荐，3.14 不兼容 sentence-transformers）
- `sentence-transformers` — BGE-M3 Embedding
- `chromadb` — 向量数据库
- `torch` — PyTorch 推理
- `mcp` — MCP Server SDK

## 资源占用

| 状态 | RAM | 说明 |
|------|:---:|------|
| Quick Search / List / Stats | ~0MB | 即用即走，纯 SQLite |
| MCP Server 空闲（懒加载前） | ~25MB | 仅 stdlib http 服务 |
| Deep Search 运行时 | ~5.2GB | BGE-M3 + Reranker 模型 |

## License

MIT
