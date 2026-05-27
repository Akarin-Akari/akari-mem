# akari-mem-mcp

阿卡林的个人记忆系统 — **BGE-M3 向量 + FTS5 关键字 + RRF 融合 + Rerank 重排** 四级搜索管线。

**Daemon + Thin Client** 架构，ONNX GPU 推理加速，闲置自动卸载 VRAM，通过 MCP 协议对接 Claude Code / Antigravity 等主流 Agent 平台。

## 核心特性

- 🔍 **四级检索管线**：BGE-M3 向量召回 → FTS5 关键字召回(jieba 中文分词) → RRF 融合排序 → BGE-Reranker-v2-M3 精排
- 🏗️ **Daemon + Thin Client 架构**：`api_server.py` 作为常驻后台 Daemon，`server.py` (MCP) 作为轻量 Thin Client 按需转发
- ⚡ **ONNX GPU 推理**：BGE-M3 FP16 量化 + CUDA Execution Provider + tight memory arena，替代 PyTorch 显著降低显存占用
- 🧹 **闲置 VRAM 管理**：IdleUnloadManager 5 分钟无请求自动卸载模型释放 VRAM，30 分钟无活动 Daemon 自我退出
- 🔒 **进程单例保障**：Socket 端口抢占 + `pythonw.exe` 无窗口启动 + `DETACHED_PROCESS` 三重保险
- 🛡️ **ctypes DLL 预加载**：主动加载 20+ CUDA DLL，防止 ONNX Runtime silent fallback 到 CPU
- 📡 **MCP 协议 + HTTP API 双模式**：stdio (Claude Code) + HTTP :37800 (Antigravity / 任意客户端)

## 架构

```
                                    ┌─────────────────────────────────────────────────────┐
┌───────────────┐  stdio MCP        │  server.py (MCP Thin Client)                        │
│  Claude Code  ├──────────────────►│    ↓ HTTP forward to Daemon                         │
│  Cherry Studio│                   └─────────────────┬───────────────────────────────────┘
└───────────────┘                                     │
                                                      ▼
┌───────────────┐  HTTP :37800      ┌─────────────────────────────────────────────────────┐
│  Antigravity  ├──────────────────►│  api_server.py (Daemon 主进程)                       │
│  任意 HTTP    │                   │                                                      │
└───────────────┘                   │  ┌──────────────┐   ┌─────────────────────────┐      │
                                    │  │ SQLite + FTS5│   │ ChromaDB (BGE-M3 1024d) │      │
                                    │  │ + jieba 分词 │   │ ONNX Runtime CUDA EP    │      │
                                    │  └──────────────┘   └─────────────────────────┘      │
                                    │                                                      │
                                    │  ┌──────────────────────────────────────────────┐    │
                                    │  │  四级混合检索管线                               │    │
                                    │  │  Vector → FTS5 → RRF Fusion → Reranker       │    │
                                    │  └──────────────────────────────────────────────┘    │
                                    │                                                      │
                                    │  ┌──────────────────────────────────────────────┐    │
                                    │  │  IdleUnloadManager                            │    │
                                    │  │  5min 无请求 → 卸载模型释放 VRAM               │    │
                                    │  │  30min 无活动 → Daemon 自我退出                │    │
                                    │  └──────────────────────────────────────────────┘    │
                                    └─────────────────────────────────────────────────────┘
```

## 双搜索策略

| 模式 | 引擎 | 速度 | 精度 | 适用场景 |
|:----:|------|:----:|:----:|----------|
| **Quick** | SQLite FTS5 (jieba 预分词) | ~5ms | ★★★ | 精确关键字、名称匹配 |
| **Deep** | BGE-M3 + FTS5 + RRF + Rerank | ~1-3s | ★★★★★ | 语义理解、模糊查询 |

**Quick Search** 直接查询 SQLite FTS5 索引，毫秒级返回，无需加载任何模型。

**Deep Search** 走完整四级管线：
1. **Vector Recall** — BGE-M3 ONNX 向量召回 (1024 维, FP16)
2. **Keyword Recall** — FTS5 关键字召回 (jieba 应用层预分词, 无需 C 扩展)
3. **RRF Fusion** — Reciprocal Rank Fusion 两路结果融合排序
4. **Rerank** — BGE-Reranker-v2-M3 精排重排序

## 快速开始

### 方式一：MCP Server (Claude Code / Cherry Studio)

```bash
# 启动 MCP server（stdio 模式，自动拉起 Daemon）
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

### 方式二：HTTP API (Daemon 自动启动)

```bash
# 直接启动 Daemon（pythonw.exe 无窗口后台运行）
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
| `search_memory(query, limit)` | 四级混合语义搜索 | ✅ 首次懒加载 |
| `save_memory(title, text, tags, project)` | 保存记忆 (自动嵌入) | ✅ 首次懒加载 |
| `get_memory(memory_id)` | 获取完整记忆内容 | ❌ 不需要 |
| `update_memory(memory_id, ...)` | 部分更新记忆 | ✅ 重新嵌入 |
| `list_memories(limit)` | 列出最近记忆 | ❌ 不需要 |
| `delete_memory(memory_id)` | 删除记忆 | ❌ 不需要 |
| `get_stats()` | 查看统计信息 | ❌ 不需要 |

> 标记为"不需要"的工具直接操作 SQLite，**秒响应**，不触发模型加载。

## ONNX GPU 推理优化

项目已从 PyTorch + sentence-transformers 迁移至 **ONNX Runtime**，显著降低显存占用和启动时间：

| 优化项 | 说明 |
|--------|------|
| **CUDA Execution Provider** | GPU 推理，自动 fallback CPU |
| **FP16 量化** | BGE-M3 模型体积 & 显存减半 |
| **Tight Memory Arena** | `arena_extend_strategy=kSameAsRequested`，按需分配显存 |
| **GC 强制释放** | 推理后 `gc.collect()` + `torch` 残留清理，防 OOM |
| **ctypes DLL 预加载** | 主动加载 20+ CUDA DLL，防 ORT silent fallback CPU |
| **Idle Unload** | 5 分钟闲置自动卸载 Embedding + Reranker 模型释放 VRAM |

## 项目文件

```
akari-mem-mcp/
├── server.py          # MCP Server (Thin Client, stdio 模式)
├── daemon_client.py   # Daemon 启动器 (pythonw.exe spawn + Socket 单例)
├── api_server.py      # HTTP API Daemon (端口 37800, IdleUnloadManager)
├── store.py           # MemoryStore 核心 (SQLite + ChromaDB + 四级管线)
├── embeddings.py      # ONNX Embedding Provider (CUDA EP + FP16 + arena)
├── rerank.py          # ONNX Reranker Provider (BGE-Reranker-v2-M3)
├── tokenizer.py       # jieba 中文分词 (应用层预分词 for FTS5)
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

## Embedding 配置

编辑 `config.json` 的 `embedding` 字段切换模式：

| 模式 | 配置 | 维度 | 说明 |
|------|------|:----:|------|
| **Local (ONNX)** | `{"mode":"local","model":"BAAI/bge-m3"}` | 1024 | ONNX GPU 推理，推荐 |
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

重排器在 Deep Search 的最后一步对召回结果精排，显著提升搜索质量。

## 环境要求

- **Python 3.10+**（3.10.5 推荐）
- `onnxruntime-gpu` — ONNX Runtime GPU 推理 (CUDA 12)
- `chromadb` — 向量数据库
- `tokenizers` — HuggingFace 分词器
- `jieba` — 中文分词
- `mcp` — MCP Server SDK
- `aiohttp` / `aiosqlite` — 异步 HTTP/SQLite

## 资源占用

| 状态 | RAM | VRAM | 说明 |
|------|:---:|:----:|------|
| Quick Search / List / Stats | ~0MB | 0 | 即用即走，纯 SQLite |
| Daemon 空闲（模型卸载后） | ~30MB | 0 | IdleUnload 已释放 |
| Deep Search 运行时 | ~1.2GB | ~2.4GB | ONNX BGE-M3 + Reranker (FP16) |

## License

MIT
