# akari-mem-mcp 项目本地约束

> 本文件覆盖全局 CLAUDE.md / rules，仅在本项目内生效。

## 🔴 CUDA 运行时环境（最重要！踩坑过多次，下次别再走弯路）

**CUDA 12 + cuDNN 9 完整运行时由 PyTorch wheel 自带，已经在 `.venv` 内：**

```
.venv/Lib/site-packages/torch/lib/
├─ cudart64_12.dll               cublas64_12.dll       cublasLt64_12.dll  (643 MB)
├─ cufft64_11.dll  (263 MB)      cufftw64_11.dll       curand64_10.dll
├─ cusparse64_12.dll             cusolver64_11.dll     cusolverMg64_11.dll
├─ nvrtc64_120_0.dll             nvJitLink_120_0.dll   nvrtc-builtins64_128.dll
└─ cudnn{64,_graph,_ops,_heuristic,_engines_precompiled,_engines_runtime_compiled,_cnn,_adv}64_9.dll
```

**绝对禁止：**
- ❌ 不要建议主人「装系统 cuDNN」、「下载 cuDNN tarball」、「装 CUDA Toolkit」
- ❌ 不要折腾 `C:\Program Files\NVIDIA GPU Computing Toolkit` 路径
- ❌ 不要 `setx PATH ...` 加 CUDA 路径（会 1024 字符截断破坏 PATH）
- ❌ 不要卸载/重装 `torch` —— `torch-2.11.0+cu128` 是 GPU 运行时的关键依赖，即使代码不直接 import torch
- ❌ 不要把 `torch-2.11.0+cu128` 换成 CPU 版（会丢失全套 CUDA + cuDNN）

**onnxruntime-gpu 1.23.2 的诡异依赖混合：**
- cublas/cudart/cusparse 是 CUDA **12**
- cufft/cusolver 是 CUDA **11**
- cuDNN 必须是 **9.x**
- 这套组合恰好被 PyTorch 2.11+cu128 完全覆盖，**所以 venv 自带就够了，不需要任何系统级安装**。

## 🔴 Thin Client + Daemon 架构（2026-05-13 后，解决多 Agent 显存爆炸）

**问题背景：** MCP stdio 协议每个 Claude Code agent 启动一个独立子进程，原 `server.py` 每个实例都加载完整 BGE-M3 + jina-reranker（~3.5GB VRAM），3+ agent 同时跑会爆 5070 的 12GB 显存。

**架构方案：**
```
Agent 1 → server.py (thin client) ─┐
Agent 2 → server.py (thin client) ─┼─→ HTTP 127.0.0.1:37800 → api_server.py (daemon 单例)
Agent 3 → server.py (thin client) ─┘                                  ↓
                                                          BGE-M3 + jina-reranker
                                                          （全局只加载一次！VRAM 共享）
```

**三个核心文件：**
| 文件 | 角色 | 说明 |
|------|------|------|
| `api_server.py` | **Daemon 主进程** | 唯一加载模型, 提供 HTTP API, Windows DETACHED_PROCESS 静默后台运行 |
| `daemon_client.py` | **Thin Client HTTP 库** | 封装 daemon spawn + HTTP 调用, urllib stdlib 实现 |
| `server.py` | **MCP Thin Client** | 每个 Claude Code agent 一个, 不加载模型, 全部 tool 转发给 daemon |

**单例机制：** 用 socket bind 抢占法（不用 file lock）。多个 thin client 同时尝试 spawn 也安全——第一个 bind 到 port 37800 的 daemon 存活，其他 spawn 进程检测到端口占用立即 `sys.exit(0)`（见 `api_server.py:main`）。

**Daemon 启动方式（关键！）：** 必须用 **`pythonw.exe`**（GUI 子系统，无控制台窗口），不是 `python.exe`。即使配合 `DETACHED_PROCESS | CREATE_NO_WINDOW`，`python.exe`（控制台子系统）仍可能留下一个空白终端窗口。`daemon_client._get_daemon_python_exe()` 自动从 venv 找 `pythonw.exe`。

**绝对禁止：**
- ❌ 不要在 `server.py` 里恢复模型加载逻辑（它必须是 thin client）
- ❌ 不要用 `sys.executable`（python.exe）spawn daemon——会有空白窗口
- ❌ 不要在 daemon spawn 时省掉 `STARTUPINFO.SW_HIDE`——四重保险一个都不能少
- ❌ 不要把 daemon 改成监听 0.0.0.0——必须绑 127.0.0.1（本地单机用）

## 🔴 测试 MCP 工具时的数据安全规则

**测试 MCP 功能时，禁止篡改、破坏、删除任何已有的记忆条目。** 正确的测试流程：
1. 先用 `save_memory` 新建一条测试专用记忆（建议 project 设为 `akari-mem-dev`、tags 包含 `test`）
2. 只对这条新建的测试记忆执行 update / delete 等操作验证
3. 测试完成后清理测试数据

**绝对禁止：**
- ❌ 不要对已有的正式记忆执行 `update_memory` 来"试一下功能"
- ❌ 不要对已有的正式记忆执行 `delete_memory` 来"清理测试"
- ❌ 不要修改已有记忆的 tags/project/text 来"顺手整理"——除非主人明确要求

**Daemon 生命周期：**
- 启动：thin client 首次启动时按需 spawn（auto_spawn=true）
- 模型加载：daemon 后台 warmup 线程异步加载（与 HTTP 服务并行）
- 闲置卸载：5 分钟无 search/save → 卸载模型释放 VRAM（`idle_unload.timeout_seconds=600`）
- 自我退出：30 分钟无任何 HTTP 请求 → daemon 进程自我退出（`daemon.idle_shutdown_seconds=1800`）
- 用户手动停：`POST http://127.0.0.1:37800/shutdown`

**新增/扩展的 HTTP endpoint：**
| Endpoint | 方法 | 说明 |
|----------|------|------|
| `/health` | GET | 含 warmup 状态 + models_loaded |
| `/search` | GET | 新增 `project` `tags` filter |
| `/deep` | GET | 新增 `project` `tags` filter，深度搜索串行化 |
| `/memory?id=N` | GET | **新增** 拿单条完整内容 |
| `/save` | POST | 改为 SQLite 即时 + ChromaDB 异步排队（不阻塞） |
| `/update` | PUT | **新增** 部分更新 (SQLite+FTS5 同步 + ChromaDB 异步) |
| `/delete?id=N` | DELETE | 同步删 SQLite + FTS5 + ChromaDB |
| `/shutdown` | POST | **新增** 优雅停机 |

**配置（config.json 的 `daemon` section）：**
```json
"daemon": {
    "host": "127.0.0.1",
    "port": 37800,
    "spawn_timeout_seconds": 60,
    "request_timeout_seconds": 300,
    "idle_shutdown_seconds": 1800
}
```

**调试时定位 daemon：**
- `data/daemon.pid` — 当前 daemon 的 PID + 端口 + 启动时间戳
- `data/daemon-stderr.log` — daemon 的 stderr（含 ORT 原生输出）
- `data/daemon-stdout.log` — daemon 的 stdout
- `data/warmup.log` — 模型加载子步骤 trace（thin client + daemon 共用同一个文件）

## 🔴 server.py 头部的 ctypes DLL 预加载（关键启动逻辑，不要乱改）

`server.py` 在 `import onnxruntime` 之前，用 `ctypes.WinDLL()` **主动按依赖顺序预加载** `torch/lib/` 下的 20 个 CUDA/cuDNN DLL。

**为什么必须 ctypes 预加载 ≠ `os.add_dll_directory`：**
- `add_dll_directory` 只对 Python 显式 LoadLibrary 起作用
- ORT 的 native loader 加载 `onnxruntime_providers_cuda.dll` 时，它的 transitive dependency（cublasLt/cufft/cudnn）按 Windows 默认 DLL search order 找，**不查 user dirs** → silent fallback CPU
- `ctypes.WinDLL(abs_path)` 把 DLL pin 进进程地址空间，ORT 后续按 base name 找时直接复用已加载模块

**历史教训：以前能跑是因为 reranker 是 sentence-transformers/torch，`import torch` 时 PyTorch 自动预加载了它的 CUDA DLL，ORT 顺带白嫖。改成 fastembed-CPU 后 torch 不再被 import，CUDA 就 silent dropout 了。**

## 🔴 架构约束（用户明确指令，不要改）

| 组件 | 后端 | 设备 |
|------|------|------|
| Embedding（BGE-M3） | **ONNX**（自导出 FP16） | **GPU**（CUDA EP） |
| Reranker（jina-reranker-v2） | **自定义 ONNX**（绕过 fastembed 拿回 SessionOptions 控制权） | **GPU**（CUDA EP，gpu_mem_limit=1GB tight arena） |
| FTS5 中文分词 | **jieba** 预分词 | CPU |
| 向量存储 | **ChromaDB**（PersistentClient） | 本地磁盘 |
| 元数据 | **SQLite** | 本地磁盘 |

**架构变更说明（2026-05-13）：**
- 主人解除了「reranker 必须跑 CPU」的旧规则，要求最低显存 + < 10s RAG 召回
- 改动详情：
  - `rerank.py` 新增 `OnnxReranker` 类，直接用 onnxruntime 加载 fastembed 缓存里的 ONNX，套用跟 BGE-M3 完全相同的 tight CUDA arena 配置（`arena_extend_strategy=kSameAsRequested` + `cudnn_conv_use_max_workspace=0` + `cudnn_conv_algo_search=HEURISTIC` + `gpu_mem_limit`）
  - `config.json`：`rerank.mode=onnx` / `device=cuda` / `gpu_mem_limit_gb=1.0` / `max_length=256`
  - `store.py` `fetch_k` 从 `limit*3` 收紧到 `max(limit*2, 10)`，rerank 输入候选 -33%
  - 不再走 `FastEmbedReranker`（它不暴露 SessionOptions，VRAM 不可控）

**架构变更说明（2026-05-14/15）：Reranker 改 CPU + INT8 节省显存**
- 主人 5070 Laptop 仅 8GB 显存，需要给本地 LLM / SD 留空间 → reranker 从 GPU FP16 改为 CPU INT8
- 改动详情：
  - `rerank.py` `OnnxReranker` 新增 `prefer_int8: bool = True` / `cpu_threads: int = 0` 参数，`_find_artifacts` 按 device + prefer_int8 排序选择 `model_int8.onnx` (267MB) > `model_fp16.onnx` (531MB) > `model.onnx` (FP32, ~1.1GB symlink)
  - `_load` CPU 路径 `sess_options.intra_op_num_threads = cpu_threads`（R9-8940HX 用 8，物理核 ÷ 2 避开 SMT 争用），`inter_op_num_threads = 1`，`execution_mode = ORT_SEQUENTIAL`
  - 用 Jina 官方静态量化的 INT8 ONNX（53/1087 ops 量化覆盖 MatMul/Gemm 热点，LayerNorm/Softmax 保留 FP32），从 hf-mirror 下载 (267MB)
  - `config.json`：`rerank.device=cpu` / `prefer_int8=true` / `cpu_threads=8` / **`cache_dir=F:/models/fastembed`**（关键！见下）
  - 性能（首次冷启动 5.37s vs 旧 GPU FP16 ~10s，VRAM 占用从 ~3.5GB 降至 ~2.5GB 仅 BGE-M3）

## ⚠️ Reranker `cache_dir` 隐式注入陷阱（2026-05-15 踩坑）

**症状**：daemon 日志稳定输出 `OnnxReranker ready: variant=fp32`，但磁盘代码、`.pyc`、端到端 `OnnxReranker(...)` 模拟全部返回 `variant=int8`。

**根因**：`env_loader.py:resolve_config()` 第 41-50 行**强行把 `HF_HOME` 注入到 `rerank.cache_dir`**（仅当 config 没有显式 cache_dir 时）。
- `.env` 设置 `HF_HOME=F:/models`（给 BGE-M3 用，正确）
- 但 fastembed cache 在 `F:/models/fastembed/`（多一层！）
- 注入后 `rerank.cache_dir = "F:/models"` → INT8/FP16 文件全 `exists=False` → 凑巧 `F:/models/.../model.onnx` 存在（HF transformers 标准 cache 也下了一份 1.1GB symlink → blob），命中 FP32 → 加载成功无报错 → bug 静默潜伏

**修复**：`config.json` 显式 `"rerank.cache_dir": "F:/models/fastembed"`，触发 env_loader 第 48 行 `if "cache_dir" not in rnk:` 守护跳过自动注入。

**绝对禁止：**
- ❌ 不要从 `config.json` 删除 `rerank.cache_dir` —— 一删 env_loader 立刻把 `HF_HOME=F:/models` 注入回来，bug 复发
- ❌ 不要把 `.env` 的 `HF_HOME` 改成 `F:/models/fastembed` —— 会破坏 BGE-M3 自导出 ONNX 路径解析
- ❌ 不要在 `env_loader.resolve_config()` 移除 cache_dir 注入逻辑而不评估影响 —— `embedding.cache_dir` 依赖它

详细调试历程、根因推理链、长期优化建议见：`docs/2026-05-15 Reranker INT8 加载错位 Bug 调试报告.md`

## 🔴 ONNX SessionOptions 关键参数

```python
sess_options.graph_optimization_level = ORT_ENABLE_EXTENDED  # NOT ORT_ENABLE_ALL
sess_options.enable_mem_pattern = False
sess_options.enable_cpu_mem_arena = False
sess_options.log_severity_level = 3
```

**`ORT_ENABLE_ALL` 会触发 SimplifiedLayerNormFusion bug**（ORT 1.23 Issue #26717 OPEN，PR #26631 OPEN 未合并），在 BGE-M3 FP16 自导出 ONNX 上 crash：
```
Attempting to get index by a name which does not exist:
InsertedPrecisionFreeCast_.../LayerNorm/Constant_output_0
```
**`ORT_ENABLE_EXTENDED` 是官方推荐的 workaround**，吞吐损失 < 5%。

**PyPI `onnxruntime-gpu` 最新版仍为 1.23.2（GitHub release 到了 1.26 但 GPU wheel 没发布），所以升级 ORT 暂时不可行，必须用 EXTENDED workaround。**

## 🔴 Tokenizer 用 `tokenizers.Tokenizer.from_file()` 不要用 `transformers.AutoTokenizer`

`transformers.AutoTokenizer.from_pretrained()` 在 MCP 子进程下会触发 lazy import 链阻塞 6+ 分钟（独立 Python 跑 19 秒，MCP 下 6 分钟，原因疑似 GIL 竞争 + Windows Defender 扫描）。

**改用 Rust 实现的 `tokenizers.Tokenizer.from_file(tokenizer_json)`，毫秒级完成。**

XLM-RoBERTa 配置：
```python
self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
self._tokenizer.enable_truncation(max_length=512)
```

## 🔴 fd 2 重定向（防 MCP stdio 管道阻塞）

`server.py` 在所有 import 之前 `os.dup2(fd, 2)` 把 OS 级 stderr 重定向到 `data/stderr.log`。原因：
- MCP stdio 管道 buffer ~64KB
- 原生库（ORT/cuDNN/TRT）写 fd 2 不走 Python，如果主机不消费会阻塞数十分钟
- 重定向后 Python logging 仍工作，但 native print 进文件不卡

## 关键文件位置

| 文件 | 用途 |
|------|------|
| `api_server.py` | **Daemon 主进程**：CUDA 预加载 + fd 2 重定向 + 模型加载 + HTTP API + 单例机制 |
| `daemon_client.py` | **Thin client HTTP 库**：pythonw.exe spawn + urllib 调用 + 失败 retry |
| `server.py` | **MCP thin client**：fd 2 重定向 + HF 离线模式 + 所有 tool 走 daemon_client |
| `embeddings.py` | OnnxEmbeddingProvider（ORT + tokenizers.from_file） |
| `rerank.py` | `OnnxReranker`（CUDA EP + tight arena）+ `FastEmbedReranker`（legacy fallback） |
| `store.py` | MemoryStore（SQLite + ChromaDB） |
| `tokenizer.py` | FTS5 jieba 中文分词（独立路径，不受 BGE-M3 影响） |
| `data/daemon.pid` | Daemon 当前 PID + 端口 + 启动时间戳（pythonw.exe 进程） |
| `data/daemon-stderr.log` | Daemon stderr（ORT 原生输出） |
| `data/daemon-stdout.log` | Daemon stdout |
| `data/warmup.log` | 启动 sub-step trace（独立于 stderr） |
| `data/stderr.log` | Thin client stderr（应该几乎是空的） |
| `F:/models/bge-m3-onnx-fp16/onnx/` | 自导出 BGE-M3 FP16 ONNX + tokenizer.json |
| `F:/models/bge-m3-onnx-fp16/load_stderr.log` | InferenceSession 创建时的 ORT 错误日志 |
| `F:/models/fastembed/models--jinaai--jina-reranker-v2-base-multilingual/snapshots/*/onnx/` | jina-reranker ONNX（fastembed 下载，OnnxReranker 直读） |
| `api_server.py.backup.before_daemon_upgrade` | 升级前的单体 api_server.py 备份 |
| `server.py.backup.before_thin_client` | 升级前的单体 server.py 备份（含 893 行模型加载逻辑） |

## 调试速查

| 症状 | 真因 | 修复 |
|------|------|------|
| 启动 thin client 时弹出空白终端窗口 | daemon_client 用了 python.exe 而不是 pythonw.exe | 检查 `_get_daemon_python_exe()` 是否在 venv 找到 `pythonw.exe` |
| 多个 agent 都在加载模型 | thin client 仍在加载模型（没走 daemon） | 检查 server.py 是否真的是 thin client 版本（应该 < 400 行） |
| daemon 起不来 | 端口 37800 被占 / pythonw.exe 缺失 / 模型路径错 | 看 `data/daemon-stderr.log` |
| thin client 报 `[daemon error]` | daemon 不可达 / spawn 失败 / HTTP 超时 | 看 `data/daemon-stderr.log` + 确认 `data/daemon.pid` 存在 |
| 第一次 deep search 慢 (8-15s) | daemon warmup 仍在加载模型 | 等 warmup ready (看 `/health` 的 `warmup` 字段) |
| daemon 突然消失 | idle_shutdown_seconds=1800 超时 | 下次 thin client 调用会自动 re-spawn，无需手动干预 |
| RAM 爆 + VRAM 不动 + 召回 50s+ | CUDA EP silent fallback CPU | 检查 api_server.py 头部 ctypes 预加载是否完整 |
| `Error 126: cublasLt64_12.dll missing` | torch/lib 未预加载到进程 | 同上 |
| `SimplifiedLayerNormFusion` crash | ORT 1.23 bug | 确认 graph_optimization_level=ORT_ENABLE_EXTENDED |
| Step 3 卡 5-30 分钟 | AutoTokenizer 在 MCP 下阻塞 | 确认用了 tokenizers.Tokenizer.from_file |
| Warmup 完全无响应 | fd 2 阻塞 MCP 管道 | 确认 api_server.py 头部 os.dup2 重定向在位 |

## 性能基准（thin client + daemon 架构下预期）

| 指标 | 目标 |
|------|------|
| Thin client 启动（daemon 已在跑） | < 200ms（仅 import mcp 库 + health check） |
| Thin client 首次启动（需 spawn daemon） | 5-15 秒（含 daemon 进程启动 + 异步 warmup） |
| Daemon 首次冷启动到模型 ready | < 5 秒 |
| 首次 search_memory（warmup 已完成） | 5-10 秒（含 reranker 冷启动 + CUDA kernel JIT，比之前 28s+ 快 3-5x） |
| 后续 search_memory | 0.5-1.5 秒（embedding GPU + reranker GPU，目标 < 10s 已远超） |
| VRAM 占用（**所有 thin client 共享**） | ~2.5 GB（仅 BGE-M3 加载，reranker idle-unload 后释放） |
| VRAM 占用（search 中） | ~3.5 GB（BGE-M3 ~2.5GB + jina-reranker tight arena ~1GB） |
| RAM 占用（daemon） | < 2 GB |
| RAM 占用（每个 thin client） | < 50 MB（只 import mcp + urllib） |

**多 agent 场景对比（5070 12GB 显存）：**

| Agent 数 | 旧架构（独立模型）| 新架构（共享 daemon） |
|---------|--------------|------------------|
| 1 | 3.5 GB VRAM | 3.5 GB VRAM |
| 3 | ~10.5 GB（接近爆显存） | **3.5 GB**（不变！） |
| 5 | 💥 17.5 GB（必爆） | **3.5 GB**（不变！） |
| 10 | 💥💥 完全无法运行 | **3.5 GB**（依然不变！） |
