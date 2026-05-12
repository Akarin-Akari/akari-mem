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
| Reranker（jina-reranker-v2） | **fastembed** | **CPU** |
| FTS5 中文分词 | **jieba** 预分词 | CPU |
| 向量存储 | **ChromaDB**（PersistentClient） | 本地磁盘 |
| 元数据 | **SQLite** | 本地磁盘 |

**禁止改动：**
- ❌ 不要把 reranker 改成 GPU
- ❌ 不要把 embedding 改回 sentence-transformers / fastembed-GPU
- ❌ 不要换掉 ONNX 后端（如改回 sentence-transformers）
- ❌ 不要在 embedding/rerank 里加内部锁（之前导致死锁）
- ❌ 不要让 search_memory 被并行调用（之前导致显存爆炸）

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
| `server.py` | MCP 入口，包含 CUDA 预加载 + fd 2 重定向 + HF 离线模式 |
| `embeddings.py` | OnnxEmbeddingProvider（ORT + tokenizers.from_file） |
| `rerank.py` | FastEmbedReranker（CPU 模式） |
| `store.py` | MemoryStore（SQLite + ChromaDB） |
| `tokenizer.py` | FTS5 jieba 中文分词（独立路径，不受 BGE-M3 影响） |
| `data/warmup.log` | 启动 sub-step trace（独立于 stderr） |
| `data/stderr.log` | 重定向后的 OS 级 stderr（ORT 原生输出） |
| `F:/models/bge-m3-onnx-fp16/onnx/` | 自导出 BGE-M3 FP16 ONNX + tokenizer.json |
| `F:/models/bge-m3-onnx-fp16/load_stderr.log` | InferenceSession 创建时的 ORT 错误日志 |

## 调试速查

| 症状 | 真因 | 修复 |
|------|------|------|
| RAM 爆 + VRAM 不动 + 召回 50s+ | CUDA EP silent fallback CPU | 检查 server.py 头部 ctypes 预加载是否完整 |
| `Error 126: cublasLt64_12.dll missing` | torch/lib 未预加载到进程 | 同上 |
| `SimplifiedLayerNormFusion` crash | ORT 1.23 bug | 确认 graph_optimization_level=ORT_ENABLE_EXTENDED |
| Step 3 卡 5-30 分钟 | AutoTokenizer 在 MCP 下阻塞 | 确认用了 tokenizers.Tokenizer.from_file |
| Warmup 完全无响应 | fd 2 阻塞 MCP 管道 | 确认 server.py 头部 os.dup2 重定向在位 |

## 性能基准（修复全部 patch 后预期）

| 指标 | 目标 |
|------|------|
| 首次冷启动到模型 ready | < 5 秒 |
| 首次 search_memory | 10-15 秒（含 reranker 冷启动 + CUDA kernel JIT） |
| 后续 search_memory | 1-3 秒 |
| VRAM 占用 | 1.5-2 GB（embedding GPU）+ 0 GB（reranker CPU） |
| RAM 占用 | < 2 GB |
