# 🚀 Akari-Mem-MCP 启动指南 ฅ'ω'ฅ

欢迎使用 **Akari-Mem-MCP**！这是一个集成了 **BGE-M3 向量搜索** 和 **SQLite FTS5 关键字搜索** 的强力记忆系统。

本指南将带你从零开始完成环境搭建与服务启动。主人加油喵！(๑•̀ㅂ•́)✧

---

## 1. 🐍 环境准备

### 基础要求
- **Python 3.10+** (推荐 3.10.x 或 3.11.x，**注意：** 3.14 目前与 `sentence-transformers` 不兼容)。
- 良好的网络环境（首次运行会从 HuggingFace 下载模型）。

### 创建虚拟环境 (推荐)
```bash
# 进入项目根目录
cd f:/claude-tools/akari-mem-mcp

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
.\venv\Scripts\activate
```

---

## 2. 📦 安装依赖

建议根据你的需求选择不同的安装方式：

### 核心安装 (仅 MCP 服务器)
```bash
pip install .
```

### 完整安装 (包含本地向量模型支持 - 推荐 ✨)
如果你打算在本地跑 BGE-M3 模型实现“深搜”，请运行：
```bash
pip install .[local]
```

---

## 3. ⚙️ 配置说明

### 3.1 环境变量 (.env)
将根目录下的 `.env.example` 复制为 `.env`：
```bash
cp .env.example .env
```

**必看配置项：**
- `AKARI_MEM_DATA_DIR`: 数据存储路径（默认 `./data`）。
- `HF_ENDPOINT`: 如果你在国内，建议取消注释并设为 `https://hf-mirror.com`。
- `HF_HOME`: 指定模型下载存放路径。

### 3.2 向量模型 (config.json)
默认配置已经开启了 **local** 模式使用 **BGE-M3**。
- 如果想节省内存/显存，可以考虑在 `config.json` 中切换为 `mode: "api"`（需 OpenAI 格式接口）。

---

## 4. 🔍 初始化与验证

首次启动建议运行一次测试，这会触发模型自动下载（可能需要几分钟）：

```bash
python server.py --test
```

如果你更换了 Embedding 模型，或者想要强制重建向量库：
```bash
python server.py --rebuild
```

---

## 5. 🛠️ 启动方式

### 方案 A：作为 MCP Server (推荐给 Claude Code)
直接在终端测试：
```bash
python server.py
```
*在 IDE/Claude 中配置时，请确保使用 `python.exe` 的完整路径。*

### 方案 B：作为 HTTP API
如果你有自己的脚本或 Web 界面想调用：
```bash
python api_server.py
```
默认端口：**37800**。

### 方案 C：命令行工具 (Skill CLI)
纯 CLI 检索：
```bash
python search_memory.py "你想搜的内容"
```

---

## 💡 浮浮酱的温馨提示
- **首次“深搜”：** 第一次进行 `deep_search` 时会加载 5.2GB 左右的模型到内存，请确保内存充足喵～
- **快速检索：** `quick_search` 不需要加载模型，毫秒级响应，适合简单关键词搜索。
- **遇到问题？** 运行 `python diagnose.py` 查看系统诊断报告喵！ฅ'ω'ฅ

祝主人玩得开心喵！(✿◡‿◡)
