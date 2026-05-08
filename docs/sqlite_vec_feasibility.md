# Phase 4: SQLite-VEC 替代方案预研报告 (Feasibility Study)

## 1. 背景与目标
在 Akari-Mem-MCP 中，目前使用 `SQLite (FTS5)` 作为主存储与关键字检索引，使用 `ChromaDB` 作为向量检索引擎。虽然实现了高性能的双引擎 RRF 混合检索，但也引入了以下问题：
- **依赖较重**：ChromaDB 引入了大量的间接依赖（如 FastAPI, grpcio 等），对于本地轻量级工具来说过于臃肿。
- **一致性问题**：双写系统（SQLite + ChromaDB）存在分布式事务一致性挑战，目前通过后台异步队列尽量缓解，但仍无法做到完全原子化。
- **备份复杂**：需要同时备份 SQLite 的 `.db` 文件和 ChromaDB 的本地存储目录。

**目标**：评估使用 `sqlite-vec` 扩展替换 `ChromaDB`，将向量存储与检索也收拢到 SQLite 中的可行性。

## 2. 技术可行性分析

### 2.1 什么是 sqlite-vec？
`sqlite-vec` 是一个开源的 SQLite C 扩展，提供极速的向量相似度检索（ANN）和向量存储能力。它可以作为 Python 库通过 `pip install sqlite-vec` 安装，并直接加载到 `sqlite3` 连接中。

### 2.2 核心优势
1. **单一事实来源 (Single Source of Truth)**：彻底消除双写问题。元数据、全文索引（FTS5）和向量索引（vec0）都在同一个 SQLite 事务中完成写入。
2. **极致轻量**：移除 ChromaDB 及其庞大的依赖树，MCP 启动速度和内存占用进一步降低。
3. **备份极简**：用户的整个记忆库（包含向量）只是一个 `akari-mem.db` 文件，方便同步与备份。
4. **无需异步队列**：由于去除了 ChromaDB 本地实例初始化的延迟和插入的开销，可以直接在主线程（或简单的协程）中完成 SQLite 原子插入，极大简化 `store.py` 和 `server.py` 的代码逻辑，移除复杂的 `_index_worker` 守护线程。

### 2.3 潜在挑战与风险
1. **跨平台兼容性**：`sqlite-vec` 是 C 扩展，虽然官方提供了 Windows/macOS/Linux 的 pre-built wheels，但在某些极其特殊的系统环境下可能会遇到编译或加载失败的问题。
2. **数据迁移 (Migration)**：必须编写迁移脚本，将用户现存的 ChromaDB 数据（或依据现有的 SQLite 文本重新生成向量）迁移到 SQLite 的 `vec0` 虚拟表中。
3. **SQL 查询改写**：混合检索逻辑需要完全用 SQL（或在 Python 内存中结合 FTS5 和 vec0 的结果）重写。例如：
   ```sql
   -- 创建 vec0 虚拟表
   CREATE VIRTUAL TABLE mem_vectors USING vec0(
       embedding float[512] -- 匹配 bge-small-zh-v1.5 的维度
   );
   
   -- 向量检索示例
   SELECT rowid, distance 
   FROM mem_vectors 
   WHERE embedding MATCH ? 
   ORDER BY distance 
   LIMIT ?
   ```

## 3. 实施建议 (Implementation Strategy)

鉴于目前系统（经过 Phase 1~3 优化）**已经极其稳定且速度极快**，`sqlite-vec` 的替换属于**高风险的底层重构（Breaking Change）**。

**建议的实施路径**：
1. **暂缓直接合并**：不要在当前的稳定版本主分支上直接实施。
2. **创建 Feature 分支**：在单独的 `feat/sqlite-vec` 分支上进行开发。
3. **重写存储层**：修改 `store.py`，移除 `chromadb` 导入，初始化时加载 `sqlite_vec.load(db)`，创建 `vec0` 虚拟表。
4. **开发迁移工具**：提供一条命令或自动检测脚本，当检测到旧版 ChromaDB 目录时，自动提取向量写入 SQLite，然后清理旧目录。

## 4. 结论
技术上**完全可行**且**极其优雅**，能将 Akari-Mem-MCP 变成一个真正"零重型依赖"的极客级插件。但考虑到迁移成本和当前系统的稳定性，建议由主人决定是否立即启动这台"大手术"。
