# Startup · 配置与启动

> **配置分两类，加载时机完全不同。** `app/config/` 是**运行时配置**（启动时真的被解析、校验），
> `app/services/**/*.yaml` 是**契约规格**（运行时不加载一个字节）。把两者混为一谈，是这个仓库最容易走错的第一步 ——
> 你会花半天找一个「改了没反应」的配置。
>
> ***没有任何热加载。*** 改任何 yaml 都必须重启进程。

`14 份运行时配置` · `21 份契约规格` · `lifespan 组合根` · `强制 SelectorEventLoop`

---

## 📚 本文结构

- 🚀 **启动** —— 四条命令，以及为什么不能用 `uvicorn app.main:app`
- ⚙️ **运行时配置一览** —— 14 份，逐个说清「管什么 + 改了会怎样」
- 🔗 **跨文件不变量** —— 改一个必须回头看另一个
- 🧯 **启动失败速查** —— 排障时直接跳这张表
- 📖 返回 **[主手册](README.md)** · ⚠️ 缺口见 **[todolist](todolist)**

---

## 🎯 一张表看懂

| 问题 | 一句话答案 |
|---|---|
| 配置分几类？ | **两类，性质完全不同**：`app/config/` 会被解析；`app/services/**/*.yaml` 一个字节都不读 |
| 运行时配置在哪？ | 全部集中在 `app/config/`：**顶层 8 份 + `datasource/` 6 份**。不再散落在各模块目录 |
| 契约规格会被读吗？ | **不会**。真正做校验的是 Python DTO；yaml 是「唯一事实来源」的声明，只有开发期 `spec_check` 会解析 |
| 改完要重启吗？ | **要**。`@lru_cache` 或启动期单例，无热加载 |
| 哪个配置错了会起不来？ | `postgre.yaml` 与 `checkpointer.yaml` 都标了 `required: true`；`casdoor.yaml` 在 `APP_ENV=prod` 下强制自检 |
| 加一个数据源要做什么？ | 只在 `datasource/` 放一个带**顶层 `type`** 的 yaml —— 目录被整目录扫描，**不用改代码** |
| 最容易被坑的地方？ | 两个跨文件不变量（见下节）+ 顶层 `type` 拼错会被**静默跳过** |

---

## 🚀 启动

```bash
# ⚠️ 用 .venv 里的 python，不要用裸 python
#    （裸 python 是旧 anaconda，SQLAlchemy 太老，直接 ImportError）
.venv/Scripts/python.exe run_server.py              # 默认 localhost:8080
.venv/Scripts/python.exe run_server.py --port 8081
.venv/Scripts/python.exe run_server.py --reload
```

`run_server.py` 只是个 argparse 壳，真正的启动逻辑在 **`app/serve.py`**。
直接跑 `app/main.py` 走的是同一份逻辑，两个入口行为一致。

### ⚠️ 不要用 `uvicorn app.main:app`

`serve.py` 用 `asyncio.run(server.serve(), loop_factory=SelectorEventLoop)` **显式指定事件循环** ——
因为 **psycopg（checkpointer 的驱动）需要 `SelectorEventLoop`**，而 **uvicorn 0.36+ 会忽略进程级 asyncio policy**，
于是 CLI 那条路拿不到正确的循环。

这是 Windows 上必须绕过 uvicorn CLI 的**具体原因**，不是风格偏好。

### 启动时序

启动顺序写在 `app/main.py` 的 lifespan 里，它是**组合根**，只负责排序 —— 每一步的位置都有注释解释理由。
装配细节在 `app/bootstrap/`（auth / event_stream / graph_runtime / pipelines / task）。

**依赖方向**（约定，非强制，见 `app/bootstrap/__init__.py`）：

```
app/common  ←  app/services  ←  app/infra  ←  app/bootstrap  ←  app/main.py
```

箭头指向「被依赖」。**`app/common` 与 `app/infra` 都不允许 import 业务模块** ——
这正是「任务 Handler 的注册点必须在 `app/bootstrap/task.py` 而不在 `app/infra`」的硬理由。

---

## ⚙️ 运行时配置一览

### 顶层 8 份（`app/config/`）

| 文件 | 管什么 | 一句话记住 |
|---|---|---|
| [`app.yaml`](./app/config/app.yaml) | 应用级开关 | — |
| [`casdoor.yaml`](./app/config/casdoor.yaml) | JWT 校验、OAuth 登录、管理员门禁 | `APP_ENV=prod` + `enabled: false` = **硬拒绝启动** |
| [`event_stream.yaml`](./app/config/event_stream.yaml) | SSE 链路的全部节流参数 | `maxlen` 与 `ttl` 管**两件不同的事，两个都要** |
| [`providers.yaml.template`](./app/config/providers.yaml.template) | 外部 provider 连接：`llm` / `web` / `rerank` 三段 | 复制成 `providers.yaml` 再填 key；真文件 gitignore |
| [`mcp.yaml`](./app/config/mcp.yaml) | MCP 外部工具来源 | 配了才连，连不上不影响启动 |
| [`model.yaml`](./app/config/model.yaml) | **选哪个模型**（六段目录 + 绑定） | `bindings` 刻意**没有通配兜底** |
| [`mock_model.yaml`](./app/config/mock_model.yaml) | 离线/测试用的假模型 | 有它才能不接真 key 跑冒烟 |
| [`stress_eval.yaml`](./app/config/stress_eval.yaml) | 压测与评测参数 | — |

### `datasource/` 6 份

| 文件 | 角色 | 缺了会怎样 |
|---|---|---|
| [`postgre.yaml`](./app/config/datasource/postgre.yaml) | 主库 + **`required: true`** | 起不来 |
| [`checkpointer.yaml`](./app/config/datasource/checkpointer.yaml) | LangGraph checkpoint + **HITL 审批账本** + **`required: true`** | 起不来；`store` 段没生效则账本退化成内存版 |
| [`redis.yaml`](./app/config/datasource/redis.yaml) | 事件流 + 任务板 | SSE 与子代理不可用 |
| [`milvus.yaml`](./app/config/datasource/milvus.yaml) | 稠密向量 | 该检索路降级 |
| [`elasticsearch.yaml`](./app/config/datasource/elasticsearch.yaml) | 关键词 + 稀疏向量 | 该检索路降级 |
| [`neo4j.yaml`](./app/config/datasource/neo4j.yaml) | 图库 | 已接为 DataSource，但**尚未被业务消费** |

> **加一个数据源不用改代码**：目录被整目录扫描，只要有顶层 `type` 就会被加载。

### 21 份契约规格（`app/services/**/*.yaml`）

`capability` · `context_builder` · `document` · `document_retrieve` · `document_type_definition` ·
`document_version` · `file` · `mcp_provider` · `memory` · `memory_api_v2` · `memory_atom` ·
`memory_retrieval` · `messages` · `pipeline` · `processor_capability` · `session` · `skill` ·
`task` · `tool` · `tool_processor` · `user`

**运行时不加载**。开发期自检：

```bash
.venv/Scripts/python.exe -m app.agent.spec_check    # 当前只覆盖 document 域
```

---

## 🔗 跨文件不变量

改一个必须回头看另一个，否则症状很隐蔽（不报错，只是行为怪）：

| 不变量 | 违反后的症状 |
|---|---|
| `event_stream.yaml` 的 `block_ms` 必须**远小于** `socket_timeout` | 阻塞读超时，事件流莫名断开 |
| `max_active_run_streams` 必须**严格小于** `stream_max_connections` | 连接池被自己的上限锁死，高并发下拿不到连接 |
| `model.yaml` 的 `bindings` 必须覆盖所有会被调到的 `(agent_type, model_role)` | 运行期报「没有可用于 … 的模型绑定」，**没有通配兜底** |
| `checkpointer.yaml` 的 `default_ttl_minutes` 是**分钟**；审批拦截器那份 TTL 是**秒** | 单位混用，TTL 差 60 倍 |

---

## 🌱 环境变量

| 变量 | 用途 | 缺省行为 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek 凭证 | 落到 yaml 里的开发 key |
| `KIMI_API_KEY` | Kimi 凭证 | 落到 yaml 里的开发 key |
| `QWEN_API_KEY` | Qwen 凭证 | 落到 yaml 里的开发 key |
| `AGENT_EVENT_STREAM_PREFIX` | 事件流键前缀 | `agent_run` |
| `APP_ENV` | 运行环境；`prod` 会开启额外保护 | 非 prod |

其它 `${VAR:-default}` 占位符在各 datasource yaml 里（dsn / 密码 / uri / token 等）。

> ⚠️ 有**两份 `expand_env` 实现**：`app/common/utils/env.py` 与 `app/infra/datasource/loader.py::_expand_env`（后者是复制的一份）。
> 改展开语义要**同时改两处**。

---

## 📡 SSE 与前端契约

**这是一个跨仓库契约**，前端同学要知道：

- SSE **遇审批主动关流**。前端须自行 `resume` 后重开，不要指望同一条连接能续上。
- 「所有 interrupt 都到齐」的**权威信号是 DB 状态**，不是事件批次 —— 事件可能丢、可能乱序，DB 不会。
- `maxlen`（保留多少条）与 `ttl`（保留多久）管**两件不同的事**，两个都要配。

---

## 🧯 启动失败速查

| 症状 | 最可能的原因 | 去哪看 |
|---|---|---|
| 启动即中止，提到 `postgresql` | `postgre.yaml` 缺失 / dsn 错 / 库没起 | `postgre.yaml` |
| psycopg 报事件循环类型不对 / checkpointer 建不起来 | 用了 `uvicorn app.main:app`，拿到默认循环而不是 `SelectorEventLoop` | 改用 `python run_server.py` |
| 启动即中止，提到证书 / `admin_role` | `casdoor.yaml` 自检没过 | `app/bootstrap/auth.py::require_ready()` |
| `APP_ENV=prod` 下拒绝启动 | `casdoor.yaml` 里 `enabled: false` | **刻意的硬拒绝，不是 bug** |
| `Timeout reading from localhost:6379` | **多半与 Redis 无关** —— 是启动期同步代码按住事件循环 | `main.py:485-505`、`app/common/loop_watchdog.py` |
| 模型解析报「没有可用于 … 的模型绑定」 | `model.yaml` 的 `bindings` 缺一条（没有通配兜底） | 报错会列出当前所有绑定 |
| 模型解析报「命中多条同等具体的绑定」 | `bindings` 有歧义，**不会按书写顺序猜** | 加 `attributes` 消歧或删掉多余的 |
| 新建的 service 模块**静默不生效** | `TOOL_SOURCE_PACKAGES` 没覆盖该包，工具没被发现 | `app/services/tool/registry/registry.py` |
| 数据源 yaml 改了没反应 | 顶层 `type` **键名拼错** → 被静默跳过 | 启动日志里的 skip warning（**唯一线索**） |
| 已经批准过的工具重启后又问一遍 | `checkpointer.yaml` 的 `store` 段没生效 → 账本退化成内存版 | `interceptor/approval.py` |
| 应用起不来 + prompt 相关报错 | `prompts/` 下有空文件 / 缺 `PROMPT` 声明（目录装配要求**每个文件都声明**） | 见 [`todolist`](todolist) A 部分第 2 条 |

---

## 🤝 相关

| 想深入看 | 去哪 |
|---|---|
| 总览与模块索引 | [`README.md`](README.md) |
| 模型目录与解析优先级 | [`README_MODEL.md`](README_MODEL.md) |
| 工具定义 / 处理器链 / HITL 审批 | [`README_TOOL.md`](README_TOOL.md) |
| 上下文流水线（13 段 / 14 处理器） | [`README_CONTEXT.md`](README_CONTEXT.md) |
| 四层记忆与契约分工 | [`README_MEMORY.md`](README_MEMORY.md) |
| 文档管线与 RAG | [`README_DOCUMENT.md`](README_DOCUMENT.md) |
