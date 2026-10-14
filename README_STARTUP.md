# START_UP.md — 配置与启动

> YTraderAgent · 配置手册与启动时序
> 覆盖范围：`app/config/**`（运行时配置，10 份）、`app/services/**/*.yaml`（HTTP 契约规格，19 份）、
> 以及 `app/main.py` 的 lifespan 启动链。
> 本文回答一个问题：**"这个服务要跑起来，哪些文件要配、按什么顺序读、改完怎么生效。"**

---

## 0. TL;DR

| 问题 | 一句话答案 |
|---|---|
| 配置分几类？ | **两类，且性质完全不同**：`app/config/` 是**运行时配置**（真的会被解析），`app/services/**/*.yaml` 是**契约规格**（运行时不加载） |
| 运行时配置在哪？ | 全部集中在 `app/config/`（顶层 4 份 + `datasource/` 6 份）。**不再散落在各模块目录** |
| 契约规格会被读吗？ | **不会**。真正做请求/响应校验的是 Python 里的 DTO；yaml 是"唯一事实来源"的声明，只有开发期的 `app/agent/spec_check.py` 会解析 |
| 改完要重启吗？ | **要**。没有任何热加载 —— `@lru_cache` 或启动期单例，改 yaml 必须重启进程 |
| 哪个配置错了会起不来？ | `app/config/datasource/postgre.yaml`（唯一 `required: true`）；`casdoor.yaml` 在 `APP_ENV=prod` 下也强制自检 |
| 加一个数据源要做什么？ | 只在 `app/config/datasource/` 放一个带顶层 `type` 的 yaml —— 目录被整目录扫描，**不用改代码** |
| 最容易被坑的地方？ | 两个**跨文件不变量**（见第 4 节）：`block_ms` 必须远小于 `socket_timeout`；`max_active_run_streams` 必须严格小于 `stream_max_connections` |
| 启动命令？ | `.venv` 的 python 跑 `python run_server.py`（默认 `localhost:8080`） |

**⚠️ 先读这一条 —— 仓库里有一批「注释失真」**

本仓库近期把运行时配置**从各模块目录收敛到了 `app/config/`**。为了支持这个收敛，
数据源加载器从 `raise ValueError` 放宽成了「没有顶层 `type` 就**跳过并告警**」
（`app/infra/datasource/loader.py:66-88`，注释里写明了这次改造的动机）。

于是下面这些文档/注释描述的是**改造前的布局**，照它们找文件会找不到：

| 位置 | 它说配置在哪 | 实际在哪 |
|---|---|---|
| `app/config/llm.yaml` 开头注释 | "放在 `app/common/llm.py` 旁边" | `app/config/llm.yaml` |
| `app/config/event_stream.yaml` 开头注释 | "放在读它的代码旁边，不在 `app/config/`" | `app/config/event_stream.yaml` |
| `README_MODEL.md` 第 1 节 | `app/agent/model/model.yaml` | `app/config/model.yaml` |

**以实际文件位置为准**（`app/agent/model/model.yaml` 与 `app/common/llm.yaml` 都已不存在）。

---

## 1. 两类 yaml：先分清这个，否则后面全错位

| | **运行时配置** | **契约规格** |
|---|---|---|
| 位置 | `app/config/**` | `app/services/**/*.yaml` |
| 数量 | 10 份 | 19 份 |
| 顶层结构 | 自由（按各加载器的 pydantic 模型） | 统一 `openapi: 3.1.0` + `info` / `servers` / `paths` / `components` |
| 启动时加载？ | **会**，全部解析 + 校验 | **不会**，一个字节都不读 |
| 谁在用 | `app/main.py` lifespan 各步骤 | Python DTO（`app/services/*/domain/schemas/*.py`）声明"照它实现" |
| 改错的后果 | **应用起不来**或行为不对 | 不影响运行；靠 `spec_check` 自检发现漂移 |
| 唯一解析它的代码 | 各自的 loader（见第 6 节） | `app/agent/spec_check.py`（**开发期脚本**，只覆盖 document 那 6 份） |

> **为什么要写契约 yaml 却不加载它？**
> 它们是**服务边界的声明**：定义了对外 HTTP 的形状、枚举取值、必填字段。
> 运行时的校验由 FastAPI + pydantic DTO 完成 —— 两者应当一致，但**一致性靠人守**，
> 目前只有 document 域有 `spec_check` 这道自动闸。其余域是已知缺口。

---

## 2. 启动时序（`app/main.py` 的 lifespan）

顺序不是随意的，每一步的位置都有理由。改启动逻辑前先读这张表。

| # | 步骤 | 位置 | 为什么在这个位置 |
|---|---|---|---|
| 0 | 事件循环卡顿哨兵 `LoopWatchdog` | `main.py:208` | **刻意最前**。启动阶段本身就有一堆同步活（建池/建表/装图），哨兵要盯的正是这个 —— 卡住超阈值就打线程栈 |
| 0.5 | Casdoor 鉴权自检 `initialize_auth()` | `main.py:224` | 不碰网络也不碰库，**失败得越早日志越好读**；prod + 未启用会直接拒绝启动 |
| 1 | 数据源 `DataSourceStartup.initialize()` | `main.py:234` | 整目录扫描 `app/config/datasource/`，建池/建表/探活/注册。**checkpointer 也是标准数据源**，所以不需要第二套连接池生命周期 |
| 1.5 | 运行事件流 `initialize_event_stream()` | `main.py:252` | **必须在装图之前** —— 节点第一次运行就要拿得到发布器 |
| 2 | PostgreSQL schema `initialize_database()` | `main.py:264` | 建表；在装图之前，因为图要用表 |
| 3 | 装配两张图 `build_pipelines()` | `main.py:277` | 文档管线 + Agent 对话，连同共享的记录器 / 产物存储 / 能力注册表。产出的只读句柄挂 `app.state`（接口/探针用，构造后冻结） |
| 4 | 图调度器 `build_graph_runtime()` + `start_graph_runtime()` | `main.py:307` | 建与启动**分两步** —— 启动时要把 dispatcher 登记进进程级访问器，而事件流读取端排在更早的位置，那时它还不存在 |
| 4.5 | 异步任务基础设施 `build_task_infrastructure()` | `app/bootstrap/task.py` | 业务 Handler 的注册点也在这里 —— 注册要 import 业务契约，而 `app/common` / `app/infra` 不允许 import 业务模块（依赖方向） |
| 5 | 应用层工厂 `build_factories()` | `app/common/factories.py` | 建 `RepositoryFactory` + `ServiceFactory` 与进程级单例池 |
| 5.1 | 启动任务基础设施 `start_task_infrastructure()` | `app/bootstrap/task.py` | 建与启动分两步，同第 4 步 |
| 5.2 | 提示词 Provider `DefaultPromptProvider` | 启动期 | 目录装配要求**每个提示词文件都声明 `PROMPT`**（有空文件 → `PromptRegistryError` → 起不来） |
| 5.5 | 模型目录 `get_model_registry()` | 启动期单例 | 解析 `model.yaml` 六段 + **做引用校验**（缺引用 / 歧义在启动期就炸，不留到调用时） |
| 6 | 收敛上次进程残留的 `running` 运行 | lifespan | 进程重启后不留假运行态（orphan-run reconciliation） |
| 6.x | 后台模型预加载 + LLM provider 预热 `warmup_chat_models()` | `main.py:502` | **必须 `await` 在 `yield` 之前** —— 预热的意义就是"接请求前把这份成本付掉"。丢进线程池是刻意的：它本身就是同步阻塞的 |
| 关闭 | 任务基础设施 → 图调度器（等在跑的任务结束）→ 事件流 → 模型工厂 → LoopWatchdog → `datasource_manager.close_all()` | lifespan `finally` | ⚠️ 关闭是**逆序**的；checkpointer 在注册顺序里排最后，因此最先被关 |

> **第 6 步那段注释值得单独读**（`main.py:485-505`）：启动时那次 provider 预热会把事件循环按住若干秒，
> 而 SSE 那条 `XREAD BLOCK` 的读超时定时器会在循环恢复的瞬间先到期，
> 把"回复早就到了"谎报成 `TimeoutError: Timeout reading from localhost:6379`。
> **那条 Redis 报错的根因不在 Redis。** 完整链路见 `app/common/llm.py` 与 `app/common/loop_watchdog.py`。

---

## 3. 运行时配置逐个说明

### 3.1 `app/config/casdoor.yaml` — 鉴权（约 246 行）

| 项 | 值 |
|---|---|
| 加载器 | `app/common/auth/config.py::load_casdoor_config()`（pydantic `CasdoorConfig`，`extra="forbid"`，`lru_cache`） |
| 接线 | `app/bootstrap/auth.py::initialize_auth()`，在 lifespan 第 0.5 步 |

**管什么**：`/api/v1` 全量端点的 JWT 校验（签名 / audience / 群组 / 角色 → 401/401/403）、
OAuth 登录流（`/auth/login` → Casdoor → `/auth/callback`）、claim 名映射、管理员端点门禁、
以及**唯一**允许从 query string 取 token 的例外。

**必须知道的几条**：

- `enabled: false` 会返回一个合成用户 —— 方便本地开发，但 **`APP_ENV=prod` + `enabled: false` 是硬拒绝启动**。
- `query_token_paths` 默认只放行 `/api/v1/runs/*/events` —— 因为 `EventSource` 不能设请求头，这是 SSE 唯一的取 token 通道。**这也是一处安全面，改之前想清楚。**
- `admin_role` 留空 = 启动报错。
- `require_ready()` 在启动期校验证书、`required_groups`、`admin_role`（fail loud）。
- ⚠️ 开发用的 `client_id` / `client_secret` 是**硬编码在 yaml 里**的。

### 3.2 `app/config/event_stream.yaml` — 运行事件流（约 120 行）

| 项 | 值 |
|---|---|
| 加载器 | `app/agent/event/config.py::load_stream_config()`（pydantic `StreamConfig`，`lru_cache`） |
| 接线 | `app/bootstrap/event_stream.py`，lifespan 第 1.5 步（**必须在装图前**） |

**管什么**：Redis Stream 上的 run 事件发布/订阅 —— 也就是 SSE 那条链路的全部节流参数。
键名形如 `<key_prefix>:<topic_id>`（例如 `agent_run:run_123`）。

**六组旋钮，每一组都对应一个具体故障**：

| key | 默认 | 作用 / 为什么是这个数 |
|---|---|---|
| `enabled` | `true` | 关掉后 `publish()` 一条命令都不发、直接返回 `False`。本地没 Redis 时用它，不必改 `redis.yaml` |
| `key_prefix` | `${AGENT_EVENT_STREAM_PREFIX:-agent_run}` | 键前缀 |
| `maxlen` | `10000` | 单条 stream 长度上限（`XADD MAXLEN ~`，近似裁剪）。`null` = 不裁剪 |
| `ttl_seconds` | `86400` | 键存活秒数，**每次写入刷新**（滑动）。⚠️ `maxlen` 与 `ttl` 管**两件不同的事，两个都要**：`maxlen` 限一条流多长，但被裁空的键**仍然存在**；一个 run 一条 stream，跑一万次就是一万个键 —— 光靠 `maxlen` 止不住键数量增长，那是**内存泄漏**，且症状隐蔽（每条流都很短，看着没问题） |
| `block_ms` | `1000` | `XREAD BLOCK` 时长。**必须明显小于 `redis.yaml` 的 `socket_timeout`（5s）**，否则被客户端读超时打断成"看起来像网络故障"的错误 |
| `idle_probe_after_reads` | `5` | **连续**空读这么多次后探活一次"这个 run 是不是不会再有事件了"。约等于 5 秒，与心跳节拍对齐。⚠️ 用「空读**次数**」而非毫秒，是为了让测试注入小数字就能触发、不必 sleep；改 `block_ms` 时这个数的含义跟着变。⚠️ 探活只在**空闲**时跑 |
| `subscriber_queue_size` | `2000` | 单订阅者待发队列上限。满了**断开这一个订阅者**（不阻塞共享循环、也不丢最旧）—— 共享循环是同一 run 所有订阅者共用的，为一个慢消费者堵住等于让所有人卡；丢最旧会让逐 token 的正文**静默缺洞**。取值要装得下一次完整补发（按 `maxlen` 定） |
| `max_active_run_streams` | `16` | 同时在订阅的 run 数上限。达到后新订阅拿 **503 + 原因**，而不是等连接池在 redis-py 内部炸 `MaxConnectionsError`（那个异常继承自 `ConnectionError`，极易被误当成"网络抖动"）。**必须严格小于 `redis.yaml` 的 `stream_max_connections`** |
| `read_max_consecutive_errors` | `3` | 连续读失败达此数就把异常抛给订阅方；没到就重发心跳继续读。⚠️ `block_ms < socket_timeout` 只保证**我们**不撑爆读超时，挡不住事件循环被别的同步代码卡住 ≥ `socket_timeout` 导致定时器抢先到期的情况（**已实测复现**：Redis 2.19 秒写入，客户端 6.00 秒报读超时）。设成 `1` = 不重试，一键回滚 |
| `write_queue_size` | `20000` | 写队列上限（**原来 10000**）。满了丢**最新**那条（已接收的照常按序写出）—— 丢尾部而非中间，正文不会被撕开空洞，且调用方拿到明确的 `False`。⚠️ 加倍的直接原因是 `message.reasoning.delta`：思维链是**同一根队列**上的第二条逐 token 通道，量级常比正文翻倍，而"一旦丢就丢到排空"意味着推理挤占的正是**正文**的余量 |

> ⚠️ 最后一条注释里还有一句关键的调参纪律：`maxlen` 与 `subscriber_queue_size`
> **互相咬合**（后者要装得下一次补发，按前者定），**没跟着一起动**。
> 要调就两处一起调，并且**先有观察**。

### 3.3 `app/config/llm.yaml` — 云端 LLM 提供方（约 58 行）

| 项 | 值 |
|---|---|
| 加载器 | `app/common/llm.py::load_llm_config()`（pydantic `LLMConfig` / `LLMProviderConfig`，`lru_cache`） |
| 接线 | `get_chat_model()` 建客户端；`warmup_chat_models()` 在 lifespan 第 6 步预热 |

**管什么**：**客户端构造**（provider 分发 / `base_url` / 模型名 / API key）。
它与 `model.yaml` 的分工是：**`llm.yaml` 管怎么建客户端，`model.yaml` 管选哪个模型**。

| key | 本机值 |
|---|---|
| `default_provider` | `kimi` |
| `providers.deepseek` | `deepseek-v4-flash` @ `https://api.deepseek.com/v1` |
| `providers.kimi` | `kimi-k2.7-code` @ `https://api.moonshot.cn/v1` |
| `providers.qwen` | `qwen3.7-flash` @ `https://maas.qianwenaiapi.com/compatible-mode/v1` |

**必须知道的几条**：

- 三家都是 **OpenAI 兼容**接口，用同一个 `ChatOpenAI` 客户端，只换 `base_url` / `model` / `api_key` —— 换一家不用换客户端类。新增一家 = 这里加一段 + `app/common/llm.py` 的枚举加一项。
- `base_url` **必须带 `/v1`**（OpenAI SDK 会自己拼 `/chat/completions`）。
- API key 走 `${VAR:-default}` 展开。⚠️ **`:-` 后面留空表示"没配就是空串"，但本机文件里硬编码了可用的开发 key。** 密钥空值**刻意不在加载期校验**，推迟到真正调用时 —— 这样只跑检索、不跑 LLM 的场景不会被一条缺失的 key 拦在门外。
- `kimi` 走 `langchain_moonshot.ChatMoonshot`（不在 `init_chat_model` 的表里），`qwen` 映射到 `openai` 适配器。

### 3.4 `app/config/model.yaml` — 模型目录（约 494 行）

| 项 | 值 |
|---|---|
| 加载器 | `app/agent/model/catalog.py::load_model_catalog()` —— **唯一不走 pydantic 的配置**：`yaml.safe_load` + 自研两段式解析（先形状检查 + 顶层键白名单，再引用解析） |
| 接线 | `build_model_registry()` / `get_model_registry()`（`app/bootstrap/pipelines.py`），启动期单例 |

**六段**（顶层键白名单）：`providers` / `credentials` / `models` / `profiles` / `aliases` / `bindings`。

**管什么**：模型目录本身 —— 有哪些模型、各自 `capabilities`（tool_calling / structured_output / streaming / vision / reasoning）、`context_window`（预算依据）、`config`、`structured_output_extra_body`，以及 `profiles`（primary + fallbacks + 重试/超时策略）、`aliases`、`(agent_type, model_role) → profile` 的 `bindings`。

**必须知道的几条**：

- **`bindings` 刻意没有通配兜底**：命中不到 = **显式报错**；命中多条同等具体 = **也报错**（见第 5 节）。
- ⚠️ **不做环境变量展开** —— 文件里不含密钥，凭证只存引用名。
- ⚠️ `structured_output_extra_body` **每家写法不同**：DeepSeek 用 `thinking: {type: disabled}`（不关的话**硬 400**），Qwen 用 `enable_thinking: false`（不关的话**静默变慢**）。同一个语义、两种拼法、两种故障形态。
- ⚠️ `reasoning: false` **不等于**"不是思考模型" —— 这个坑在 `README_MODEL.md` 里有长说明。
- `pricing:` 段**全部被注释掉** → `estimated_cost` 恒为 `None`，**成本估算是关的**。
- 包含记忆后台任务的绑定：`memory_atom_extract` / `memory_atom_dedup` / `memory_scene_consolidate` / `memory_persona_extract`。

### 3.5 `app/config/datasource/*.yaml` — 数据源（6 份）

**统一加载机制**：`app/infra/datasource/loader.py::DataSourceConfigLoader.load_all()`
对传入目录（`main.py` 给的是 `./app/config/datasource/`）做 `glob("*.yaml")`，
每份文件 `yaml.safe_load` + `${VAR:-default}` 展开 + 用 `TypeAdapter(DataSourceDefinition)`
按顶层 `type` 做**判别式联合**校验（`app/infra/datasource/models.py`，全部 `extra="forbid"`）。

**三条判定规则**（读 `loader.py:52-88`）：

1. **顶层没有 `type` 键** → **跳过 + 打 warning 列出文件名**（不是报错）。
   > 这条是**近期放宽**的：以前是 `raise ValueError`，等于宣布"这个目录只许放数据源"，
   > 于是 `llm.yaml` / `event_stream.yaml` / `model.yaml` 为了躲开它被挪到了各模块目录下。
   > 现在 `app/config/` 是"部署时要改的东西"该待的地方，不该因为一个加载器的假设把配置打散到三处。
   > ⚠️ 宽松化的代价：`tpye: postgresql` 这种**键名拼错也会被静默跳过**，之后只在取 `postgresql` 时报一句更差的错。**warning 日志是这里唯一的线索。**
2. **`type` 不认识** → 当作可选数据源，warning + 跳过。
3. **`postgresql` 是唯一强制类型** → 缺了 / 错了 = **配置错误，直接中止启动**。

| 文件 | type | required | 关键 config | 干什么 |
|---|---|---|---|---|
| `postgre.yaml` | `postgresql` | ✅ **true** | `dsn`（asyncpg）、`pool_size: 10`、`max_overflow: 20`、`checkpointer_pool_size: 5` | 主 SQLAlchemy 异步连接池。**唯一致命项** |
| `checkpointer.yaml` | `postgresql_checkpointer` | ✅ **true** | `derive_from: postgresql`、`pool_min/max: 1/5`、`store.default_ttl_minutes: 10080`、`sweep_interval_minutes: 60` | LangGraph checkpointer（**独立的 psycopg 池**，与业务池分开）+ 同池上的 LangGraph `store` |
| `redis.yaml` | `redis` | ❌ false | `max_connections: 10`、`stream_max_connections: 20`、`socket_timeout: 5`、`decode_responses: false` | 两套池：短命令池 + **事件流订阅专用长连接池** |
| `milvus.yaml` | `milvus` | ❌ false | `uri`、`token`、`db_name` | 向量库 |
| `elasticsearch.yaml` | `elasticsearch` | ❌ false | `hosts`、`verify_certs: false`、`request_timeout: 5` | 全文检索 |
| `neo4j.yaml` | `neo4j` | ❌ false | `uri: bolt://…`、`max_connection_pool_size: 100` | 图库 |

**三个必须知道的点**：

- **`checkpointer.yaml` 的 `derive_from: postgresql`** 避免重复写一遍密码 —— 它从 postgre 那份继承连接信息。
- ⚠️ **单位陷阱**：`checkpointer.yaml` 的 `default_ttl_minutes` 是**分钟**（10080 = 7 天），而审批拦截器那份配置的 TTL 是**秒**。同一套系统里两种单位。
- **`checkpointer.yaml` 的 `store` 段不是可选的装饰** —— 它是 **HITL 审批账本**的持久化后端
  （`app/agent/tools/processor/interceptor/approval.py`）。**不传 `store`，账本就退化成重启即丢的内存版**，症状是"已经批准过的工具重启后又来问一遍"。
- ⚠️ `redis.yaml` 的 `decode_responses: false`（返回 bytes，无损默认）；`verify_certs: false` 是因为开发容器跑的是**无 TLS 的 HTTP**。

---

## 4. 跨文件不变量（改一个必须回头看另一个）

这些约束**没有任何代码强制**，只写在注释里。破坏它们的症状全都长得像"网络抖动"。

| 不变量 | 涉及文件 | 违反后的症状 |
|---|---|---|
| `block_ms` **≪** `socket_timeout` | `event_stream.yaml:45` ↔ `redis.yaml` | 命令被客户端读超时打断，报成像网络故障的错误。默认 1s vs 5s，留 4s 余量 |
| `max_active_run_streams` **严格小于** `stream_max_connections` | `event_stream.yaml:77` ↔ `redis.yaml` | 池满时 redis-py **立刻抛、不排队** —— 取等号时一次重试就能撞出 `MaxConnectionsError`。默认 16 < 20 |
| `subscriber_queue_size` 要装得下一次完整补发 | `event_stream.yaml:69`，按 `maxlen` 定 | 重连补发时把订阅者顶掉 |
| `sweep_interval_minutes` 与 `store.default_ttl_minutes` 的相对大小 | `checkpointer.yaml` | 清扫不及时 / 清扫过于频繁 |
| LLM provider 的 `base_url` 必须带 `/v1` | `llm.yaml` | SDK 拼出错误路径 |

---

## 5. 契约规格清单（`app/services/**/*.yaml`，19 份）

**统一结构**：`openapi: 3.1.0` + `info` / `servers` / `[tags]` / `[security]` / `paths` / `components`。
**运行时零加载**。唯一的解析者是开发期脚本 `python -m app.agent.spec_check`（**只覆盖 document 那 6 份**）。

| 文件 | 行数 | 契约 | 主要端点 |
|---|---|---|---|
| `session/session.yaml` | 554 | Agent Runtime API | `/sessions`、`/{id}/runs`、`/runs/{run_id}/events`、`/resume`、`/cancel` |
| `session/messages.yaml` | 253 | Messages API | `/sessions/{session_id}/messages` |
| `task/task.yaml` | 536 | Async Task Management | `/tasks`、`/lineage`、`/cancel`、`/retry`、`/reschedule` |
| `tool/tool.yaml` | 644 | Tool Definition Management | `/tools`、`/discovery/sync`、`/{tool_id}/documentation` |
| `tool/tool_processor.yaml` | 547 | Tool Processor Management | `/processors`、`/tools/{tool_id}/processors`、`/order` |
| `capability/capability.yaml` | 367 | Capability Management | `/capabilities`、`/capabilities/{id}` |
| `context/context_builder.yaml` | 585 | Context Builder Catalog（只读） | `/context-builders`、`/{type}/pipeline` |
| `model`（无独立 yaml） | — | 模型目录由 `model.yaml` 承载，**不是** OpenAPI 契约 | — |
| `memory/memory.yaml` | 397 | Memory Search API（L0 工具） | `/memory/search` |
| `memory/memory_api_v2.yaml` | 1310 | Memory API v2.1.0（管理/查询） | `/memory/retrieval`、`/query/l0..l3`、`/extraction/l1..l3` |
| `memory/memory_atom.yaml` | 341 | Memory Atom (L1) Search | `/memory-atom/search` |
| `memory/memory_retrieval.yaml` | 1231 | Memory Retrieval API | `/retrieval/l0..l3` |
| `document/document.yaml` | 706 | Document Service | `/documents`、`/{id}/versions` |
| `document/document_retrieve.yaml` | 800 | Document Retrieve | `/documents/retrieve` |
| `document/document_type_definition.yaml` | 1408 | Document Definition Management | `/document-type-definitions`、`/document-processors` |
| `document/document_version.yaml` | 1307 | Document Version | `/document-versions/{vid}/reprocess`、`/qa-datasets:generate` |
| `document/pipeline.yaml` | 1562 | Document Pipeline | `/pipeline-runs`、`/{run_id}/nodes`、`/document-progress` |
| `document/processor_capability.yaml` | 485 | Document Processor Capability | `/document-processor-capabilities` |
| `file/file.yaml` | 648 | File Service | `/files`、`/{file_id}/content` |
| `evaluate/evaluation.yaml` | 1349 | RAG Evaluation | `/evaluation/datasets`、`/metrics`、`/runs`、`/results` |
| `user/user.yaml` | 480 | User & Authentication | `/auth/login`、`/auth/callback`、`/auth/refresh`、`/auth/me` |

**几处值得单独记的**：

- **`context_builder.yaml` 枚举最重**：它声明了固定的 `ContextBuilderType`
  （`agent_invoke` / `planner` / `classifier` / `compaction` / `memory` / `sub_agent`）、
  固定的 **11 段 `StageId` 流水线**、`StageKind`、`ExecutionMode`、边类型 ——
  **都是在文档化"服务端流水线，客户端不可改"**。这份 yaml 是那套流水线对外的唯一说明。
- **只有 5 份带顶层 `security:`**：`file.yaml`、`memory_api_v2.yaml`、`memory_retrieval.yaml`、`messages.yaml`、`session.yaml`；其余依赖 Casdoor 保护的整棵路由树。
- **`messages.yaml` 声明 `Message`，Python 类却叫 `AgentMessage`**（表前缀 `agent_`）——
  `app/common/openapi.py::pin_contract_component_names` 在 FastAPI 生成完文档后把组件名改回 `Message`。**契约名与类名是故意不一致的。**
- **`memory.yaml` / `memory_atom.yaml` 一份文件担两个角色**：既是 HTTP 契约，**又是暴露给主模型的工具输入 schema**（`memory_atom.search`）。
- `user.yaml` 是 `/auth/callback` 契约的唯一声明处。

---

## 6. 加载器机制总表

| 配置 | 加载器 | 校验 | 环境变量展开 | 缓存 |
|---|---|---|---|---|
| `casdoor.yaml` | `app/common/auth/config.py` | pydantic `extra="forbid"` | ✅ | `lru_cache` |
| `event_stream.yaml` | `app/agent/event/config.py` | pydantic `extra="forbid"` | ✅ | `lru_cache` |
| `llm.yaml` | `app/common/llm.py` | pydantic `extra="forbid"` | ✅ | `lru_cache` |
| `model.yaml` | `app/agent/model/catalog.py` | **自研**：顶层键白名单 + 引用解析 | ❌（刻意） | 启动期单例 |
| `datasource/*.yaml` | `app/infra/datasource/loader.py` | pydantic `TypeAdapter` 判别式联合 | ✅ | 启动期加载一次 |

**共同点**：

- **无热加载**。要么 `@lru_cache`（进程内缓存到死），要么启动期单例。**改 yaml 必须重启。**
- ⚠️ **有两份 `expand_env` 实现**：`app/common/utils/env.py::expand_env` 与
  `app/infra/datasource/loader.py::_expand_env`（后者是复制的一份）。改展开语义要**同时改两处**。

---

## 7. 环境变量

| 变量 | 用途 | 缺省行为 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek 凭证 | 落到 yaml 里的开发 key |
| `KIMI_API_KEY` | Kimi 凭证 | 落到 yaml 里的开发 key |
| `QWEN_API_KEY` | Qwen 凭证 | 落到 yaml 里的开发 key |
| `AGENT_EVENT_STREAM_PREFIX` | 事件流键前缀 | `agent_run` |
| `APP_ENV` | 运行环境；`prod` 会开启额外保护 | 非 prod |

其它 `${VAR:-default}` 形式的占位符在各 datasource yaml 里（dsn / 密码 / uri / token 等）。

---

## 8. 启动

```bash
# ⚠️ 用 .venv 里的 python，不要用裸 python
#    （裸 python 是旧 anaconda，SQLAlchemy 太老，直接 ImportError）
.venv/Scripts/python.exe run_server.py                 # 默认 localhost:8080
.venv/Scripts/python.exe run_server.py --port 8081
.venv/Scripts/python.exe run_server.py --reload
```

`run_server.py` 只是个 argparse 壳，真正的启动逻辑在 **`app/serve.py`**。
`app/main.py` 直接运行走的是同一份逻辑，两个入口行为一致。

**⚠️ 不要用 `uvicorn app.main:app`**

`serve.py` 用 `asyncio.run(server.serve(), loop_factory=SelectorEventLoop)` 显式指定事件循环 ——
因为 **psycopg（checkpointer 的驱动）需要 `SelectorEventLoop`**，而
**uvicorn 0.36+ 会忽略进程级 asyncio policy**，于是 CLI 那条路拿不到正确的循环。
这是 Windows 上必须绕过 uvicorn CLI 的具体原因，不是风格偏好。

**依赖方向**（约定，非强制，见 `app/bootstrap/__init__.py`）：

```
app/common  ←  app/services  ←  app/infra  ←  app/bootstrap  ←  app/main.py
```

箭头指向"被依赖"。**`app/common` 与 `app/infra` 都不允许 import 业务模块** ——
这正是「任务 Handler 的注册点必须在 `app/bootstrap/task.py` 而不在 `app/infra`」的硬理由。

**契约自检**（开发期）：

```bash
.venv/Scripts/python.exe -m app.agent.spec_check    # 只覆盖 document 那 6 份 spec
```

---

## 9. 启动失败速查

| 症状 | 最可能的原因 | 去哪看 |
|---|---|---|
| 启动即中止，提到 `postgresql` | `postgre.yaml` 缺失 / dsn 错 / 库没起 | `postgre.yaml`；这是**唯一** required |
| psycopg 报事件循环类型不对 / checkpointer 建不起来 | 用了 `uvicorn app.main:app`，拿到的是默认循环而不是 `SelectorEventLoop` | 改用 `python run_server.py`，见第 8 节 |
| 启动即中止，提到证书 / `admin_role` | `casdoor.yaml` 自检没过 | `app/bootstrap/auth.py::require_ready()` |
| `APP_ENV=prod` 下拒绝启动 | `casdoor.yaml` 里 `enabled: false` | 刻意的硬拒绝，不是 bug |
| 出现 `Timeout reading from localhost:6379` | **多半与 Redis 无关** —— 是启动期同步代码按住事件循环 | `main.py:485-505`、`app/common/loop_watchdog.py` |
| 模型解析报"没有可用于 … 的模型绑定" | `model.yaml` 的 `bindings` 缺一条（**没有通配兜底**） | 报错信息会列出当前所有绑定 |
| 模型解析报"命中多条同等具体的绑定" | `bindings` 有歧义，**不会按书写顺序猜** | 加 `attributes` 消歧或删掉多余的 |
| 新建的 service 模块**静默不生效** | `TOOL_SOURCE_PACKAGES` 没覆盖该包，工具没被发现 | `app/services/tool/registry/registry.py` |
| 数据源 yaml 改了没反应 | 顶层 `type` **键名拼错** → 被静默跳过 | 启动日志里的 skip warning（唯一线索） |
| 已经批准过的工具重启后又问一遍 | `checkpointer.yaml` 的 `store` 段没生效 → 账本退化成内存版 | `interceptor/approval.py` |
| 应用起不来 + prompt 相关报错 | `prompts/` 下有空文件 / 缺 `PROMPT` 声明（目录装配要求**每个文件都声明**） | 见 `todolist` 第 2 条的历史记录 |

---

## 10. 相关文档

| 想深入看 | 去哪 |
|---|---|
| 总览与模块索引 | [`README.md`](./README.md) |
| 模型目录与解析优先级 | [`README_MODEL.md`](README_MODEL.md) |
| 工具定义 / 处理器链 / HITL 审批 | [`README_TOOL.md`](README_TOOL.md) |
| 上下文流水线（11 段 / 12 处理器） | [`README_CONTEXT.md`](README_CONTEXT.md) |
| 四层记忆与契约分工 | [`README_MEMORY.md`](README_MEMORY.md) |
| 文档管线与 RAG | [`README_DOCUMENT.md`](README_DOCUMENT.md) |
