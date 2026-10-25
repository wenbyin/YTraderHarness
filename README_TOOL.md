# Tool · 工具、处理器链与审批

> 一个工具不是「一个函数 + 一段 docstring」，而是一条**可配置的调用链**：
> 定义 → 发现 → 能力暴露 → 处理器链（洋葱折叠）→ 真正调 service 方法 → 落 `tool_execution`。
>
> 主张：**改工具给模型看的说明、换它的前置/后置处理器、调它们的顺序 —— 都不发版、不重启。**
> 但「处理器**实现**」是只读目录 —— 你只能在运行时挂载已有的处理器，不能在运行时新建一个。

`9 个内置处理器` · `6 张表` · `洋葱折叠` · `HITL 持久化账本` · `可见 == 可执行`

---

## 📚 本文结构

- 🎯 **一张表看懂**
- 🏗️ **阅读地图**
- 🧅 **处理器链与洋葱折叠**
- 🔐 **HITL 审批** —— 重启不重复问的机制
- 🧮 **执行守卫**
- 📡 **HTTP 全清单**
- 📖 返回 **[主手册](README.md)** · ⚠️ 缺口见 **[todolist](todolist)**

---

## 🎯 一张表看懂

| 问题 | 一句话答案 |
|---|---|
| 工具怎么定义？ | 把方法挂在一个 service 类上，加 `@tool` 装饰器；装饰器同时生成 langchain 工具与注册表描述符 |
| 工具从哪来？ | 三路扫描：`@tool` 自注册（import 时）+ `ServiceMethodScanner` 遍历 `TOOL_SOURCE_PACKAGES` 触发未 import 模块 + `MCPToolScanner` 按 `mcp.yaml` 取 `tools/list` |
| 模型看到的函数名是什么？ | `tool_key` 的点号换成下划线（`capability.list` → `capability_list`），因为 OpenAI 函数名不允许点号 |
| 一次工具调用经过什么？ | `select_specs` 鉴权 → 构造 `ToolInvocation` → 读 `tool_processor` 表的链 → 折叠洋葱 → `_base_call` 真正调 service 方法 |
| 危险工具谁来管？ | `approval_interceptor`（HITL）：挂 `BEFORE`，调 `interrupt()` 挂起图，等 `POST /runs/{id}/resume`；账本落 PostgresStore，**重启不重复问** |
| 重复调用怎么管？ | `execution_guard_interceptor`（AROUND）：按 scope（SESSION / RUN / USER / GLOBAL）× execution（METHOD / INVOKE）计数，**一次执行一行** |
| 有工具级超时/重试吗？ | 有（`timeout_interceptor` / `retry_interceptor`），但**默认不挂** —— 图节点级已有重试与超时，工具级只给确实需要的工具显式挂 |
| 运行时要重启吗？ | 改说明、改链 —— **不用**。改处理器实现、加新工具、改 `input_schema` —— **要** |

---

## 🏗️ 阅读地图

```
app/services/tool/                    ← 定义 / 注册 / 发现 / 管理
├── registry/
│   ├── decorator.py      ← @tool 装饰器、ToolDescriptor、source_ref/tool_key 推导
│   ├── scanner.py        ← 两个扫描器（service 方法 / langchain 工具）
│   ├── discoverer.py     ← 装配扫描器、合并清单、冲突检测
│   ├── registry.py       ← ToolRegistry（按 tool_key 索引）+ TOOL_SOURCE_PACKAGES
│   ├── spec.py           ← ToolSpec / ToolDescriptor 数据与推导规则常量
│   ├── schema.py         ← 函数签名 → JSON Schema
│   ├── docstring.py      ← docstring → 参数/返回值说明
│   ├── selection.py      ← select_specs（可见 = 可执行）+ wire_name / tool_key_of
│   └── processor_registry.py ← 处理器注册表
├── application/          ← 查询 / 发现同步 / 文档 / 处理器 / 分类服务
├── models/sqlalchemy/    ← 6 张表
├── repository/           ← 接口 + *_impl 实现
└── domain/schemas/       ← DTO / 枚举

app/agent/tools/
├── time_tool.py          ← 最简工具示例（刻意不写 TOOL_SOURCE_REF）
├── catalog_tool.py       ← 组合工具示例（写了 TOOL_SOURCE_REF）
├── plan_tool.py          ← enter_plan_mode / exit_plan_mode
├── task_tool.py          ← create_team（唯一真实派发入口）+ 4 个占位工具
└── processor/
    ├── base.py           ← BaseToolProcessor + 三类子类
    ├── context.py        ← ToolInvocation / ToolOutcome / ToolCallDenied
    ├── interceptor/      ← approval.py（HITL）/ auth.py / cache.py / execution_guard.py / timeout.py / retry.py
    ├── transformer/      ← privacy.py
    ├── observer/         ← events.py / metrics.py
    └── runtime/          ← events.py / cache.py / metrics.py

app/infra/agent/
├── tool_executor.py      ← 执行入口（ToolExecutor.execute）
├── tool_exposure.py      ← 暴露给模型（RegistryToolExposureSource）
└── chain.py              ← 链编译与洋葱折叠（build_chain / compile_chain）
```

### 6 张表

| 表 | 作用 |
|---|---|
| `tool_definition` | 工具本体（`tool_key` / 描述 / `input_schema` / 源信息） |
| `tool_definition_classification` | 工具 ↔ 能力树映射（**整体替换**语义） |
| `tool_documentation` | **运行时改的**工具级说明 |
| `tool_parameter_documentation` | **运行时改的**逐参数 `description_override` |
| `tool_processor` | 处理器链（`processor_key` / `sequence` / `hook` / `execution_mode` / `config`） |
| `tool_execution` | 执行权威记录 + **幂等账本** |

> 说明文本拆两张表是刻意的：**代码拥有 schema，运维拥有措辞**。
> `input_schema` 的类型 / 必填 / 默认**运行时改不了**，只有 `description` 能改。

---

## 🧅 处理器链与洋葱折叠

### 9 个内置处理器

按反射从 `app.agent.tools.processor` 自动发现：

| `processor_key` | 类型 | 支持的 hook | 模式 |
|---|---|---|---|
| `approval_interceptor` | INTERCEPTOR | BEFORE | SYNC |
| `auth_interceptor` | INTERCEPTOR | BEFORE | SYNC |
| `cache_interceptor` | INTERCEPTOR | BEFORE + AFTER + AROUND | SYNC |
| `execution_guard_interceptor` | INTERCEPTOR | AROUND | SYNC |
| `timeout_interceptor` | INTERCEPTOR | AROUND | SYNC |
| `retry_interceptor` | INTERCEPTOR | AROUND | SYNC |
| `privacy_transformer` | TRANSFORMER | BEFORE + AFTER | SYNC |
| `event_observer` | OBSERVER | BEFORE + AFTER | ASYNC |
| `metrics_observer` | OBSERVER | BEFORE + AFTER | ASYNC |

### 折叠模型

`compile_chain` 从链尾往头递归包：每一层把 `inner` 当作自己的 `next_call`，
**最内层就是真正调 service 方法的 `_base_call`**。

| hook | 位置 | 能干什么 |
|---|---|---|
| `BEFORE` | 外层 | 可直接返回 `ToolOutcome` **短路**整条链（缓存命中 / 审批拒绝 / 鉴权拒绝 → 工具根本不跑） |
| `AFTER` | 内层 | 先拿到 `outcome`，可以改写它 |
| `AROUND` | 占**两行**（进入标记 + 无操作退出标记） | 把下游整个包起来 |
| `ASYNC` | `create_task` 甩出去，**不 await** | 监控不阻塞工具调用 |

> ⚠️ **每次工具调用都重新从 DB 读链，没有任何缓存**（`ToolExecutor._load_chain`）。
> 所以改完**下一次调用就生效** —— 这是「运维友好」的实现基础。

### 写入即校验

所有校验在**写链时**跑完，改错了当场被拒，而不是等运行时出事：

| 错误 | 结果 |
|---|---|
| 未知 `processor_key` | **404** |
| hook 不被该处理器支持 | **409** |
| 执行模式不支持 | **409** |
| AROUND + ASYNC 组合 | **409**（另有 DB `CHECK` 约束兜底） |
| `config` 不过该处理器的 `config_model` | **400**（pydantic） |
| **洋葱区间交叉** | **409** |
| 重排时 id 集合不全 | **409**（「链已过期」） |

**三层防护**：应用层校验 → DB `CHECK` 约束（防手改库）→ 执行期 `build_chain._validate` 再兜一次。

> 洋葱嵌套用**栈式校验**（`validate_onion`）保证区间正确嵌套而非交叉 ——
> 交叉意味着 AROUND 的进入与退出挂到了不同层，折叠出的调用顺序不是你想的那个。

---

## 🔐 HITL 审批

| 环节 | 机制 |
|---|---|
| 触发 | 工具挂了 `approval_interceptor` 且本次调用命中审批条件 |
| 挂起 | 调 langgraph `interrupt()`，图停在这里 |
| 恢复 | `POST /runs/{id}/resume` |
| 账本 | 「始终允许」落 **LangGraph PostgresStore** → **重启不重复问** |
| 幂等键 | 审批记录的幂等键决定「同一次调用」的识别 |
| 超时 | 两个 TTL **单位不同**（见下） |

> ⚠️ **代码里显式标注了一条**：少调一次 `interrupt()` 会让后面的中断**错位** ——
> 这也是 react 图把工具扇出成「一个 tool_call 一个分支」的原因。

> ⚠️ **两个容易误读的点**：`checkpointer.yaml` 的 `default_ttl_minutes` 是**分钟**，
> 审批拦截器那份配置的 TTL 是**秒**。混用就差 60 倍。

> ⚠️ **`store` 段不是装饰**：不传 `store`，账本退化成重启即丢的内存版，
> 症状是「已经批准过的工具重启后又问一遍」。

---

## 🧮 执行守卫

替代旧的「幂等 claim」，挂在 `AROUND`：

| 维度 | 取值 |
|---|---|
| `scope` | `SESSION` / `RUN` / `USER` / `GLOBAL` |
| `execution` | `METHOD` / `INVOKE` |
| 状态 | `success` / `error` / `unknown` —— **各自独立配额** |
| 超限 | 按各自的 `action` 拦截 |

**`tool_execution` 表就是幂等的基础**：唯一约束 `(session_id, run_id, tool_key, param_hash)` +
一条 `INSERT ... ON CONFLICT DO UPDATE WHERE status IN retryable` **原子抢占**。
重放与并发重复调用交给数据库，应用层不做「先查后写」那种有竞态的判断。

`param_hash = sha256(canonical_json(模型给的原始参数))`，在**链跑之前**算。

> ⚠️ 未挂守卫的工具**没有执行记录** —— 守卫是链上的 `AROUND`，链不存在时它不跑。
> 上线前需给需要保护的工具挂两条 `AROUND`。

---

## 📡 HTTP 全清单

| 资源 | 端点 |
|---|---|
| 工具查询 / 文档覆盖 | `GET /tools`、`GET /tools/{id}`、`PUT /tools/{id}/documentation` |
| 处理器链 | `GET /tools/{id}/processors`、`POST`、`PATCH …/{processor_id}`、`DELETE`、`PUT …/processors`、`PUT …/processors/order` |
| 工具 ↔ 能力 | `PUT /tools/{id}/classifications`（**整体替换**） |
| 发现同步 | `POST /tools/discovery/sync` → create / update / unchanged / missing |
| 处理器目录（只读） | `GET /processors`、`GET /tool-processors` |
| 能力树 | `/api/v1/capabilities` |
| MCP | `/api/v1/mcp-providers` |

### 发现同步的两条边界

- ✅ **能**发现**新加的**工具（模块从未被 import，扫描器会 import 它）
- ❌ **不能**拾取对**已 import 模块**的代码修改 —— Python 不会重新执行已加载的模块。**改代码仍然要重启。**

> ⚠️ 新增业务路由必须注册到 `protected_router`，且**注册顺序有约束**，不满足会静默匹配错。
> 详见 `app/api/v1/router.py` 的模块注释。

---

## 🚀 验证

```bash
.venv/Scripts/python.exe -m app.agent.tool_smoke              # 发现 + 注册表 + tool_key 推导
.venv/Scripts/python.exe -m app.agent.tool_processor_smoke    # 链折叠 + 洋葱校验
```

---

## 🤝 相关

| 想深入看 | 去哪 |
|---|---|
| 工具暴露给模型时怎么按能力筛 | [`README_CONTEXT.md`](README_CONTEXT.md) |
| 审批挂起后事件流怎么表现 | [`README_STARTUP.md`](README_STARTUP.md) |
| `create_team` 怎么发布任务板 | [`README_SUB_AGENT.md`](README_SUB_AGENT.md) |
| 总览 | [`README.md`](README.md) |
