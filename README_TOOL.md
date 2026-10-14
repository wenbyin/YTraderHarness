# TOOL_README.MD — Tool 子系统

> YTraderHarness · Tool 子系统事实手册
> 覆盖范围：`app/services/tool/`（定义 / 注册 / 发现 / 管理）、`app/agent/tools/processor/`（处理器链 / 拦截）、`app/infra/agent/`（执行器 / 暴露）、`app/api/v1/{tools,tool_processors,processors,capabilities}.py`。
> 本文只描述**代码里真实存在**的东西；注释与代码冲突处一律以代码为准并显式标注。

---

## 0. TL;DR

| 问题 | 一句话答案 |
|---|---|
| 工具怎么定义？ | 把方法挂在一个 service 类上，加 `@tool` 装饰器；装饰器同时生成 langchain 工具与注册表描述符 |
| 工具从哪来？ | 两路扫描：`@tool` 自注册（import 时）+ `ServiceMethodScanner` 遍历 `TOOL_SOURCE_PACKAGES` 触发未 import 模块 |
| 模型看到的函数名是什么？ | `tool_key` 的点号会被换成下划线（`capability.list` → `capability_list`），因为 OpenAI 函数名不允许点号 |
| 一次工具调用会经过什么？ | 处理器链：`select_specs` 鉴权 → 构造 `ToolInvocation` → 加载 `tool_processor` 表里的链 → 折叠洋葱 → `_base_call` 真正调 service 方法 |
| 危险工具谁来管？ | `approval_interceptor`（HITL）：挂 `BEFORE`，调 langgraph `interrupt()` 挂起图，等 `POST /runs/{id}/resume`；批准账本落 PostgresStore，重启不重复问 |
| 幂等靠什么？ | `tool_execution` 表唯一约束 `(session_id, run_id, tool_key, param_hash)` + 一条 `INSERT ... ON CONFLICT DO UPDATE WHERE status IN retryable` 原子抢占 |
| 有工具级超时/重试吗？ | **没有**，且是刻意不设（重试在模型运行层与图节点层） |

---

## 1. 阅读地图

```
app/services/tool/                    ← 定义 / 注册 / 发现 / 管理
├── registry/
│   ├── decorator.py      ← @tool 装饰器、ToolDescriptor、source_ref/tool_key 推导
│   ├── scanner.py        ← 两个扫描器（service 方法 / langchain 工具）
│   ├── discoverer.py     ← 装配两个扫描器、合并清单、冲突检测
│   ├── registry.py       ← ToolRegistry（按 tool_key 索引）+ TOOL_SOURCE_PACKAGES
│   ├── spec.py           ← ToolSpec / ToolDescriptor 数据与推导规则常量
│   ├── schema.py         ← 函数签名 → JSON Schema
│   ├── docstring.py      ← docstring → 参数/返回值说明
│   ├── selection.py      ← select_specs（可见=可执行）+ wire_name / tool_key_of
│   └── processor_registry.py ← 处理器注册表
├── application/          ← 查询 / 发现同步 / 文档 / 处理器 / 分类 服务
├── models/sqlalchemy/    ← 6 张表
├── repository/           ← 接口 + *_impl 实现
└── domain/schemas/       ← DTO / 枚举

app/agent/tools/
├── time_tool.py          ← 最简工具示例（刻意不写 TOOL_SOURCE_REF）
├── catalog_tool.py       ← 组合工具示例（写了 TOOL_SOURCE_REF）
└── processor/
    ├── base.py           ← BaseToolProcessor + 三类子类
    ├── context.py        ← ToolInvocation / ToolOutcome / ToolCallDenied
    ├── interceptor/      ← approval.py（HITL）/ auth.py / cache.py
    ├── transformer/      ← privacy.py
    ├── observer/         ← events.py / metrics.py
    └── runtime/          ← events.py / cache.py / metrics.py

app/infra/agent/
├── tool_executor.py      ← 执行入口（ToolExecutor.execute）
├── tool_exposure.py      ← 暴露给模型（RegistryToolExposureSource）
└── chain.py              ← 链编译与洋葱折叠（build_chain / compile_chain）
```

---

## 2. 全貌

```mermaid
flowchart TB
    subgraph DEF["工具来源"]
        SVC["service 类上的 @tool 方法<br/>（import 时 __set_name__ 自注册）"]
        CMP["app/agent/tools/ 下的<br/>langchain @tool 组合工具"]
    end

    subgraph SCAN["发现（按需，非启动期）"]
        MS["ServiceMethodScanner<br/>遍历 TOOL_SOURCE_PACKAGES"]
        LS["LangchainToolScanner<br/>扫 app.agent.tools"]
        DISC["ToolDiscoverer.default().discover()"]
    end

    REG["ToolRegistry<br/>dict[tool_key → ToolSpec]"]

    subgraph EXEC["一次调用"]
        EXP["tool_exposure.for_wire()<br/>tool_key 点号→下划线"]
        SEL["selection.select_specs<br/>可见集合 == 可执行集合"]
        INV["ToolInvocation<br/>arguments 私有副本 + caller 快照"]
        CHAIN["chain.build_chain / compile_chain<br/>按 sequence 折叠洋葱"]
        BASE["_base_call<br/>factory.scope 造 service + 调方法"]
    end

    EXE[("tool_execution 表<br/>UNIQUE(session,run,tool_key,param_hash)")]
    PRC[("tool_processor 表<br/>挂在 tool_id 上的链")]

    SVC --> MS
    CMP --> LS
    MS --> DISC
    LS --> DISC
    DISC --> REG
    REG --> EXP --> SEL --> INV --> CHAIN --> BASE
    PRC --> CHAIN
    INV --> EXE
    BASE --> OUT["ToolOutcome<br/>value / error / duration_ms / cached"]
```

**一句话架构**：工具是「注册表里的描述符 + 表里挂的处理器链 + 执行器里的洋葱折叠」三件套。模型只看到 `wire_name`，执行只认 `tool_key`，两者是**同一集合**的两个投影。

---

## 3. 工具如何定义

### 3.1 装饰器

```python
tool(func=None, *, key=None, name=None, description=None)
```

定义在 `app/services/tool/registry/decorator.py:270`。返回 `ToolDescriptor(StructuredTool)` —— **同时是 langchain 工具（模型侧）与描述符（调用点侧）**。

生命周期：

| 时机 | 行为 | 位置 |
|---|---|---|
| 类体执行 `__set_name__` | 从 `owner` 推 `source_ref`，写进注册表 | `decorator.py:146` |
| 实例访问 `__get__` | 返回绑定方法 | `decorator.py:190` |
| 类访问 `__get__` | 返回工具本体 | `decorator.py:190` |

约束：

- `owner` 必须存在（方法必须挂在类上，不能是模块级函数）—— `spec.py:144` 校验。
- 同步/异步皆可，按 `iscoroutinefunction` 落到 `func` / `coroutine` 字段 —— `decorator.py:319`。
- 参数推荐**单一 pydantic 请求模型**；否则回退到按签名合成 schema —— `schema.py:44` `resolve_request_model`。
- `description` 声明优先，否则回退 docstring —— `decorator.py:334`。

### 3.2 最小真实示例（`app/agent/tools/time_tool.py:54`）

```python
@tool
async def current_time(timezone: str = DEFAULT_TIMEZONE) -> dict:
    """
    查询当前的日期与时间。用于回答"今天几号""现在几点""今天星期几"。
    ...
    """
    zone = _resolve_timezone(timezone)
    now = datetime.now(zone)
    return {...}
```

组合工具（langchain 原生 `@tool`，跨 service）见 `app/agent/tools/catalog_tool.py:32` 的 `describe_platform`。

### 3.3 四个身份的推导规则

| 字段 | service 方法 | langchain 工具 | 规则出处 |
|---|---|---|---|
| `source_type` | `LOCAL_METHOD`（硬编码） | `LOCAL_METHOD`（硬编码） | `scanner.py:82`、`:161` |
| `source_ref` | `derive_source_ref(owner.__module__)` | 模块级 `TOOL_SOURCE_REF`，缺省则同样推导 | `decorator.py:165`、`scanner.py:213`、`registry.py:203` |
| `entrypoint` | `owner.__module__:owner.__qualname__.method` | 底层函数 `__module__:__qualname__` | `spec.py:268`、`spec.py:251` |
| `tool_key` | `f"{source_ref}.{name}"`（显式 `key` 优先） | `f"{source_ref}.{name}"` | `decorator.py:168`、`scanner.py:224` |

`source_ref` 的分组规则（`spec.py:39`）：

```
app.x                    → x
app.services.<模块>      → <模块>            （取第三段）
其它                     → 取最后一段
```

**`TOOL_SOURCE_REF` 的优先级**：模块属性为非空 `str` 时采用，否则推导 —— `scanner.py:213-219`。变量名常量 `SOURCE_REF_VARIABLE = "TOOL_SOURCE_REF"` 在 `spec.py:36`。

> ⚠️ `time_tool.py` **刻意不写** `TOOL_SOURCE_REF` —— 它的 docstring 明说：写了就等于给它单开一个"来源"。因此 `agent.current_time` 的 `source_ref` 是推导出来的 `agent`，而不是某个自定义分组。这条设计直接导致了 `tool_smoke` 的一条断言过期（见 §14）。

### 3.4 `SourceType` 枚举

`enum.py:41` 定义了 `LOCAL_METHOD` / `MCP` / `OPENAPI`，但**只有 `LOCAL_METHOD` 有扫描器**。`MCP` / `OPENAPI` 仅占枚举值，属预留。

---

## 4. 发现机制

### 4.1 扫描的包

`TOOL_SOURCE_PACKAGES`（`registry.py:53`）：

```python
app.services.document.application
app.services.file.application
app.services.memory.application
app.services.tool.application
app.agent.tools
```

### 4.2 时机与装配

- **按需，非启动期预热**。`ToolDiscoverer.default()` 装配两个扫描器（`discoverer.py:43`），`discover()` **每次重扫、无状态**（`discoverer.py:36`）。
- `@tool` 方法在 import 时已自注册；`ServiceMethodScanner.scan()` 额外遍历包，触发**尚未 import 的模块**完成登记（`scanner.py:87`、`registry.py:143`）——这是 `TOOL_SOURCE_PACKAGES` 存在的理由：否则一个没被任何人 import 的模块里的工具永远不会出现。

### 4.3 失败处理

| 情形 | 行为 | 位置 |
|---|---|---|
| 包不存在 | `logger.warning` 跳过 | `registry.py:263` |
| 包内某模块 import 崩 | **异常上抛，不吞** | `registry.py:255` |
| 未安装 langchain | 跳过 langchain 扫描 | `scanner.py:_import_base_tool` |
| 两个来源抢同一 `tool_key` | 直接 `ValueError` | `discoverer.py:115`、`scanner.py:185` |

---

## 5. 从函数签名生成 JSON Schema

`schema.py` + `docstring.py`。

| 事项 | 规则 | 位置 |
|---|---|---|
| 类型映射 | `str/int/float/bool/list/dict` → `string/integer/number/boolean/array/object` | `schema.py:34` `_JSON_TYPES` |
| 嵌套 pydantic 模型 | 用其 `model_json_schema()`，`$defs` 提到根并改名防冲突 | `_hoist_defs`，`schema.py:167` |
| 认不出的类型 | 回退 `"string"` | `schema.py` |
| **顶层 `Enum`** | 不在映射表里 → 会被降级成 `string`（局限） | 同表 |
| 必填 | 无默认值的参数进 `required` | `schema.py:145-148` |
| 默认值 | `schema["default"] = parameter.default` | `schema.py:146` |
| 参数描述 | 取自 pydantic 字段 `description`，按 JSON Pointer 抽取 | `extract_parameter_descriptions`，`schema.py:300` |
| docstring 解析 | 只认**顶格 Google 风格** `Args` / `Returns` | `docstring.py:58` |
| 返回值说明 | DTO（输出模型 docstring）优先，否则 docstring 的 `Returns` | `resolve_returns`，`docstring.py:135` |

### 5.1 `tool_parameter_documentation` 是覆盖，不是替换

它按 `parameter_path`（JSON Pointer，如 `/timezone`）存 `description_override`，生效值 = `override or source`（`_assembly.py:134-144`、模型 `tool_parameter_documentation.py:29`）。

**`input_schema` 的所有权仍在代码**——覆盖只改"说明文字"，改不了类型、必填、默认值。这是刻意的边界：schema 是代码契约，文案是运营资产。

---

## 6. 注册表与工具选择

### 6.1 `ToolRegistry`

- 结构：`_specs: dict[str, ToolSpec]`，**按 `tool_key` 索引**（`registry.py:75`），进程级单例 `get_tool_registry()`。
- 重复登记（`registry.py:98-111`）：

| 情形 | 行为 |
|---|---|
| 同 `key` + 同 `entrypoint` | 覆盖（热重载场景） |
| 同 `key` + 不同 `entrypoint` | `ValueError` |

### 6.2 `selection.py` —— 「可见 == 可执行」的唯一判据

`select_specs(tool_keys)`（`selection.py:4`）是**"哪些工具可用"的唯一定义**，被"暴露给模型"与"实际执行允许集"两处共用——这是保证「模型看得到的一定能执行、模型看不到的一定不能执行」的机制基础。

规则：

- 按传入顺序**去重保序**；
- 不在注册表、或没有 `ToolDescriptor` 的 key → **跳过 + WARNING**（`selection.py:169`）。

### 6.3 `wire_name` / `tool_key_of`

`selection.py:69-118`：

- `wire_name("capability.list")` → `"capability_list"`（点号 → 下划线）。原因是 OpenAI 的函数名必须匹配 `^[a-zA-Z0-9_-]+$`，点号非法。
- 反向 `tool_key_of()` **只在本次暴露集合内**查找；多对一时返回 `None` 并报 WARNING（不猜）。

---

## 7. 数据模型（6 张表）

### 7.1 `tool_definition`

| 列 | 含义 |
|---|---|
| `id` | PK |
| `tool_key` | **业务唯一键** |
| `name` / `source_type` / `source_ref` / `entrypoint` | 身份四件套 |
| `input_schema` / `output_schema` / `source_docstring` (JSONB) | **schema 快照** |
| `manifest_hash` | sha256 摘要，用于判漂移 |
| `source_state` | `ACTIVE` / `MISSING` |
| `discovered_at` / `last_synced_at` | 时间戳 |

约束：`UNIQUE(tool_key)`，索引 `source_type` / `source_state`。

> **注意：没有版本号列**。版本漂移靠 `manifest_hash` 判定——同一 `tool_key` 的 schema 变了，hash 就变。

### 7.2 其余五张表

| 表 | 关键列 | 约束 |
|---|---|---|
| `tool_documentation` | `tool_id`(PK, FK CASCADE), `description_override`, `return_description_override`, `updated_by` | 与工具一对一 |
| `tool_parameter_documentation` | `id`, `tool_id`(FK), `parameter_path`(JSON Pointer), `description_override` | `UNIQUE(tool_id, parameter_path)` |
| `tool_definition_classification` | `tool_id`(PK, FK CASCADE), `classification_id`(PK, FK→`capability.id` CASCADE) | 双列主键；**该表不在 `models.sql`**，是手写补的（`tool_classification.py:22`） |
| `tool_execution` | `id`, `session_id`, `run_id`, `tool_key`, `param_hash`, `status`, `result`, `error` | `UNIQUE(session_id, run_id, tool_key, param_hash)`；`CHECK status IN (...)`；部分索引覆盖 unsettled；**`run_id` 刻意无 FK** |
| `tool_processor` | `id`, `tool_id`(FK CASCADE), `processor_key`, `sequence`, `execution_mode`, `hook`, `config`(JSONB), `enabled` | `UNIQUE(tool_id, sequence)`；`CHECK(sequence >= 1)`、`CHECK(AROUND ⇒ SYNC)` |

两张 SQL 脚本 vs ORM 的差异：`tool_processor.sql` 带 `updated_at` 触发器，但 ORM **故意不镜像**（统一用 `onupdate`，`tool_processor.py:57`）——改这两处的人要知道它们本来就不打算一致。

---

## 8. 工具 ↔ 能力树（capability）

**关联字段**：`tool_definition_classification.classification_id` → `capability.id`。

**挂载方式**：`PUT /tools/{tool_id}/classifications`（`tools.py:251`）→ `replace_tool_classifications`：先校验 `classification_ids` 在 capability 表里存在（否则 `NotFoundError`），再 **delete + reinsert** 整体替换（`tool_classification_service_impl.py:61`、`tool_classification_impl.py:77`）。

**DELETE 引用检查**在 capability 侧（`capability_service_impl.py:195`）：

| 情形 | 行为 |
|---|---|
| 有子节点 | **永远拒绝**（409，`force` 也解不开） |
| 被工具引用且 `force=false` | 409，错误里给出引用数（`"正被 N 个工具引用"`） |
| `force=true` | 放行，关联随 `ON DELETE CASCADE` 消失 |

改 `parent_id` 时 `_assert_no_cycle` 拒绝成环，**步数上限 64** 防死循环（`capability_service_impl.py:264`）。

> 📌 **能力树本体在 `app/services/capability/`**，不在 `app/services/tool/`。Tool 这边只持有多对多关联表。理解这一点能避免在错误目录里找 capability 的 CRUD。

---

## 9. 处理器链

### 9.1 阶段（hook）与执行方式

| 枚举 | 取值 | 位置 |
|---|---|---|
| hook | `BEFORE` / `AFTER` / `AROUND` | `enum.py:122-124` |
| execution_mode | `SYNC`（参与结果）/ `ASYNC`（旁路不等） | `enum.py:140-141` |

**硬约束**：`AROUND ⇒ SYNC`、`TRANSFORMER ⇒ SYNC`。两层校验：类定义期（`base.py:176-201`）+ DB `CHECK`（`ck_tool_processor_around_sync`）。

### 9.2 顺序与洋葱折叠

- 链 = `tool_processor` 表里 `tool_id` 对应的行，按 **`sequence` 升序**（`schemas/processor.py:176`，从 1 连续）。
- 折叠从**链尾向前**，每层的 `next_call` 包住其下游（`chain.py:278-336`）。
- **`AROUND` 在链上是两条记录**（`chain.py:87` `_AROUND_RECORDS = 2`）：第一次出现 = 进入包裹其后全部；第二次 = 执行期 **no-op**（`chain.py:307-313`）。这就是"AROUND 为什么占两行"。
- `validate_onion`（`tool_processor_service_impl.py:55-128`）保证 AROUND 区间**洋葱嵌套、不得交叉**。

### 9.3 处理器基类

`BaseToolProcessor`（`base.py:86`）**没有抽象方法**，靠注册表结构区分类型。三类子类（`base.py:435-483`）：

| 类别 | 能做什么 |
|---|---|
| `ToolInterceptor` | 改**控制流**（放行 / 短路 / 拒绝） |
| `ToolTransformer` | 只改**数据** |
| `ToolObserver` | **只读旁路**（`processor_type` 由继承定死，不可声明） |

身份与参数：

- `name` —— 注册键，落 `processor_key`，**永不复用**（`base.py:127`，改名的纪律）；
- `display_name` / `description`；
- `config_model` —— pydantic 模型，转成 JSON Schema 作为配置契约；
- `supported_hooks` / `supported_execution_modes`；
- `get_config_schema()`（`base.py:212`）。

三个语义方法（`base.py:336-403`）：

| 方法 | 返回 | 说明 |
|---|---|---|
| `before()` | `ToolInvocation`（放行）或 `ToolOutcome`（短路，不调工具） | 可改参数 |
| `after()` | 改写后的 `ToolOutcome`（无 `next_call`） | 可改结果 |
| `around()` | 包整次调用，可 0 / 多次 `next_call()` | 可重试 / 计时 / 缓存 |

未实现的方法默认抛 `NotImplementedError`（`_unsupported`）。

### 9.4 处理器上下文（`context.py`）

| 对象 | 字段 |
|---|---|
| `ToolInvocation` | frozen；`arguments`（**可变私有副本**）、`tool_id`、`context`（共享区）、`CALLER_CONTEXT_KEY = "caller"` |
| `ToolOutcome` | `value` / `error` / `duration_ms` / `cached` |
| `ToolCallDenied` | 拒绝信号 |
| `AsyncToolCall` | ASYNC 观察者句柄 |

> ⚠️ `runtime` 经**参数注入**，**禁止存 `self.runtime`**（`base.py:106`）。原因是处理器实例在进程内共享/复用，把请求级 runtime 挂到 self 上会造成串台。

---

## 10. HITL 审批（`interceptor/approval.py`）

### 10.1 触发与决策

- 挂 **`BEFORE`**（`approval.py:623`）。
- **哪些工具需要审批**：该工具的链上是否挂了 `approval_interceptor` **且配了 `prompt`**。规则即数据，不在代码里写死白名单。
- `ApprovalDecision`（`approval.py:247`）：`approve` / `reject` / `always`。
  - `always` 需要 `ApprovalConfig.always_allowed = True`，否则**降级为「仅这次」**。

### 10.2 账本

| 项 | 值 |
|---|---|
| 协议 | `ApprovalLedger` |
| 生产实现 | `StoreApprovalLedger`，走 `runtime.store`（即 `AsyncPostgresStore`），`approval.py:358` |
| 兜底实现 | `InMemoryApprovalLedger`（**不跨重启**，会打 warning） |
| 命名空间 | `("tool_approval", run_id)`（`approval.py:225`） |
| 键 | `tool_key:suffix`；`TOOL_WIDE_KEY = "*"` 表示"本次运行内该工具永放行"（`:222,:231`） |

### 10.3 幂等键（关键）

幂等键 = `runtime.tool_call_id`（模型给的稳定 id）+ `run_id`。`run_id` 取自 `runtime.context.pipeline_run_id`，**不是 thread**（`approval.py:914-954`）。

> 🐛 历史修正：审批命名空间**曾误用 `thread_id`**，后果是"始终允许"会**泄漏到同会话的后续运行**。已改为 `run_id`（`approval.py:192-203`）。这类 bug 不报错、只在安全审计时暴露，是 HITL 最典型的坑。

### 10.4 挂起与恢复

1. `before()` 调 langgraph `interrupt()`（`approval.py:684`）→ 图停，产出 `waiting_input`。
2. 恢复走 `POST /runs/{run_id}/resume`（`api/v1/sessions.py`）。
3. **两者必须同上线**——只上一边会出现"停了没人能恢复"。

### 10.5 重启会不会重复问？

**不会**（前提成立时）。批准记录落在持久 store（与 checkpointer 同连接池的 `AsyncPostgresStore`）。重跑时账本命中 → 直接放行、**不再调 `interrupt()`**（`approval.py:662-679`）。

前提：**一个工具一个分支 `Send` 扇出**——否则同一 run 内多个分支的 `tool_call_id` 语义会打架。

### 10.6 超时与 TTL（两个容易误读的点）

| 配置 | 默认 | 真相 |
|---|---|---|
| `wait_seconds` | `300`（`approval.py:551`） | **只下发给前端显示，服务端不判**。真超时属运行生命周期清理任务 |
| `on_timeout` | `DENY` | **只在拿不到 runtime（无 interrupt 能力）时生效**，不是真超时分支（`approval.py:567`） |
| `always_ttl_seconds` | `3600` | store 接口收**分钟**，故传给 store 前 `/60`（`approval.py:385-396`）；忘了除会让有效期长 60 倍 |

---

## 11. 另外三个内置处理器

### 11.1 `auth`（`interceptor/auth.py`）

- 挂 `BEFORE`；依据 `invocation.context["caller"]` 的 `roles` / `groups`（运行建立时的 claim 快照）。
- `AuthConfig.require_roles`（任一命中即可）/ `require_group`（精确匹配，**不做 owner 宽松**）/ `on_missing = "deny"`（**fail-closed**）。
- 拒绝 → `ToolCallDenied`，**文案刻意含糊**（不告诉调用方差哪个角色）。
- 空配置仍挂载 → 挂载期 `ValidationError`（400，`auth.py:98-129`）。

### 11.2 `cache`（`interceptor/cache.py`）

| 项 | 值 |
|---|---|
| 挂载 | `BEFORE + AFTER`，或 `AROUND`（`chain_smoke` 专门验证"同一个键占多个 hook 位置"） |
| `ttl_seconds` | 默认 `300`；**`0` = 不缓存** |
| `key_fields` | 默认**全部参数**；需要隔离时显式列出 |
| 缓存键 | `sha256(canonical_json(tool_key + 选定参数))`（`cache.py:226-262`） |
| 只缓存 | **成功**结果（失败/已缓存不回填） |
| 后端 | `InProcessToolCache`（`runtime/cache.py`，LRU + `monotonic` TTL，**进程内、重启清零**） |

### 11.3 `privacy`（`transformer/privacy.py`）

- 挂 `BEFORE` 抹**参数**、`AFTER` 抹**结果**（均 `SYNC`）。
- `PrivacyTarget`：`PHONE` / `EMAIL` / `ID_CARD` / `BANK_CARD`。**长模式先跑**，防短模式先匹配造成二次误抹。
- **默认不含银行卡** —— 因为卡号正则容易误伤金额与订单号。
- **是正则，不是 NER**：属最佳努力，不保证召回。`keep_length`（是否保长度）、`max_depth`（递归深度）可配。

---

## 12. 观测：两套 events + metrics

### 12.1 events

`ToolEventObserver`（ASYNC，BEFORE + AFTER）向进程内 `ToolEventBus` 发布：

| 阶段 | 事件 |
|---|---|
| BEFORE | `STARTED` |
| AFTER | `SUCCEEDED` / `FAILED` / `DENIED` |

**`DENIED` 与 `FAILED` 严格区分**：`DENIED` = 工具**根本没跑**（被 auth/审批拦下），`FAILED` = 跑了但出错（`events.py:159-190`）。

其它性质：

- `ToolEvent` 默认 **`include_arguments = False`** —— 参数可能含业务数据，默认不带走。
- `to_payload()` 保证 JSON 可序列化（SSE 要用）。
- 订阅用 `async with bus.subscribe()`，退出即退订；队列满**丢最旧**并按 2 的幂告警（`runtime/events.py:127-265`）。

### 12.2 metrics

`ToolMetricsObserver`（ASYNC）记：`calls` / `errors` / `denied` / `errors_by_type`（按异常类型分桶）/ 耗时。

- **`denied` 单列，不算 failure**。
- `error_rate = errors / calls`（`runtime/metrics.py:80-101`）。
- 前后两半靠 `invocation.context["metrics.started_at"]` 传开始时刻，AFTER 用完即 `pop`（`observer/metrics.py:58,:133`）——**这正是 BEFORE 临时键必须被清理的原因**。

### 12.3 落库与否

两套 observer **都只活在进程内，重启清零**。没有 Prometheus 导出，也没有把事件写进 DB。

---

## 13. 执行链路（按调用顺序）

```mermaid
sequenceDiagram
    participant N as react tools 节点
    participant E as ToolExecutor
    participant S as selection
    participant C as chain
    participant F as ServiceFactory
    participant DB as tool_execution

    N->>E: execute(tool_key, arguments, tool_id, tool_call_id, caller_identity, runtime)
    E->>S: select_specs([tool_key])  ← 校验可执行
    E->>E: 构造 ToolInvocation（私有 arguments 副本 + caller 快照）
    E->>DB: claim(...)  ← 原子抢占，None = 放弃
    E->>C: _load_chain(tool_id) → build_chain → CompiledChain
    Note over E: ASYNC BEFORE 观察者 create_task（旁路，不 await）
    E->>C: compile_chain 折叠洋葱
    C->>C: 各 BEFORE → 最里层
    C->>F: factory.scope → 造 service → 调方法
    F-->>C: 返回值 / 异常
    C->>C: 各 AFTER（由内向外）
    Note over E: ASYNC AFTER 观察者 create_task
    E-->>N: ToolOutcome → 转 ToolMessage 回灌模型
```

逐步：

1. react 图的 tools 节点（`graph/react/tools.py:1046`）→ `get_tool_executor().execute(...)`（`tool_executor.py:143`）。
2. `select_specs([tool_key])` 校验可执行（`:185`）。
3. 构造 `ToolInvocation`（私有 arguments 副本、塞 caller 快照，`:212-219`）。
4. `_load_chain(tool_id)` → `build_chain`（跳过已删实现、未 `enabled` 的行，`:160-217`）→ `CompiledChain`（`sync_entries` + `async_before/after`）。
5. ASYNC BEFORE 观察者先 `create_task` 起掉（`:231-241`）；`compile_chain` 折叠；最里层 `_base_call` 造 service + 调方法（经 `factory.scope`，`:300-389`）。
6. 同步链执行完，ASYNC AFTER 观察者起掉。
7. 结果 `ToolOutcome` 回灌模型（tools 节点转成 `ToolMessage`）。

### 13.1 工具如何"暴露"给模型

- `RegistryToolExposureSource.resolve`（`tool_exposure.py`）调 `select_specs` —— 保证「**可见 == 可执行**」是同一集合。
- `for_wire()` 把 `tool_key` 经 `wire_name()`（`selection.py:69`）转下划线：`capability.list_capabilities` → `capability_list_capabilities`，原因是 OpenAI 函数名须匹配 `^[a-zA-Z0-9_-]+$`。
- 反向 `tool_key_of()` **只在本次暴露集合内**找（`selection.py:84`）。
- **副本不污染注册表描述符**：`for_wire` 用 `model_copy`，改的是副本。

---

## 14. 错误分类与幂等

### 14.1 执行状态机

`ToolExecutionStatus`（`enum.py:144-207`）：

```
PENDING / RUNNING / SUCCESS / FAILED / UNKNOWN / CANCELLED / REJECTED / INTERRUPT
```

`RETRYABLE_EXECUTION_STATUSES = {FAILED, CANCELLED, REJECTED, INTERRUPT}`（`:224`，**fail-closed**：不在集合里的一律不重试）。

| 观察 | 含义 |
|---|---|
| `FAILED` / `UNKNOWN` **当前无写入方** | 未分类异常一律落 `RUNNING` —— 无法确定副作用，**防误重试** |
| `INTERRUPT` | 仅支持**执行前**中断（`enum.py:180`） |

### 14.2 异常 → `ToolOutcome`

工具异常被 `_base_call` 收进 `outcome.error`（`tool_executor.py:373`），**不向上抛**。唯一例外：`asyncio.CancelledError` 上抛（`:369`）。

### 14.3 超时

**本子系统没有工具级超时**。executor 未用 `wait_for`；`ApprovalConfig.wait_seconds` 只用于显示。超时在**模型运行层**（`ModelRuntimePolicy.timeout`）与**图节点层**处理。

### 14.4 幂等

唯一约束 `uq_tool_execution_identity = (session_id, run_id, tool_key, param_hash)`。

`claim()`（`tool_execution_impl.py:45-102`）用**一条语句**完成「查 + 占」：

```sql
INSERT ... ON CONFLICT DO UPDATE ... WHERE status IN (retryable)
```

返回 `None` = 放弃执行。这条设计专门挡 **`Send` 扇出并发重复调用** —— 两个分支同时要调同一个工具时，只有一个能抢占成功。

---

## 15. HTTP API 全清单

全部挂在 `protected_router`（前缀 `/api/v1`，统一 `Depends(current_user)`）。

### 15.1 `tools.py`（`tool.yaml`）

| 方法 | 路径 | 职责 |
|---|---|---|
| POST | `/tools/discovery/sync` | 扫描并把发现结果同步进 `tool_definition` |
| GET | `/tools` | 列出工具（分页） |
| GET | `/tools/{tool_id}` | 工具详情（含 manifest） |
| GET | `/tools/{tool_id}/documentation` | 取文档（生效值） |
| PUT | `/tools/{tool_id}/documentation` | 覆盖文档 |
| GET | `/tools/{tool_id}/classifications` | 取能力关联 |
| PUT | `/tools/{tool_id}/classifications` | 整体替换能力关联 |

### 15.2 `tool_processors.py` + `processors.py`（`tool_processor.yaml`）

| 方法 | 路径 | 职责 |
|---|---|---|
| GET | `/processors` | 列出**代码实现**的处理器 |
| GET | `/processors/{processor_key}` | 取处理器定义（含 config schema） |
| GET | `/tools/{tool_id}/processors` | 取该工具挂的链 |
| POST | `/tools/{tool_id}/processors` | 追加一条 |
| PATCH | `/tools/{tool_id}/processors/{processor_id}` | 改一条 |
| DELETE | `/tools/{tool_id}/processors/{processor_id}` | 删一条 |
| PUT | `/tools/{tool_id}/processors` | **整体替换**链 |
| PUT | `/tools/{tool_id}/processors/order` | 重排（`sequence`） |

错误码：未注册 404 / hook·mode 不兼容或链过期 409 / config 不合法 400。

### 15.3 `capabilities.py`

`GET|POST /capabilities`、`GET|PATCH|PUT|DELETE /capabilities/{capability_id}`。

### 15.4 注册顺序约束（不满足会静默匹配错）

| 约束 | 位置 |
|---|---|
| `/tools/discovery/sync` 必须注册在 `/tools/{tool_id}` **之前** | `tools.py:15`（文件内实际顺序 `:66` vs `:143`） |
| `/tools/{tool_id}/processors/order` 必须注册在 `/tools/{tool_id}/processors` **之前** | `tool_processors.py`（实际 `:58` vs `:85`） |
| `tools_router` 先于 `tool_processors_router` | `router.py` |
| `processor_capabilities_router` 先于 `document_definitions_router` | `router.py` |
| `capabilities_router` 最后 | `router.py` |

---

## 16. 已知缺口、桩、注释失真

### 16.1 刻意预留（不是 bug）

| 项 | 说明 |
|---|---|
| `SourceType.MCP` / `OPENAPI` | 枚举存在，**无扫描器**（`enum.py:41`） |
| `ToolExecutionStatus.FAILED` / `UNKNOWN` | 无写入方，保留语义（`enum.py:164-178`） |
| 无重试 / 超时拦截器 | **刻意不设**（`interceptor/__init__.py:15`）：图节点级已有重试与超时，重复一层只会放大重试次数 |
| 无 reaper | 没有清理 `PENDING` / `RUNNING` 遗留行的任务（`tool_execution.py` 注释） |

### 16.2 契约与实现的不一致

| 项 | 说明 |
|---|---|
| `tool_definition_classification` 不在 `models.sql` | 与其它 5 张表**来源不一**，由 ORM 手写（`tool_classification.py:22`） |
| `tool.yaml` 的 `Classification` schema 是 `Capability` 的**手工复制** | `tool.yaml:457` 自注 `"Kept in sync by hand"`，无跨文件 `$ref` → 字段易漂移 |
| `tool_processor.sql` 的 `updated_at` 触发器 vs ORM | ORM **故意不镜像**，统一用 `onupdate`（`tool_processor.py:57`） |

### 16.3 未确认

- `tool_execution` 的**服务实现完整度**未逐行核对（接口基类里的 `NotImplementedError` 属正常 ABC 模式，非功能缺失）。
- `graph/react/tools.py:1046-1222` 里「异常 → `ToolExecutionStatus`」的具体映射（`decide`）与审批 `interrupt` 在节点内的完整装配，未在本次阅读范围内。

### 16.4 实测发现的过期断言（**2 条**）

| 脚本 | 失败断言 | 根因 | 是否链路缺陷 |
|---|---|---|---|
| `app.agent.tool_smoke` | `组合工具走的是 TOOL_SOURCE_REF 模块级变量` | 断言写 `all(m.source_ref == "agent.platform")`，但组合工具现在有 **2 个**：`agent.current_time`（`source_ref="agent"`，因 `time_tool.py` 刻意不写 `TOOL_SOURCE_REF`）+ `agent.platform.describe_platform`（`source_ref="agent.platform"`） | **否**，断言过窄 |
| `app.agent.prompt_smoke` | `两个 model_role 都登记了` | 断言 `keys == {CLASSIFICATION, SUMMARIZATION}`，但 `found_prompts()` 现返回 **7 个**键（另加 5 个 `memory.*` 提示词） | **否**，目录正常增长 |

修法（留给维护者）：`tool_smoke` 改成断言"组合工具的 `source_ref` 由**模块路径推导**，只有显式声明 `TOOL_SOURCE_REF` 的才是自定义分组"；`prompt_smoke` 改成断言"含 `CLASSIFICATION` / `SUMMARIZATION` 且 `REASONING` 不在其中"，而不是精确集合相等。**不要**为了测试绿去给 `time_tool.py` 加 `TOOL_SOURCE_REF` —— 那会破坏刻意设计的分组。

---

## 17. 实测验证

全部用 `./.venv/Scripts/python.exe -B -m <module>`（`-B` 阻止写 `.pyc`）。

| 脚本 | 结果 | 覆盖 |
|---|---|---|
| `app.agent.tool_smoke` | 部分通过（1 条断言过期，见 §16.4） | 工具定义/发现/文档覆盖/能力树 CRUD/关联/成环拒绝/引用检查 |
| `app.agent.tool_processor_smoke` | ✅ 全部通过 | cache 短路成 `ToolOutcome` 并标 `cached`、底层只执行一次、`ttl=0` 不缓存、指标记次数与失败率、错误按类型分桶、BEFORE 临时键被清理、BEFORE 发 STARTED / AFTER 发 SUCCEEDED、事件默认不带参数且可 JSON 序列化、退出订阅上下文自动退订 |
| `app.infra.agent.chain_smoke` | ✅ 全部通过 | 半条 AROUND 被 `enabled` 关掉、同一个键占多个 hook 位置、AROUND+ASYNC 被兜住、三条 AROUND 记录报错、实现已删的行跳过并留痕、空链时 base 直接被调用 |
| `app.infra.agent.tool_execution_smoke` | ✅ 全部通过 | 真处理器（cache 第一次未命中/第二次命中）、ASYNC 观察者被起掉且返回值不被等待 |

**量化指标口径**（由 `ToolMetricsObserver` 给出）：`calls`、`errors`、`denied`、`error_rate`、`errors_by_type`、耗时。**注意这些是进程内计数，重启清零，不落库** —— 要做长期观测必须在外部接一层。

---

## 18. 设计取舍（读代码时的"为什么"）

1. **`TOOL_SOURCE_PACKAGES` 而不是全仓扫描**。显式列出 5 个包，好处是"工具从哪来"可枚举、可审查，代价是新增来源要改常量。全仓扫描会让任何被 import 的模块里的 `@tool` 都能悄悄进注册表——那是供应链风险。
2. **`source_ref` 由模块路径推导**。让分组（`agent.platform`、`memory`…）与代码结构对齐，改目录就改分组。代价是需要显式声明自定义分组时得写 `TOOL_SOURCE_REF`（且写了就得接受它成为独立分组）。
3. **`select_specs` 是唯一判据**。把"能看见"与"能执行"绑到同一函数，消灭"模型看到但执行时 403"这类不一致。
4. **AROUND 占两条记录**。用同一个数据结构表达洋葱进出，避免引入第二种链式描述格式。代价是 `_AROUND_RECORDS = 2` 这个隐式约定必须靠 `validate_onion` 守住。
5. **没有工具级重试/超时**。重试要在**知道副作用是否发生**的层级做。工具层不知道，模型运行层和图节点层知道。在工具层加会得到"重试了但有副作用"的最坏组合。
6. **`FAILED`/`UNKNOWN` 留空**。宁可把它记成 `RUNNING`（不确定）也不给一个错误的终态——因为终态会驱动自动重试。
7. **审批命名空间用 `run_id` 而不是 `thread_id`**。HITL 的授权粒度必须是"这一次运行"，不是"这个会话"。用 thread 会让一次"始终允许"永久污染整个会话。
8. **`DENIED` 单列不进 failure**。被拦下不是故障，把它算进错误率会让"安全策略生效"看起来像"系统不稳定"。
9. **参数文档是覆盖层，schema 是代码**。让运营能改文案，但改不了类型契约。

---

## 19. 文件索引

| 关注点 | 先读 |
|---|---|
| 工具定义 | `app/services/tool/registry/decorator.py`、`app/agent/tools/time_tool.py`、`app/agent/tools/catalog_tool.py` |
| 发现 | `app/services/tool/registry/{discoverer,scanner,registry}.py` |
| Schema | `app/services/tool/registry/{schema,docstring}.py` |
| 选择/命名 | `app/services/tool/registry/selection.py` |
| 表结构 | `app/services/tool/models/sqlalchemy/`、`models.sql`、`tool_processor.sql` |
| 能力树 | `app/services/capability/`、`app/api/v1/capabilities.py` |
| 处理器契约 | `app/agent/tools/processor/{base,context}.py` |
| HITL | `app/agent/tools/processor/interceptor/approval.py` |
| 链编译 | `app/infra/agent/chain.py` |
| 执行入口 | `app/infra/agent/tool_executor.py` |
| 暴露给模型 | `app/infra/agent/tool_exposure.py` |
| 观测 | `app/agent/tools/processor/{observer,runtime}/` |
| 验证 | `app/agent/tool_smoke.py`、`app/agent/tool_processor_smoke.py`、`app/infra/agent/{chain_smoke,tool_execution_smoke}.py` |
