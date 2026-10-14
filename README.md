# YTraderHarness

> 一个**面向生产**的 Agent 运行时：把「模型怎么选、上下文怎么拼、工具怎么放行、审批怎么落地、记忆怎么提炼」
> 全部做成**显式配置 + 显式失败**的数据，而不是散在 prompt 和 if-else 里的隐性约定。
>
> Python 3.14 · FastAPI · LangGraph · PostgreSQL(pgvector) · Redis Stream · Milvus · Elasticsearch · Neo4j

---

## 一句话定位

大多数 agent 框架把"能跑通一次对话"当作终点；这个项目把**跑通之后的事**当作起点：
改一个工具的说明要不要发版、给某个工具换前置校验要不要重启、模型漏配会不会静默降级、
审批记录重启后还在不在、流式输出里混进模型的私有标记会不会原样发给用户。

这些问题的共同答案是同一条立场 —— **静默失败是最贵的失败**。
本仓库几乎每个子系统都在反复做同一个选择：**宁可显式炸，也不要悄悄跳过**（见 [`README_DOCUMENT.md` §15.1](README_DOCUMENT.md)）。

---

## 四个核心能力

### 一、工具定义与拦截器链**运行时**可改

**主张：改工具给模型看的说明、换它的前置/后置处理器、调它们的顺序 —— 都不发版、不重启。**

#### 改「说明」：`PUT /tools/{tool_id}/documentation`

可以覆盖工具级 `description`、返回值说明，以及**逐个参数**的 `description_override`。

| 事实 | 说明 |
|---|---|
| 生效时机 | **下一次模型调用**（每次 build context 时现读，无缓存） |
| 覆盖面 | **只改描述文本** —— `input_schema` 的类型 / 必填 / 默认**归代码所有**，运行时改不了 |
| 隔离性 | `for_wire` 返回描述符的**副本**，进程内注册表**永不被改写** —— 运行时编辑不会污染其他会话或并发 run |

#### 改「链」：`/tools/{tool_id}/processors`

| 操作 | 端点 | 说明 |
|---|---|---|
| 读链 | `GET /tools/{tool_id}/processors` | 按 `sequence` 升序 |
| 追加 | `POST …/processors` | `sequence` 只是位置**建议**，实际插入位置会重算 |
| 改一项 | `PATCH …/processors/{processor_id}` | 只能改 `execution_mode` / `hook` / `config` / `enabled` —— **改不了 `processor_key` 与 `sequence`** |
| 删一项 | `DELETE …/processors/{processor_id}` | 其余自动重排为 1..N |
| 整体替换 | `PUT …/processors` | 数组顺序 = 执行顺序 |
| 重排 | `PUT …/processors/order` | 必须给出**完整** id 集合，否则 409「链已过期」 |

**⚠️ 每次工具调用都重新从 DB 读链，没有任何缓存**（`ToolExecutor._load_chain`）。
所以改完**下一次调用就生效** —— 这是"运维友好"的实现基础。

**⚠️ 一条重要边界**：处理器**实现**是只读目录（`/processors`）。
你只能在运行时**挂载已有的**处理器并调整它们的顺序与配置，**不能在运行时新建一个处理器** —— 那需要发版。

#### 六个内置处理器

按反射从 `app.agent.tools.processor` 自动发现：

| processor_key | 类型 | 支持的 hook | 执行模式 |
|---|---|---|---|
| `approval_interceptor` | INTERCEPTOR | BEFORE | SYNC |
| `auth_interceptor` | INTERCEPTOR | BEFORE | SYNC |
| `cache_interceptor` | INTERCEPTOR | BEFORE + AFTER + AROUND | SYNC |
| `privacy_transformer` | TRANSFORMER | BEFORE + AFTER | SYNC |
| `event_observer` | OBSERVER | BEFORE + AFTER | ASYNC |
| `metrics_observer` | OBSERVER | BEFORE + AFTER | ASYNC |

#### 洋葱折叠模型

`compile_chain` 从链尾往头递归包：每一层把 `inner` 作为自己的 `next_call`，
**最内层就是真正调 service 方法的 `_base_call`**。

- **BEFORE** 在外层：可以直接返回 `ToolOutcome` **短路**整条链（缓存命中 / 审批拒绝 / 鉴权拒绝 → 工具根本不跑）
- **AFTER** 在内层：先拿到 `outcome`，可以改写它
- **AROUND** 占**两行**（进入标记 + 无操作退出标记），把下游整个包起来
- **ASYNC 观察者**用 `create_task` 甩出去，**不 await** —— 监控不阻塞工具调用

#### 写入即校验：改错了当场被拒，而不是等运行时出事

这是「运维友好」里最实的一块。所有校验在**写链时**跑完：

| 错误 | 结果 |
|---|---|
| 未知 `processor_key` | **404** |
| hook 不被该处理器支持 | **409** |
| 执行模式不支持 | **409** |
| AROUND + ASYNC 组合 | **409**（另有 DB `CHECK` 约束兜底） |
| `config` 不过该处理器的 `config_model` | **400**（pydantic） |
| **洋葱区间交叉**（区间未正确嵌套） | **409** |
| 重排时 id 集合不全 | **409** |

**三层防护**：应用层校验 → DB `CHECK` 约束（防手改库）→ **执行期** `build_chain._validate` 再兜一次
（半条链 / AROUND+ASYNC 会在真正调用前抛错）。

> 洋葱嵌套用**栈式校验**（`validate_onion`）保证区间正确嵌套而非交叉 ——
> 交叉的区间意味着 AROUND 的进入与退出挂到了不同的层，折叠出来的调用顺序不是你想的那个。

#### 发现同步：`POST /tools/discovery/sync`

重新扫描 `TOOL_SOURCE_PACKAGES`，把 `tool_definition` 表与代码对齐（create / update / unchanged / missing）。

⚠️ **能**发现**新加的**工具（模块从未被 import，扫描器会 import 它）；
⚠️ **不能**拾取对**已 import 模块**的代码修改 —— Python 不会重新执行已加载的模块。**改代码仍然要重启。**

---

### 二、工具级监控

**主张：每一次工具调用都能回答"跑了没、成功没、花了多久、因为什么失败、是不是重复调用"。**

要分清**两层**，它们的持久性完全不同。

#### (a) 进程内实时（重启清零）

| | 内容 |
|---|---|
| **事件** `ToolEventBus` | `STARTED` / `SUCCEEDED` / `FAILED` / **`DENIED`** |
| **指标** `ToolMetricsRegistry`（按 tool_key） | `calls`、`errors`、`denied`、`total/min/max/avg_duration_ms`、`error_rate`、`errors_by_type`、`last_called_at` |

> ⚠️ `DENIED` 与 `FAILED` 是**两件事**：`DENIED` = 工具**根本没跑**（被 auth / approval 短路），
> `FAILED` = 跑了但报错。指标里 `denied` 也**单独计数、不计入 `errors`** —— 否则"审批拒绝"会污染错误率。
> 默认 `include_arguments=False`（不把参数带进事件）。

#### (b) 落库的权威记录 —— `tool_execution` 表

| 列 | 含义 |
|---|---|
| `session_id` / `run_id` | 关联到会话与运行（**刻意不建外键**） |
| `tool_key` | 工具身份 |
| **`param_hash`** | `sha256(canonical_json(模型给的原始参数))`，在**链跑之前**算 —— 幂等键 |
| `status` | 8 值枚举：`PENDING` / `RUNNING` / `SUCCESS` / `FAILED` / `UNKNOWN` / `CANCELLED` / `REJECTED` / `INTERRUPT` |
| `result` | JSONB，**经处理器链脱敏之后**的结果 |
| `error` | JSONB `{type, message}` |

**这张表就是幂等的基础**：唯一约束 `(session_id, run_id, tool_key, param_hash)` +
一条 `INSERT ... ON CONFLICT DO UPDATE WHERE status IN retryable` **原子抢占**。
重放与并发重复调用交给数据库，应用层不做"先查后写"那种有竞态的判断。

#### (c) 与 run 级记录的关系

- `agent_run` 是 run 级**状态**（`queued/running/waiting_input/completed/failed/cancelled`），**不是可观测表**
- react 图的 `node_span` 走 `NodeLogRecorder` —— **只打日志、不落库**（对比：文档管线走 `SqlAlchemyRunRecorder` → `document_pipeline_node_run` **表**）
- `tool_execution.run_id` ↔ `agent_run.run_id` 是同一套运行标识，但是**两张独立的表**，`agent_run` 不汇总 `tool_execution`

#### ⚠️ 运维现实（必须知道）

当前 tool 级可观测性的**边界**：

| 限制 | 后果 |
|---|---|
| 指标**没有查询接口** | `snapshot()` / `for_tool()` 只有测试和 smoke 脚本在用，**没有路由暴露** |
| **没有 Prometheus 导出** | 依赖里没有 Prometheus；注释建议接 Pushgateway 或换掉该模块 |
| **重启清零** | 事件与指标都是进程内单例 |
| **`tool_execution` 没有 API 读** | 只有拒绝消息回查时内部读一次 |
| **不记 latency、不记 token 成本** | `tool_execution` 没有时长列；工具层没有 token 记账 |
| **没有 reaper** | 僵死的 `PENDING` / `RUNNING` 行不会被清理 |
| `FAILED` / `UNKNOWN` **没有写入方** | 实际落到 `RUNNING`（枚举值存在但没人写） |

**一句话总结**：**实时诊断能力强，历史可查询能力弱。**
要做真正的 tool 级监控面板，得自己接一层 —— 数据是有的（`tool_execution` + 指标快照），只是没有出口。

---

### 三、渐进式工具暴露：按树型业务分类现算

**主张：不给模型看全部工具。每一轮运行开始时，按业务分类**现算**一遍它该看哪些。**

#### 链路

```
用户输入
  └─ builder 节点（每个 run 只跑一次）
       ├─ 从 DB 现读活跃能力候选集  CapabilityService.list_capabilities(is_active=True)
       │    └─ 读不到 → 降级为空集 + 记录 note（不炸）
       ├─ 一次 LLM 调用（agent_type="react" + ModelRole.CLASSIFICATION）
       │    └─ 候选集**渲染进提示词**：每行 "code：name —— description"
       ├─ 产出 capability_needs（限制在候选集内，未知落 other）
       └─ tool_keys = 这些能力 code 映射到的工具
  └─ 写入 context_policy.tool_keys（随 checkpoint 持久化）
       └─ 每次模型调用：CapabilityProcessor 重算「可绑定的工具对象」（纯内存、零 DB）
```

#### 两个必须分清的粒度

| | 何时算 | 运维改数据的生效点 |
|---|---|---|
| **key 集合** `context_policy.tool_keys` | **每 run 一次**（builder 只跑一次，图里没有回边） | **下一次 run** —— 当前 run 已快照，不受影响 |
| **可绑定工具对象** | **每次模型调用**重算（INITIAL 与每次 CONTINUATION 各一次） | 立即 |

> 为什么这样做得通：`select_specs` 是**纯内存函数、零 DB 查询**，所以每次模型调用重算的代价可以忽略。

#### 「可见 == 可执行」是构造出来的，不是靠约定

**可见侧**（`CapabilityProcessor` → `RegistryToolExposureSource.resolve`）与
**执行侧**（tools 节点 `_wrap_tool_call`）调用的是**同一个纯函数 `select_specs`，传同一份 `tool_keys`**。
因此"模型看得见的工具集合"与"允许执行的工具集合"**按构造相等**。

两条诚实的边界：

1. **在集合里 ≠ 一定会执行。** 通过名称暴露门之后，下游还有：目录解析失败（**fail-closed**）、
   幂等账本不可用、**参数相同的重复调用被抢占**、处理器链拒绝（审批/鉴权/隐私）。这些都会正确地拒绝。
2. **`wire_name` 是多对一的**（`a_b.c` 与 `a.b_c` 都会被换成 `a_b_c`）。命中这种碰撞时，
   工具可能"可见但被执行侧判为未暴露" —— 有 **WARNING**，不是静默选择。

#### 运维能改什么（不发版）

| 改什么 | 端点 |
|---|---|
| 能力树增删改（`code` 存在性、`name`/`description`、`parent_id`、`sort_order`、`is_active`） | `/api/v1/capabilities` |
| 工具 ↔ 能力映射（**整体替换**语义） | `PUT /tools/{tool_id}/classifications` |
| 工具说明覆盖 | `PUT /tools/{tool_id}/documentation` |

⚠️ 改能力的 `name` / `description` **会直接改变分类提示词的内容** —— 那正是分类器看到的候选集描述。
改措辞可以改变模型的选择倾向，这是特性也是风险。

#### ⚠️ 两条必须说清的边界

1. **树形结构用于「组织」，门禁本身是「平」的。**
   `is_active` 只是被查询那一行上的过滤（`CapabilityModel.is_active == is_active`），
   **不会递归到子树**；工具查找也是 `code IN (...)` **精确匹配、不向下递归**。
   **想停掉一整块，得逐个节点停用** —— 停用一个父节点不会连带停用它的子节点。

2. **「能配上」≠「能用」。** DB 可以把一个能力映射到某个 `tool_key`，
   但如果代码里没有注册对应的描述符，`select_specs` 会**静默跳过（WARNING）**。
   所以"改数据不改代码"只覆盖**选择**维度，**不覆盖可执行入口**维度。加一个**新工具**仍然要发版。

#### 缺口

`modules` **恒为空**（能力表里没有字段区分"业务域"与"能力域"，模型分流时会误判，于是干脆不分）；
`capability_config` **只落字段、没有消费者**；`exposed_tools` 通道在 react 图里**恒空**
（`tool_source` 恒在，只走 fallback 分支）；`conversation_id` 传进来了但**被忽略**（预留给未来的按会话/租户过滤）。

---

### 四、RAG 流水线：全链条可配置、可拓展

**主张：从文件到可检索索引的每一个环节都是「注册表 + 配置选实现」，换实现不动调用方，加组件不碰中心清单。**

#### 写入链路（LangGraph 状态图，`document_ingestion_pipeline`）

11 个 spec 节点，编译后 16 个（含 3 个扇出汇合屏障）。每个节点按**角色**查注册表选实现：

| # | 节点 | 角色 | 已注册实现 |
|---|---|---|---|
| 2 | `parsing` | `LOADER` | `TEXT_TXT_COMMON_PARSER`（**唯一**，`.txt` / `.md`） |
| 3 | `normalization` | `NORMALIZE` | `TEXT_TXT_COMMON_NORMALIZER`（**13 步**，可逐项开关，幂等，**从不删内容**） |
| 4 | `cleaning` | `CLEAN` | `TEXT_TXT_COMMON_CLEANER`（**7 类规则**，**会删行**） |
| 5 | `metadata_extraction` | `ENRICH_METADATA` | ⚠️ **无实现**（passthrough） |
| 6 | `chunking` | `CHUNK` | **3 个**：递归分块（512/64，中文优先分隔符）/ 不分块 / GraphRAG 分块（1200/100） |
| 7 | `chunk_enrichment` | `CHUNK_ENRICH` | `TEXT_COMMON_CHUNK_ENRICHMENT` |
| 8 | `embedding` | `EMBED`（按 `output_kind` 选） | **3 个**：BGE-small-zh 稠密(512维) / 词法稀疏(20万维词表) / 图抽取（⚠️ `embed()` 抛 `NotImplementedError`） |
| 9 | `indexing` | `IndexBackend` | `Elasticsearch` / `Milvus`（**仅在注入了客户端时注册**） |

> **归一与清洗有严格的边界**：归一化是**等价变换**（只动控制字符与空白，从不删内容），
> 清洗**会删行**。把两者混为一谈会让"内容莫名少了"无从定位。

#### 配置是数据库行，不是代码 —— `DTD → Processor / Retrieval → Binding`

| 层 | 表 | 关键字段 |
|---|---|---|
| 类型定义 | `document_type_definition` | `type_key`（唯一不可变）、`analysis_mode`、**融合策略**、**精排策略** |
| 处理链 | `document_processor_definition` | `file_extension`、`loader_name`/`config`、`processor_name`/`config` |
| 检索链 | `document_retrieval_definition` | `retrieval_type`、`chunker_name`/`config`、`retriever_name`/`config`、`storage_backend`/`config` |
| 绑定 | `document_processor_retrieval_binding` | 配置层的 N:N |
| 物理落点 | `document_index_binding` | 把**物理索引名**与配置解耦 |

**一个文档类型如何落到具体组件链**：纯函数 `select_retrievals` ——
绑定了具体 processor 就只走它绑定的检索；否则走该 DTD 下**全部活跃**检索；`reprocess` 可以取交集收窄。

#### 「可拓展」：加一个组件是**放手即用**

**关键：没有装饰器，没有中心注册清单。** 靠**反射发现**：

- `import_package_modules(*CAPABILITY_PACKAGES)` 先 import 角色包下所有模块
  （`__subclasses__()` 只看得到已 import 的类）
- `leaf_subclasses(RoleBase, skip_abstract=True)` 深度优先收集**叶子子类**

**加一个 chunker / parser / embedder / cleaner / normalizer / enricher 的完整步骤**：

1. 新建一个类，继承对应角色基类
2. 设 `name`（全局唯一、永不复用的 key）
3. **可选**：声明 `config_model`（一个 pydantic 模型）—— `get_config_schema()` 会自动产出
   `model_json_schema()`，**前端表单因此自动生成，不需要改前端**
4. 放进该角色的目录。**完成。**

注册表、能力目录接口（`GET /document-processor-capabilities`）、目录 API **全部自动收录**。

> **唯一需要碰中心清单的情况**：新增一个**角色**（而不是角色下的实现）时，
> 把新角色包的名字追加进 `CAPABILITY_PACKAGES`。

**两道硬错误闸守着"静默二选一"**：

| 冲突 | 结果 | 为什么必须炸 |
|---|---|---|
| 两个实现用了同一个 key | **`ValueError`** | 字典推导会让后者**静默覆盖**前者，症状是"某能力凭空消失" |
| 两个 embedder 是同一个 `output_kind` | **`ValueError`** | 写入与查询**只按形态**选 embedder，"返回第一个"等于按模块名排序静默二选一 —— 而不同模型的向量空间不可比，混用**不报错**，只让召回率下降 |

检索侧同理可拓展：加 `Fusioner` 子类（设 `fusioner_type`）或 `Reranker` 子类（设 `reranker_types`）。
但 **Filter 与 Cutter 必须保持单一实现** —— 多于一个直接抛 `RetrieveStageNotFound`。

#### 读取链路：`POST /documents/retrieve`

四段式，优先级 **请求参数 > DTD 配置 > 类默认值**：

| 段 | 配置来源 | 默认 |
|---|---|---|
| **融合** | DTD `fusion_strategy` | 开启，RRF，`top_k=50`，`rrf_k=60` |
| **过滤** | 类默认（无 DTD 列） | 开启，无分数阈值 |
| **精排** | DTD `rerank_strategy` | 开启；默认实现是**本地 cross-encoder** `BAAI/bge-reranker-v2-m3`（50 候选 × 512 字符，进程内） |
| **裁剪** | 类默认（无 DTD 列） | `top_k=10` |

**多路召回**：`retrieval_type` → 容器名（见下），查询向量按 `dense_vector`/`sparse_vector` 映射到 embedder
—— **写查必须同一个模型**，这正是上面那道 `ValueError` 要防的。内容**从不存进索引**，召回后再从 PostgreSQL 回填。

#### 容器名是契约，且只有一份

`pipeline/containers.py` 是 `retrieval_type → 物理容器名` 的**唯一副本**，写入期与查询期共用。

> **为什么硬编码而不让用户配**：查询期必须**一次查到全部文档**。允许每个类型/版本各配一个容器，
> 查询就要跨 N 个容器做 federation 再融合 —— 而用户配错的表现是"**检索搜不到"，不是报错**。
> 名字逐条写出来而不用 f-string 拼，也是为了让人一眼看出**这是个契约**（改了等于全量重建 + 数据迁移）。

#### 失败语义：该炸的炸，该降级的降级，但**绝不静默**

| 炸 | 降级（但必须记录） |
|---|---|
| 能力/后端查不到 → `CapabilityNotFound`（继承 `LookupError`，**不是 400**） | 后端故障 / 超时 / 未实现 → 该路进 **`degraded_routes`**，其余路照常 |
| 能力 key 冲突、同形态多 embedder → `ValueError` | `llm`/`custom` 精排 → 直通 + WARNING |
| DTD 缺失/未激活 → **404**（整请求，不是跳过） | 精排模型加载失败 → 退化为原序 + WARNING |
| `retrieval_config` 键写错 → **422** | 调用方指名了但 DTD 没有的检索类型 → **静默筛除**（这是刻意的"筛选"语义，**不进** `degraded_routes`） |

> `degraded_routes == []` 意味着**所有路都参与了**。这是一条很好用的断言。

#### 缺口

Graph 检索**端到端未实现**（`embed()` 抛 `NotImplementedError`，容器映射里 `graph` 被注释掉）；
`retrieval_type=document` 的召回**未实现**（没有外部容器，本轮直接跳过）；
`index/fiass.py` 是**0 字节**（文件名疑似 `faiss` 笔误，不存在该后端）；
`ENRICH_METADATA` 无实现；`llm`/`custom` 精排是直通桩；层级分块未做（`parent_chunk_id` 恒 `None`）；
**只支持 `.txt` / `.md`**（`format_detection` 节点是占位，不按 MIME 路由）；
**逻辑删除不清外部索引**（删了文档，ES/Milvus 里的 chunk 仍可被召回）。

---

## 其他值得注意的机制

除上面四项外，这些也是本仓库区别于"裸用 LangGraph"的地方：

| 机制 | 一句话 | 详见 |
|---|---|---|
| **模型选择拒绝兜底** | `(agent_type, model_role, attributes)` 等值匹配；匹配不到**报错**，平局**也报错** —— `bindings` 刻意无通配 `*` | [`README_MODEL.md`](README_MODEL.md) |
| **上下文是固定 11 段流水线** | 顺序写死在 `PIPELINE_TEMPLATE`，**import 期断言**枚举与模板一致；12 个处理器 | [`README_CONTEXT.md`](README_CONTEXT.md) |
| **处理器不能改对话** | 只能 `add_section` / `add_history` / 声明工具；回写 state 走白名单，**只有 `summary` 与 `memory` 两个键** | [`README_CONTEXT.md`](README_CONTEXT.md) |
| **HITL 审批 + 持久化账本** | `interrupt()` 挂起图，"始终允许"落 LangGraph store，**重启不重复问**；代码里显式标注了「少调一次 `interrupt()` 会让后面的中断错位」 | [`README_TOOL.md`](README_TOOL.md) |
| **模型私有标记的字符流闸** | deepseek 思考模式把工具调用写成 DSML 标记当正文吐出，**全程不报错**；用有状态闸跨分片拦截（含全角/半角竖线） | `app/agent/model/leaked_tool_call.py` |
| **四层记忆，可重建性不同** | L0 是索引（可重建），**L1 是权威数据（不可重建）**；L2/L3 由 L1 抽象 | [`README_MEMORY.md`](README_MEMORY.md) |
| **SSE 遇审批主动关流** | **跨仓库前端契约**：前端须自行 `resume` 后重开；"所有 interrupt 到齐"的权威信号是 **DB 状态**，不是事件批次 | [`README_STARTUP.md`](README_STARTUP.md) |
| **契约先行** | 19 份 OpenAPI 规格**运行时不加载**；契约名与类名可不同，有专门的翻译函数（`AgentMessage`→`Message`） | [`README_STARTUP.md`](README_STARTUP.md) |
| **历史运行按冻结拓扑渲染** | `topology_fingerprint`（sha256）供 CI 比对 golden；不写死节点列表 | [`README_DOCUMENT.md`](README_DOCUMENT.md) |
| **启动期就炸** | Casdoor 自检、模型引用完整性、流水线模板一致性、提示词目录装配；配一个常驻 `LoopWatchdog` 抓事件循环卡顿 | [`README_STARTUP.md`](README_STARTUP.md) |

---

## 架构总览

### 分层与依赖方向

```
app/common  ←  app/services  ←  app/infra  ←  app/bootstrap  ←  app/main.py
```

箭头指向"被依赖"。**`app/common` 与 `app/infra` 都不允许 import 业务模块** ——
这正是「任务 Handler 的注册点必须在 `app/bootstrap/task.py` 而不在 `app/infra`」的硬理由。

### 目录树

```
app/
├── main.py            # FastAPI app + lifespan（组合根，只负责排序）
├── serve.py           # uvicorn 启动（显式 SelectorEventLoop，见 START_UP.md §8）
├── bootstrap/         # 装配层：auth / event_stream / graph_runtime / pipelines / task
├── api/v1/            # HTTP 路由（契约实现）+ router.py（public / protected 两棵树）
├── agent/             # LLM 运行时（图侧，不做业务持久化）
│   ├── graph/react/   # ReAct 图：nodes / state / routing / catalog / runtime
│   ├── state/         # AgentClassification 与枚举
│   ├── context/       # ContextBuilder 流水线（11 段 + 12 处理器）
│   ├── model/         # ModelRegistry / resolver / adapters / runtime
│   ├── prompt/        # 节点功能提示词（分类、记忆、摘要）
│   ├── tools/         # 处理器链契约（interceptor / observer / transformer）+ @tool
│   └── event/         # Redis Stream 事件实现（publisher / reader / sse / run_stream）
├── services/          # 业务模块，各自 domain / application / models / repository 分层
│   ├── capability/    # 能力树（工具暴露的分类法）
│   ├── document/      # 摄入流水线 + RAG 检索
│   ├── file/          # 文件与产物存储
│   ├── memory/        # L0–L3 记忆
│   ├── prompt/        # 职责提示词（业务数据）
│   ├── session/       # session / run / message + ReAct 调度 handler
│   ├── task/          # 异步任务持久化（tasks / outbox）
│   ├── tool/          # 工具定义注册与同步
│   └── user/          # Casdoor 鉴权 + 用户检索
├── infra/             # 共享运行时内核（不 import 业务）
│   ├── graph/         # GraphRegistry / dispatcher / checkpointer / runner
│   ├── datasource/    # DataSource 抽象 + postgres / redis / es / milvus / neo4j / checkpointer
│   ├── task/          # 任务 worker / dispatcher / outbox / retry
│   └── agent/         # 工具执行器 + 暴露 + 链折叠
├── common/            # 共享内核：工厂、模型、鉴权、schema、任务协议、工具函数
└── config/            # 运行时配置（casdoor / llm / model / event_stream / datasource/*）
```

### 两张 LangGraph 图

都在 `app/bootstrap/pipelines.py` 装配，注册进同一个 `GraphRegistry`，由同一个
`InProcessGraphDispatcher` 运行（一个 asyncio 队列 + 一个串行 worker；每个 run spawn 子任务，
**因此取消针对的是 run，不是 worker**）。

**图 A · 文档摄入管线** — `document_ingestion_pipeline`（见"核心能力四"）

**图 B · Agent ReAct 图** — `agent_react` v1.3.0：

```
START → builder → (execution_plan == plan ? planner → agent : agent)
agent → (有 tool_calls ? 每个 tool_call 一个 Send 扇出到 tools → agent : END)
```

| 节点 | 职责 | 用哪类模型 |
|---|---|---|
| `builder` | 输入分类 + **能力选择 → 工具集**（每 run 一次） | 快模型 |
| `agent` | 主循环 | 推理模型 |
| `planner` | ⚠️ 占位（plan-and-execute 待做） | — |
| `tools` | 扇出：每个 tool_call 一个分支，各自过处理器链 | — |

`ReactState` 是 `TypedDict`（**刻意不用 Pydantic**）：`messages` 用 `add_messages` reducer，其余是
"最后写入者胜出"的通道。路由函数是**纯函数**。

> **为什么要扇出成"一个 tool_call 一个分支"**：HITL 的 resume 语义依赖它 ——
> 一分支一个中断，审批账本跳过时才不会让后面的中断错位。

### 一次对话的完整链路

```
POST /sessions/{id}/runs
  → 存消息、建 run 记录（session 域）
  → InProcessGraphDispatcher.dispatch()（入队 → 子任务）
  → builder 节点：从 DB 现读能力候选 → 一次分类 LLM 调用 → capability_needs → tool_keys
  → agent 节点：ContextBuilder 跑 11 段流水线
       → 拼出一条 SystemMessage（各处理器 section 按序注入）
       → ModelResolver：「(agent_type, model_role, attributes)」→ profile
       → ModelRuntime：候选链 / 重试 / 超时 / fallback
       → 流式产出 → leaked_tool_call 字符流闸 → 事件发布
  → 有 tool_calls？ → tools 扇出 → 处理器链（auth → approval[可能 interrupt] → cache → 真正调）
  → 事件经 Redis Stream → SSE 下发；遇审批**主动关流**（前端自行 resume 后重开）
```

---

## 配置速查

完整说明见 **[`README_STARTUP.md`](README_STARTUP.md)**。要点先放这里：

### 两类 yaml，性质完全不同

| | 运行时配置 | 契约规格 |
|---|---|---|
| 位置 | `app/config/**` | `app/services/**/*.yaml` |
| 数量 | 10 | 19 |
| 启动加载 | **会**（解析 + 校验） | **不会**，一个字节都不读 |
| 改错后果 | 起不来 / 行为不对 | 不影响运行 |
| 热加载 | **无** —— 改完必须重启 | — |

### 运行时配置一览

| 文件 | 管什么 | 一句话记住 |
|---|---|---|
| [`casdoor.yaml`](./app/config/casdoor.yaml) | JWT 校验、OAuth 登录、管理员门禁 | `APP_ENV=prod` + `enabled: false` = **硬拒绝启动** |
| [`event_stream.yaml`](./app/config/event_stream.yaml) | SSE 链路的全部节流参数 | `maxlen` 与 `ttl` 管**两件不同的事，两个都要** |
| [`llm.yaml`](./app/config/llm.yaml) | **怎么建客户端**（provider / base_url / key） | 三家都 OpenAI 兼容，`base_url` 必须带 `/v1` |
| [`model.yaml`](./app/config/model.yaml) | **选哪个模型**（六段目录 + 绑定） | `bindings` 刻意**没有通配兜底** |
| [`datasource/*.yaml`](./app/config/datasource/) | 6 个数据源 | `postgre.yaml` 是**唯一强制**；加数据源不用改代码 |

**最容易踩的三条**：

1. **`checkpointer.yaml` 的 `store` 段不是装饰** —— 它是 HITL 审批账本的持久化后端。**不传 `store`，账本退化成重启即丢的内存版**，症状是"已经批准过的工具重启后又问一遍"。
2. **单位陷阱**：`checkpointer.yaml` 的 `default_ttl_minutes` 是**分钟**，而审批拦截器那份配置的 TTL 是**秒**。
3. **数据源 yaml 顶层 `type` 拼错会被静默跳过**（loader 已从 `raise` 放宽为 skip + warning）—— `tpye: postgresql` 之后只在取 `postgresql` 时报一句更差的错，**启动日志里的 warning 是唯一线索**。

> ⚠️ 仓库里有一批**注释失真**：`llm.yaml` / `event_stream.yaml` 的开头注释与 `README_MODEL.md` 第 1 节
> 描述的是**配置收敛到 `app/config/` 之前**的布局，照它们找文件会找不到。细节见 [`README_STARTUP.md` §0](README_STARTUP.md)。

---

## 快速开始

```bash
# ⚠️ 用 .venv 里的 python，不要用裸 python
#    （裸 python 是旧 anaconda，SQLAlchemy 太老，直接 ImportError）
.venv/Scripts/python.exe run_server.py            # 默认 localhost:8080
.venv/Scripts/python.exe run_server.py --port 8081

# ⚠️ 不要用 `uvicorn app.main:app`
#    psycopg 需要 SelectorEventLoop，而 uvicorn 0.36+ 会忽略 asyncio policy
```

**前置**：Postgres（**唯一必需**）+ Redis（SSE）。Milvus / ES / Neo4j 可选，缺失时对应能力降级。
本机开发环境见 `app/config/datasource/`，重置数据用 `reset_data.py`。

**契约自检**（开发期）：`.venv/Scripts/python.exe -m app.agent.spec_check`（目前只覆盖 document 那 6 份 spec）

启动失败先查 [`README_STARTUP.md` §9 启动失败速查](README_STARTUP.md)。

---

## 已知边界（诚实清单）

这份 README 只描述**代码里真实存在**的东西。以下能力目前**不工作或只有一半**，细节在各模块手册的
「已知缺口」章节：

| 缺口 | 症状 |
|---|---|
| **工具指标无出口** | 事件/指标只在进程内，**无查询接口、无 Prometheus、重启清零**；`tool_execution` 无 API 读、无 latency、无 token 记账、无 reaper |
| **运行时改不了处理器实现** | 只能挂载已有处理器；新增处理器要发版 |
| **能力树的 `is_active` 不级联** | 停用父节点**不会**停用子树；工具查找按 code 精确匹配、不向下递归 |
| **加新工具要发版** | 能力树可改"选择"，但 tool_key 没有注册描述符就会被 `select_specs` 静默跳过 |
| **发现同步拾取不了代码修改** | 只能发现"从未 import 的新模块"，改已加载模块仍需重启 |
| `modules` 恒为空 / `capability_config` 无消费者 | 分类五维里只有 3 维真正起作用 |
| Graph 检索 | 端到端未实现；`index/fiass.py` 是 **0 字节** |
| `retrieval_type=document` | 召回未实现（无外部容器） |
| 文档格式 | **只支持 `.txt` / `.md`**；`format_detection` 是占位 |
| 逻辑删除 | **不清外部索引** —— 删了文档，chunk 仍可被召回 |
| 成本估算 | `model.yaml` 的 `pricing:` 段全部被注释 → `estimated_cost` 恒为 `None` |
| 结构化输出不能流式 | `_can_stream` 要求 `response_format is None` |
| Context 的三个 Source 端口 | `ProfileSource` / `KnowledgeSource` / `MemorySource` **全仓零实现** |
| L2 / L3 消费侧 | 场景与画像**只有写入方**，还没有检索/注入 |
| L1 背景对话 | 只喂增量，从对话中间开始时指代没有先行词（漏抽或猜，**两类都不报错**） |
| `planner` 节点 | 占位，plan-and-execute 未实现 |
| Neo4j | 已接为 DataSource 但**尚未被消费** |
| `tool_span` | react 图的节点级 span **只打日志不落库**（文档管线才落表） |
| 契约一致性 | 只有 document 域有 `spec_check` 自动闸，其余域靠人守 |

更大的方向性未做项（多 agent 协作、sandbox、agent eval、trace）见仓库根 `todolist`。

---

## 阅读建议

第一次读这个仓库，建议的顺序：

1. **本文的「四个核心能力」** → 拿到它和别的 harness 的差异面
2. **[`README_STARTUP.md`](README_STARTUP.md)** → 把它跑起来，知道配置在哪
3. **挑一个你关心的子系统手册**，并且**先读它的「已知缺口与注释失真」章节** —— 那节列的是当前**不能**工作的能力，比正面描述更能校准预期
4. 再回到代码，从 `app/main.py` 的 lifespan 往下走 —— 它是组合根，每一步的位置都有注释解释理由

> 本仓库的注释密度很高，而且**大量注释在解释"为什么"，包括"为什么不是另一种写法"**。
> 这不是装饰 —— 它是这个项目最主要的交付物之一。
