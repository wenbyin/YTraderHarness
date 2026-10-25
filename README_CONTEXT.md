# Context · 上下文装配

> Context **不是记忆、不是知识库**。它是**装配层**：把散落在图 state 里的东西（历史、摘要、记忆、工具、能力）
> 按一条**写死的段序**拼成一条 `ModelRequest`。
>
> 它最容易被误解的地方是「处理器」这个词 —— 这里的处理器**不能改对话**。它们只能声明式读图状态、
> 往工作集里加 section / history / tools；想影响下一轮 state，只能走**白名单**（只有 `summary` 与 `memory` 两个键）。

`13 段流水线` · `14 个处理器` · `PIPELINE_TEMPLATE 显式字面量` · `回写白名单` · `一条 SystemMessage`

---

## 📚 本文结构

- 🎯 **一张表看懂**
- 🏗️ **阅读地图** —— 找文件先看这张表
- 🔁 **13 段流水线** —— 顺序即契约
- 🧩 **处理器的能与不能** —— 边界在这里
- 📡 **调用方式**
- 📖 返回 **[主手册](README.md)** · ⚠️ 缺口见 **[todolist](todolist)**

---

## 🎯 一张表看懂

| 问题 | 一句话答案 |
|---|---|
| Context 是什么？ | **装配层**：把图 state 里的东西拼成一条 `ModelRequest` |
| 谁触发它？ | Agent 图的 LLM 节点（`agent_node` / `agent_builder_node`），每次调模型前调一次 `builder.build()` |
| 流水线有几段？ | **13 个 `StageId`**，顺序由 `PIPELINE_TEMPLATE` 显式写死，import 期用断言锁住 |
| 有几个处理器？ | **14 个**（`MEMORY_RETRIEVER` 阶段上挂了 2 个） |
| 有几套 Builder？ | 声明 6 种 `ContextBuilderType`，实现 4 种：`agent_invoke` / `classifier` / `planner` / `subagent` |
| 处理器能改对话吗？ | **不能**。回写图 state 走 `context_state_updates` 白名单，**只有 `summary` 与 `memory` 两个键** |
| 注入段最终在哪？ | 全部进**一条** `SystemMessage`（`system_prompt` 之后，按处理器顺序），`## {label}` 由 `system_message()` 统一包 |
| token 怎么数？ | 默认 `HeuristicTokenCounter`（`chars/3.0` + 每消息 4 开销），**不联网、不用 tiktoken** |
| 唯一自己调 LLM 的地方？ | `SummaryCompactor`（`CONTEXT_COMPACTOR` 阶段）—— 这也是 `COMPACTION` 必须独立成一个 builder type 的理由 |

---

## 🏗️ 阅读地图

```
app/common/context/enum.py     ← StageId / ContextBuilderType / StageKind 定义处

app/agent/context/
├── registry.py                ← found_builders（反射发现）/ create_context_builder
├── updater.py                 ← ContextUpdater —— ⚠️ 仅 Protocol，无实现
├── builder/
│   ├── base.py                ← ContextBuilder 抽象 + _stage_* 方法 + build()（约 880 行）
│   ├── contracts.py           ← PIPELINE_TEMPLATE / LLM_INVOCATION_STAGES / ContextBuildRequest
│   ├── agent_invoke.py        ← 生产主路径 Builder
│   ├── classifier.py          ← 分类器专用窄 Builder（3 阶段）
│   ├── planner.py             ← ContextBuilderType.PLANNER
│   └── subagent.py            ← ContextBuilderType.SUB_AGENT
└── processor/
    ├── pipeline.py            ← 流水线编排 / begin / run_stage / _phase（约 800 行）
    ├── context.py             ← ContextBuildContext（工作集）+ ProcessorExecution + ContextSnapshot
    ├── spec.py                ← ContextProcessorSpec（inputs 绑定 / when 门控 / optional）
    ├── base.py                ← ContextProcessor 基类（唯一抽象方法 process）
    ├── state.py               ← resolve_inputs / resolve_state_path（dot-path）
    ├── policy.py              ← ContextPolicy + ContextPolicyResolver
    ├── turns.py               ← 轮次/窗口/摘要节奏的纯函数数学
    ├── budget.py              ← TokenBudget + TokenCounter + HeuristicTokenCounter
    ├── summarization.py       ← SummaryCompactor（全仓唯一在上下文构建中调 LLM 之处）
    ├── dependencies.py        ← ContextDependencies（model_registry / tool_source / *_source）
    └── processors/            ← 14 个处理器实现
```

---

## 🔁 13 段流水线

**顺序写死在 `PIPELINE_TEMPLATE`（显式字面量，不是 `tuple(StageId)`）。**

> 为什么不用枚举推导：那会把「枚举里有哪些成员」和「流水线按什么顺序跑」绑成一件东西 ——
> 有人为了排版挪一下枚举成员，流水线顺序就**静默**变了。写成字面量的代价是「加了新阶段忘了往这里加」，
> 那个由 import 期断言接住。

| # | `StageId` | 处理器 | 干什么 |
|---|---|---|---|
| 1 | `CONTEXT_RESOLVER` | `ContextResolveProcessor` | 解析运行时上下文（agent_type / 属性 / 策略来源） |
| 2 | `TASK_RESOLVER` | `TaskResolverProcessor` | 计划模式下解析当前任务（⚠️ 由 gate 决定跑不跑） |
| 3 | `SKILL_LOADER` | `SkillLoaderProcessor` | 装载任务对应的 skill（⚠️ 由 gate 决定跑不跑） |
| 4 | `PROFILE_LOADER` | `UserProfileProcessor` | 用户档案段 |
| 5 | `SUMMARY_LOADER` | `SummaryProcessor` | 历史摘要段 |
| 6 | `HISTORY_SELECTOR` | `HistoryProcessor` | 按窗口选历史消息 |
| 7 | `MEMORY_RETRIEVER` | `MemoryProcessor` + `MemoryProfileProcessor` | **两个处理器**：记忆段 + 画像段 |
| 8 | `KNOWLEDGE_RETRIEVER` | `RetrievalProcessor` | RAG 知识段 |
| 9 | `CAPABILITY_LOADER` | `CapabilityProcessor` | **重算可绑定工具对象**（纯内存、零 DB） |
| 10 | `PROMPT_ASSEMBLER` | `PromptAssemblerProcessor` | 职责提示词 |
| 11 | `TOKEN_BUDGETER` | `TokenBudgetProcessor` | 预算计算与超限处置 |
| 12 | `CONTEXT_COMPACTOR` | `CompactionProcessor` | 压缩（**唯一会自己调 LLM 的阶段**） |
| 13 | `CONTEXT_VALIDATOR` | `ValidationProcessor` | 结构校验 |

> `TASK_RESOLVER` / `SKILL_LOADER` 是 1.1.0 的**显式插入**（不是追加到末尾）—— 它们必须跑在
> `PROMPT_ASSEMBLER` 之前。这是一次有意的流水线契约变更：所有 builder 的 `enabled_stages` 与测试一起更新。

### INITIAL 与 CONTINUATION

| 相位 | 何时 | 跑哪些 |
|---|---|---|
| `INITIAL` | 一轮对话的第一次模型调用 | 全 13 段 |
| `CONTINUATION` | 工具调用之后的每次续调 | 只跑必要段（历史不重选、摘要不重算） |

轮次 / 窗口 / 摘要节奏的数学在 `turns.py`，**全是纯函数** —— 想验证「第几轮该压摘要」不用起服务。

---

## 🧩 处理器的能与不能

### 能做

- 往工作集里 `add_section` / `add_history` / 声明工具
- 声明式读图状态（`inputs` 用 **dot-path 绑定**，如 `context_policy.tool_keys`）
- 通过 `when` 门控决定自己跑不跑，标 `optional` 决定失败是否致命
- 通过 `context_state_updates` **白名单**回写 —— **只有 `summary` 与 `memory` 两个键**

### 不能做

| 不能 | 为什么 |
|---|---|
| 直接改 `messages` | 对话只由 `HISTORY_SELECTOR` 段与 `ModelRequest` 组装决定；处理器改 messages 会让「谁改的」无法定位 |
| 改别的处理器的输出 | 每个处理器只拿到工作集的一个**受控视图** |
| 回写任意 state 键 | 白名单只有两个键，其余一律丢弃（**静默丢弃不是遗漏，是边界**） |

### 失败语义

- 标了 `optional` → 记录 note 后跳过，流水线继续
- 没标 → 抛错，整次构建失败

> ⚠️ 三个 source 端口（`ProfileSource` / `KnowledgeSource` / `MemorySource`）**全仓零实现**，
> 因此 `PROFILE_LOADER` 与 `KNOWLEDGE_RETRIEVER` 当前只能走 state 兜底路径。
> 详见 [`todolist`](todolist)。

---

## 📡 调用方式

真实消费方式是**进程内调用**，不是 HTTP：

```python
builder = create_context_builder(ContextBuilderType.AGENT_INVOKE)
request = await builder.build(build_request)   # → ModelRequest
```

`context_builder.yaml` 里声明的三个端点**当前未接路由** —— 它是「契约先行」的产物，
规格先行、路由后置（见 [`todolist`](todolist)）。

---

## 🚀 验证

```bash
# 流水线模板与枚举的一致性由 import 期断言保证 —— 导入即验证
.venv/Scripts/python.exe -c "import app.agent.context.builder.contracts; print('pipeline ok')"

# 处理器发现（应为 14 个）
.venv/Scripts/python.exe -c "import app.agent.context.processor.registry"
```

---

## 🤝 相关

| 想深入看 | 去哪 |
|---|---|
| 注入的「记忆」从哪来 | [`README_MEMORY.md`](README_MEMORY.md) |
| 工具对象怎么按能力重算 | [`README_TOOL.md`](README_TOOL.md) |
| 职责提示词从哪来 | `app/services/prompt/` + `app/agent/prompt/` |
| 总览 | [`README.md`](README.md) |
