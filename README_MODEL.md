# Model · 模型选择

> 模型不是一个「全局默认值 + 覆盖」，而是一张**显式目录**：provider、凭证、模型、参数档（profile）、别名、
> 绑定分六段声明。选模型是**等值匹配** —— 匹配不到就报错，平局也报错。
>
> `bindings` 里**刻意没有通配 `*` 兜底**。理由是：漏配一个 `(agent_type, model_role)` 时，
> 「悄悄用默认模型跑起来」比「启动/调用时炸掉」贵得多 —— 你不会知道自己在用哪个模型回答用户。

`6 段目录` · `10 个 ModelRole` · `14 条绑定` · `等值匹配 + 无兜底` · `DSML 泄漏闸`

---

## 📚 本文结构

- 🎯 **一张表看懂**
- 🗂️ **`model.yaml` 六段**
- 🧭 **解析优先级**
- ⚠️ **六个 provider 特有的坑**
- 🛡️ **DSML 泄漏闸** —— 本仓库最凶的一个坑
- 📖 返回 **[主手册](README.md)** · ⚠️ 缺口见 **[todolist](todolist)**

---

## 🎯 一张表看懂

| 问题 | 一句话答案 |
|---|---|
| 模型从哪配？ | `app/config/model.yaml`：provider / 凭证引用 / 模型 / 参数档 / 别名 / 绑定，六段 |
| 怎么选模型？ | 按 `(agent_type, model_role, attributes)` **等值匹配** bindings，取 specificity 最高者，**平局直接报错**、**无兜底** |
| 谁能被调到？ | 只有 bindings 里声明过的组合。当前 **14 条**，`*` 兜底**刻意不存在** |
| 一次 `ainvoke` 是几次 provider 调用？ | **不一定是一次** —— 候选链上每个候选可重试 `max_retries` 次，还可能有 fallback |
| 重试/超时数值在哪？ | profile 的 `max_retries` / `timeout` / `total_timeout` / `retry_backoff` / `max_retry_backoff` |
| 结构化输出能流式吗？ | **不能**。`_can_stream` 要求 `response_format is None`，结构化输出一律退回 `ainvoke` |
| 成本能估吗？ | **不能**。`pricing:` 段全部被注释 → `estimated_cost` 恒为 `None` |
| 指标落哪？ | 默认 observer **只打标准库 logging，不落库**；落库靠 LLM Node 把 `ModelResponse.metadata` 写进既有 `node_span` |
| 最凶的坑？ | deepseek 思考模式把工具调用写成 `<\|DSML\|…\|>` 标记当正文发给用户，**全程不报错** |

---

## 🏗️ 阅读地图

```
app/agent/model/
├── catalog.py        ← model.yaml 六段解析 + 启动期引用校验（约 1020 行）
├── registry.py       ← ModelRegistry（按 id / alias / profile 检索）
├── resolver.py       ← (agent_type, model_role, attributes) → ModelProfile
├── factory.py        ← 由 profile 实例化 adapter（剔除非 SDK 参数）
├── provider.py       ← ProviderAdapter 抽象（create_client / ainvoke / aclose）
├── adapters.py       ← LangChainProviderAdapter（约 1070 行）
├── runtime.py        ← ModelRuntime：候选链 / 重试 / 超时 / fallback / 观测（约 1140 行）
├── types.py          ← DTO + ModelRole 常量 + 校验（约 1250 行）
├── errors.py         ← 异常树 + ErrorDisposition + HTTP 启发式
├── credentials.py    ← 凭证解析（env / literal / null）
├── redaction.py      ← 日志脱敏
├── leaked_tool_call.py ← DSML 泄漏闸
├── observer.py       ← ModelCallObserver（默认只打日志）
├── streaming_smoke.py
└── v3_reroute_smoke.py
```

---

## 🗂️ `model.yaml` 六段

| 段 | 管什么 |
|---|---|
| `providers` | 提供方声明（三家都 OpenAI 兼容） |
| `credentials` | 凭证引用：`env` / `literal` / `null` 三种解析方式 |
| `models` | 模型条目（id / 能力 / 上下文窗口 / 是否支持工具） |
| `profiles` | **参数档**：温度、`max_retries`、`timeout`、`total_timeout`、backoff —— 换档不改绑定 |
| `aliases` | 别名（如 `reasoning_default`） |
| `bindings` | **谁能被调到**：`(agent_type, model_role)` → profile |

### 10 个 `ModelRole`

| 角色 | 语义 |
|---|---|
| `reasoning` | 主循环推理 |
| `summarization` | 摘要 / 上下文压缩 |
| `classification` | 输入分类（走快模型） |
| `evaluation` | 评测相关 |
| `vision` | 📌 无绑定使用 |
| `fast` | 📌 无绑定使用 |
| `memory.atom.extract` | L1 抽取（后台任务，不在任何图里） |
| `memory.atom.dedup` | L1 去重 |
| `memory.scene.consolidate` | L2 场景整合 |
| `memory.persona.extract` | L3 画像生成 |

> ⚠️ **抽取与去重是两个角色、两个 `agent_type`**，因为它们取舍不同：抽取要「读得多、写得全」，
> 去重要「判得准」。挤成一个就没法只调其中一个的模型。

### `agent_type` 的命名口径

`agent_type` 对的是**产物 / 服务**，`model_role` 对的是**动作**。所以是 `memory_atom_extract`
而不是 `memory_l1_extract`。

---

## 🧭 解析优先级

```
(agent_type, model_role, attributes)
        │
        ├─ 在 bindings 里找**等值**匹配
        │     ├─ 命中 0 条  → ❌ 报错（列出当前所有绑定）
        │     ├─ 命中 ≥2 条且 specificity 相同 → ❌ 报错（不按书写顺序猜）
        │     └─ 命中 1 条  → profile
        │
        └─ profile → factory → adapter（剔除非 SDK 参数）
```

**运行时（`runtime.py`）不是一次调用**：

| 环节 | 说明 |
|---|---|
| 候选链 | `_candidate_chain` 生成候选序列（含 fallback） |
| 重试 | 每个候选可重试 `max_retries` 次，按 `retry_backoff` / `max_retry_backoff` |
| 超时 | 单次 `timeout` + 总计 `total_timeout` |
| 降级 | 按 `errors.py` 的 `ErrorDisposition` 决定换候选还是直接抛 |

---

## ⚠️ 六个 provider 特有的坑

1. **`base_url` 必须带 `/v1`** —— 三家都 OpenAI 兼容，少一段就是 404。
2. **结构化输出不能流式** —— `_can_stream` 要求 `response_format is None`，否则退回 `ainvoke`。
3. **`disable_thinking` 必须有 `extra_body`** —— `ModelDefinition.__post_init__` 强制校验（目前所有模型都没开这个开关）。
4. **工具调用映射不是一一对应** —— 各家的 `tool_call` 结构差异在 `adapters.py` 里收口。
5. **HTTP 状态码不能直接当错误类型** —— `errors.py` 用启发式 + `ErrorDisposition`，因为同一状态码在不同 provider 含义不同。
6. **凭证可能出现在日志里** —— `redaction.py` 负责脱敏，改日志格式时别忘了它。

---

## 🛡️ DSML 泄漏闸

**deepseek 思考模式会把工具调用写成 `<|DSML|…|>` 标记当正文吐出来，而且全程不报错。**

后果：用户会在聊天窗口里看到一串本该是工具调用的内部标记。

修法在 `leaked_tool_call.py`：一个**有状态的**字符流闸，跨分片拦截（模型标记可能被切在 chunk 边界上，
所以要带状态；还要同时匹配全角与半角竖线）。

> 这是「静默失败是最贵的失败」最典型的一例 —— 没有任何异常，只有用户体验坏掉。

---

## 🚀 验证

```bash
.venv/Scripts/python.exe -m app.agent.model.streaming_smoke     # 流式与非流式行为
.venv/Scripts/python.exe -m app.agent.model.v3_reroute_smoke    # 候选链重路由
```

---

## 🤝 相关

| 想深入看 | 去哪 |
|---|---|
| 模型建在哪个客户端上（`llm.yaml`） | [`README_STARTUP.md`](README_STARTUP.md) |
| 每次模型调用的上下文怎么拼 | [`README_CONTEXT.md`](README_CONTEXT.md) |
| 记忆抽取用的是哪两个角色 | [`README_MEMORY.md`](README_MEMORY.md) |
| 总览 | [`README.md`](README.md) |
