# Plan · 计划模式

> 计划模式做的事只有一件：**把一句复杂请求变成一张可调度的 DAG**，然后交给执行层。
>
> 它是**子图**，不是节点。原因是规划本身要做工具循环（查技能目录、查能力目录），
> 那些中间推理**一旦进主对话 `messages`**，会被用户看到、被 `message_recorder` 落库、被前端时间线渲染。
> 所以规划必须有一个**隔离的 state 边界**。

`1 个子图` · `3 个节点` · `PlanDraft → A2A Task` · `拒绝制校验` · `无 HTTP 端点`

---

## 📚 本文结构

- 🎯 **一张表看懂**
- 🧩 **先分清五个「计划」** —— 命名撞车很严重
- 🔁 **图拓扑**
- ✅ **输出契约与校验**
- 🎬 **进入 / 退出计划模式**
- 📖 返回 **[主手册](README.md)** · ⚠️ 边界见 **[todolist](todolist)**

---

## 🎯 一张表看懂

| 问题 | 一句话答案 |
|---|---|
| 计划模式是什么？ | 一个 `plan_status="plan"` 的运行态；先把请求拆成 **DAG 任务**，再逐条执行 |
| planner 是节点还是子图？ | **子图**。编译后作为父图节点 `planner` 挂载（`graph/react/graph.py:147`） |
| 为什么要拆子图？ | 规划要做工具循环，中间推理**不能**进主对话 `messages` |
| 计划怎么变成任务？ | 模型输出 `PlanDraft` JSON → `validate_plan()` 校验 DAG → `to_a2a_tasks()` 转 A2A Task |
| planner 会写任务板吗？ | **不写**。只发 `plan.published` 事件 + 写父图 state；**发布任务板在 `create_team`** |
| 谁决定进入计划模式？ | 两条入口、同一条执行路径：分类器合成 `enter_plan_mode` 调用 / 模型自己调用 |
| 计划存在哪？ | `ReactState.plan`（快照）+ `ReactState.tasks`（A2A canonical dict 列表，拓扑序） |
| 有没有 HTTP 端点？ | **没有**。计划任务只活在 graph state / Redis 任务板 / 事件流里 |
| 一个任务能跑多大？ | `message` 与 `metadata` 各 **≤ 64 KiB**，超出直接拒绝，**不静默截断** |
| 实现完整度 | 3 个节点全实现；**33 个 planner pytest 通过（1.39s）** |

---

## 🧩 先分清五个「计划」

命名撞车是这个子系统最大的阅读障碍：

| 名字 | 是什么 |
|---|---|
| `PlanDraft` / `TaskDraft` | **模型输出的**契约形状（只有 `id` / `message` / `metadata`） |
| `ReactState.plan` | 父图里的**计划快照** |
| `ReactState.tasks` | 父图里的 **A2A Task canonical dict 列表**（拓扑序） |
| Redis **任务板** | `create_team` 审批通过后发布的**执行态权威** |
| `/api/v1/tasks` | ⚠️ **另一套异步作业**，与计划任务无关（只是名字撞了） |

---

## 🏗️ 阅读地图

```
app/agent/graph/planner/          1,255 行 —— 子图本体
├── catalog.py        142  图名/版本/节点目录/边/排除工具/模型路由键
├── state.py           97  PlannerInput / PlannerOutput / PlannerState / ToolPayload
├── nodes.py          838  planner_llm / planner_tool / task_generator
├── routing.py         74  route_after_planner_llm（Send 扇出）
└── graph.py           82  装配（input_schema / output_schema 在这里定）

app/agent/plan/                   2,476 行 —— 计划层（不含 LangGraph 依赖）
├── contracts.py      453  TaskDraft / PlanDraft / normalize_payload / validate_plan
├── a2a.py            196  TaskDraft → A2A Task → canonical dict
├── listener.py       722  claim 规则 + State/Redis 两种数据源 + Lua CAS
├── publish.py        391  Redis 任务板 + publish_plan（degraded 语义）
├── checks.py         191  纯函数：pending / ready / 并行判据 / 找环
└── task_prompt.py    448  任务指令文本的**唯一**渲染入口
```

> `app/agent/plan/` **不 import LangGraph** —— 计划层的契约、校验、任务板都是纯业务逻辑，
> 可以脱离图单独测试。这是刻意的分层。

---

## 🔁 图拓扑

```
planner_llm ──(有 tool_calls)──▶ planner_tool ──(Command)──▶ planner_llm
     │
     └──(无 tool_calls / 探索够了)──▶ task_generator ──▶ END
```

| 节点 | 职责 |
|---|---|
| `planner_llm` | ReAct 探索（查技能目录、能力目录），不产出最终计划 |
| `planner_tool` | 执行工具 + **Command 守门**（拦下会跳父图的工具） |
| `task_generator` | 严格 JSON 输出 `PlanDraft`，失败保留旧计划 |

**扇出的三条硬规则**：planner 用 `input_schema=PlannerInput` / `output_schema=PlannerOutput` ——
否则父图 `context_policy` 漏进规划阶段（主 agent 的职责提示词与工具集污染规划），
且 `messages` 会漏回父图（工具循环被当成用户可见回复）。

---

## ✅ 输出契约与校验

模型只输出这个形状 —— **执行态字段一个都不写**（`status` / `claim` / `blocked` / `messageId` 全由转换器补齐，
模型写它们只会制造第二份真相）：

```json
{
  "id": "plan_1",
  "message": "先查一下…再…",
  "tasks": [
    {
      "id": "t1",
      "message": "…",
      "metadata": { "skill_key": "research", "blocked_by": [], "block_policy": "all_succeeded", "capabilities": [] }
    }
  ]
}
```

```
PlanDraft  ──validate_plan()──▶  Kahn 拓扑排序  ──to_a2a_tasks()──▶  A2A Task
             DAG / 体积 / id 格式               id = <plan_id>:<local_id>
             自依赖 / 悬空依赖 / 环              blocked_by ↔ blocked 正反边
```

### 校验是**拒绝制**，不是裁剪制

| 情况 | 结果 |
|---|---|
| `message` 或 `metadata` > 64 KiB | ❌ 整份计划作废（`PlanValidationError`） |
| 有环 / 自依赖 / 引用不存在的依赖 | ❌ 整份计划作废 |
| `id` 格式不合法 | ❌ 整份计划作废 |
| `skill_key` 不在目录里 | ⚠️ 降级成 `plan.notes` 里的 warning |
| 未知 `capability` code | ⚠️ 降级成 warning |

> **为什么不裁剪**：一个被悄悄删掉两个任务的计划，比一个明确报错的计划危险得多。

---

## 🎬 进入 / 退出计划模式

**两条入口，一条执行路径**：

| 入口 | 怎么进 |
|---|---|
| 分类器合成 | `builder` 节点判定要规划 → **合成一条 `enter_plan_mode` 调用**塞进 tools 节点 |
| 模型自调用 | 模型自己调 `enter_plan_mode` |

合成那条是刻意设计的：它走的是**和模型自己调用完全相同**的代码路径 —— 少一条路径就少一处要排查的地方。

- `enter_plan_mode` 是**幂等**的（重复进入不会叠 state）
- `exit_plan_mode` 结束计划模式
- **重规划的唯一判据**在 `plan_state.py:86`（`plan_needs_replan`）—— 全仓只有这一处判断

| 开关 | 效果 |
|---|---|
| `YTRADER_PLAN_STOP_AFTER_PLANNER=1` | `planner → END`，只规划不执行（调试计划质量用） |

---

## 📡 观测

| 事件 / span | 说明 |
|---|---|
| `plan.published` | planner 结束时**唯一**发出的事件（不含任务板） |
| 节点 span | 走 `NodeLogRecorder` —— **只打日志、不落库** |

> 计划任务**没有 HTTP 面**。外部系统想看进度只能订阅事件流 —— 这是已知能力缺口，
> 不是遗漏（见 [`todolist`](todolist)）。

---

## 🚀 验证

```bash
.venv/Scripts/python.exe -m pytest tests/agent/graph/planner -q     # 33 个用例
```

---

## 🤝 相关

| 想深入看 | 去哪 |
|---|---|
| 谁来执行这些任务、怎么并行 | [`README_SUB_AGENT.md`](README_SUB_AGENT.md) |
| `create_team` 工具与审批 | [`README_TOOL.md`](README_TOOL.md) |
| 规划阶段的上下文怎么拼（`PLANNER` builder） | [`README_CONTEXT.md`](README_CONTEXT.md) |
| 总览 | [`README.md`](README.md) |
