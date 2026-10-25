# Sub-Agent · 子代理

> 子代理是**一个 worker 子图**：`task_listener → llm_node ↔ tool_node`，一次执行**一个**已 claim 的任务。
>
> 整个设计的核心问题是「**谁跑哪个任务**」。答案不是「worker 抢」，而是**父图先 CAS 预 claim，再按 `Send` 分发** ——
> 因为一旦多个 worker 各自读板抢同一批 ready，重复执行就是必然的，而且**不会报错**。

`1 个 worker 子图` · `预 claim + Send 扇出` · `Redis 任务板 TTL 7 天` · `fail-closed` · `结果只走任务板`

---

## 📚 本文结构

- 🎯 **一张表看懂**
- 🧩 **先分清三个「任务」**
- 🎬 **端到端时序**
- 🔒 **三条并发正确性保证**
- 🚨 **Fail-closed 矩阵**
- 📖 返回 **[主手册](README.md)** · ⚠️ 边界见 **[todolist](todolist)**

---

## 🎯 一张表看懂

| 问题 | 一句话答案 |
|---|---|
| 子代理是什么？ | 一个 **worker 子图**，一次执行**一个**已 claim 的任务 |
| 怎么被拉起？ | `create_team` 审批通过 → `subagent_dispatch` 预 claim N 个任务 → `Command(goto=[Send(...)])` 扇出 N 个分支 |
| 一个分支跑几个任务？ | **一个**。`single_task=True`，完成即退出；下一波由 dispatch 重新扇出 |
| 谁 claim 任务？ | **父图** dispatch 预 claim（`owner={run_id}:worker:{i}`），worker 只「收养」，不再抢 |
| 为什么要预 claim？ | ① 两个 worker 同时抢板会互相踩 ② `Send` 的 payload **取代 state**，worker 看不到父图 `tasks` |
| 任务板是必须的吗？ | **是**。没有 Redis 任务板 → `subagent_dispatch` 退回主 agent 顺序执行（**fail-closed**） |
| 任务板读不到怎么办？ | **抛 `RuntimeError`**，整轮 failed。不假装「没有 ready task」，也不假装「派发成功」 |
| worker 的结果去哪？ | Redis 任务板的 `artifacts` + `metadata.result_summary`；**不进**主对话 `messages` |
| 主对话怎么看到结果？ | 全部任务终态后，父图 `task_listener` 起一次**汇总回合**，渲染成 HumanMessage 交给主 agent |
| 实现完整度 | 3 个节点全实现；**42 个相关 pytest 通过（1.50s）** |

---

## 🧩 先分清三个「任务」

| 名字 | 是什么 |
|---|---|
| **A2A 计划任务** | planner 产出的 DAG 节点，活在 graph state / Redis 任务板 / 事件流里 |
| **Redis 任务板** | 执行态**权威**（TTL 7 天，按 `plan_id` 索引）；graph state 只是投影 |
| `/api/v1/tasks` | ⚠️ **另一套异步作业**（tasks / outbox），与计划任务无关 |

---

## 🏗️ 阅读地图

```
app/agent/graph/subagent/         1,056 行 —— worker 子图
├── catalog.py        124  图名/版本/节点目录/边/模型路由键
├── state.py          156  SubAgentState / SubAgentOutput（空）/ ToolPayload
├── nodes.py          574  task_listener / llm_node / tool_node
├── routing.py         83  route_after_listener / route_after_llm
└── graph.py           93  装配（output_schema 为空 schema，关键）

app/agent/graph/react/
├── subagent_dispatch_node.py   507   预 claim + Send 扇出 + 事件补发
├── graph.py                    worker 子图挂载 + join 静态边
└── task_listener_node.py       顺序执行 / 团队决策 / 汇总回合

app/agent/plan/
├── listener.py    722  claim 规则 / State+Redis 两种源 / Lua CAS / complete_task
├── publish.py     391  Redis 任务板（幂等 upsert / TTL / complete）
├── checks.py      191  pending / ready / has_parallel_ready / find_dependency_cycle
└── task_prompt.py 448  任务指令与结果渲染的**唯一**入口
```

---

## 🎬 端到端时序

```
1. planner 产出 DAG（tasks 只在 state 里，Redis 上什么都没有）
2. task_listener 发现 ≥2 个 ready task
        └─ 且本轮暴露了 create_team → 不 claim，写 team_decision="pending" 回主 agent
3. 主 agent 决定开团队 → 调 create_team
        └─ 走审批（interrupt）→ 通过后才执行
4. create_team 发布任务板（publish_plan）
5. 图跳到 subagent_dispatch：
        ├─ 从任务板预 claim 最多 subagent_workers 个 ready task（CAS）
        └─ Command(goto=[Send(...)]) 扇出 N 个分支
6. subagent_worker ×N：task_listener → llm_node ↔ tool_node → 完成即退出
7. join 回 subagent_dispatch → 下一波 或 收尾
8. 全部终态 → task_listener 起汇总回合（结果取自任务板 artifact）
```

### 关键顺序（错了就出事故）

| 顺序 | 不这么做会怎样 |
|---|---|
| **审批在发布之前** | 未批准就发板 → 任务在板上但没人有权跑 |
| **预 claim 在 Send 之前** | worker 各自抢板 → 重复执行 |
| **结果写板在退出之前** | 分支退出后没机会回写 → 任务永久 `WORKING` |
| **汇总在全部终态之后** | 部分结果被当完整结果交给主 agent |

---

## 🔒 三条并发正确性保证

| 保证 | 做法 | 不这么做 |
|---|---|---|
| **谁跑哪个任务是确定的** | 父图先 CAS 预 claim，再按 `Send` 分发（`owner={run_id}:worker:{i}`） | 多个 worker 各自读板抢同一批 ready，**重复执行** |
| **任务板是执行态权威** | Redis（TTL 7 天，按 `plan_id` 索引）；graph state 只是投影 | `Send` payload 取代 state，worker 根本看不到父图 `tasks` |
| **结果只走任务板** | worker 写 `artifacts` + `result_summary`；父图在全部终态后起一次汇总回合 | worker transcript 串进主对话，多 worker 并发写 `messages` |

- CAS 用 **Redis Lua** 保证原子性
- 任务板 upsert 是**幂等**的，且**不覆盖运行态**

### 上下文隔离

worker 用 `ContextBuilderType.SUB_AGENT`（同一套 13 段），但 **`output_schema` 是空 schema**。

| 隔离的四个理由 | 不隔离会怎样 |
|---|---|
| worker transcript 不该进主对话 | 用户会看到 4 个并行推理过程串在一起 |
| 多 worker 并发写 `tasks` / `plan` | 最后写者赢，或 `InvalidUpdateError` |
| tool scope 命名空间独立 | 与主图算出同一个 `branch_key`，观测行被合并 |
| 上下文预算独立 | 主对话的长历史挤掉 worker 的工具返回 |

---

## 🚨 Fail-closed 矩阵

| 情况 | 行为 |
|---|---|
| 没有 Redis 任务板 | **不派发**，退回主 agent 顺序执行 |
| 读板失败 / 快照不可用 | **抛 `RuntimeError`**，整轮 failed |
| `create_team` 审批被拒 | 不切模式、不起 worker（正确行为） |
| 没有 ready task | 回 `task_listener` 收尾，不是「空转」 |
| worker 崩溃 | ⚠️ 任务停在 `WORKING` —— **lease 只写不续租**，见 [`todolist`](todolist) |

> **fail-closed 的含义是**：宁可退回「慢但确定」的顺序执行，也不假装并行成功。

### 任务状态机

`is_ready` 的三个条件：依赖全部 `COMPLETED`、本身是 `PENDING`、未被 claim。

**空转防护**：`task_listener → agent → task_listener` 这条环，靠「团队决策是**一次性**的」断开 ——
主 agent 那一轮结束后若仍是 `pending`（没调 `create_team`），listener 写 `declined` 并继续顺序 claim。

---

## 📡 观测

| 事件 | 谁发 |
|---|---|
| `subagent.dispatched` / `subagent.completed` / `subagent.failed` | **父图补发** |

> ⚠️ 为什么由父图补发：worker 是并发分支，各自发事件会乱序且无法保证都发出。
> 两条刻意的规则：worker **不发** `plan.*` 事件；worker **不发**自身生命周期事件。

---

## 🚀 验证

```bash
.venv/Scripts/python.exe -m pytest tests/agent/graph/subagent -q     # 42 个用例
```

---

## 🤝 相关

| 想深入看 | 去哪 |
|---|---|
| 任务怎么被拆出来的 | [`README_PLAN.md`](README_PLAN.md) |
| `create_team` 的审批与幂等 | [`README_TOOL.md`](README_TOOL.md) |
| worker 的上下文怎么拼 | [`README_CONTEXT.md`](README_CONTEXT.md) |
| 总览 | [`README.md`](README.md) |
