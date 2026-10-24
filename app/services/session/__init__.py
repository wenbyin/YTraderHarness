"""
Agent 运行时会话（session）与运行（run）。

它回答的是 `app/agent/graph/react/runtime.py` 里**明确悬置**的那个问题
--------------------------------------------------------------------
那份 docstring 写着"本次不写 dispatcher handler"，理由是：

    `RunSnapshot.subject_id` 是**非空 UUID**，而一次 agent 运行的身份是
    thread_id（一个 str）。于是"一次 agent 运行的主体是什么"必须先回答：
    是会话表里的那一行（那要先有会话表与它的 UUID），还是先拿
    pipeline_run_id 顶上？

本模块给出的答案是**前者**：一次运行的主体是它所属的会话那一行。

    会话（agent_session）  长期存在，跨多次运行。主体身份在这。
    运行（agent_run）      一次执行，单向派生自会话。运行历史在这。

这与 document 模块的形状对齐（那边是 document_version / pipeline_run）。
`new_thread_id(namespace, subject_id: UUID, pipeline_run_id: UUID)` 正好
要求一个 UUID 主体 —— 会话的 session_id 就是它。

与 document 模块的分工
----------------------
    app/infra/graph/      两边共用的运行时（runner / registry / recorder /
                          dispatcher / checkpointer），本模块不重复造
    document 模块         数据处理流水线的运行
    本模块                Agent 对话的运行

⚠️ 表名为什么不叫 session
------------------------
`session` 在这个仓库里已经有两个很重的含义：数据库会话
（`AsyncSession` / `session_factory`，见 every repository）与 HTTP 会话。
再叫 session 会让"这个 session 是哪一个"成为每处代码都要想一遍的问题。
因此表名与 ORM 类都带 `agent_` 前缀，只有**对外契约**（session.yaml 的
路径与 DTO）保留 session 这个词 —— 那是 API 的词汇，不是数据库的。

⚠️ 实现到哪一步
--------------
已经做：会话、运行、消息三张表，以及它们的完整读写入 ——

    ①② 建会话 / 查会话
    ②b 会话列表（严格本人 + 分页筛选）与会话软删（归档）
    ③  建运行并**真的派发**：建出来的运行会跑 `agent_react` 图
       （application/agent_run_dispatch.py），状态从 queued 走到
       completed / failed
    ④  运行事件流（SSE）：`GET /runs/{run_id}/events`
    ⑤  会话消息：落库 + `GET /sessions/{id}/messages`（游标分页）
    ⑥  中断恢复（resume）：`POST /runs/{run_id}/resume`
    ⑦  取消（cancel）：`POST /runs/{run_id}/cancel`

ⓘ 归档（DELETE /sessions/{id}）是软删：只改 status，不清理 checkpoint /
记忆索引；归档时会先取消未终态运行。

消息分两处写（时机不同，见 application/agent_run_service_impl.py 与
message_recorder.py）：用户消息在**接收时**、与运行行同一事务；assistant
消息在**运行终局确定时**、由派发 handler 从图终态里取本轮新增的那些。

没做的部分（例如取消对跨进程运行无能为力、没有 unarchive）在各自的
位置都有注释说明，不留"看起来能跑其实空转"的假象。

⚠️ 已知缺口（每一处都在代码里写了）
--------------------------------
1. **没有查运行的接口**：规格里没有它，前端建完运行只能靠 SSE 事件流
   看进展，或者查库。
2. **模型生成到一半报错，已生成的文字会丢**：agent_node 整段生成完才产出
   AIMessage。见 message_recorder.py 的模块 docstring。
3. **失败后重试会重复用户消息**：checkpoint 与 agent_message 表各一份重复。
   见 agent_run_dispatch.py 的缺口清单。
4. **进程被杀时 agent_run 会停在 running**（document 那边有启动收敛，
   这边还没有）；本轮消息也不会写 —— 与运行停在 running 是同一件事的两面。

⚠️ 冒烟脚本
--------
    python -m app.services.session.agent_run_smoke    派发 + 落消息（不需要数据库）
    python -m app.services.session.message_smoke      翻页（**需要**本地 PostgreSQL）
"""
