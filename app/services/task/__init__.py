"""
Task 模块：统一异步任务基础设施的持久化与对外契约。

分层与仓库其余模块一致：

    domain/schemas     对外契约（APIModel）
    models/sqlalchemy  tasks / task_outbox 两张表
    repository         TaskStore（仓储契约与 Postgres 实现）
    application        TaskService（业务提交任务的唯一入口）

⚠️ 共同抽象不在这里，在 app/common/task
---------------------------------------
Task 模型、状态、端口（TaskStore / TaskDispatcher / TaskHandlerRegistry）
与异常都定义在 app/common/task —— 那样 infra/task 的运行时不依赖本模块
也能工作，且依赖方向保持单向：

    app/common/task  <-  app/services/task  <-  app/infra/task

本模块只负责"把 Task 存下来、按状态机推进、把它暴露成 HTTP 契约"。

另外：**本模块不是业务模块**。它不懂 HITL、Run、Memory、Tool ——
那些是各自模块的事，它们通过注册 Handler 接入（见 infra/task/registry.py）。
"""
