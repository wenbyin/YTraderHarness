"""
Mock Model Stress Eval 模块。

契约权威是 ``design_agent_eval/openapi.yaml``，表结构权威是
``design_agent_eval/schema.sql``。本模块只做一件事：

    HTTP 契约 -> StressRunService -> StressRunStore -> PostgreSQL

分层与仓库其余模块一致：

    domain/schemas     对外契约（APIModel）与枚举
    domain/entities    持久化行的 Python 形态（不认识 ORM）
    models/sqlalchemy  stress_runs / stress_run_metrics 两张表
    repository         StressRunStore（契约 + Postgres Core 实现）
    application        StressRunService 与本地模拟执行器

⚠️ 当前阶段不发出真实压力请求
---------------------------
``simulated_runner`` 在**本进程内**推进 queued -> running -> completed，
并用配置推导出一份确定性的聚合快照。它是给前端联调用的占位实现，
不是压测结果；指标里 ``aggregation_algorithm`` 固定写
``simulated-v1``，前端可以据此标注。接真实 Runner 时只需替换
``SimulatedStressRunner.schedule`` 这一处，API 与表结构不用改。
"""
