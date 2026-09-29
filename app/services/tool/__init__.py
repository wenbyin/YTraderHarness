"""
Tool Definition Management 模块。

职责
----
    - 从代码 / 运行时来源发现并同步工具定义
    - 暴露代码侧的工具元数据（只读）
    - 管理人工的说明文字覆盖
    - 管理业务分类（供 LLM 与 JEV 筛选工具）

不负责（见 tool.yaml 的 Non-responsibilities）
    - Group / Permission / Tag / Processor chain

分层（与 document 模块一致）
---------------------------
    domain/       实体与对外契约
    models/       SQLAlchemy ORM
    repository/   持久化（ABC + Impl）
    application/  服务（ABC + Impl）
    registry/     @tool 自注册与发现

规格
----
    models.sql    建表语句（本模块另补了一张关联表，见 models/sqlalchemy/）
    tool.yaml     OpenAPI 契约
"""
