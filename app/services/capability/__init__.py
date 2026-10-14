"""
业务能力（capability）模块。

从 app/services/tool 拆出来的独立模块：能力是一棵**树**，
有自己的生命周期（增删改查、层级、启用状态），与工具的定义
同步、说明覆盖没有关系，此前只是恰好被工具用着才挤在一起。

与工具的关系
------------
    本模块         能力本体（capability 表）
    tool 模块      工具 ↔ 能力的关联（tool_definition_classification 表）

关联表留在 tool 模块，因为它的主语是工具：那条路径是
/tools/{tool_id}/classifications，删掉一个工具就该顺带解除它的
关联（见 tool 模块的 models.sql）。本模块因此**不认识**工具，
只有一处例外：删除能力前要数一数还有多少工具在用它
（CapabilityRepository.count_links）—— 那是"我能不能被删"的
判断，属于本模块。

⚠️ 与 document 模块的 capability 不是一回事
--------------------------------------------
    app/services/document/domain/schemas/capability.py
        处理器**能力目录**：平台有哪些处理能力、参数怎么填
        表是 document_processor_definition 等，没有 capability 表

    本模块
        给 LLM / JEV 筛工具用的**业务分类树**
        表就是 capability

两者同名不同物。看到 capability 时先确认是哪一个 —— 本模块的
一律带 Capability* 前缀（CapabilityService / CapabilityRepository），
document 那边是 ProcessorCapability*。
"""
