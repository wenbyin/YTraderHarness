"""
组合工具示例：把两个 service 的能力拼成一个工具给模型。

它演示组合工具的三个要点：

    1. 用 **langchain 的 @tool**（不是服务方法上那个）
    2. 靠 **应用级工厂** 拿 service —— 这里没有 Request
    3. 一次调用里可以碰**多个** service，把结果拼起来

真实用途：Agent 在动手之前先问一句"你能做什么"。模型看不见
工具表以外的能力，也不该把整张表塞进上下文，因此给一个
汇总入口是划算的。
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.common.factories import get_service_factory
from app.services.document.application.capability_service import (
    ProcessorCapabilityService,
)
from app.services.tool.application.tool_query_service import (
    ToolQueryService,
)

#: 覆盖推导出来的 source_ref（推导会给 "agent"）。
#: 写这一行的原因见包 docstring —— langchain 的 @tool 声明不了它。
TOOL_SOURCE_REF = "agent.platform"


@tool
async def describe_platform(tool_limit: int = 20) -> dict:
    """
    汇总平台当前可用的工具与文档处理能力。

    用于回答"你能做什么"这类问题，避免把整张工具表塞进上下文。
    """

    factory = get_service_factory()

    # ---- 服务 1：能力目录 ------------------------------------
    #
    # 它不碰数据库（读的是启动时反射出来的静态表），因此用
    # create() 而不是 scope() —— 不占连接、不开事务。
    capabilities = factory.create(ProcessorCapabilityService)

    catalog = await capabilities.list_capabilities()

    # ---- 服务 2：工具表 --------------------------------------
    #
    # 只读，所以 begin=False：开一段会话但不显式开事务。
    # 写明只读是有意义的 —— 默认的 scope() 会开事务，而这里
    # 一次写入都没有。
    async with factory.scope(
            ToolQueryService, begin=False
    ) as tool_service:
        page = await tool_service.list_tools(
            page=1, page_size=tool_limit
        )

    # ---- 拼成一个结果 ----------------------------------------
    #
    # 返回值必须是可 JSON 序列化的 —— 它要进对话消息。
    # 服务的响应是 pydantic 模型，用 model_dump(mode="json")
    # 转过去，而不是直接塞对象。
    return {
        "tool_count": page.total,
        "tools": [
            {
                "key": item.tool_key,
                "name": item.name,
                "description": item.description,
            }
            for item in page.items
        ],
        "processor_capabilities": [
            {
                "key": entry.processor_key,
                "role": entry.role.value,
            }
            for entry in catalog.items
        ],
    }
