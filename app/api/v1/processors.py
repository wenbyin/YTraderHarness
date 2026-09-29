"""
处理器**实现**目录接口（只读）。

对应 tool_processor.yaml 的 Processor Registry 那组：
    GET /processors
    GET /processors/{processor_key}

这里列的是**代码里注册了哪些处理器**，不是"哪些被挂上了"。
挂载在另一个文件（tool_processors.py）。

只读是刻意的：处理器实现归代码。管理面能做的是"把已注册的实现
挂到某个 Tool 上"，不是"造一个处理器"—— 因此这个文件里没有
任何一个写方法，将来也不该有。
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.tool.application.processor_definition_service import (
    ProcessorDefinitionService,
)
from app.services.tool.domain.schemas import (
    ExecutionMode,
    PagedProcessorDefinitionResponse,
    ProcessorDefinition,
    ProcessorHook,
)

router = APIRouter()


@router.get(
    "/processors",
    response_model=PagedProcessorDefinitionResponse,
    tags=["Processor Registry"],
)
async def list_processors(
        keyword: str | None = Query(
            default=None,
            description="按 processor_key / 名称 / 说明模糊匹配。",
        ),
        enabled: bool | None = Query(
            default=None,
            description=(
                "按注册表可用性过滤。⚠️ 本接口列出的条目全部来自"
                "注册表，因此 registry_available 恒为 true；"
                "传 false 会得到空集。"
            ),
        ),
        hook: ProcessorHook | None = Query(
            default=None,
            description="只返回支持该 hook 的处理器。",
        ),
        execution_mode: ExecutionMode | None = Query(
            default=None,
            description="只返回支持该执行方式的处理器。",
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
        service: ProcessorDefinitionService = Service[
            ProcessorDefinitionService
        ],
):
    """
    分页列出已注册的处理器实现。

    不碰数据库 —— 数据源是进程内的注册表，因此这个接口在
    数据源全挂的时候也照样能用。
    """

    try:
        return await service.list_definitions(
            keyword=keyword,
            enabled=enabled,
            hook=hook,
            execution_mode=execution_mode,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/processors/{processor_key}",
    response_model=ProcessorDefinition,
    tags=["Processor Registry"],
)
async def get_processor(
        processor_key: str,
        service: ProcessorDefinitionService = Service[
            ProcessorDefinitionService
        ],
):
    """
    取一个处理器实现。

    config_schema 是它的参数结构（JSON Schema），前端据此生成表单。
    挂载时服务端用**同一个 pydantic 模型**校验，因此两边规则不会漂移。
    """

    try:
        return await service.get_definition(processor_key)

    except ValueError as exc:
        raise to_http_error(exc) from exc
