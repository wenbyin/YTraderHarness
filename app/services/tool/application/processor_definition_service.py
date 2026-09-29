"""
处理器**实现**目录的契约（只读）。

对应 tool_processor.yaml：
    GET /processors
    GET /processors/{processor_key}

这里列的每一行都来自代码侧的注册表，不是数据库 ——
管理面能改的只有"把哪个实现挂到哪个 Tool 上"，
不能创建也不能修改实现本身。
"""

from __future__ import annotations

from abc import abstractmethod

from app.common.service.base import BaseService

from ..domain.schemas import (
    ExecutionMode,
    PagedProcessorDefinitionResponse,
    ProcessorDefinition,
    ProcessorHook,
)

__all__ = ["ProcessorDefinitionService"]


class ProcessorDefinitionService(BaseService):

    @abstractmethod
    async def list_definitions(
            self,
            *,
            keyword: str | None = None,
            enabled: bool | None = None,
            hook: ProcessorHook | None = None,
            execution_mode: ExecutionMode | None = None,
            page: int = 1,
            page_size: int = 50,
    ) -> PagedProcessorDefinitionResponse:
        """
        分页列出已注册的处理器实现。

        keyword 在三处模糊匹配：key / name / description。

        ⚠️ `enabled` 的语义见实现里的说明 —— 它筛的是"注册表可用性"，
        而本接口列出的条目**全都来自注册表**，因此传 false 会得到空集。

        Raises:
            无。这是只读目录，没有"不存在"的分页。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_definition(
            self,
            processor_key: str,
    ) -> ProcessorDefinition:
        """
        取一个实现。

        Raises:
            NotFoundError: 该 key 没有注册任何实现。
        """

        raise NotImplementedError
