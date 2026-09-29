"""
处理能力目录的查询契约。

为什么单独一个服务
------------------
它读的是**进程启动时反射出来的静态能力表**（PipelineRegistry），
不碰数据库、不开 session、不持有仓储 —— 与 DocumentRuntimeService、
PipelineQueryService 的依赖形状完全不同，塞进去只会把干净的
只读服务污染成"有一半依赖用不上"。

只读，无副作用。

与配置资源的分工
----------------
    GET /document-processors              配置资源（可增删改的数据库行）
    GET /document-processor-capabilities  能力目录（代码里的实现，只读）

前者的 loader_name / processor_name 等字段取值，正是后者的
processor_key。两者是"配置引用能力"的关系。
"""

from __future__ import annotations

from abc import abstractmethod

from langchain_core.tools import tool

from app.common.service.base import BaseService

from ..domain.schemas.capability import (
    ProcessorCapabilityCatalogResponse,
    ProcessorCapabilityEntry,
)
from ..domain.schemas.enum import ModalityType, ProcessorRole

__all__ = ["ProcessorCapabilityService"]


class ProcessorCapabilityService(BaseService):
    """处理能力目录的只读查询。"""

    @abstractmethod
    async def list_capabilities(
            self,
            *,
            role: ProcessorRole | None = None,
            modality_type: ModalityType | None = None,
            file_extension: str | None = None,
    ) -> ProcessorCapabilityCatalogResponse:
        """
        列出能力目录。

        modality_type 对**所有 role** 都成立 —— 它是能力的适用域，
        不是 loader 专属。audio 类型的文档只该选到 audio 的能力。

        file_extension 只对 role=loader 有意义 —— 后缀是文件边界的
        概念，清洗之后就没有后缀了。

        响应里的 support_suffix 是**过滤后** loader 条目的后缀并集。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_capability(
            self,
            processor_key: str,
    ) -> ProcessorCapabilityEntry:
        """
        按 key 查单个能力。

        Raises:
            NotFoundError: 该 key 在当前构建里不存在。
                它可能来自历史 PipelineConfigSnapshot ——
                调用方应回退到显示原始 key，而不是丢弃配置。
        """
        raise NotImplementedError
