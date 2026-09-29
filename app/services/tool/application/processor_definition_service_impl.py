"""
ProcessorDefinitionService 的默认实现。

**只持有注册表，不碰数据库**
----------------------------
与 document 模块的 ProcessorCapabilityServiceImpl 同一取舍：
服务工厂按"__init__ 里有没有仓储注解"决定要不要开事务
（见 common/service/factory.py 的 _build_kwargs），而本服务没有任何
仓储参数，因此 `Service[ProcessorDefinitionService]` 不会占用数据库
连接。一个纯内存目录不该为了读几条类属性去借一条连接。

数据源是**进程级快照**
----------------------
注册表构造后即冻结，因此这里没有任何缓存需求 —— 每次请求重建一遍
DTO 的成本远低于一次数据库往返。
"""

from __future__ import annotations

from app.common.service.base import ServiceContext

from ..domain.schemas import (
    ExecutionMode,
    PagedProcessorDefinitionResponse,
    ProcessorDefinition,
    ProcessorHook,
)
from ..registry.processor_registry import (
    ToolProcessorRegistry,
    get_processor_registry,
)
from ._assembly import build_processor_definition
from .processor_definition_service import ProcessorDefinitionService

__all__ = ["ProcessorDefinitionServiceImpl"]


class ProcessorDefinitionServiceImpl(ProcessorDefinitionService):
    """只读目录。持有的是注册表，不是 session。"""

    def __init__(
            self,
            registry: ToolProcessorRegistry | None = None,
            *,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        # 留一个注入口（与 ToolDiscoveryServiceImpl 的 discoverer 同理）：
        # 用例要能把目录收窄到几个假实现上，否则它会随"别人在别处
        # 加了个处理器"而失败，而失败信息指向不了真正的原因。
        #
        # 工厂不会注入它（不是仓储契约、也不在池子里），
        # 因此生产上走默认值这条分支。
        self._registry = (
            registry if registry is not None
            else get_processor_registry()
        )

    # ========================================================
    # 查询
    # ========================================================

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

        entries = [
            build_processor_definition(processor)
            for processor in self._registry.all()
        ]

        if keyword:
            entries = [
                e for e in entries if self._matches(e, keyword)
            ]

        if enabled is not None:
            # 规格里这个参数筛的是"注册表可用性"，而这里列出的条目
            # **全部来自注册表**，因此 registry_available 恒为 true：
            #
            #     enabled=true  -> 全部
            #     enabled=false -> 空
            #
            # 看着像废话，但它是契约的字面实现。真正的用途在挂载记录
            # 那一侧：链上可能有"实现已被删掉"的行，那种行才需要
            # 前端按 registry_available 分流。这里照做是为了让两边
            # 的字段语义一致，而不是让这个参数在运行时撒谎。
            entries = [
                e for e in entries
                if e.registry_available is enabled
            ]

        if hook is not None:
            entries = [
                e for e in entries if hook in e.supported_hooks
            ]

        if execution_mode is not None:
            entries = [
                e for e in entries
                if execution_mode in e.supported_execution_modes
            ]

        # total 是**过滤之后**的总数，不是本页条数 —— 与列表接口的
        # 其余部分一致（见 schemas/processor.py 的字段说明）。
        total = len(entries)

        offset = (page - 1) * page_size

        return PagedProcessorDefinitionResponse(
            items=entries[offset:offset + page_size],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def get_definition(
            self,
            processor_key: str,
    ) -> ProcessorDefinition:

        # require 而不是 get：查不到抛 NotFoundError（→404），
        # 与挂载写路径共用同一条消息。
        return build_processor_definition(
            self._registry.require(processor_key)
        )

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _matches(
            entry: ProcessorDefinition,
            keyword: str,
    ) -> bool:
        """key / name / description 三处任一命中即算匹配。"""

        needle = keyword.strip().lower()

        if not needle:
            return True

        haystack = " ".join(
            filter(
                None,
                (
                    entry.key,
                    entry.name,
                    entry.description,
                ),
            )
        ).lower()

        return needle in haystack
