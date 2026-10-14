"""
ToolClassificationService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext
from app.services.capability.application._assembly import (
    build_capability,
)
from app.services.capability.domain.entities import (
    Capability as CapabilityEntity,
)
from app.services.capability.domain.schemas import Capability
from app.services.capability.repository.capability import (
    CapabilityRepository,
)

from ..domain.schemas import ToolClassificationUpdate
from ..repository.tool_classification import (
    ToolClassificationRepository,
)
from ..repository.tool_definition import ToolDefinitionRepository
from .tool_classification_service import ToolClassificationService


class ToolClassificationServiceImpl(ToolClassificationService):

    def __init__(
            self,
            *,
            tool_classification_repository: (
                ToolClassificationRepository
            ),
            tool_definition_repository: ToolDefinitionRepository,
            capability_repository: CapabilityRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._links = tool_classification_repository
        self._definitions = tool_definition_repository
        self._capabilities = capability_repository

    async def get_tool_classifications(
            self,
            tool_id: UUID,
    ) -> list[Capability]:

        await self._require_tool(tool_id)

        entities = await self._links.list_for_tool(tool_id)

        return self._to_dtos(entities)

    async def replace_tool_classifications(
            self,
            tool_id: UUID,
            request: ToolClassificationUpdate,
    ) -> list[Capability]:

        await self._require_tool(tool_id)

        # 一次查回来比逐个 get 好，而且能一次报出**全部**无效 id，
        # 而不是让调用方改一个再撞一个。
        #
        # 这个校验查的是能力表，走的是 capability 模块的仓储 ——
        # 关联仓储只管"挂上/摘掉"，不该兼职判断被挂的东西存不存在。
        wanted = list(dict.fromkeys(request.classification_ids))

        if wanted:
            known = {
                c.capability_id
                for c in await self._capabilities.list()
            }

            unknown = [i for i in wanted if i not in known]

            if unknown:
                raise NotFoundError(
                    "能力不存在："
                    + ", ".join(str(i) for i in unknown)
                )

        await self._links.replace_tool_classifications(
            tool_id, wanted
        )

        entities = await self._links.list_for_tool(tool_id)

        return self._to_dtos(entities)

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _to_dtos(
            entities: list[CapabilityEntity],
    ) -> list[Capability]:
        """
        关联仓储交出来的是**实体**（capability_id / created_at …），
        而这两个接口的 response_model 是 DTO（id / 没有时间戳）。

        转换必须在这里做完：漏掉这一步，实体就会直接撞上 FastAPI 的
        响应校验，报的是 'response.0.id Field required' —— 那句话指向
        响应体，离"有人在某处少写了一层组装"很远。组装逻辑复用
        capability 模块的 build_capability，不另写一份（两份迟早漂移）。
        """

        return [build_capability(e) for e in entities]

    async def _require_tool(self, tool_id: UUID) -> None:
        if await self._definitions.get(tool_id) is None:
            raise NotFoundError(f"Tool 定义不存在：{tool_id}")
