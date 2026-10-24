"""
SkillService 的默认实现。

事务边界由调用方（ServiceFactory.scope() 持有的 session）决定，
本服务内部不 commit / rollback。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypeVar
from uuid import UUID, uuid4

from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import ServiceContext
from app.services.capability.domain.entities import (
    Capability as CapabilityEntity,
)
from app.services.capability.repository.capability import (
    CapabilityRepository,
)

from ..domain.entities import Skill as SkillEntity
from ..domain.schemas import (
    CapabilityRef,
    PagedSkillResponse,
    SkillCapabilityUpdate,
    SkillCreate,
    SkillDetail,
    SkillStatus,
    SkillUpdate,
)
from ..repository.skill import SkillRepository
from ..repository.skill_capability import (
    SkillCapabilityRepository,
)
from ._assembly import (
    build_capability_ref,
    build_skill_detail,
    build_skill_summary,
)
from .skill_service import SkillService
from ...tool.registry import tool

_T = TypeVar("_T")


class SkillServiceImpl(SkillService):

    def __init__(
            self,
            *,
            skill_repository: SkillRepository,
            skill_capability_repository: (
                SkillCapabilityRepository
            ),
            capability_repository: CapabilityRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._skills = skill_repository
        self._links = skill_capability_repository
        self._capabilities = capability_repository

    # ========================================================
    # Skill
    # ========================================================

    @tool
    async def list_skills(
            self,
            *,
            status: SkillStatus | None = None,
            keyword: str | None = None,
            capability_id: UUID | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> PagedSkillResponse:

        offset = (page - 1) * page_size

        entities, total = await self._skills.list_paged(
            status=status,
            keyword=keyword,
            capability_id=capability_id,
            offset=offset,
            limit=page_size,
        )

        return PagedSkillResponse(
            items=[build_skill_summary(e) for e in entities],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def create_skill(
            self,
            request: SkillCreate,
    ) -> SkillDetail:

        existing = await self._skills.get_by_key(
            request.skill_key
        )

        if existing is not None:
            raise ConflictError(
                f"Skill 标识已存在：{request.skill_key}"
            )

        now = datetime.now(timezone.utc)

        entity = await self._skills.add(
            SkillEntity(
                skill_id=uuid4(),
                skill_key=request.skill_key,
                name=request.name,
                description=request.description,
                tags=list(request.tags),
                examples=list(request.examples),
                instructions=request.instructions,
                status=request.status,
                created_at=now,
                updated_at=now,
            )
        )

        # 新建的 Skill 还没有绑定，详情页的 capabilities 是空列表。
        return build_skill_detail(entity, [])

    async def get_skill(
            self,
            skill_id: UUID,
    ) -> SkillDetail:

        entity = await self._require_skill(skill_id)

        capabilities = await self._links.list_for_skill(
            skill_id
        )

        return build_skill_detail(entity, capabilities)

    async def update_skill(
            self,
            skill_id: UUID,
            request: SkillUpdate,
    ) -> SkillDetail:

        current = await self._require_skill(skill_id)

        provided = request.model_fields_set

        if not provided:
            raise ValueError("至少提供一个要更新的字段。")

        entity = SkillEntity(
            skill_id=current.skill_id,
            # skill_key 不在契约里；这里显式沿用原值，防止将来有人
            # 给 SkillUpdate 加上该字段后意外改掉稳定标识。
            skill_key=current.skill_key,
            name=self._pick(request, "name", current.name),
            description=self._pick(
                request, "description", current.description
            ),
            tags=self._pick(
                request, "tags", list(current.tags)
            ),
            examples=self._pick(
                request, "examples", list(current.examples)
            ),
            instructions=self._pick(
                request, "instructions", current.instructions
            ),
            status=self._pick(
                request, "status", current.status
            ),
            created_at=current.created_at,
            updated_at=datetime.now(timezone.utc),
        )

        updated = await self._skills.update(entity)

        capabilities = await self._links.list_for_skill(
            skill_id
        )

        return build_skill_detail(updated, capabilities)

    async def delete_skill(
            self,
            skill_id: UUID,
    ) -> None:

        await self._require_skill(skill_id)

        await self._skills.delete(skill_id)

    # ========================================================
    # 能力绑定
    # ========================================================

    async def list_skill_capabilities(
            self,
            skill_id: UUID,
    ) -> list[CapabilityRef]:

        await self._require_skill(skill_id)

        capabilities = await self._links.list_for_skill(
            skill_id
        )

        return self._to_refs(capabilities)

    async def replace_skill_capabilities(
            self,
            skill_id: UUID,
            request: SkillCapabilityUpdate,
    ) -> list[CapabilityRef]:

        await self._require_skill(skill_id)

        wanted = list(dict.fromkeys(request.capability_ids))

        # 一次查回来比逐个 get 好，而且能一次报出**全部**无效 id，
        # 而不是让调用方改一个再撞一个。这个校验查的是能力表，
        # 走 capability 模块的仓储 —— 关联仓储只管挂上/摘掉。
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

        await self._links.replace_for_skill(
            skill_id, wanted
        )

        capabilities = await self._links.list_for_skill(
            skill_id
        )

        return self._to_refs(capabilities)

    # ========================================================
    # internals
    # ========================================================

    async def _require_skill(
            self,
            skill_id: UUID,
    ) -> SkillEntity:

        entity = await self._skills.get(skill_id)

        if entity is None:
            raise NotFoundError(
                f"Skill 不存在：{skill_id}"
            )

        return entity

    @staticmethod
    def _pick(
            request: SkillUpdate,
            field: str,
            current: _T,
    ) -> _T:
        """
        局部更新取值：没传保持原值；显式传 null 直接拒绝。

        skill 的这几列在库里都是 NOT NULL。如果放任 null 走到
        flush，用户看到的是数据库完整性错误 —— 一个与请求无关的
        500。在服务层拒绝，错误码与原因都对得上。
        """

        if field not in request.model_fields_set:
            return current

        value = getattr(request, field)

        if value is None:
            raise ValueError(f"{field} 不能为 null。")

        return value

    @staticmethod
    def _to_refs(
            capabilities: list[CapabilityEntity],
    ) -> list[CapabilityRef]:

        return [
            build_capability_ref(c) for c in capabilities
        ]
