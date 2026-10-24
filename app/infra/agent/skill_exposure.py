"""Skill 源：skill_key -> SkillSnapshot。

它是 `app/agent/context/processor/sources.py::SkillSource` 的唯一实现，
被 SkillLoader 阶段调用。只在 task 模式（有 current_task）时才查库；
普通对话不会走到这里。

为什么不把 SkillDetail 直接交给处理器
----------------------------------
处理器属于 Context Builder，不应该认识 `app.services.skill` 的领域
对象。这里做一次转换，把它收成统一的 `SkillSnapshot`（instructions +
capabilities + 版本），TaskResolver/SkillLoader/将来的 SubAgent 都只
依赖这一个结构。
"""

from __future__ import annotations

from app.agent.context.processor.sources import SkillSource
from app.agent.context.skills import SkillCapability, SkillSnapshot

__all__ = ["ServiceSkillSource"]


class ServiceSkillSource(SkillSource):
    """按 skill_key 从 SkillService 取 ACTIVE 技能快照。"""

    async def load(self, skill_key: str) -> SkillSnapshot | None:

        key = str(skill_key or "").strip()

        if not key:
            return None

        from app.common.factories import get_service_factory
        from app.services.skill.application import SkillService
        from app.services.skill.domain.schemas import SkillStatus

        factory = get_service_factory()

        async with factory.scope(SkillService, begin=False) as service:

            page = await service.list_skills(
                status=SkillStatus.ACTIVE,
                keyword=key,
                page_size=50,
            )

            summary = next(
                (
                    item for item in page.items
                    if item.skill_key == key
                ),
                None,
            )

            if summary is None:
                return None

            detail = await service.get_skill(summary.id)

        return SkillSnapshot(
            skill_key=detail.skill_key,
            name=detail.name,
            description=detail.description or "",
            instructions=detail.instructions or "",
            capabilities=tuple(
                SkillCapability(
                    code=item.code,
                    name=item.name or "",
                )
                for item in detail.capabilities
            ),
            updated_at=detail.updated_at,
        )
