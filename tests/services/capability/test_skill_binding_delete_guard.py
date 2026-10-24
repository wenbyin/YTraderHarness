"""
删除能力前必须先解除 Skill 绑定。

skill_capability 的外键是 ON DELETE RESTRICT，而 capability 删除
接口原本只统计工具引用。没有这道检查时，用户会撞到一个没有上下文
的外键错误（或者更糟：如果哪天有人把外键改成 CASCADE，绑定会
无声消失，对应 Skill 的工具可见性悄悄缩小）。

这里同时钉住：force 不能绕过这道关。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.common.exceptions import ConflictError
from app.services.capability.application.capability_service_impl import (
    CapabilityServiceImpl,
)
from app.services.capability.domain.entities import Capability


def _capability() -> Capability:
    now = datetime.now(timezone.utc)

    return Capability(
        capability_id=uuid4(),
        code="announcement.search",
        name="公告检索",
        description=None,
        parent_id=None,
        sort_order=0,
        is_active=True,
        created_at=now,
        updated_at=now,
    )


class _FakeCapabilityRepository:
    """只实现 delete_capability 用到的四个方法。"""

    def __init__(
            self,
            capability: Capability,
            *,
            children: int = 0,
            skill_links: int = 0,
            tool_links: int = 0,
    ) -> None:
        self._capability = capability
        self._children = children
        self._skill_links = skill_links
        self._tool_links = tool_links
        self.deleted = None

    async def get(self, capability_id):
        if capability_id == self._capability.capability_id:
            return self._capability
        return None

    async def count_children(self, capability_id):
        return self._children

    async def count_skill_links(self, capability_ids):
        return self._skill_links

    async def count_links(self, capability_ids):
        return self._tool_links

    async def delete(self, capability_id):
        self.deleted = capability_id


def _service(repository) -> CapabilityServiceImpl:
    return CapabilityServiceImpl(
        capability_repository=repository,
    )


async def test_skill_binding_blocks_delete_even_with_force() -> None:
    capability = _capability()
    repository = _FakeCapabilityRepository(
        capability, skill_links=2,
    )
    service = _service(repository)

    for force in (False, True):
        with pytest.raises(ConflictError) as excinfo:
            await service.delete_capability(
                capability.capability_id, force=force,
            )

        assert "Skill" in str(excinfo.value)

    assert repository.deleted is None


async def test_unbound_capability_deletes() -> None:
    capability = _capability()
    repository = _FakeCapabilityRepository(capability)
    service = _service(repository)

    await service.delete_capability(capability.capability_id)

    assert repository.deleted == capability.capability_id


async def test_tool_links_still_require_force() -> None:
    capability = _capability()
    repository = _FakeCapabilityRepository(
        capability, tool_links=3,
    )
    service = _service(repository)

    with pytest.raises(ConflictError):
        await service.delete_capability(capability.capability_id)

    await service.delete_capability(
        capability.capability_id, force=True,
    )

    assert repository.deleted == capability.capability_id
