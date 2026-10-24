"""
SkillService 的行为测试（内存假仓储）。

验的是服务层的判断：
    - skill_key 冲突
    - PATCH 保留 skill_key、拒绝空请求与显式 null
    - 绑定全量替换 + 能力存在性校验（失败时不改动旧绑定）
    - 列表的状态过滤与分页

不碰数据库：仓储逻辑（EXISTS、排序）由真实库的冒烟覆盖。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.common.exceptions import ConflictError, NotFoundError
from app.services.capability.domain.entities import Capability
from app.services.skill.application.skill_service_impl import (
    SkillServiceImpl,
)
from app.services.skill.domain.schemas import (
    SkillCapabilityUpdate,
    SkillCreate,
    SkillStatus,
    SkillUpdate,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _capability(
        code: str,
        name: str,
        *,
        sort_order: int = 0,
) -> Capability:
    now = _now()
    return Capability(
        capability_id=uuid4(),
        code=code,
        name=name,
        description=None,
        parent_id=None,
        sort_order=sort_order,
        is_active=True,
        created_at=now,
        updated_at=now,
    )


def _create_request(
        key: str,
        *,
        status: SkillStatus = SkillStatus.DRAFT,
        name: str | None = None,
        description: str = "检索公告",
) -> SkillCreate:
    return SkillCreate(
        skill_key=key,
        name=name or key,
        description=description,
        instructions=f"执行 {key} 的 SOP。",
        status=status,
    )


class _FakeSkillRepository:
    """只实现服务用到的 skill 表操作。"""

    def __init__(self) -> None:
        self.rows: dict[UUID, object] = {}

    async def add(self, skill):
        self.rows[skill.skill_id] = skill
        return skill

    async def update(self, skill):
        self.rows[skill.skill_id] = skill
        return skill

    async def delete(self, skill_id):
        self.rows.pop(skill_id, None)

    async def get(self, skill_id):
        return self.rows.get(skill_id)

    async def get_by_key(self, skill_key):
        for row in self.rows.values():
            if row.skill_key == skill_key:
                return row
        return None

    async def list_paged(
            self,
            *,
            status=None,
            keyword=None,
            capability_id=None,
            offset=0,
            limit=20,
    ):
        rows = sorted(
            self.rows.values(),
            key=lambda r: r.skill_key,
        )

        if status is not None:
            rows = [r for r in rows if r.status is status]

        if keyword:
            rows = [
                r for r in rows
                if keyword in r.skill_key
                or keyword in r.name
                or keyword in r.description
            ]

        # capability_id 过滤是仓储实现的事（EXISTS），
        # 这个假仓储不建模关联，因此忽略它。
        total = len(rows)

        return rows[offset:offset + limit], total


class _FakeSkillCapabilityRepository:
    def __init__(
            self,
            capabilities: dict[UUID, Capability],
    ) -> None:
        self._capabilities = capabilities
        self.links: dict[UUID, list[UUID]] = {}

    async def list_for_skill(self, skill_id):
        return [
            self._capabilities[cid]
            for cid in self.links.get(skill_id, [])
            if cid in self._capabilities
        ]

    async def replace_for_skill(
            self,
            skill_id,
            capability_ids,
    ):
        self.links[skill_id] = list(capability_ids)


class _FakeCapabilityRepository:
    def __init__(
            self,
            capabilities: list[Capability],
    ) -> None:
        self._capabilities = {
            c.capability_id: c for c in capabilities
        }

    async def list(self, **kwargs):
        return list(self._capabilities.values())


def _build(
        capabilities: list[Capability] | None = None,
):
    capabilities = capabilities or [
        _capability("announcement.search", "公告检索", sort_order=1),
        _capability("financial.data", "金融数据", sort_order=2),
    ]

    by_id = {c.capability_id: c for c in capabilities}

    skills = _FakeSkillRepository()
    links = _FakeSkillCapabilityRepository(by_id)
    capabilities_repo = _FakeCapabilityRepository(capabilities)

    service = SkillServiceImpl(
        skill_repository=skills,
        skill_capability_repository=links,
        capability_repository=capabilities_repo,
    )

    return service, skills, links, capabilities


async def test_create_duplicate_key_conflicts() -> None:
    service, *_ = _build()

    await service.create_skill(_create_request("announcement_search"))

    with pytest.raises(ConflictError):
        await service.create_skill(
            _create_request("announcement_search")
        )


async def test_create_defaults_to_draft_with_no_bindings() -> None:
    service, *_ = _build()

    detail = await service.create_skill(
        _create_request("announcement_search")
    )

    assert detail.status is SkillStatus.DRAFT
    assert detail.capabilities == []
    assert detail.skill_key == "announcement_search"
    # A2A AgentSkill 的必填 tags 存空数组；examples 可选。
    assert detail.tags == []
    assert detail.examples == []


async def test_update_keeps_key_and_rejects_empty_or_null() -> None:
    service, *_ = _build()

    created = await service.create_skill(
        _create_request("announcement_search")
    )

    updated = await service.update_skill(
        created.id,
        SkillUpdate(name="公告检索 v2"),
    )

    assert updated.id == created.id
    assert updated.skill_key == created.skill_key
    assert updated.name == "公告检索 v2"
    assert updated.instructions == created.instructions
    assert updated.status is created.status

    with pytest.raises(ValueError):
        await service.update_skill(created.id, SkillUpdate())

    with pytest.raises(ValueError):
        await service.update_skill(
            created.id, SkillUpdate(name=None)
        )


async def test_a2a_metadata_roundtrip_and_null_semantics() -> None:
    service, *_ = _build()

    created = await service.create_skill(
        SkillCreate(
            skill_key="announcement_search",
            name="公告检索",
            description="检索公告",
            tags=["金融", "公告检索"],
            examples=["检索最近一个月的股权质押公告"],
            instructions="按公司名检索公告。",
        )
    )

    assert created.tags == ["金融", "公告检索"]
    assert created.examples == [
        "检索最近一个月的股权质押公告"
    ]

    updated = await service.update_skill(
        created.id,
        SkillUpdate(tags=["金融"]),
    )

    assert updated.tags == ["金融"]
    # examples 没传 = 保持原值。
    assert updated.examples == [
        "检索最近一个月的股权质押公告"
    ]

    # tags / examples 在库里是 NOT NULL：显式 null 必须被拒绝，
    # 清空请传空数组。
    with pytest.raises(ValueError):
        await service.update_skill(
            created.id, SkillUpdate(tags=None)
        )

    with pytest.raises(ValueError):
        await service.update_skill(
            created.id, SkillUpdate(examples=None)
        )

    cleared = await service.update_skill(
        created.id, SkillUpdate(tags=[], examples=[])
    )

    assert cleared.tags == []
    assert cleared.examples == []


async def test_replace_bindings_validates_before_writing() -> None:
    service, _, links, capabilities = _build()

    created = await service.create_skill(
        _create_request("announcement_search")
    )

    first, second = capabilities

    replaced = await service.replace_skill_capabilities(
        created.id,
        SkillCapabilityUpdate(
            capability_ids=[first.capability_id],
        ),
    )

    assert [c.code for c in replaced] == [first.code]
    assert links.links[created.id] == [first.capability_id]

    # 未知 id 必须整批拒绝，且旧绑定保持不变 ——
    # 部分写入会让调用方以为替换成功了。
    with pytest.raises(NotFoundError):
        await service.replace_skill_capabilities(
            created.id,
            SkillCapabilityUpdate(
                capability_ids=[
                    second.capability_id, uuid4(),
                ],
            ),
        )

    assert links.links[created.id] == [first.capability_id]

    # 空数组 = 清空。
    cleared = await service.replace_skill_capabilities(
        created.id,
        SkillCapabilityUpdate(capability_ids=[]),
    )

    assert cleared == []
    assert links.links[created.id] == []


async def test_list_filters_status_and_pages() -> None:
    service, *_ = _build()

    await service.create_skill(
        _create_request("alpha", status=SkillStatus.ACTIVE)
    )
    await service.create_skill(
        _create_request("beta", status=SkillStatus.ACTIVE)
    )
    await service.create_skill(
        _create_request("gamma", status=SkillStatus.DRAFT)
    )

    page = await service.list_skills(
        status=SkillStatus.ACTIVE,
        page=1,
        page_size=1,
    )

    assert page.total == 2
    assert page.page == 1
    assert page.page_size == 1
    assert [s.skill_key for s in page.items] == ["alpha"]
    # 摘要带 tags（发现用），但不带 instructions / capabilities。
    assert page.items[0].tags == []
    assert "instructions" not in page.items[0].model_dump()
    assert "capabilities" not in page.items[0].model_dump()
    assert "examples" not in page.items[0].model_dump()

    second = await service.list_skills(
        status=SkillStatus.ACTIVE,
        page=2,
        page_size=1,
    )

    assert [s.skill_key for s in second.items] == ["beta"]

    searched = await service.list_skills(keyword="gam")

    assert [s.skill_key for s in searched.items] == ["gamma"]


async def test_missing_skill_is_not_found() -> None:
    service, *_ = _build()

    with pytest.raises(NotFoundError):
        await service.get_skill(uuid4())

    with pytest.raises(NotFoundError):
        await service.list_skill_capabilities(uuid4())

    with pytest.raises(NotFoundError):
        await service.delete_skill(uuid4())
