"""
`AgentSessionService.list_sessions` 的服务层行为。

不连数据库：假仓储把“严格归属 / 过滤 / 排序 / 分页”用 Python 复现一遍，
验的是**服务层怎么传参、怎么校验、怎么组装**。真 SQL 的形状由仓储本身
保证（条件构造只有一处）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from app.common.service.base import ServiceContext
from app.services.session.application.agent_session_service_impl import (
    AgentSessionServiceImpl,
)
from app.services.session.domain.entities import (
    AgentSession as AgentSessionEntity,
)
from app.services.session.domain.schemas import SessionStatus

_NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def _entity(
        *,
        user_id: str | None,
        status: str = "active",
        agent_id: str = "stock_research",
        updated_at: datetime | None = None,
) -> AgentSessionEntity:

    return AgentSessionEntity(
        session_id=uuid4(),
        agent_id=agent_id,
        status=status,
        user_id=user_id,
        created_at=_NOW,
        updated_at=updated_at or _NOW,
    )


class _FakeSessionRepository:
    """只实现 list_sessions 用到的 `list_page`。"""

    def __init__(self, entities: list[AgentSessionEntity]) -> None:
        self._entities = entities
        self.calls: list[dict] = []

    async def list_page(
            self,
            *,
            user_id: str,
            status: str | None,
            agent_id: str | None,
            updated_after: datetime | None,
            updated_before: datetime | None,
            offset: int,
            limit: int,
    ) -> tuple[list[AgentSessionEntity], int]:

        self.calls.append({
            "user_id": user_id,
            "status": status,
            "agent_id": agent_id,
            "updated_after": updated_after,
            "updated_before": updated_before,
            "offset": offset,
            "limit": limit,
        })

        matched = [
            entity
            for entity in self._entities
            if entity.user_id == user_id
            and (status is None or entity.status == status)
            and (agent_id is None or entity.agent_id == agent_id)
            and (
                updated_after is None
                or entity.updated_at >= updated_after
            )
            and (
                updated_before is None
                or entity.updated_at <= updated_before
            )
        ]

        matched.sort(
            key=lambda entity: (entity.updated_at, entity.session_id),
            reverse=True,
        )

        return matched[offset:offset + limit], len(matched)


class _UnusedRunRepository:
    """list_sessions 不碰运行仓储。"""


def _service(
        entities: list[AgentSessionEntity],
) -> tuple[AgentSessionServiceImpl, _FakeSessionRepository]:

    sessions = _FakeSessionRepository(entities)

    service = AgentSessionServiceImpl(
        agent_session_repository=sessions,  # type: ignore[arg-type]
        agent_run_repository=_UnusedRunRepository(),  # type: ignore[arg-type]
        ctx=ServiceContext(),
    )

    return service, sessions


async def test_only_own_sessions_and_default_active() -> None:
    mine_active = _entity(user_id="user-1")
    mine_archived = _entity(user_id="user-1", status="archived")
    other = _entity(user_id="user-2")
    legacy = _entity(user_id=None)

    service, sessions = _service(
        [mine_active, mine_archived, other, legacy]
    )

    page = await service.list_sessions(user_id="user-1")

    assert [item.session_id for item in page.items] == [
        mine_active.session_id,
    ]
    assert page.total == 1

    # 默认 active 是服务层决定的；仓储拿到的是字符串。
    assert sessions.calls[0]["status"] == SessionStatus.ACTIVE.value

    # 严格本人：NULL 存量会话不出现，别人的也不出现。
    assert sessions.calls[0]["user_id"] == "user-1"


async def test_filters_and_pagination_are_forwarded() -> None:
    entities = [
        _entity(user_id="user-1", updated_at=_NOW - timedelta(days=i))
        for i in range(5)
    ]
    service, sessions = _service(entities)

    page = await service.list_sessions(
        user_id="user-1",
        status=SessionStatus.ARCHIVED,
        agent_id="stock_research",
        updated_after=_NOW - timedelta(days=3),
        updated_before=_NOW,
        page=2,
        page_size=2,
    )

    call = sessions.calls[0]

    assert call["status"] == "archived"
    assert call["agent_id"] == "stock_research"
    assert call["offset"] == 2
    assert call["limit"] == 2
    assert call["updated_after"].tzinfo is not None

    # 没有 archived 的样本 -> 空页，但 total 是过滤后的真实计数。
    assert page.items == []
    assert page.total == 0


async def test_empty_user_id_short_circuits() -> None:
    service, sessions = _service([_entity(user_id=None)])

    page = await service.list_sessions(user_id="")

    assert page.items == []
    assert page.total == 0
    assert sessions.calls == []


async def test_naive_datetime_is_rejected() -> None:
    service, sessions = _service([])

    with pytest.raises(ValueError, match="时区"):
        await service.list_sessions(
            user_id="user-1",
            updated_after=datetime(2026, 10, 18, 0, 0),
        )

    assert sessions.calls == []


async def test_reversed_time_range_is_rejected() -> None:
    service, sessions = _service([])

    with pytest.raises(ValueError, match="updated_after"):
        await service.list_sessions(
            user_id="user-1",
            updated_after=_NOW,
            updated_before=_NOW - timedelta(days=1),
        )

    assert sessions.calls == []


async def test_aware_datetime_is_normalized_to_utc() -> None:
    service, sessions = _service([])

    shanghai = timezone(timedelta(hours=8))
    value = datetime(2026, 10, 18, 20, 0, tzinfo=shanghai)

    await service.list_sessions(
        user_id="user-1",
        updated_after=value,
    )

    forwarded = sessions.calls[0]["updated_after"]

    assert forwarded == value
    assert forwarded.tzinfo == timezone.utc


async def test_page_shape_uses_session_dto() -> None:
    """
    items 是契约 DTO，不是实体 —— 实体多出的 meta / created_at 不该外泄。
    """

    service, _ = _service([_entity(user_id="user-1")])

    page = await service.list_sessions(user_id="user-1")

    dumped = page.items[0].model_dump()

    assert set(dumped) == {
        "session_id",
        "agent_id",
        "status",
        "user_id",
        "updated_at",
    }
