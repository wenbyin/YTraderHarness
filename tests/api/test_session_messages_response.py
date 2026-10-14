"""
`GET /sessions/{id}/messages` 的**线上形状**：`reasoning_content` 真的出现在 JSON 里。

⚠️ 为什么必须真打一次 HTTP
----------------------
消息那套冒烟脚本（`app/services/session/message_smoke.py`）调的是**服务层**，
拿回来的 DTO 是 Python 对象 —— 字段名写错、类型不对、被 `response_model`
挡掉，那些用例**全都不会红**。真正决定前端看到什么的是 FastAPI 那一步
序列化，所以这里必须经过它。

（本仓库既有的一处教训：实体被当 DTO 返回过，服务层用例全绿，只有真打
接口才炸。）

装置刻意**不碰数据库**：`service_dependency` 只在请求时调一次
`get_service_factory()`，把它替成一个"不需要事务"的假工厂即可 —— 于是
这条用例验的是路由 -> DTO -> 序列化那一段，毫秒级、不依赖本地 Postgres。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import app.services.dependencies as dependencies_module
from app.api.v1.sessions import router as sessions_router
from app.common.auth import current_user
from app.services.session.application._assembly import build_message_page
from app.services.session.domain.entities import AgentMessage as AgentMessageEntity

_THINKING = "先看要不要查工具，再答。"
_SESSION_ID = uuid4()


def _entity(*, seq: int, reasoning: str | None) -> AgentMessageEntity:
    now = datetime.now(timezone.utc)

    return AgentMessageEntity(
        message_id=uuid4(),
        session_id=_SESSION_ID,
        run_id=uuid4(),
        seq=seq,
        role="assistant",
        kind="chat",
        content=[{"type": "text", "text": f"第 {seq} 条"}],
        reasoning_content=reasoning,
        status="completed",
        created_at=now,
        updated_at=now,
    )


class _FakeMessageService:
    """只要 `list_messages` —— 路由用到的全部。"""

    def __init__(self, entities: list[AgentMessageEntity]) -> None:
        self._entities = entities

    async def list_messages(self, session_id: Any, **kwargs: Any) -> Any:
        return build_message_page(
            session_id=session_id,
            entities=self._entities,
            limit=50,
            has_more_before=False,
            has_more_after=False,
        )


class _FakeFactory:
    """
    `needs_session=False` 是关键：走 `factory.create(contract)` 那条分支，
    不 open 事务，于是这条用例不需要数据库。真工厂那边的 `scope` 会去开
    一段 AsyncSession。
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    def needs_session(self, contract: Any) -> bool:
        return False

    def create(self, contract: Any) -> Any:
        return self._service


def _app(monkeypatch: pytest.MonkeyPatch, entities: list[AgentMessageEntity]) -> FastAPI:
    app = FastAPI()
    app.include_router(sessions_router, prefix="/api/v1")

    app.dependency_overrides[current_user] = lambda: SimpleNamespace(subject="user-1")

    monkeypatch.setattr(
        dependencies_module,
        "get_service_factory",
        lambda: _FakeFactory(_FakeMessageService(entities)),
    )

    return app


async def _get(app: FastAPI) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/api/v1/sessions/{_SESSION_ID}/messages")


async def test_reasoning_content_is_on_the_wire(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    有思维链的那条带着它，没思维链的那条是 `null`（而不是缺字段）。

    ⚠️ "null 而不是缺字段"值得单列：少了它前端的两条读法
    （`msg.reasoning_content ?? content 里的推理块`）在类型上分不出
    "没有推理"与"这个后端版本还没有这个字段"。
    """

    app = _app(
        monkeypatch,
        [
            _entity(seq=1, reasoning=_THINKING),
            _entity(seq=2, reasoning=None),
        ],
    )

    response = await _get(app)

    assert response.status_code == 200

    messages = response.json()["messages"]

    assert messages[0]["reasoning_content"] == _THINKING
    assert messages[1]["reasoning_content"] is None

    # 字段名与 messages.yaml 的 `Message.reasoning_content` 逐字一致 ——
    # 这里断言的是**线上**那个名字，不是 DTO 的。
    assert "reasoning_content" in messages[0]

    # 正文块数组里不许再出现推理（两个容器不重叠）。这条在服务层已经
    # 验过，但漏到线上正是最难发现的那种，所以线上再看一眼。
    assert [block["type"] for block in messages[0]["content"]] == ["text"]
