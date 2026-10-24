"""
`GET /sessions` 与 `DELETE /sessions/{id}` 的**线上形状**。

与 `test_session_messages_response.py` 同一套装置：假工厂 + `current_user`
override，真打 ASGI。验的是路由 -> DTO -> 序列化那一段，不碰数据库。

这里额外断言两件只有走 HTTP 才能看见的事：

    GET 的 JSON 是 {"items": [...], "total": n}
    DELETE 的 204 真的**没有响应体**（不是 "null" / 空对象）
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
from app.common.exceptions import NotFoundError
from app.services.session.domain.schemas import (
    Session,
    SessionPage,
    SessionStatus,
)

_SESSION_ID = uuid4()


def _session(*, status: str = "active") -> Session:
    return Session(
        session_id=_SESSION_ID,
        agent_id="stock_research",
        status=status,
        user_id="user-1",
        updated_at=datetime.now(timezone.utc),
    )


class _FakeSessionService:
    """只实现路由用到的两个方法。"""

    def __init__(
            self,
            *,
            page: SessionPage | None = None,
            delete_error: Exception | None = None,
    ) -> None:
        self._page = page or SessionPage(items=[], total=0)
        self._delete_error = delete_error

        self.list_kwargs: dict[str, Any] | None = None
        self.delete_kwargs: dict[str, Any] | None = None

    async def list_sessions(self, **kwargs: Any) -> SessionPage:
        self.list_kwargs = kwargs
        return self._page

    async def delete_session(self, session_id: Any, **kwargs: Any) -> None:
        self.delete_kwargs = {"session_id": session_id, **kwargs}

        if self._delete_error is not None:
            raise self._delete_error


class _FakeFactory:
    """`needs_session=False` -> 走 create 分支，不 open 事务。"""

    def __init__(self, service: Any) -> None:
        self._service = service

    def needs_session(self, contract: Any) -> bool:
        return False

    def create(self, contract: Any) -> Any:
        return self._service


def _app(
        monkeypatch: pytest.MonkeyPatch,
        service: _FakeSessionService,
) -> FastAPI:

    app = FastAPI()
    app.include_router(sessions_router, prefix="/api/v1")

    app.dependency_overrides[current_user] = lambda: SimpleNamespace(
        subject="user-1",
    )

    monkeypatch.setattr(
        dependencies_module,
        "get_service_factory",
        lambda: _FakeFactory(service),
    )

    return app


async def _request(
        app: FastAPI,
        method: str,
        url: str,
) -> httpx.Response:

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
    ) as client:
        return await client.request(method, url)


async def test_list_response_shape_and_default_filters(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    默认 status=active；JSON 是 items/total；身份用 subject 原样传下去。
    """

    service = _FakeSessionService(
        page=SessionPage(items=[_session()], total=1),
    )
    app = _app(monkeypatch, service)

    response = await _request(app, "GET", "/api/v1/sessions")

    assert response.status_code == 200

    body = response.json()

    assert list(body) == ["items", "total"]
    assert body["total"] == 1
    assert body["items"][0]["session_id"] == str(_SESSION_ID)
    assert body["items"][0]["status"] == "active"

    # 服务层拿到的是 subject 字符串，不是 _owner_id 归一出来的 None ——
    # 后者会让查询变成 user_id IS NULL，把存量公共会话漏出来。
    assert service.list_kwargs is not None
    assert service.list_kwargs["user_id"] == "user-1"
    assert service.list_kwargs["status"] is SessionStatus.ACTIVE
    assert service.list_kwargs["page"] == 1
    assert service.list_kwargs["page_size"] == 20


async def test_list_passes_filters_through(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSessionService()
    app = _app(monkeypatch, service)

    response = await _request(
        app,
        "GET",
        (
            "/api/v1/sessions?status=archived&agent_id=stock_research"
            "&updated_after=2026-10-01T00%3A00%3A00Z"
            "&updated_before=2026-10-19T00%3A00%3A00Z"
            "&page=2&page_size=5"
        ),
    )

    assert response.status_code == 200
    assert service.list_kwargs is not None
    assert service.list_kwargs["status"] is SessionStatus.ARCHIVED
    assert service.list_kwargs["agent_id"] == "stock_research"
    assert service.list_kwargs["page"] == 2
    assert service.list_kwargs["page_size"] == 5
    assert service.list_kwargs["updated_after"] is not None
    assert service.list_kwargs["updated_before"] is not None


async def test_delete_is_204_with_empty_body(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSessionService()
    app = _app(monkeypatch, service)

    response = await _request(
        app, "DELETE", f"/api/v1/sessions/{_SESSION_ID}",
    )

    assert response.status_code == 204
    assert response.content == b""

    assert service.delete_kwargs is not None
    assert service.delete_kwargs["session_id"] == _SESSION_ID
    assert service.delete_kwargs["user_id"] == "user-1"


async def test_delete_not_found_maps_to_404(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSessionService(
        delete_error=NotFoundError("会话不存在"),
    )
    app = _app(monkeypatch, service)

    response = await _request(
        app, "DELETE", f"/api/v1/sessions/{_SESSION_ID}",
    )

    assert response.status_code == 404


def test_openapi_contains_new_operations(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _FakeSessionService())

    schema = app.openapi()

    assert "get" in schema["paths"]["/api/v1/sessions"]
    assert "delete" in schema["paths"]["/api/v1/sessions/{session_id}"]
    assert "SessionPage" in schema["components"]["schemas"]

    page = schema["components"]["schemas"]["SessionPage"]
    assert set(page["required"]) == {"items", "total"}
