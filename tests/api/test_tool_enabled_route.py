"""
`PATCH /tools/{tool_id}/enabled` 的线上形状。

与服务层单测分开：这里验的是路由 -> 请求体 -> DTO 序列化那一段
（与 test_skill_routes_response.py 同一套装置：假工厂 + 真打 ASGI）。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

import app.services.dependencies as dependencies_module
from app.api.v1.tools import router as tools_router
from app.services.tool.domain.schemas import ToolEnabledState

_TOOL_ID = uuid4()


class _FakeStateService:

    def __init__(self) -> None:
        self.calls: list[tuple[UUID, bool]] = []

    async def set_enabled(
            self,
            tool_id: UUID,
            enabled: bool,
            *,
            updated_by: str | None = None,
    ) -> ToolEnabledState:

        self.calls.append((tool_id, enabled))

        return ToolEnabledState(
            tool_key="document.list",
            enabled=enabled,
        )


class _FakeFactory:

    def __init__(self, service: Any) -> None:
        self._service = service

    def needs_session(self, contract: Any) -> bool:
        return False

    def create(self, contract: Any) -> Any:
        return self._service


def _app(
        monkeypatch: pytest.MonkeyPatch,
        service: _FakeStateService,
) -> FastAPI:

    app = FastAPI()
    app.include_router(tools_router, prefix="/api/v1")

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
        *,
        json: Any = None,
) -> httpx.Response:

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
    ) as client:
        return await client.request(method, url, json=json)


async def test_disable_roundtrip(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeStateService()
    app = _app(monkeypatch, service)

    response = await _request(
        app,
        "PATCH",
        f"/api/v1/tools/{_TOOL_ID}/enabled",
        json={"enabled": False},
    )

    assert response.status_code == 200
    assert response.json() == {
        "tool_key": "document.list",
        "enabled": False,
    }
    assert service.calls == [(_TOOL_ID, False)]


def test_openapi_exposes_enabled_operation_and_fields(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _FakeStateService())

    schema = app.openapi()

    operation = schema["paths"][
        "/api/v1/tools/{tool_id}/enabled"
    ]["patch"]

    assert "requestBody" in operation

    schemas = schema["components"]["schemas"]

    assert "ToolEnabledUpdate" in schemas
    assert "ToolEnabledState" in schemas

    # 列表与详情都要带运行期开关；manifest 不应该有它
    # （manifest 是代码侧事实，开关来自 tool_definition_state）。
    assert "enabled" in schemas["ToolSummary"]["properties"]
    assert "enabled" in schemas["ToolDefinitionDetail"]["properties"]
    assert "enabled" not in schemas["ToolManifest"]["properties"]
