"""
Skill 路由的**线上形状**。

与 test_sessions_list_delete_response.py 同一套装置：假工厂 + 真打
ASGI。验的是路由 -> DTO -> 序列化那一段，不碰数据库。

为什么不能只测服务层：服务返回的实体如果被直接当 DTO 用，单测全绿，
只有响应模型校验才会发现字段对不上（见 tools 模块那次
`response.0.id Field required` 的教训）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

import app.services.dependencies as dependencies_module
from app.api.v1.skills import router as skills_router
from app.common.exceptions import NotFoundError
from app.services.skill.domain.schemas import (
    CapabilityRef,
    PagedSkillResponse,
    SkillCreate,
    SkillDetail,
    SkillStatus,
    SkillSummary,
    SkillUpdate,
)

_SKILL_ID = uuid4()
_CAPABILITY_ID = uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ref() -> CapabilityRef:
    return CapabilityRef(
        id=_CAPABILITY_ID,
        code="announcement.search",
        name="公告检索",
    )


def _summary() -> SkillSummary:
    now = _now()

    return SkillSummary(
        id=_SKILL_ID,
        skill_key="announcement_search",
        name="上市公司公告检索",
        description="检索公告",
        tags=["金融", "公告检索"],
        status=SkillStatus.ACTIVE,
        created_at=now,
        updated_at=now,
        # 只读容量度量：响应里必须有，写请求里绝不能有。
        instructions_tokens=128,
        prompt_tokens=196,
        token_encoding="o200k_base",
        token_computed_at=now,
    )


def _detail() -> SkillDetail:
    summary = _summary()

    return SkillDetail(
        **summary.model_dump(),
        examples=["检索最近一个月的股权质押公告"],
        instructions="按公司名与时间范围检索公告。",
        capabilities=[_ref()],
    )


class _FakeSkillService:
    """只实现路由用到的七个方法。"""

    def __init__(self) -> None:
        self.list_kwargs: dict[str, Any] | None = None
        self.replace_kwargs: dict[str, Any] | None = None
        self.created_request: Any = None
        self.deleted: UUID | None = None
        self.get_error: Exception | None = None

    async def list_skills(self, **kwargs: Any) -> PagedSkillResponse:
        self.list_kwargs = kwargs

        return PagedSkillResponse(
            items=[_summary()],
            page=kwargs["page"],
            page_size=kwargs["page_size"],
            total=1,
        )

    async def create_skill(self, request: Any) -> SkillDetail:
        self.created_request = request
        return _detail()

    async def get_skill(self, skill_id: UUID) -> SkillDetail:
        if self.get_error is not None:
            raise self.get_error

        return _detail()

    async def update_skill(
            self, skill_id: UUID, request: Any,
    ) -> SkillDetail:
        return _detail()

    async def delete_skill(self, skill_id: UUID) -> None:
        self.deleted = skill_id

    async def list_skill_capabilities(
            self, skill_id: UUID,
    ) -> list[CapabilityRef]:
        return [_ref()]

    async def replace_skill_capabilities(
            self, skill_id: UUID, request: Any,
    ) -> list[CapabilityRef]:
        self.replace_kwargs = {
            "skill_id": skill_id,
            "capability_ids": request.capability_ids,
        }

        return [_ref()]


class _FakeFactory:
    """needs_session=False -> 走 create 分支，不 open 事务。"""

    def __init__(self, service: Any) -> None:
        self._service = service

    def needs_session(self, contract: Any) -> bool:
        return False

    def create(self, contract: Any) -> Any:
        return self._service


def _app(
        monkeypatch: pytest.MonkeyPatch,
        service: _FakeSkillService,
) -> FastAPI:

    app = FastAPI()
    app.include_router(skills_router, prefix="/api/v1")

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


async def test_list_shape_and_filters(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSkillService()
    app = _app(monkeypatch, service)

    response = await _request(
        app,
        "GET",
        "/api/v1/skills?status=ACTIVE&q=公告&page=2&page_size=5",
    )

    assert response.status_code == 200

    body = response.json()

    assert set(body) == {"items", "page", "page_size", "total"}
    assert body["page"] == 2
    assert body["page_size"] == 5
    assert body["total"] == 1

    item = body["items"][0]

    assert set(item) == {
        "id",
        "skill_key",
        "name",
        "description",
        "tags",
        "status",
        "created_at",
        "updated_at",
        "instructions_tokens",
        "prompt_tokens",
        "token_encoding",
        "token_computed_at",
    }
    assert item["id"] == str(_SKILL_ID)
    assert item["tags"] == ["金融", "公告检索"]
    assert item["status"] == "ACTIVE"
    assert isinstance(item["created_at"], str)

    # 容量是只读派生值：列表页也要带，且数字与口径成对出现。
    assert item["instructions_tokens"] == 128
    assert item["prompt_tokens"] == 196
    assert item["token_encoding"] == "o200k_base"
    assert isinstance(item["token_computed_at"], str)

    assert service.list_kwargs is not None
    assert service.list_kwargs["status"] is SkillStatus.ACTIVE
    assert service.list_kwargs["keyword"] == "公告"
    assert service.list_kwargs["page"] == 2
    assert service.list_kwargs["page_size"] == 5


async def test_detail_and_capability_shape(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSkillService()
    app = _app(monkeypatch, service)

    detail = await _request(
        app, "GET", f"/api/v1/skills/{_SKILL_ID}",
    )

    assert detail.status_code == 200

    body = detail.json()

    assert body["instructions"] == "按公司名与时间范围检索公告。"
    assert body["tags"] == ["金融", "公告检索"]
    assert body["examples"] == [
        "检索最近一个月的股权质押公告"
    ]
    assert isinstance(body["capabilities"], list)

    # SkillDetail 继承 SkillSummary，容量字段不能被 allOf 吞掉。
    assert body["instructions_tokens"] == 128
    assert body["prompt_tokens"] == 196
    assert body["prompt_tokens"] > body["instructions_tokens"]
    assert body["token_encoding"] == "o200k_base"

    capability = body["capabilities"][0]

    # CapabilityRef 只带识别绑定所需的三个字段；
    # 多带字段说明有人在组装层把完整实体直接丢出来了。
    assert set(capability) == {"id", "code", "name"}
    assert capability["id"] == str(_CAPABILITY_ID)

    bindings = await _request(
        app,
        "GET",
        f"/api/v1/skills/{_SKILL_ID}/capabilities",
    )

    assert bindings.status_code == 200
    assert bindings.json() == [
        {
            "id": str(_CAPABILITY_ID),
            "code": "announcement.search",
            "name": "公告检索",
        }
    ]


async def test_create_parses_a2a_metadata(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSkillService()
    app = _app(monkeypatch, service)

    response = await _request(
        app,
        "POST",
        "/api/v1/skills",
        json={
            "skill_key": "announcement_search",
            "name": "上市公司公告检索",
            "description": "检索公告",
            "instructions": "按公司名检索公告。",
            "tags": ["金融", "公告检索"],
            "examples": ["检索最近一个月的股权质押公告"],
        },
    )

    assert response.status_code == 201

    request = service.created_request

    assert request is not None
    assert request.tags == ["金融", "公告检索"]
    assert request.examples == [
        "检索最近一个月的股权质押公告"
    ]


def test_write_contracts_reject_capacity_fields() -> None:
    """
    容量字段是**只读**的：SkillCreate / SkillUpdate 是
    additionalProperties: false。前端如果把 GET 到的值回填进 PATCH，
    后端必须 422，而不是静默忽略或当真写库。
    """

    base = {
        "skill_key": "announcement_search",
        "name": "上市公司公告检索",
        "description": "检索公告",
        "instructions": "按公司名检索公告。",
    }

    for field in (
        "instructions_tokens",
        "prompt_tokens",
        "token_encoding",
        "token_computed_at",
    ):
        with pytest.raises(ValidationError):
            SkillCreate(**base, **{field: 1})

        with pytest.raises(ValidationError):
            SkillUpdate(**{field: 1})


async def test_replace_bindings_parses_uuids(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSkillService()
    app = _app(monkeypatch, service)

    response = await _request(
        app,
        "PUT",
        f"/api/v1/skills/{_SKILL_ID}/capabilities",
        json={"capability_ids": [str(_CAPABILITY_ID)]},
    )

    assert response.status_code == 200
    assert [c["code"] for c in response.json()] == [
        "announcement.search"
    ]

    assert service.replace_kwargs is not None
    assert service.replace_kwargs["skill_id"] == _SKILL_ID
    assert service.replace_kwargs["capability_ids"] == [
        _CAPABILITY_ID
    ]


async def test_delete_is_204_with_empty_body(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSkillService()
    app = _app(monkeypatch, service)

    response = await _request(
        app, "DELETE", f"/api/v1/skills/{_SKILL_ID}",
    )

    assert response.status_code == 204
    assert response.content == b""
    assert service.deleted == _SKILL_ID


async def test_not_found_maps_to_404(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _FakeSkillService()
    service.get_error = NotFoundError("Skill 不存在")

    app = _app(monkeypatch, service)

    response = await _request(
        app, "GET", f"/api/v1/skills/{_SKILL_ID}",
    )

    assert response.status_code == 404


def test_openapi_contains_operations_and_schemas(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, _FakeSkillService())

    schema = app.openapi()

    paths = schema["paths"]

    assert "get" in paths["/api/v1/skills"]
    assert "post" in paths["/api/v1/skills"]
    assert "patch" in paths["/api/v1/skills/{skill_id}"]
    assert "delete" in paths["/api/v1/skills/{skill_id}"]
    assert (
        "put" in paths["/api/v1/skills/{skill_id}/capabilities"]
    )

    schemas = schema["components"]["schemas"]

    assert "SkillDetail" in schemas
    assert "PagedSkillResponse" in schemas
    assert set(
        schemas["PagedSkillResponse"]["required"]
    ) == {"items", "page", "page_size", "total"}

    # 运行态 OpenAPI 也必须把容量字段列为 required —— 若 Pydantic
    # 给了 default，这里会静默少四项，与 skill.yaml 和前端 TS 漂移。
    assert {
        "instructions_tokens",
        "prompt_tokens",
        "token_encoding",
        "token_computed_at",
    } <= set(schemas["SkillSummary"]["required"])
