"""
DeepSeek web search / fetch 工具的单测。

不访问真实 DeepSeek：HTTP 那一层由本地 aiohttp TestServer 或
`_request_*` 的假实现替换；配置、参数校验、响应归一化与注册表
发现各自独立验证。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from app.agent.tools import web_tool as module
from app.agent.tools.web_tool import WebSettings


def _settings(**overrides: Any) -> WebSettings:
    values = {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "default_count": 5,
        "max_count": 20,
        "max_content_chars": 4000,
        "max_fetch_chars": 12000,
    }
    values.update(overrides)

    return WebSettings(**values)


async def test_search_builds_payload_and_normalizes(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_request(settings, payload):
        captured["settings"] = settings
        captured["payload"] = payload

        return {
            "results": [
                {
                    "title": "DeepSeek 发布搜索 API",
                    "url": "https://example.com/a",
                    "summary": "摘要",
                    "date": "2026-10-20",
                    "rerank_score": 2.4,
                    "authority": 3,
                    "content": "x" * 200,
                }
            ]
        }

    monkeypatch.setattr(module, "_request_search", fake_request)
    monkeypatch.setattr(
        module, "get_web_settings",
        lambda: _settings(max_content_chars=100),
    )

    result = await module.search.coroutine(
        query="  DeepSeek 搜索 API  ",
        count=999,
        need_content=True,
        ori_question="怎么接搜索 API",
    )

    assert captured["payload"] == {
        "query": "DeepSeek 搜索 API",
        "count": 20,
        "ori_question": "怎么接搜索 API",
        "need_content": True,
    }

    assert result["query"] == "DeepSeek 搜索 API"
    assert result["count"] == 1

    item = result["results"][0]

    assert item["title"] == "DeepSeek 发布搜索 API"
    assert item["url"] == "https://example.com/a"
    assert item["rerank_score"] == 2.4
    assert item["content"] == "x" * 100 + "…"
    assert item["content_truncated"] is True


async def test_empty_search_results_are_not_an_error(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_request(settings, payload):
        return {"results": []}

    monkeypatch.setattr(module, "_request_search", fake_request)
    monkeypatch.setattr(module, "get_web_settings", _settings)

    result = await module.search.coroutine(query="没有结果的关键词")

    assert result == {
        "query": "没有结果的关键词",
        "count": 0,
        "results": [],
    }


async def test_search_query_validation() -> None:
    for bad in ("", "   ", "x" * 129):
        with pytest.raises(ValueError):
            await module.search.coroutine(query=bad)


async def test_search_ori_question_validation() -> None:
    with pytest.raises(ValueError):
        await module.search.coroutine(
            query="ok", ori_question="   "
        )


async def test_fetch_returns_title_and_truncated_content(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_request(settings, payload):
        captured["payload"] = payload

        return {
            "title": "页面标题",
            "content": "x" * 200,
        }

    monkeypatch.setattr(module, "_request_fetch", fake_request)
    monkeypatch.setattr(
        module, "get_web_settings",
        lambda: _settings(max_fetch_chars=100),
    )

    result = await module.fetch.coroutine(
        url="https://example.com/page"
    )

    assert captured["payload"] == {
        "url": "https://example.com/page"
    }
    assert result["url"] == "https://example.com/page"
    assert result["title"] == "页面标题"
    assert result["content"] == "x" * 100 + "…"
    assert result["content_truncated"] is True


async def test_fetch_rejects_invalid_urls() -> None:
    for bad in ("", "   ", "ftp://example.com/a", "example.com/a"):
        with pytest.raises(ValueError):
            await module.fetch.coroutine(url=bad)

    with pytest.raises(ValueError):
        await module.fetch.coroutine(
            url="https://example.com/" + "x" * 8200
        )


async def test_fetch_empty_content_is_an_error(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_request(settings, payload):
        return {"title": "空页面", "content": ""}

    monkeypatch.setattr(module, "_request_fetch", fake_request)
    monkeypatch.setattr(module, "get_web_settings", _settings)

    with pytest.raises(RuntimeError):
        await module.fetch.coroutine(
            url="https://example.com/empty"
        )


async def test_missing_api_key_fails_before_network() -> None:
    settings = _settings(api_key="")

    for call in (module._request_search, module._request_fetch):
        with pytest.raises(RuntimeError) as excinfo:
            await call(settings, {"query": "x"})

        assert "DEEPSEEK_API_KEY" in str(excinfo.value)


def test_origin_accepts_chat_and_site_base() -> None:
    assert (
        module._origin("https://api.deepseek.com/v1")
        == "https://api.deepseek.com"
    )
    assert (
        module._origin("https://api.deepseek.com/")
        == "https://api.deepseek.com"
    )
    assert (
        module._origin("")
        == "https://api.deepseek.com"
    )


def test_http_error_messages_are_actionable() -> None:
    assert "DEEPSEEK_API_KEY" in module._http_error(401, "")
    assert "限流" in module._http_error(429, "")
    assert "bad request" in module._http_error(400, "bad request")


def test_normalize_rejects_malformed_response() -> None:
    with pytest.raises(RuntimeError):
        module._normalize_results(
            {"results": "not-a-list"},
            need_content=False,
            max_content_chars=10,
        )


def test_settings_load_from_yaml(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "web.yaml"

    path.write_text(
        "web:\n"
        "  api_key: ${WEB_TEST_KEY}\n"
        "  max_count: 7\n"
        "  max_fetch_chars: 777\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_CONFIG_PATH", path)
    monkeypatch.setenv("WEB_TEST_KEY", "from-yaml")

    module.get_web_settings.cache_clear()

    try:
        settings = module.get_web_settings()

        assert settings.api_key == "from-yaml"
        assert settings.max_count == 7
        assert settings.max_fetch_chars == 777
    finally:
        module.get_web_settings.cache_clear()


async def test_request_search_http_contract() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["path"] = request.path
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = await request.json()

        return web.json_response(
            {
                "results": [
                    {"title": "t", "url": "https://example.com"}
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/v1/tools/search", handler)

    server = TestServer(app)
    await server.start_server()

    try:
        settings = _settings(
            base_url=str(server.make_url("/")).rstrip("/"),
            api_key="secret",
        )

        raw = await module._request_search(
            settings, {"query": "x", "count": 3}
        )
    finally:
        await server.close()

    assert captured["path"] == "/v1/tools/search"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"] == {"query": "x", "count": 3}
    assert raw["results"][0]["url"] == "https://example.com"


async def test_request_fetch_http_contract() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        captured["path"] = request.path
        captured["body"] = await request.json()

        return web.json_response(
            {"title": "t", "content": "# body"}
        )

    app = web.Application()
    app.router.add_post("/v1/tools/fetch", handler)

    server = TestServer(app)
    await server.start_server()

    try:
        settings = _settings(
            base_url=str(server.make_url("/")).rstrip("/"),
            api_key="secret",
        )

        raw = await module._request_fetch(
            settings, {"url": "https://example.com/a"}
        )
    finally:
        await server.close()

    assert captured["path"] == "/v1/tools/fetch"
    assert captured["body"] == {"url": "https://example.com/a"}
    assert raw == {"title": "t", "content": "# body"}


async def test_request_maps_auth_error() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=401, text="unauthorized")

    app = web.Application()
    app.router.add_post("/v1/tools/search", handler)

    server = TestServer(app)
    await server.start_server()

    try:
        with pytest.raises(RuntimeError) as excinfo:
            await module._request_search(
                _settings(
                    base_url=str(server.make_url("/")).rstrip("/")
                ),
                {"query": "x"},
            )
    finally:
        await server.close()

    assert "DEEPSEEK_API_KEY" in str(excinfo.value)


def test_registered_as_web_tools() -> None:
    from app.services.tool.registry import get_tool_registry

    specs = get_tool_registry().scan(
        packages=("app.agent.tools",)
    )

    keys = {spec.tool_key for spec in specs}

    assert "web.search" in keys
    assert "web.fetch" in keys
