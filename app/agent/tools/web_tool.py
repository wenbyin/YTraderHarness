"""
Web 工具：接 DeepSeek 的 Search & Fetch API。

端点（2026-10 DeepSeek 官方 API）
--------------------------------
    POST https://api.deepseek.com/v1/tools/search
    POST https://api.deepseek.com/v1/tools/fetch
    Authorization: Bearer <DEEPSEEK_API_KEY>

Search 请求：`query`、`count`、`ori_question`、`need_content`。
Search 响应：`{"results": [{title, url, summary, content?, date,
rerank_score, authority}, ...]}`；无命中时 results 是空数组，
**不是错误**。

Fetch 请求：`{"url": "https://..."}`（HTTP/HTTPS，最长 8192）。
Fetch 响应：`{title, content}`，content 是 Markdown。

⚠️ 不要把 llm.yaml 的 base_url 直接拿来拼
--------------------------------------
llm.yaml 的 deepseek base_url 是 `https://api.deepseek.com/v1`（chat
端点），而 Search / Fetch 的路径挂在**站点根**上。`_origin()` 会还原
origin，所以配置写站点根或带 `/v1` 都行。

⚠️ 用 aiohttp，不用 httpx
-------------------------
httpx 默认读系统代理（Windows 注册表里常有 127.0.0.1:7890），会把出站
请求塞进代理。本仓在 ES / Casdoor / rerank 三处都踩过这个坑。aiohttp
不读系统代理；这里也显式 `trust_env=False`。

⚠️ 只读
-------
search 只搜索；fetch 只读取给定 URL 的正文。两者都不写任何本地状态。
Fetch 不能替代 RAG：内部文档仍走 `document.retrieve_documents`。

⚠️ 工具 key
-----------
模块在 `TOOL_SOURCE_PACKAGES` 的 `app.agent.tools` 下。`TOOL_SOURCE_REF`
显式写成 `"web"`，因此 tool_key 是 `web.search` / `web.fetch`。
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.common.utils.env import expand_env

__all__ = [
    "WebSettings",
    "fetch",
    "get_web_settings",
    "search",
]

logger = logging.getLogger(__name__)

#: 显式来源；不写会按模块路径推导成 `agent`，tool_key 就不是 `web.*`。
TOOL_SOURCE_REF = "web"

#: 配置文件（真文件 gitignore；模板入库）。
_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "web.yaml"
)

#: DeepSeek 端点路径。
_SEARCH_PATH = "/v1/tools/search"
_FETCH_PATH = "/v1/tools/fetch"

#: DeepSeek 对 query / ori_question 的长度上限。
_MAX_QUERY_CHARS = 128

#: Fetch 的 URL 长度上限。
_MAX_URL_CHARS = 8192

#: DeepSeek 的 count 上限。
_API_MAX_COUNT = 50


class WebSettings(BaseModel):
    """`web.yaml` 的形状。"""

    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    timeout_ms: int = Field(default=20_000, ge=1, le=120_000)
    max_retries: int = Field(default=1, ge=0, le=3)
    default_count: int = Field(default=5, ge=1, le=_API_MAX_COUNT)
    max_count: int = Field(default=20, ge=1, le=_API_MAX_COUNT)
    max_content_chars: int = Field(
        default=4000, ge=100, le=100_000
    )
    max_fetch_chars: int = Field(
        default=12_000, ge=100, le=200_000
    )


@lru_cache(maxsize=1)
def get_web_settings() -> WebSettings:
    """
    读 `app/config/web.yaml`；缺的项从 llm.yaml 的 deepseek provider
    兜底，最后回落 `DEEPSEEK_API_KEY`。

    进程内缓存；测试改配置时调 `get_web_settings.cache_clear()`。
    """

    defaults = WebSettings()
    fallback = _from_llm_config()

    if not _CONFIG_PATH.exists():
        merged = {**defaults.model_dump(), **fallback}
        return WebSettings.model_validate(merged)

    raw = yaml.safe_load(
        _CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}

    section = expand_env(raw.get("web") or {})

    # 逐项兜底：只配了 api_key 的人不该因此丢掉端点；空的 section
    # 值也不该把 llm.yaml 的 key 覆盖成空串。
    merged = defaults.model_dump()

    for key, value in fallback.items():
        if value:
            merged[key] = value

    for key, value in section.items():
        if value is not None and value != "":
            merged[key] = value

    settings = WebSettings.model_validate(merged)

    if not settings.api_key.strip() and fallback.get("api_key"):
        settings = settings.model_copy(
            update={"api_key": fallback["api_key"]}
        )

    return settings


def _from_llm_config() -> dict[str, str]:
    """
    从 llm.yaml 的 deepseek provider 取 key 与站点根。

    读不到就返回空 dict —— 真正的"没凭据"错误推迟到工具调用时，
    不在 import / 装配期炸（与 rerank_api.py 同一立场）。
    """

    try:
        from app.common.llm import load_llm_config

        provider = load_llm_config().providers.get("deepseek")

    except Exception as exc:  # noqa: BLE001 - 兜底而已

        logger.debug(
            "读 llm.yaml 的 deepseek provider 失败（%s: %s），"
            "web 工具将没有默认端点/凭据。",
            type(exc).__name__,
            exc,
        )

        return {}

    if provider is None:
        return {}

    return {
        "base_url": _origin(provider.base_url),
        "api_key": provider.api_key,
    }


def _origin(base_url: str) -> str:
    """`https://api.deepseek.com/v1` -> `https://api.deepseek.com`。"""

    trimmed = (base_url or "").strip()

    if not trimmed:
        return "https://api.deepseek.com"

    parts = urlsplit(trimmed)

    if not parts.scheme or not parts.netloc:
        return trimmed.rstrip("/")

    return f"{parts.scheme}://{parts.netloc}"


@tool
async def search(
        query: str,
        count: int = 5,
        need_content: bool = False,
        ori_question: str | None = None,
) -> dict[str, Any]:
    """
    搜索公开网页，返回标题、URL、摘要、日期、相关性分数与权威度。

    用于外部、时效性事实（新闻、公告、价格、版本等）；内部文档请用
    RAG 检索工具，不要用本工具替代内部知识库。

    支持 DeepSeek 搜索操作符：`site:`、`-site:`、`"精确短语"`、
    `lang:`、`after:` / `before:`。`count` 取值 1~20；需要网页正文时
    传 `need_content=true`（正文会按配置截断）。

    ⚠️ 需要配置 DEEPSEEK_API_KEY；没有凭据时本工具会返回错误，
    不要改用记忆或猜测来补外部事实。
    """

    settings = get_web_settings()

    q = str(query or "").strip()

    if not q or len(q) > _MAX_QUERY_CHARS:
        raise ValueError(
            f"query 必须是 1~{_MAX_QUERY_CHARS} 个字符的非空字符串。"
        )

    upper = min(settings.max_count, _API_MAX_COUNT)
    n = max(1, min(int(count), upper))

    payload: dict[str, Any] = {
        "query": q,
        "count": n,
    }

    if ori_question is not None:
        oq = str(ori_question).strip()

        if not oq or len(oq) > _MAX_QUERY_CHARS:
            raise ValueError(
                f"ori_question 必须是 1~{_MAX_QUERY_CHARS} 个字符。"
            )

        payload["ori_question"] = oq

    if need_content:
        payload["need_content"] = True

    raw = await _request_search(settings, payload)

    results = _normalize_results(
        raw,
        need_content=need_content,
        max_content_chars=settings.max_content_chars,
    )

    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@tool
async def fetch(url: str) -> dict[str, Any]:
    """
    抓取一个公开网页，返回标题与 Markdown 正文。

    只在搜索结果摘要不够时使用；需要先定位 URL 时先用网页搜索。
    仅支持 HTTP/HTTPS 公网地址。正文按配置截断，返回值里会标明是否
    截断。内部文档请用 RAG 检索工具，不要用本工具读取内部资料。

    ⚠️ 部分站点会被上游 fetch 端点拒绝（实测 Wikipedia 返回 404）。
    遇到这种情况改用 web_search 获取摘要，不要反复重试同一个 URL。
    """

    settings = get_web_settings()

    target = str(url or "").strip()

    if not target or len(target) > _MAX_URL_CHARS:
        raise ValueError(
            f"url 必须是 1~{_MAX_URL_CHARS} 个字符的非空字符串。"
        )

    parts = urlsplit(target)

    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(
            "url 必须是合法的 http/https 公网地址。"
        )

    raw = await _request_fetch(settings, {"url": target})

    title = raw.get("title")
    content = raw.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "DeepSeek web fetch 没有返回正文（content 为空或类型不对）。"
        )

    truncated = len(content) > settings.max_fetch_chars

    if truncated:
        content = content[: settings.max_fetch_chars] + "…"

    return {
        "url": target,
        "title": title if isinstance(title, str) else None,
        "content": content,
        "content_truncated": truncated,
    }


async def _request_search(
        settings: WebSettings,
        payload: dict[str, Any],
) -> dict[str, Any]:
    return await _post_json(
        settings,
        _SEARCH_PATH,
        payload,
        operation="search",
    )


async def _request_fetch(
        settings: WebSettings,
        payload: dict[str, Any],
) -> dict[str, Any]:
    return await _post_json(
        settings,
        _FETCH_PATH,
        payload,
        operation="fetch",
    )


async def _post_json(
        settings: WebSettings,
        path: str,
        payload: dict[str, Any],
        *,
        operation: str,
) -> dict[str, Any]:
    """发一次 Search / Fetch 请求；429 / 5xx / 网络错误按配置重试。"""

    api_key = settings.api_key.strip()

    if not api_key:
        raise RuntimeError(
            "DeepSeek web API 没有凭据：请设置 DEEPSEEK_API_KEY，"
            "或在 app/config/web.yaml 里配置 api_key。"
        )

    url = (
        _origin(settings.base_url).rstrip("/")
        + path
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(
        total=settings.timeout_ms / 1000.0
    )

    last_error = ""

    #: 这次操作的目标（fetch 的 URL / search 的 query），只用于错误文案。
    #: 上游 404 只说 "failed to fetch URL"，不带给的是哪个 URL，排查时
    #: 还得回去翻 tool_call 参数。
    target = str(
        payload.get("url") or payload.get("query") or ""
    ).strip()

    # trust_env=False：aiohttp 默认不读系统代理，这里显式写出来，
    # 避免将来有人改成 True。
    async with aiohttp.ClientSession(
            timeout=timeout,
            trust_env=False,
    ) as session:

        for attempt in range(settings.max_retries + 1):

            try:
                async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                ) as response:

                    body = await response.text()

                    if response.status == 200:
                        return _parse_json(body)

                    message = _http_error(
                        response.status,
                        body,
                        operation,
                        target=target,
                    )

                    if (
                            response.status == 429
                            or response.status >= 500
                    ):
                        last_error = message

                        if attempt < settings.max_retries:
                            await asyncio.sleep(
                                0.5 * (2 ** attempt)
                            )
                            continue

                    raise RuntimeError(message)

            except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
            ) as exc:

                last_error = (
                    f"DeepSeek web {operation} 请求失败"
                    f"（target={target[:200]}）："
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt < settings.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue

                raise RuntimeError(last_error) from exc

    raise RuntimeError(
        last_error or f"DeepSeek web {operation} 请求失败。"
    )


def _parse_json(body: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "DeepSeek web API 返回的不是合法 JSON。"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            "DeepSeek web API 响应顶层不是对象。"
        )

    return value


def _http_error(
        status: int,
        body: str,
        operation: str = "search",
        *,
        target: str = "",
) -> str:
    detail = (body or "").strip().replace("\n", " ")[:300]

    suffix = f"（target={target[:200]}）" if target else ""

    if status in (401, 403):
        return (
            f"DeepSeek web {operation} 鉴权失败（HTTP {status}）"
            f"{suffix}：检查 DEEPSEEK_API_KEY。"
        )

    if status == 429:
        return (
            f"DeepSeek web {operation} 被限流（HTTP 429，5 QPS）"
            f"{suffix}：稍后重试，或降低调用频率。"
        )

    if operation == "fetch" and status == 404:
        return (
            f"DeepSeek web fetch 失败（HTTP 404）{suffix}："
            f"{detail or '（空响应体）'}。"
            "该 URL 可能不被 fetch 端点支持（实测 Wikipedia 会 404）；"
            "改用 web_search 获取摘要，不要反复重试同一个 URL。"
        )

    return (
        f"DeepSeek web {operation} 失败（HTTP {status}）{suffix}："
        f"{detail or '（空响应体）'}"
    )


def _normalize_results(
        raw: dict[str, Any],
        *,
        need_content: bool,
        max_content_chars: int,
) -> list[dict[str, Any]]:
    """
    原始响应 -> 稳定、可 JSON 化的结果列表。

    字段缺失时保留 None，而不是丢键：调用方（模型）能看到"这个字段
    没返回"，而不是把"没有"误当成"空字符串"。
    """

    items = raw.get("results")

    if not isinstance(items, list):
        raise RuntimeError(
            "DeepSeek web search 响应里没有 results 数组"
            f"（实际是 {type(items).__name__}）。"
        )

    results: list[dict[str, Any]] = []

    for item in items:

        if not isinstance(item, dict):
            continue

        if not item.get("url") and not item.get("title"):
            continue

        result: dict[str, Any] = {
            key: item.get(key)
            for key in (
                "title",
                "url",
                "summary",
                "date",
                "rerank_score",
                "authority",
            )
        }

        if need_content:
            content = item.get("content")

            if isinstance(content, str) and (
                    len(content) > max_content_chars
            ):
                content = content[:max_content_chars] + "…"
                result["content_truncated"] = True

            result["content"] = content

        results.append(result)

    return results
