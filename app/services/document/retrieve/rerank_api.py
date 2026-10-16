"""
第三方 HTTP 精排（`RerankerType.API`）。

接的是阿里云的精排服务（默认 `qwen3.7-text-rerank`），端点与凭据全在
`app/config/rerank.yaml` 里 —— 请求里只带 model id。

为什么另开一个文件而不是塞进 rerank.py
--------------------------------------
`rerank.py` 顶上挂着 `FlagEmbedding` / torch 那条 import 链（本地
cross-encoder）。把 HTTP 实现放同一个文件，会让"只想发一次 HTTP"也把
torch 拖进内存。注册表是**按模块**反射的（registry.py 的
`import_package_modules(recursive=False)`），多一个兄弟文件不花钱。

⚠️ 用 aiohttp，**不要用 httpx**
------------------------------
httpx 的 `trust_env` 默认为 True，会去读**环境变量与 Windows 注册表**里
的系统代理。装了 Clash 之类工具的机器上注册表里有 127.0.0.1:7890，
于是出站请求会被塞进那个代理 —— 本仓库已经在 ES / Casdoor / ES 索引
三处记录过这个坑（见 infra/datasource/elasticsearch.py 的注释）。
aiohttp 不读系统代理。

端到端形状（本机实测，2026-10-16）
--------------------------------
    端点    dashscope（默认）{站点根}/api/v1/services/rerank/text-rerank/text-rerank
            ✅ 实测 200；maas host 与 dashscope.aliyuncs.com 都可以
            openai  {base_url}{path}
            ❌ 实测 404（/rerank 与 /reranks 都试过），保留它只为
               将来别的服务商
    认证    Authorization: Bearer <api_key>
    请求    dashscope {model, input: {query, documents}, parameters: {top_n, return_documents}}
    响应    dashscope {"output": {"results": [{"index", "relevance_score"}]}, "usage": {...}}

⚠️ `base_url` 可能带 `/compatible-mode/v1`（llm.yaml 里 qwen 的 chat
端点就是它），而原生路径挂在**站点根**上 —— 直接拼会 404 而且 404 不带
任何提示。`_origin()` 负责还原。

两个 parser 的容错都写成"取不到分数就当作整批失败"（降级），而不是把
0 分当成"不相关"—— 后者会让一次接口改版表现为"精排把结果排坏了"。
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from pydantic import BaseModel, Field

from app.common.utils.env import expand_env

from ..domain.schemas import RerankerConfig
from ..domain.schemas.enum import RerankerType
from .base import Reranker, RetrieveCandidate, normalize_scores

__all__ = [
    "ApiReranker",
    "ApiRerankerSettings",
    "get_reranker_settings",
]

logger = logging.getLogger(__name__)


#: 配置文件路径（与 llm.yaml 同一层，真文件 gitignore、模板入库）。
_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "rerank.yaml"
)

#: dashscope 协议下的固定路径（原生 text-rerank 端点）。
_DASHSCOPE_PATH = (
    "/api/v1/services/rerank/text-rerank/text-rerank"
)

#: 每段候选送进去时的截断长度。
#:
#: 与 `rerank.py` 的 `_MAX_CHARS` 是同一个量、同一个理由：精排只看这段
#: 正文与 query 的相关性，送全文只是多付 token。这份语料的 chunk 多在
#: 500 字上下，512 基本不截断；真超了也只截掉尾巴。
_MAX_CHARS = 512


class ApiRerankerSettings(BaseModel):
    """`rerank.yaml` 的形状。"""

    base_url: str
    api_key: str
    protocol: str = "openai"
    path: str = "/rerank"
    model: str | None = None
    timeout_ms: int = Field(default=20_000, ge=1, le=120_000)
    max_retries: int = Field(default=1, ge=0, le=5)
    max_documents: int = Field(default=50, ge=1, le=1000)


@lru_cache(maxsize=1)
def get_reranker_settings() -> ApiRerankerSettings:
    """
    读 `app/config/rerank.yaml`；缺的项从 `llm.yaml` 的 qwen provider 兜。

    进程内缓存：它在每次精排调用里被读一次，而文件不会中途变。
    测试要改配置时调 `get_reranker_settings.cache_clear()`。

    ⚠️ 文件不存在时**不抛**，而是走"从 llm.yaml 兜"这条路 —— 抛的话，
    任何没配 rerank.yaml 的环境（CI、别人的机器）里，只要有一轮阶梯配了
    `api` 精排就会在装配期炸；而正确的表现是"这次精排降级"（与 rerank /
    改写的一贯口径一致：能力缺口不该让检索失败）。
    """

    defaults = _defaults()

    if not _CONFIG_PATH.exists():

        logger.info(
            "没有 %s —— 第三方精排用 llm.yaml 的 qwen provider 兜底"
            "（端点与凭据同源）。要另外配就复制 rerank.yaml.template。",
            _CONFIG_PATH.name,
        )

        merged = {
            **defaults.model_dump(),
            **{
                key: value
                for key, value in _from_llm_config().items()
                if value
            },
        }

        return ApiRerankerSettings.model_validate(merged)

    raw = yaml.safe_load(
        _CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}

    section = expand_env(raw.get("rerank") or {})

    # 逐项兜底而不是整体替换：只配了 model 的人不该因此丢掉端点和 key。
    fallback = {
        key: value
        for key, value in _from_llm_config().items()
        if value and not section.get(key)
    }

    merged = {**defaults.model_dump(), **fallback, **section}

    return ApiRerankerSettings.model_validate(merged)


def _from_llm_config() -> dict[str, str]:
    """
    从 `llm.yaml` 的 qwen provider 取端点与凭据。

    用户口径是"用现有 QWEN_API_KEY" —— 而那个 key 的实际值就住在
    llm.yaml 里（`${QWEN_API_KEY:-sk-...}`，该文件 gitignore）。让这里
    直接复用它，比让运维把同一个 key 抄进第二个配置文件好：
    抄一份就有两处会漂，而漂的表现是"精排突然 401"。

    读不到（文件缺失 / 没有 qwen）就返回空 dict —— 调用方会走到
    "没凭据"那条路，最终表现为降级而不是装配期崩溃。
    """

    try:
        from app.common.llm import load_llm_config

        provider = load_llm_config().providers.get("qwen")

    except Exception as exc:  # noqa: BLE001 —— 兜底而已，不该炸

        logger.debug(
            "读 llm.yaml 的 qwen provider 失败（%s: %s），"
            "第三方精排将没有默认端点/凭据。",
            type(exc).__name__,
            exc,
        )

        return {}

    if provider is None:
        return {}

    return {
        "base_url": provider.base_url,
        "api_key": provider.api_key,
    }


def _defaults() -> ApiRerankerSettings:
    """内置默认值（与 rerank.yaml.template 一致）。"""

    return ApiRerankerSettings(
        base_url="https://maas.qianwenaiapi.com/compatible-mode/v1",
        api_key="",
        # 实测：这台 host 上 OpenAI 兼容的 `/rerank` 与 `/reranks` 都是
        # 404，而原生的 text-rerank 路径返回 200（响应形状见
        # _parse_response）。所以默认走原生协议。
        protocol="dashscope",
        path="/rerank",
        model="qwen3.7-text-rerank",
    )


def _origin(base_url: str) -> str:
    """
    把 OpenAI 兼容端点还原成**站点根**。

    `llm.yaml` 里 qwen provider 的 base_url 是
    `https://maas.qianwenaiapi.com/compatible-mode/v1` —— 那是 chat 的
    端点，而 rerank 的原生路径挂在站点根上。不还原就会拼成
    `.../compatible-mode/v1/api/v1/services/...`，实测返回 404（那个
    404 不带任何提示，看起来像"服务没开"）。

    传入的若本来就是站点根（没有这些后缀），原样返回。
    """

    trimmed = base_url.rstrip("/")

    for suffix in ("/compatible-mode/v1", "/compatible-mode", "/v1"):

        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]

    return trimmed


class ApiReranker(Reranker):
    """
    调第三方 HTTP 精排服务重排候选。

    失败一律**降级**：保持融合顺序、原样返回，并把原因写进 `degraded`
    出参（调用方据此把这一段标成 executed=False）。不抛异常 ——
    精排挂掉不该让一次检索失败，但也不能静默（否则"精排没跑"与
    "精排跑了但没改善排序"在数据上完全一样）。
    """

    reranker_types = (RerankerType.API,)

    async def rerank(
            self,
            *,
            query: str,
            candidates: list[RetrieveCandidate],
            top_n: int,
            config: RerankerConfig,
            degraded: dict[str, str] | None = None,
    ) -> list[RetrieveCandidate]:

        if not candidates:
            return []

        settings = get_reranker_settings()

        window = min(top_n, settings.max_documents)

        if window < min(top_n, len(candidates)):
            # 送不满就少送几段（并在 trace 里留下痕迹）：整批被服务端拒
            # 掉比"少排几段"严重得多。
            logger.warning(
                "精排窗口被服务端上限收窄：要 %d 段、只送 %d 段"
                "（rerank.yaml 的 max_documents=%d）。",
                top_n,
                window,
                settings.max_documents,
            )

        selected = candidates[:window]
        rest = candidates[window:]

        try:
            scored = await self._score_remote(
                query=query,
                candidates=selected,
                settings=settings,
                model=config.model or settings.model,
            )

        except Exception as exc:  # noqa: BLE001 —— 契约：失败即降级

            logger.warning(
                "第三方精排不可用（%s: %s），本轮按融合顺序返回。",
                type(exc).__name__,
                exc,
            )

            if degraded is not None:
                degraded["reason"] = f"{type(exc).__name__}: {exc}"

            return list(candidates)

        # 服务端只回它打过分的那几段；**缺的那些**接在后面（不丢）。
        ranked = [
            candidate.normalized(score)
            for candidate, score in scored
        ]

        ranked_ids = {
            candidate.document_chunk_id for candidate in ranked
        }

        tail = [
            candidate
            for candidate in selected
            if candidate.document_chunk_id not in ranked_ids
        ]

        # 与本地那条路完全一致的收尾：归一化到 [0,1]，**不截断**
        # （最终条数归 cutter.top_k）。
        return normalize_scores(ranked + tail + list(rest))

    # ========================================================
    # HTTP
    # ========================================================

    async def _score_remote(
            self,
            *,
            query: str,
            candidates: list[RetrieveCandidate],
            settings: ApiRerankerSettings,
            model: str | None,
    ) -> list[tuple[RetrieveCandidate, float]]:
        """
        打分并返回 (候选, 分数) 列表，已按分数降序。

        这里**必须**把顺序与分数对回去（按响应里的 `index`）—— 服务端
        返回的是"第几段得了多少分"，而候选里可能有内容重复的块。
        """

        if not model:
            raise ValueError(
                "没给精排模型：rerank.yaml 的 model 与请求里的 "
                "RerankerConfig.model 都是空的。"
            )

        documents = [
            (candidate.content or "")[:_MAX_CHARS]
            for candidate in candidates
        ]

        url, headers, payload = self._build_request(
            query=query,
            documents=documents,
            model=model,
            top_n=len(candidates),
            settings=settings,
        )

        raw = await self._post(url=url, headers=headers, payload=payload, settings=settings)

        scored = self._parse_response(raw, candidates)

        scored.sort(key=lambda pair: pair[1], reverse=True)

        return scored

    def _build_request(
            self,
            *,
            query: str,
            documents: list[str],
            model: str,
            top_n: int,
            settings: ApiRerankerSettings,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:

        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }

        if settings.protocol == "dashscope":

            url = f"{_origin(settings.base_url)}{_DASHSCOPE_PATH}"

            payload = {
                "model": model,
                "input": {"query": query, "documents": documents},
                "parameters": {
                    "top_n": top_n,
                    # 不要回带正文：正文我们手上就有，回带只是多付一次
                    # 传输（与本地精排的 _MAX_CHARS 是同一个考虑）。
                    "return_documents": False,
                },
            }

            return url, headers, payload

        base = settings.base_url.rstrip("/")

        url = f"{base}{settings.path}"

        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        return url, headers, payload

    async def _post(
            self,
            *,
            url: str,
            headers: dict[str, str],
            payload: dict[str, Any],
            settings: ApiRerankerSettings,
    ) -> dict[str, Any]:
        """
        POST 一次，5xx / 超时重试 `max_retries` 次。

        4xx **不重试**：那是我方请求写错了（模型名、文档数、鉴权），
        重试只会把同一个错误付两遍钱。它的信息足够定位，直接抛。
        """

        timeout = aiohttp.ClientTimeout(
            total=settings.timeout_ms / 1000
        )

        last: Exception | None = None

        for attempt in range(settings.max_retries + 1):

            try:
                async with aiohttp.ClientSession(
                    timeout=timeout,
                ) as session:

                    async with session.post(
                        url, headers=headers, json=payload,
                    ) as response:

                        text = await response.text()

                        if response.status >= 500:
                            raise RuntimeError(
                                f"精排服务 {response.status}：{text[:200]}"
                            )

                        if response.status >= 400:
                            raise ValueError(
                                f"精排请求被拒 {response.status}："
                                f"{text[:200]}"
                            )

                        return json.loads(text)

            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:

                last = exc

                if attempt >= settings.max_retries:
                    break

                logger.warning(
                    "精排第 %d 次失败（%s: %s），重试。",
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )

        raise last if last is not None else RuntimeError("精排未知失败")

    def _parse_response(
            self,
            raw: dict[str, Any],
            candidates: list[RetrieveCandidate],
    ) -> list[tuple[RetrieveCandidate, float]]:
        """
        从回包里取出 (候选, 分数)。

        ⚠️ 取不到 `results` 一律**抛**（= 降级），不要静默当 0 分：
        接口改版时，静默的表现是"精排把最相关的排到了最后"，读起来像
        检索质量问题，而真实原因是这一层已经认不出回包了。
        """

        results: Any

        if isinstance(raw.get("output"), dict):

            results = raw["output"].get("results")

        else:

            results = raw.get("data") or raw.get("results")

        if not isinstance(results, list):
            raise ValueError(
                "精排回包里找不到 results 列表"
                f"（顶层键：{sorted(raw)[:8]}）。"
                "形状变了要改 _parse_response 与 rerank.yaml 的协议。"
            )

        scored: list[tuple[RetrieveCandidate, float]] = []

        for item in results:

            if not isinstance(item, dict):
                continue

            index = item.get("index")
            score = item.get(
                "relevance_score", item.get("score")
            )

            if not isinstance(index, int) or not isinstance(score, (int, float)):
                raise ValueError(
                    "精排结果缺少 index / relevance_score："
                    f"{str(item)[:120]}"
                )

            if not 0 <= index < len(candidates):
                raise ValueError(
                    f"精排回包的 index={index} 超出候选范围"
                    f"（共 {len(candidates)} 段）。"
                )

            scored.append((candidates[index], float(score)))

        if not scored and candidates:
            raise ValueError("精排回包的 results 是空的。")

        return scored
