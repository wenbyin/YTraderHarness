"""
工具结果缓存。

两种挂法都支持，因为它们解决的问题不同
--------------------------------------
    AROUND 一条          查缓存 + 命中短路 + 未命中回填，一个实现搞定
    BEFORE + AFTER 两条  拆成两半：BEFORE 查（命中即短路），
                         AFTER 回填

为什么两种都要支持：AROUND 更省事，但如果链上已经有别的 AROUND
（比如重试），把缓存也做成 AROUND 会让它们互相嵌套 ——
"缓存的是哪一次尝试的结果"就变成一个需要想的问题。拆成
BEFORE / AFTER 时缓存落在调用链的固定位置，与重试无关。

⚠️ 挂 BEFORE+AFTER 两条时，两边的 config 是**各自一份**
------------------------------------------------------
所以 `ttl_seconds` 在两半上都写一份。键（key_fields）也必须一致，
否则会出现"存进去的键与查的键不是一个"——表现是命中率恒为 0，
且没有任何报错。

⚠️ 凭什么认定"同一个调用"
------------------------
键 = 工具键 + 选定参数的规范化 JSON 的摘要。因此：

    参数不同        不命中（对）
    参数相同、链不同  命中（对：缓存的是**工具的结果**）
    参数含不可序列化对象  用 repr 兜底（可能不稳定，见 _cache_key）

**不含**租户 / 用户 / 会话。这是刻意的：判据由调用方通过
`key_fields` 决定 —— 需要按租户隔离时就把租户字段列进去。
默认取全部参数，是因为"少算了隔离维度"比"多算了"危险得多。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from ..base import ToolInterceptor
from ..context import (
    AsyncToolCall,
    ToolInvocation,
    ToolOutcome,
)
from ..runtime.cache import get_tool_cache
from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

__all__ = [
    "CacheConfig",
    "CacheToolProcessor",
]

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """缓存参数。title / description 是对外契约（前端表单）。"""

    ttl_seconds: int = Field(
        default=300,
        ge=0,
        le=86_400,
        title="有效期",
        description=(
            "缓存多少秒。0 表示**不缓存**（等于把这个处理器关掉，"
            "但保留在链上）。超过一天的值不建议：工具结果多半与"
            "实时数据有关，缓存一整天的东西没人敢用。"
        ),
    )

    key_fields: list[str] | None = Field(
        default=None,
        title="参与判重的参数",
        description=(
            "哪些参数决定「是不是同一个调用」。留空 = **全部参数**。"
            "需要按租户/用户隔离时，把它们列进来 —— "
            "不隔离会让 A 用户拿到 B 用户的结果。"
        ),
    )


class CacheToolProcessor(ToolInterceptor):
    """
    命中即短路返回缓存值；未命中则调用并在成功后回填。

    只缓存**成功**的结果：把失败也缓存起来会让一次偶发故障在
    TTL 内变成持续故障，而调用方看到的是"这个工具一直坏着"。
    真要抑制故障时的重复调用，那是**熔断**（另一个处理器），
    不是缓存 —— 两者的失效条件不一样，混在一起会互相掩盖。
    """

    name = "cache_interceptor"

    display_name = "结果缓存"

    description = (
        "缓存工具调用结果，命中时直接返回、不执行工具。"
    )

    config_model: ClassVar[type[BaseModel]] = CacheConfig

    #: BEFORE 短路、AFTER 回填、AROUND 两者合一 —— 三种都支持。
    #: 但**只有 SYNC**：一个不被 await 的 BEFORE 无法短路（它的
    #: 返回值没人等），那会让"配了缓存却没生效"变成一个静默事实。
    supported_hooks: ClassVar[frozenset[ProcessorHook]] = (
        frozenset({
            ProcessorHook.BEFORE,
            ProcessorHook.AFTER,
            ProcessorHook.AROUND,
        })
    )

    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset({ExecutionMode.SYNC})

    # ========================================================
    # BEFORE：查
    # ========================================================

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation | ToolOutcome:

        parsed = self.config_model.model_validate(config)

        key = self._cache_key(invocation, parsed)

        hit, value = get_tool_cache().get(key)

        if not hit:
            return invocation

        logger.debug(
            "%s：%s 命中缓存。", self.get_key(), invocation.tool_key
        )

        # cached=True 是必须的：观测侧靠它区分"真的执行了一次"与
        # "一次命中"（见 context.py 的说明）。
        return ToolOutcome(value=value, cached=True)

    # ========================================================
    # AFTER：回填
    # ========================================================

    async def after(
            self,
            invocation: ToolInvocation,
            outcome: ToolOutcome,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:

        if outcome.error is not None or outcome.cached:
            # 失败不缓存（理由见类 docstring）；已经来自缓存的结果
            # 不需要再存一遍（那只会刷新 TTL，"缓存永不失效"
            # 会以这种方式悄悄发生）。
            return outcome

        parsed = self.config_model.model_validate(config)

        get_tool_cache().put(
            self._cache_key(invocation, parsed),
            outcome.value,
            ttl_seconds=parsed.ttl_seconds,
        )

        return outcome

    # ========================================================
    # AROUND：查 + 调用 + 回填
    # ========================================================

    async def around(
            self,
            invocation: ToolInvocation,
            next_call: AsyncToolCall,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:

        parsed = self.config_model.model_validate(config)

        key = self._cache_key(invocation, parsed)

        hit, value = get_tool_cache().get(key)

        if hit:
            logger.debug(
                "%s：%s 命中缓存（AROUND）。",
                self.get_key(),
                invocation.tool_key,
            )
            return ToolOutcome(value=value, cached=True)

        outcome = await next_call()

        if outcome.error is None:
            get_tool_cache().put(
                key,
                outcome.value,
                ttl_seconds=parsed.ttl_seconds,
            )

        return outcome

    # ========================================================
    # internals
    # ========================================================

    @staticmethod
    def _cache_key(
            invocation: ToolInvocation,
            config: CacheConfig,
    ) -> str:
        """
        工具键 + 选定参数的摘要。

        用 sha256 而不是把参数字符串直接当键：参数可能很长
        （整段正文），而缓存字典的键会被一直持有 —— 摘要定长。

        ⚠️ `default=str` 是兜底，不是保证
        ---------------------------------
        参数里出现不可 JSON 序列化的对象时（日期、UUID、自定义类）
        会退化成 str()，而**默认的 repr 含内存地址**（`<Foo at 0x7f…>`）
        —— 那样同一个逻辑对象每次算出的键都不同，表现为命中率恒为 0。

        因此：参数请用可序列化的形态（UUID / datetime 的 str 形式）。
        这里不抛异常是因为"缓存算不出键"不该让整个调用失败 ——
        穿透到底层重算总是安全的，只是没省下那次调用。
        """

        if config.key_fields is None:
            selected = dict(invocation.arguments)
        else:
            selected = {
                field: invocation.arguments[field]
                for field in config.key_fields
                if field in invocation.arguments
            }

        canonical = json.dumps(
            {
                "tool_key": invocation.tool_key,
                "arguments": selected,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
