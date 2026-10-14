"""
L0 记忆检索的对外契约。

逐字段对应 memory.yaml 的 components/schemas：
    MemorySearchRequest / MemorySearchHit / DegradedRoute /
    MemorySearchResponse

⚠️ 这个请求模型**同时是工具的入参 schema**
--------------------------------------
`@tool` 打在服务实现的方法上时，schema 由方法的那个 pydantic 模型参数
推导（见 tool/registry/schema.py::resolve_request_model）。所以这个类的
字段就是模型能填的全部 —— 也正因如此，`user_id` / `org_id` /
`session_id` **一个都不在这里**：

    模型能填的东西 = 越权的可能性

`session_id` 走 HTTP 的 query 参数（或由工具的运行时注入），身份类字段
从 access token（或运行时）取。理由写在 memory.yaml 的 info.description。

⚠️ 刻意没有 total / 分页
--------------------
这是一轮召回，不是可分页的资源集合。条数由 limit 决定，"还有 400 条"
对调用方不是一个可行动的事实 —— 它会换一组词再问一次。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from app.common.schemas import APIModel
from app.services.session.domain.schemas.enum import MessageRole

from .enum import (
    MemoryRecallRoute,
    MemorySearchMode,
    MemorySearchScope,
)

__all__ = [
    "DegradedRoute",
    "MemorySearchHit",
    "MemorySearchRequest",
    "MemorySearchResponse",
]


#: 一次召回的默认条数与上界。
#:
#: 上界是 50 而不是"不设"：这个接口的调用方之一是**模型**，而它会把
#: 结果整段塞进上下文。没有上界时"多搜一点"是个免费的选项，于是它会
#: 选；真正付代价的是那一轮的 token 预算。
DEFAULT_LIMIT = 10
MAX_LIMIT = 50


class MemorySearchRequest(APIModel):
    """
    一次召回请求。

    也是暴露给模型的工具入参（见模块 docstring）。

    ⚠️ 字段顺序 = 模型看到的顺序
    --------------------------
    `query` 在最前，因为它是唯一必填的那个；其余都有默认值。工具 schema
    里必填字段的排布会影响模型填参数的顺序，把可选的放前面会让它先纠结
    无关的开关。
    """

    query: str = Field(
        min_length=1,
        description=(
            "自然语言查询，用用户的语言。\n\n"
            "关键词路用它切词、向量路用它算向量 —— 两侧都用与建索引时"
            "同一个分词器 / 同一个模型。"
        ),
    )

    scope: MemorySearchScope = Field(
        default=MemorySearchScope.SESSION,
        description=(
            "召回范围。session=只搜当前会话；user=搜该用户的所有会话。\n\n"
            "⚠️ user 放开的只是**会话维度**：user_id 由运行时注入，"
            "填不了。"
        ),
    )

    mode: MemorySearchMode = Field(
        default=MemorySearchMode.HYBRID,
        description="跑哪几路召回。某一路没跑成时看响应里的 degraded_routes。",
    )

    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description="最多返回几条。在**融合之后**截断。",
    )

    roles: list[MessageRole] | None = Field(
        default=None,
        description=(
            "只搜这些角色。null = 两种都要（默认）。\n\n"
            "默认两种都要是刻意的：“我说过什么”和“你说过什么”都是正当"
            "问题，为了拿到默认值而必须把两个角色都写出来只是噪声。"
        ),
    )

    time_from: datetime | None = Field(
        default=None,
        description="消息 created_at 的下界（含）。null = 不限。",
    )

    time_to: datetime | None = Field(
        default=None,
        description="消息 created_at 的上界（含）。null = 不限。",
    )

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        """
        空白串不算"有查询"。

        `min_length=1` 拦不住 `"   "`，而它的后果不是"返回空"那么无害：
        关键词路的 `tsquery_text` 会切出零个词元、于是这一路降级进
        degraded_routes —— 调用方会看到一条"关键词后端有问题"的告警，
        而真实原因只是它自己传了个空格。

        所以在这里断掉，并且**返回去空白后的值**：模型偶尔会带上前后
        空格，为此报错不值当。
        """

        stripped = value.strip()

        if not stripped:
            raise ValueError("query 不能是空白字符串。")

        return stripped

    @field_validator("time_to")
    @classmethod
    def _check_window(
            cls, value: datetime | None, info: Any
    ) -> datetime | None:
        """
        时间窗的两端不能反。

        反了的窗口在 SQL 里是**空结果**，而不是错误 —— 调用方会以为
        "那段时间什么都没说过"。而它真正的含义是"你写反了"。
        """

        start = info.data.get("time_from")

        if value is not None and start is not None and value < start:
            raise ValueError(
                f"time_to（{value.isoformat()}）早于 time_from"
                f"（{start.isoformat()}）：时间窗写反了。"
            )

        return value


class MemorySearchHit(APIModel):
    """
    一条命中。

    ⚠️ text 是**片段**，不是整条消息
    -----------------------------
    要全文就拿 message_id 去 messages.yaml 取。这里返回整条消息会让
    响应体积随对话长度膨胀，而"一次工具调用往上下文里倒五条长消息"
    比"倒五句相关的"更糟。
    """

    message_id: UUID = Field(description="命中的 agent_message 行。")

    session_id: UUID = Field(
        description="来自哪个会话。scope=user 时可能与调用者当前会话不同。",
    )

    seq: int = Field(
        description=(
            "会话内序号。取邻居（“紧接着说了什么”）与 messages.yaml "
            "翻页都按它。允许有空洞。"
        ),
    )

    role: MessageRole = Field(description="user / assistant。")

    text: str = Field(
        description=(
            "命中的片段，**原文**（不是建索引用的分词串）。"
            "片段长度由服务选择，目的是让命中处带着上下文出现。"
        ),
    )

    created_at: datetime = Field(description="消息被记录的时间。")

    score: float = Field(
        description=(
            "融合分，越大越相关。**只在本次响应内可比** —— 关键词分来自"
            " ts_rank、向量分来自余弦距离，跨响应或跨 mode 比较它们没有"
            "意义。"
        ),
    )

    matched_by: list[MemoryRecallRoute] = Field(
        description=(
            "这条被哪几路召回到。只有向量路找到的那些是关键词搜索本来"
            "漏掉的 —— 所以这个字段也是“hybrid 到底值不值”的便宜检查。"
        ),
    )


class DegradedRoute(APIModel):
    """
    这一轮**没能参与**的一路召回。

    ⚠️ 刻意不复用 document 模块那个 DegradedRoute
    ----------------------------------------
    那个按 `document_retrieval_definition_id` 定位，因为文档侧有一张
    检索定义表可以指。记忆只有固定的两路、没有那张表，所以这里
    `route` 本身就是全部身份。

    空列表 = 每一路都跑了。
    """

    route: MemoryRecallRoute = Field(description="没参与的那一路。")

    reason: str = Field(
        description=(
            "没参与的原因（后端异常信息，或“尚未接通”）。同时打一条"
            " warning 日志 —— 放进响应里是为了**不静默**。"
        ),
    )


class MemorySearchResponse(APIModel):
    """
    一轮召回的结果。

    `results` 按融合分从高到低。空结果 + 空 degraded_routes 的含义是
    "语料里确实没有" —— 那是一个正当答案，不是错误。
    """

    search_id: UUID = Field(
        description="本次调用的标识，用于日志与链路追踪对齐。",
    )

    query: str = Field(
        description=(
            "回显查询串。一次工具调用的入参与结果在 transcript 里是分开"
            "的，回显让读到结果的人不必往上翻就知道问的是什么。"
        ),
    )

    mode: MemorySearchMode = Field(description="请求的模式，原样回显。")

    scope: MemorySearchScope = Field(description="实际生效的召回范围。")

    results: list[MemorySearchHit] = Field(description="命中，按分数降序。")

    degraded_routes: list[DegradedRoute] = Field(
        description="没参与的那些路。空 = 每一路都跑了。",
    )
