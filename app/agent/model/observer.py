"""
调用级观测：一次 `runtime.ainvoke` 与它内部的每一次 provider attempt。

这不是第二套日志 / 追踪系统
--------------------------
本模块**没有**引入新的 tracing / logging 框架：它只有一个 Protocol，
默认实现用标准库 `logging`。要接 OpenTelemetry / LangSmith / 观测表，
实现这个 Protocol 即可 —— 而不是在 Runtime 里埋 SDK 调用。

接线方式（与既有观测的关系）
--------------------------
    Runtime 在每次 attempt 与整次调用结束时调 observer
        ↓
    默认实现往标准库 logging 写（本地开发看得见）
        ↓
    LLM Node 另外拿到 `ModelResponse.metadata`（含逐次 attempt），
    写进现有 node_span 的 metrics —— 于是 provider 级信息顺着
    **已有的**图观测落库，不需要 Model 模块认识 infra/graph。

⚠️ 观察者**绝不能**让模型调用失败
--------------------------------
每个回调都被 `emit_attempt` / `emit_call` 包住，异常只记日志。
理由是这类"基础设施回调"抛错时，真正失败的是一次本来成功的模型调用
（和同一套 `_safe_record` 在 infra/graph 里的取舍一致）。

⚠️ 观察者**绝不能**在这里做重 IO
-------------------------------
回调是同步的，且跑在模型调用的路径上（在 attempt 的计时窗口内）。
在这里同步写库会把数据库的延迟算进模型延迟，并直接拖慢每一次调用。
需要落库就交给 LLM Node 的 span，或做成队列异步消费。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .types import ModelAttempt, ModelCallSummary

__all__ = [
    "CompositeModelObserver",
    "LoggingModelObserver",
    "ModelCallContext",
    "ModelCallObserver",
    "NullModelObserver",
    "emit_attempt",
    "emit_call",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelCallContext:
    """
    一次调用的**稳定身份**，在 attempt 回调里充当"这次调用是谁"。

    与 ModelCallSummary 的区别：summary 是**结果**（要等调用结束才
    存在），这个是**身份**（调用一开始就有）。attempt 回调发生在
    调用过程中，因此只能拿到这个。
    """

    invocation_id: str
    agent_type: str
    model_role: str
    profile_id: str


@runtime_checkable
class ModelCallObserver(Protocol):
    """
    模型调用的观察者。

    两个方法各对应一件真实的事：

        on_attempt   一次 provider attempt 结束（成功或失败）
                     —— "试了几次、每次多久、每次什么错"
        on_call      整次 runtime.ainvoke 结束
                     —— "这次调用最终成不成、降级没有、花了多少"
    """

    def on_attempt(
            self,
            attempt: ModelAttempt,
            *,
            context: ModelCallContext,
    ) -> None:
        ...

    def on_call(self, summary: ModelCallSummary) -> None:
        ...


class NullModelObserver:
    """什么都不做 —— 显式表达"这次不要观测"。"""

    def on_attempt(
            self,
            attempt: ModelAttempt,
            *,
            context: ModelCallContext,
    ) -> None:
        return None

    def on_call(self, summary: ModelCallSummary) -> None:
        return None


class LoggingModelObserver:
    """
    用标准库 logging 记录调用结论。

    ⚠️ 只打**结论**，不打正文
    ------------------------
    没有消息内容、没有 prompt、没有 metadata 原文（只有键名与计数）。
    日志不该是"模型说了什么"的第二份存储 —— 那是 checkpoint 与
    观测行的职责，而日志既没有访问控制也没有保留策略。

    级别：

        attempt 成功        DEBUG（正常情况下没人要看）
        attempt 失败        WARNING（失败一次可能马上就好了，所以不是 ERROR）
        call 成功且未降级    INFO
        call 成功但降级过    WARNING（结果可用，但可靠性打了折 —— 值得看）
        call 失败           ERROR

    ⚠️ 正文在**别处**，而且比这里低一级
    ----------------------------------
    "模型看到了什么 / 回了什么"由 `adapters.py` 的 `_log_request` /
    `_log_response` 打 —— 压在 **DEBUG**（本类成功路径的 INFO 之下一级），
    并且先过 `redaction.redact_text`。

    所以两处的分工是：默认级别下日志里只有这里的**结论**（谁答的、
    多久、几次、降级没有），一个字的正文都没有；把级别调到 DEBUG 才
    看得到提示词与响应。要加正文相关的日志，加在那边，不要加在这里 ——
    这里一旦掺进正文，"默认级别下没有正文"这条保证就不再成立了。
    """

    def on_attempt(
            self,
            attempt: ModelAttempt,
            *,
            context: ModelCallContext,
    ) -> None:

        if attempt.success:
            logger.debug(
                "model attempt ok: invocation=%s profile=%s %s attempt=%d "
                "candidate=%d latency=%dms",
                context.invocation_id,
                context.profile_id,
                attempt.model_id,
                attempt.attempt,
                attempt.candidate_index,
                attempt.latency_ms,
            )
            return

        logger.warning(
            "model attempt failed: invocation=%s profile=%s %s attempt=%d "
            "candidate=%d latency=%dms error=%s(%s)",
            context.invocation_id,
            context.profile_id,
            attempt.model_id,
            attempt.attempt,
            attempt.candidate_index,
            attempt.latency_ms,
            attempt.error_type,
            attempt.error_code,
        )

    def on_call(self, summary: ModelCallSummary) -> None:

        if not summary.success:

            logger.error(
                "model call failed: invocation=%s agent_type=%s role=%s "
                "profile=%s attempts=%d latency=%dms error=%s(%s)",
                summary.invocation_id,
                summary.agent_type,
                summary.model_role,
                summary.profile_id,
                summary.provider_attempts,
                summary.latency_ms,
                summary.error_type,
                summary.error_code,
            )

            return

        if summary.degraded:

            logger.warning(
                "model call degraded: invocation=%s agent_type=%s role=%s "
                "profile=%s answered_by=%s attempts=%d latency=%dms "
                "incompatible=%s",
                summary.invocation_id,
                summary.agent_type,
                summary.model_role,
                summary.profile_id,
                summary.model,
                summary.provider_attempts,
                summary.latency_ms,
                list(summary.incompatible_candidates),
            )

            return

        logger.info(
            "model call ok: invocation=%s agent_type=%s role=%s profile=%s "
            "answered_by=%s latency=%dms tokens=%s cost=%s",
            summary.invocation_id,
            summary.agent_type,
            summary.model_role,
            summary.profile_id,
            summary.model,
            summary.latency_ms,
            summary.usage.total_tokens if summary.usage else None,
            summary.estimated_cost,
        )


class CompositeModelObserver:
    """把一次调用扇出给多个观察者（顺序执行）。"""

    def __init__(self, observers: Sequence[ModelCallObserver]) -> None:
        self._observers = tuple(observers)

    def on_attempt(
            self,
            attempt: ModelAttempt,
            *,
            context: ModelCallContext,
    ) -> None:

        for observer in self._observers:
            _safe(observer.on_attempt, attempt, context=context)

    def on_call(self, summary: ModelCallSummary) -> None:

        for observer in self._observers:
            _safe(observer.on_call, summary)


# ============================================================
# 安全调用（Runtime 只用这两个入口）
# ============================================================


def emit_attempt(
        observer: ModelCallObserver,
        attempt: ModelAttempt,
        *,
        context: ModelCallContext,
) -> None:
    """
    通知一次 attempt，**吞掉观察者的异常**。

    理由见模块 docstring：观测失败不该让一次成功的模型调用失败。
    """

    _safe(observer.on_attempt, attempt, context=context)


def emit_call(
        observer: ModelCallObserver,
        summary: ModelCallSummary,
) -> None:
    """通知整次调用结束，同样吞掉观察者的异常。"""

    _safe(observer.on_call, summary)


def _safe(
        callback: Any,
        *args: Any,
        **kwargs: Any,
) -> None:

    try:
        callback(*args, **kwargs)

    except Exception:

        logger.exception(
            "模型调用观察者抛错，已忽略（业务结果不受影响）：%r",
            callback,
        )
