"""
执行守卫与执行记录共用的纯函数。

放在应用层而不是仓储或 domain：它们**不定形状**（形状是
models/sqlalchemy 与 domain/schemas 的事），只是把"怎么算"定在一处
—— 这些算法今天都各只有一个正确实现，抄第二份就一定会漂。

四件事：

    param_hash_of    参数 -> 判重 / INVOKE 计数的那一段
    execution_error  异常 -> error 列的 {"type","message"}
    execution_result 返回值 -> result 列能存的 JSON 形态
    match_key_of     维度组合 -> 计数与加锁用的定长摘要

另外还有两个**判定**纯函数（decide_existing / decide_new）。它们
不碰数据库，因此守卫的策略矩阵可以脱离 Postgres 单独验证。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.common.utils import canonical_json, sha256_text

from ..domain.entities import ToolExecution
from ..domain.schemas.enum import (
    ExecutionDecision,
    ExecutionMatch,
    ExecutionScope,
    GuardAction,
    ToolExecutionStatus,
)
from ..domain.schemas.execution import ExecutionStatusLimit

__all__ = [
    "decide_existing",
    "decide_new",
    "execution_error",
    "execution_result",
    "match_key_of",
    "param_hash_of",
    "scope_key_of",
]

logger = logging.getLogger(__name__)


# ============================================================
# 参数摘要
# ============================================================


def param_hash_of(
        arguments: Mapping[str, Any],
) -> str:
    """
    调用参数 -> 判重 / INVOKE 计数的那一段。

    与 cache 拦截器同一套（`canonical_json` + `sha256_text`，见
    processor/interceptor/cache.py 的 `_cache_key`）：规范化 JSON 保证
    "键序不同但内容相同"算出同一个摘要，sha256 让键定长。

    ⚠️ 算的是**模型给的原始参数**，不是处理器改写之后的
    --------------------------------------------------
    `_wrap_tool_call` 在驱动处理器链**之前**算这个值，因此
    privacy_transformer 的抹除、BEFORE 拦截器补的默认值都不影响它。

    这是刻意的：判重的语义是"**模型**要求做同一件事两次"，而处理器
    的改写是**我们**对同一次要求做的加工 —— 把它算进去会让"同一个
    要求"因为一次脱敏配置的变更而换了键，历史记录就对不上了。

    ⚠️ 代价要说清楚：两次**加工结果不同**的调用会被判成同一次。
    例如某个 BEFORE 处理器按当前时间补了一个参数，模型两次发出同样
    的调用，第二次会被拒。这种处理器与幂等记录天然冲突，配的时候
    要意识到（与 cache 拦截器面对的是同一个问题）。

    参数含不可 JSON 序列化的对象时退化成 `str()` —— 与 cache 那条
    一样的兜底（见 digest.canonical_json 的 default_str）。模型给的
    参数本来就是 JSON 形态，走不到那条路；真走到了说明调用方不是
    工具节点，那时"两个不同对象算出同一个键"的风险由调用方负责。
    """

    return sha256_text(
        canonical_json(dict(arguments))
    )


# ============================================================
# 维度与计数键
# ============================================================


def scope_key_of(
        scope: ExecutionScope,
        *,
        session_id: UUID | None,
        run_id: UUID | None,
        user_id: str | None,
) -> str:
    """
    维度 -> 计数的那个"值"。

    SESSION / RUN / USER / GLOBAL 分别取会话、运行、用户与常量
    `'*'`。取值缺失时抛 ValueError —— 守卫会把这条错误转成
    fail-closed 的拒绝；**不要**兜底造一个值，那会把不同的会话 /
    用户混进同一个计数桶。
    """

    if scope is ExecutionScope.SESSION:
        if session_id is None:
            raise ValueError(
                "scope=SESSION 需要 session_id，但调用方没有提供。"
            )
        return str(session_id)

    if scope is ExecutionScope.RUN:
        if run_id is None:
            raise ValueError(
                "scope=RUN 需要 run_id，但调用方没有提供。"
            )
        return str(run_id)

    if scope is ExecutionScope.USER:
        if not user_id:
            raise ValueError(
                "scope=USER 需要 user_id，但调用方没有提供。"
            )
        return str(user_id)

    return "*"


def match_key_of(
        *,
        tool_key: str,
        scope: ExecutionScope,
        scope_key: str,
        execution: ExecutionMatch,
        param_hash: str,
) -> str:
    """
    五段维度 -> 定长摘要。

    为什么用摘要而不是多列组合：
        加锁（pg_advisory_xact_lock 要 bigint）、计数查询、索引
        都只需要一个键；多列组合比较容易把 NULL / 大小写之类的
        差异带进判定。

    为什么把 tool_key 放进来：
        守卫是**按工具**配置的。GLOBAL 的含义是「所有用户上的这个
        工具」，不是「所有工具共享一个额度」。
    """

    return sha256_text(
        canonical_json({
            "tool_key": tool_key,
            "scope": scope.value,
            "scope_key": scope_key,
            "execution": execution.value,
            "param_hash": param_hash,
        })
    )


# ============================================================
# 判定（纯函数）
# ============================================================


def decide_existing(
        existing: ToolExecution,
) -> tuple[ExecutionDecision, str | None]:
    """
    同一次工具调用（run_id + tool_call_id）已经有记录时怎么处理。

    这是 replay / resume 的那条路，只回答**确定性**的三种：

        SUCCESS         REPLAY —— 直接回放当时的结果，不再执行
        PENDING/RUNNING BLOCK  —— 别人正在跑，或崩溃遗留；fail-closed
        其它            EXECUTE —— 交给 status_counts 的配额判定

    UNKNOWN / FAILED 不再在这里单独判：它们已经被 status_counts
    计入 unknown / error 配额，由配额决定「这次还让不让试」。
    INTERRUPT / REJECTED / CANCELLED / BLOCKED 不占额度，重新判定。
    """

    if existing.status is ToolExecutionStatus.SUCCESS:
        return ExecutionDecision.REPLAY, None

    if existing.status in (
            ToolExecutionStatus.PENDING,
            ToolExecutionStatus.RUNNING,
    ):
        return (
            ExecutionDecision.BLOCK,
            "同一次工具调用已经有一条执行中的记录"
            "（可能是另一个分支正在跑，或上一次崩溃遗留），"
            "本次未执行。",
        )

    return ExecutionDecision.EXECUTE, None


def decide_new(
        *,
        counts: Mapping[ToolExecutionStatus, int],
        success: ExecutionStatusLimit,
        error: ExecutionStatusLimit,
        unknown: ExecutionStatusLimit,
) -> tuple[ExecutionDecision, str | None]:
    """
    一次**新调用**（或可重试的旧调用）怎么处理。

    三种状态各自独立：

        success_count = SUCCESS + PENDING + RUNNING
        error_count   = FAILED
        unknown_count = UNKNOWN

    判定顺序：success -> unknown -> error。某个状态达到上限且动作是
    BLOCK 就停；ALLOW 只是「这一条不拦」，继续看下一个。

    ⚠️ success 包含 PENDING / RUNNING
    ---------------------------------
    在途尝试先占一个成功名额。否则 `success.limit=1` 时，两个
    并发分支都会看到「还没有成功」而都执行 —— 对退款这类操作是
    灾难。落终态后 PENDING 会变成 SUCCESS / FAILED / UNKNOWN，
    名额随状态一起转移。
    """

    success_count = (
        counts.get(ToolExecutionStatus.SUCCESS, 0)
        + counts.get(ToolExecutionStatus.PENDING, 0)
        + counts.get(ToolExecutionStatus.RUNNING, 0)
    )

    checks = (
        ("成功", success, success_count),
        ("unknown", unknown,
         counts.get(ToolExecutionStatus.UNKNOWN, 0)),
        ("失败", error, counts.get(ToolExecutionStatus.FAILED, 0)),
    )

    for label, rule, count in checks:

        if rule.limit is None or count < rule.limit:
            continue

        if rule.action is GuardAction.BLOCK:
            return (
                ExecutionDecision.BLOCK,
                f"相同维度的{label}次数已经达到上限 "
                f"{rule.limit} 次（当前 {count} 次），本次未执行。",
            )

        # ALLOW：只告警不拦，继续看下一个状态。

    return ExecutionDecision.EXECUTE, None


# ============================================================
# 错误规范化
# ============================================================


def execution_error(
        error: BaseException,
) -> dict[str, str]:
    """
    异常 -> `error` 列的 `{"type": ..., "message": ...}`。

    只存类型名与那段文案，**不存堆栈** —— 堆栈进日志。与 checkauth
    的"给模型的文案刻意含糊、细节留服务端日志"是同一条口径
    （processor/interceptor/auth.py 的 `_deny`）：这段东西会经
    ToolMessage 回到模型、进而可能被转述给用户，而堆栈里有路径、
    有内部结构。

    异常类型名留着是因为它对排查有用，且不泄露部署结构。
    """

    return {
        "type": type(error).__name__,
        "message": str(error),
    }


# ============================================================
# 返回值规范化
# ============================================================

#: 递归规范化的深度上限。
#:
#: 防御**自引用**的容器（A 里放 B，B 里又放回 A）。`json.dumps` 遇到
#: 那种结构会自己抛 ValueError，而这里是手写递归 —— 不设上限就是
#: RecursionError，那会一路冒到节点外，把一次成功的调用变成崩图。
#:
#: 与 registry/schema.py 的 MAX_SCHEMA_DEPTH 是同一类防御。
_MAX_RESULT_DEPTH = 12


class _NotJsonable(Exception):
    def __init__(self, type_name: str) -> None:
        super().__init__(type_name)
        self.type_name = type_name


def _jsonable(value: Any, depth: int = 0) -> Any:
    """
    递归把一个值变成 JSON 能表达的东西；认不出来就抛 `_NotJsonable`。

    ⚠️ 为什么必须递归，而不是试一次 `json.dumps`
    ------------------------------------------
    第一版是"试着 dumps 一下，失败就整份降级"，结果**本仓最常见的
    返回形状恰好是最先降级的那个**：`capability.list_capabilities`
    返回 `list[Capability]`（pydantic 实体列表），落到"容器里有异物"
    那条路，`result` 列存的是一段 repr 文本 —— 而它作为一个 JSONB 列
    的意义正是可查询。

    ⚠️ "能变成什么"的判据锚在 pydantic 上
    ------------------------------------
    标量那几条（datetime / date / UUID / Decimal 的形态）与
    `model_dump(mode="json")` 给出的一致 —— 同一份值不论裹不裹在
    pydantic 里，存进库的形状都一样。
    """

    if depth > _MAX_RESULT_DEPTH:
        raise _NotJsonable(f"嵌套超过 {_MAX_RESULT_DEPTH} 层（或自引用）")

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item, depth + 1) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, (UUID, Decimal)):
        return str(value)

    raise _NotJsonable(type(value).__name__)


def _render_for_storage(value: Any) -> str:
    """
    降级用的文本形态 —— 与 ToolMessage 的 `_render` 同一个规则。

    str 直用；其余 JSON 化；`default=str` 兜住不可序列化的对象。
    这个函数自己也可能炸（repr 会抛的对象），调用方再兜一层。
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False, default=str)


def _degrade(value: Any, reason: str) -> str:
    """
    降级成一段文本，并留下 WARNING。

    ⚠️ 这一段是"能记多少算多少"：`_render_for_storage` 也可能炸，
    所以再兜一层。两层的存在理由是同一个 —— 记账失败绝不能影响
    工具调用本身的成败。
    """

    logger.warning(
        "工具返回值记入执行记录时降级成文本（%s），"
        "result 列里存的是这段文本，不是结构化数据。",
        reason,
    )

    try:
        return _render_for_storage(value)

    except Exception:  # noqa: BLE001 - 见 docstring

        logger.warning(
            "连降级渲染也失败了，result 列里只留一个占位。",
            exc_info=True,
        )

        return f"<{type(value).__name__}：无法记入 result 列>"


def execution_result(value: Any) -> Any:
    """
    工具返回值 -> `result` 列能存的形态。

    ⚠️ 与 ToolMessage 的 `_render` 分工不同：那个产出**给模型看的
    正文**（一个字符串），本函数产出**入库的结构**（JSONB）。

    ⚠️ 降级**不静默**：认不出来的叶子会让**整份**值降级成文本，读库
    的人从值本身分辨不出来，所以打 WARNING 并带上类型名。

    ⚠️ **永不抛**
    ------------
    走到这里工具已经跑过了。为了记一笔账而把一次成功的调用变成崩节点，
    是拿观测去推翻事实。
    """

    try:
        return _jsonable(value)

    except _NotJsonable as exc:
        return _degrade(
            value, f"类型 {exc.type_name} 不知道怎么变成 JSON"
        )

    except RecursionError:
        return _degrade(value, "规范化时递归过深")
