"""任务板：把 planner 产出的 A2A task 发布到 Redis。

为什么需要任务板
----------------
主图顺序执行时，`tasks` 放在 graph state 里就够了。一旦要引入 worker
（子图 / 独立进程），state 就不再是共享的权威：

    Send 的 payload 取代 state     分支读不到父图的 tasks
    多个 worker 并发               各自持有副本会重复认领
    跨 run / 崩溃恢复              内存里的 claim 无法续命

因此任务板是**执行态的权威**，graph state 只是投影。见
`design_planner/task.md` §5.3 / §7.6。

键结构（本模块的对外契约）
------------------------
    agent_task:{task_id}                 HASH
        task          canonical A2A Task JSON（MessageToDict）
        state         TASK_STATE_SUBMITTED / WORKING / ...
        context_id    会话 id
        plan_id
        revision
        claim         认领信息 JSON（初始 {}）
        updated_at    ISO8601（UTC）

    agent_tasks:{context_id}             SET of task_id（按会话枚举）
    agent_plan:{plan_id}:{revision}      HASH（ready 标记）
        ready         1
        context_id / plan_id / revision / task_count / updated_at
    agent_plan_tasks:{plan_id}:{revision} SET of task_id（按计划枚举）

⚠️ 与事件流是两个东西
-------------------
`agent_run:{run_id}` 是实时 UI 的事件流（600 秒过期、有 maxlen），
不能当任务板。任务要跨 run 存活，TTL 按计划算（默认 7 天）。

⚠️ 幂等：已存在的任务不覆盖
------------------------
同一 task_id 已经存在时**不重写** `task` / `state` / `claim`。理由：
revision 重放、planner 重试都不应该把已认领 / 已完成的任务打回
SUBMITTED，也不应该丢掉 claim。代价是「同一 id 的内容变更」不会生效 ——
重规划应当产生新的 revision，而不是复用旧任务行。

⚠️ 发布失败**不掩盖**
-------------------
`RedisTaskBoard.publish` 让异常往上抛；`publish_plan` 把它收成
`PublishResult(degraded=True, reason=...)`，由节点写进 notes / span。
当前（只有发布、没有派发）失败不影响本轮结束；接上 worker 之后，
任务板是调度权威，这一步必须 fail-closed —— 那时把这里的容忍改成抛出。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence, runtime_checkable

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "PLAN_KEY_PREFIX",
    "PLAN_TASKS_PREFIX",
    "PublishResult",
    "RedisTaskBoard",
    "TASK_INDEX_PREFIX",
    "TASK_KEY_PREFIX",
    "TaskBoard",
    "build_task_board",
    "publish_plan",
]

logger = logging.getLogger(__name__)


TASK_KEY_PREFIX = "agent_task"
TASK_INDEX_PREFIX = "agent_tasks"
PLAN_KEY_PREFIX = "agent_plan"
PLAN_TASKS_PREFIX = "agent_plan_tasks"

#: 任务板 TTL（秒）。跟计划，不跟 run。
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class PublishResult:
    """一次发布的结果。`degraded=True` 时必须让调用方看得见。"""

    published: int
    skipped: int
    degraded: bool = False
    reason: str = ""

    def as_metadata(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "skipped": self.skipped,
            "degraded": self.degraded,
            "reason": self.reason,
        }


@runtime_checkable
class TaskBoard(Protocol):
    """
    任务板端口：发布 + 完成。

    `claim` 仍由 `plan/listener.py` 的 `RedisTaskSubscriber` 走 Lua CAS
    （它需要读-选-抢的循环）。完成没有那个形状 —— worker 手里已经有
    完整 task dict，直接写回即可，所以放在这里。
    """

    async def publish(
            self,
            *,
            context_id: str,
            plan_id: str,
            revision: int,
            tasks: Sequence[dict[str, Any]],
            ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> PublishResult:
        ...

    async def complete(
            self,
            *,
            task_id: str,
            task: Mapping[str, Any],
    ) -> bool:
        """
        把 worker 执行完的 task 写回任务板。

        Returns
        -------
        bool
            True = 行存在且已更新；False = 行不存在（任务板过期 / 键错）
            —— 调用方必须把 False 当失败看，而不是"完成成功"。
        """
        ...


class RedisTaskBoard:
    """Redis 实现：幂等 upsert 任务行 + 索引 + plan ready 标记。

    客户端用 `app.common.redis_factory` 的 asyncio 短查询客户端
    （`optional_client()`），不自己建连。
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    async def publish(
            self,
            *,
            context_id: str,
            plan_id: str,
            revision: int,
            tasks: Sequence[dict[str, Any]],
            ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> PublishResult:

        stamp = _iso(datetime.now(timezone.utc))
        index_key = task_index_key(context_id)
        plan_tasks = plan_tasks_key(plan_id, revision)

        published = 0
        skipped = 0
        task_ids: list[str] = []

        for task in tasks:

            task_id = str(task.get("id") or "").strip()

            if not task_id:
                skipped += 1
                continue

            task_ids.append(task_id)
            key = task_key(task_id)

            # 已存在 -> 保留运行期状态（claim / COMPLETED 等），不覆盖。
            if await self._client.hexists(key, "task"):
                skipped += 1
                continue

            metadata = task.get("metadata")

            if not isinstance(metadata, dict):
                metadata = {}

            claim = metadata.get("claim")

            if not isinstance(claim, dict):
                claim = {}

            await self._client.hset(key, mapping={
                "task": json.dumps(task, ensure_ascii=False),
                "state": _state_of(task),
                "context_id": context_id,
                "plan_id": plan_id,
                "revision": str(revision),
                "claim": json.dumps(claim, ensure_ascii=False),
                "updated_at": stamp,
            })
            await self._client.expire(key, ttl_seconds)

            published += 1

        if task_ids:
            await self._client.sadd(index_key, *task_ids)
            await self._client.sadd(plan_tasks, *task_ids)

        # 索引与 plan 标记即使本次全部 skipped 也刷新 TTL：重放不应该让
        # 一个还在用的计划因为一次幂等发布而过期。
        await self._client.expire(index_key, ttl_seconds)
        await self._client.expire(plan_tasks, ttl_seconds)

        plan_key = plan_key_of(plan_id, revision)
        await self._client.hset(plan_key, mapping={
            "ready": "1",
            "context_id": context_id,
            "plan_id": plan_id,
            "revision": str(revision),
            "task_count": str(len(tasks)),
            "updated_at": stamp,
        })
        await self._client.expire(plan_key, ttl_seconds)

        return PublishResult(published=published, skipped=skipped)

    async def complete(
            self,
            *,
            task_id: str,
            task: Mapping[str, Any],
    ) -> bool:
        """
        幂等写回完成态。**不刷新也不缩短 TTL** —— 生命周期跟计划走。

        行不存在时返回 False（不创建）：任务板是权威，凭空补一行
        会让父图的同步看到一个从未发布过的任务。
        """

        tid = str(task_id or "").strip()

        if not tid:
            return False

        key = task_key(tid)

        if not await self._client.exists(key):
            logger.warning(
                "任务板完成写回失败：%s 不存在（可能已过期或未发布）。",
                key,
            )
            return False

        metadata = task.get("metadata")
        claim = (
            metadata.get("claim")
            if isinstance(metadata, Mapping)
            else None
        )

        await self._client.hset(key, mapping={
            "task": json.dumps(dict(task), ensure_ascii=False, default=str),
            "state": _state_of(dict(task)),
            "claim": json.dumps(
                dict(claim) if isinstance(claim, Mapping) else {},
                ensure_ascii=False,
            ),
            "updated_at": _iso(datetime.now(timezone.utc)),
        })

        return True


def build_task_board() -> TaskBoard | None:
    """
    按应用级 Redis 工厂造一个任务板；Redis 未装配时返回 None。

    不抛：本地没起 Redis 是合法状态（`redis.yaml` 的 required=false）。
    需要「必须有任务板」的调用方应当把 None 当成失败，而不是静默继续。
    """

    try:
        from app.common.factories import get_redis_factory

        client = get_redis_factory().optional_client()

    except Exception:  # noqa: BLE001 - 未装配 Redis 是正常状态
        client = None

    if client is None:
        return None

    return RedisTaskBoard(client)


async def publish_plan(
        *,
        board: TaskBoard | None,
        context_id: str,
        plan_id: str,
        revision: int,
        tasks: Sequence[dict[str, Any]],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> PublishResult:
    """
    发布一份计划，失败收成 `degraded` 而不是抛。

    当前阶段（只发布、不派发）失败只影响可见性；接上 worker 之后，
    这里要改成 fail-closed（任务板不可用就不派发）。
    """

    if board is None:
        return PublishResult(
            published=0,
            skipped=0,
            degraded=True,
            reason="Redis 任务板未装配（redis.yaml 未启用或连接不可用）。",
        )

    try:
        return await board.publish(
            context_id=context_id,
            plan_id=plan_id,
            revision=revision,
            tasks=tasks,
            ttl_seconds=ttl_seconds,
        )

    except Exception as exc:  # noqa: BLE001 - 见 docstring 的失败语义

        logger.warning(
            "任务板发布失败（%s: %s），计划仍写入 state，但 worker 不可见。",
            type(exc).__name__,
            exc,
        )

        return PublishResult(
            published=0,
            skipped=0,
            degraded=True,
            reason=f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# 键名（发布端与 listener 必须用同一套）
# ============================================================


def task_key(task_id: str) -> str:
    return f"{TASK_KEY_PREFIX}:{task_id}"


def task_index_key(context_id: str) -> str:
    return f"{TASK_INDEX_PREFIX}:{context_id}"


def plan_key_of(plan_id: str, revision: int) -> str:
    return f"{PLAN_KEY_PREFIX}:{plan_id}:{revision}"


def plan_tasks_key(plan_id: str, revision: int) -> str:
    return f"{PLAN_TASKS_PREFIX}:{plan_id}:{revision}"


# ============================================================
# internals
# ============================================================


def _state_of(task: dict[str, Any]) -> str:
    """A2A canonical dict 的 status.state；兼容字符串形态。"""

    status = task.get("status")

    if isinstance(status, dict):
        return str(status.get("state") or "")

    if isinstance(status, str):
        return status

    return ""


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
