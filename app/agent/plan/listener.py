"""任务监听 / 领取的公共逻辑。

同一段逻辑服务两种图：

    主图 task_listener_node      source = graph state 的 `tasks`
    子图 listener                source = Redis 任务板

所以这里**不认识 LangGraph**：它只认一份任务列表和两个动作：

    claim_next(owner)   找到第一个 ready task，标记 WORKING，返回它

“循环监听”由**图上的回边**表达：

    listener -> model -> tool -> listener -> ...

listener 每次只 claim 一次；不要在节点里写 `while True` —— 那会让当前
super-step 永远不结束，父图的 fan-out/join 也永远等不到它。子图和主图
共享的是这里的 `claim_next`，不是一段阻塞循环。

任务形状
--------
输入 / 输出都是 A2A canonical dict（`MessageToDict(Task)`）：

    id
    status.state          TASK_STATE_SUBMITTED / WORKING / COMPLETED / ...
    metadata.blocked_by   依赖的 A2A task id
    metadata.claim        {"owner", "token", "claimed_at", "lease_expires_at"}

它不解析 planner 的 local id / message payload —— 那是转换器与执行节点的
事；listener 只关心“谁可以开始”。
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from a2a.helpers.proto_helpers import new_data_part, new_text_part
from a2a.types import Artifact
from google.protobuf.json_format import MessageToDict

__all__ = [
    "ClaimedTask",
    "DEFAULT_LEASE_SECONDS",
    "RedisTaskSubscriber",
    "StateTaskSubscriber",
    "TERMINAL",
    "TaskListener",
    "TaskSubscriber",
    "WORKING",
    "blocked_by",
    "build_task_subscriber",
    "complete_task",
    "has_unfinished",
    "is_ready",
    "select_ready",
    "status_state",
    "task_id",
    "task_position",
]

logger = logging.getLogger(__name__)


#: A2A 状态值。兼容 snake / 大写两种写法（老数据、手写 task）。
SUBMITTED = frozenset({"TASK_STATE_SUBMITTED", "submitted"})
WORKING = frozenset({"TASK_STATE_WORKING", "working"})
COMPLETED = frozenset({"TASK_STATE_COMPLETED", "completed"})
TERMINAL = frozenset({
    "TASK_STATE_COMPLETED", "completed",
    "TASK_STATE_FAILED", "failed",
    "TASK_STATE_CANCELED", "canceled",
    "TASK_STATE_REJECTED", "rejected",
})

DEFAULT_LEASE_SECONDS = 300

#: Redis 任务板的键前缀。发布端与 listener 必须用同一套。
TASK_KEY_PREFIX = "agent_task"
TASK_INDEX_PREFIX = "agent_tasks"


@dataclass(frozen=True)
class ClaimedTask:
    """一次 claim 的结果。"""

    #: 已标记 WORKING / 写入 claim 的任务 dict。
    task: dict[str, Any]

    #: state source 使用：替换后的完整任务列表；Redis source 为 None。
    tasks: list[dict[str, Any]] | None = None

    #: "state" | "redis"，用于观测。
    source: str = "state"


@runtime_checkable
class TaskSubscriber(Protocol):
    """listener 的数据源。主图与子图只换实现，不换调用方式。"""

    async def claim_next(self, *, owner: str) -> ClaimedTask | None:
        """claim 第一个 ready task；没有可 claim 的返回 None。"""
        ...

    async def snapshot(self) -> list[dict[str, Any]] | None:
        """
        当前任务列表的只读快照。

        Redis source 读到的是**任务板最新状态**（可能包含别的 worker
        刚写完的完成态）；读失败时抛，由调用方决定 fail-closed。
        state source 返回本地列表。
        """
        ...


class TaskListener:
    """把 subscriber 收成 listener 需要的**一次** claim。

    留这一层是为了让子图与主图共用同一段“有 task 就返回、没有就结束”
    的语义；真正的数据源差异在 subscriber。
    """

    def __init__(self, subscriber: TaskSubscriber) -> None:
        self._subscriber = subscriber

    async def claim_next(self, *, owner: str) -> ClaimedTask | None:
        return await self._subscriber.claim_next(owner=owner)

    async def snapshot(self) -> list[dict[str, Any]] | None:
        return await self._subscriber.snapshot()


# ============================================================
# 选择规则（纯函数，两种 source 共用）
# ============================================================


def task_id(task: Mapping[str, Any]) -> str:
    return str(task.get("id") or "")


def status_state(task: Mapping[str, Any]) -> str:
    status = task.get("status")

    if isinstance(status, Mapping):
        return str(status.get("state") or "")

    # 兼容手写 / 早期形状：status 直接是字符串。
    if isinstance(status, str):
        return status

    return ""


def metadata_of(task: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = task.get("metadata")

    return metadata if isinstance(metadata, Mapping) else {}


def blocked_by(task: Mapping[str, Any]) -> list[str]:
    raw = metadata_of(task).get("blocked_by")

    if isinstance(raw, str):
        return [raw] if raw else []

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw if str(item)]

    return []


def is_ready(
        task: Mapping[str, Any],
        *,
        by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """这个 task 现在能不能开始。"""

    if status_state(task) not in SUBMITTED:
        return False

    claim = metadata_of(task).get("claim")

    # 空 dict / None 才算未认领；有 owner 的 claim 不抢。
    if isinstance(claim, Mapping) and claim:
        return False

    for dep in blocked_by(task):

        blocker = by_id.get(dep)

        if blocker is None:
            return False

        if status_state(blocker) not in COMPLETED:
            return False

    return True


def select_ready(
        tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """按输入顺序选第一个 ready task（输入应是拓扑序）。"""

    by_id = {
        str(task.get("id")): task
        for task in tasks
        if task.get("id")
    }

    for task in tasks:
        if is_ready(task, by_id=by_id):
            return copy.deepcopy(dict(task))

    return None


def has_unfinished(tasks: Sequence[Mapping[str, Any]]) -> bool:
    """还有没有非终态任务。"""

    return any(status_state(task) not in TERMINAL for task in tasks)


def task_position(
        task: Mapping[str, Any],
        tasks: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """当前任务在任务列表里的位置（1-based）与总数。"""

    tid = task_id(task)

    for index, item in enumerate(tasks, start=1):

        if task_id(item) == tid:
            return index, len(tasks)

    return 0, len(tasks)


def complete_task(
        tasks: Sequence[Mapping[str, Any]],
        *,
        task_id_value: str,
        summary: str = "",
        result_text: str = "",
        result_data: Any = None,
        context_id: str | None = None,
        now: datetime | None = None,
) -> list[dict[str, Any]]:
    """
    把任务标成 COMPLETED，并把最终成果写进 A2A `artifacts`。

    分工（与 A2A 语义一致）：

        history      worker 理解任务所需的指令 / 参数 / 对话过程
        artifacts    最终交付物（完整文本或结构化结果）
        metadata     claim、result_summary 等调度元信息

    ⚠️ 这是 state 侧的完成；Redis 任务板的完成应由 TaskBoard 做。
    两种 source 共用同一个“如何表示完成”的形状。
    """

    finished = now or datetime.now(timezone.utc)
    updated: list[dict[str, Any]] = []

    for task in tasks:

        item = copy.deepcopy(dict(task))

        if task_id(item) == task_id_value:

            status = item.get("status")

            if not isinstance(status, dict):
                status = {}

            status["state"] = "TASK_STATE_COMPLETED"
            status["timestamp"] = _iso(finished)
            item["status"] = status

            metadata = item.get("metadata")

            if not isinstance(metadata, dict):
                metadata = {}

            metadata["claim"] = {}

            if summary.strip():
                metadata["result_summary"] = summary.strip()

            # ---- 最终成果：A2A artifacts -------------------------
            artifact_id = f"{task_id_value}:result:1"
            artifact_text = result_text if result_text.strip() else summary.strip()

            parts: list[Any] = []

            if result_data is not None:
                parts.append(
                    new_data_part(
                        result_data, media_type="application/json",
                    )
                )

            if artifact_text:
                parts.append(new_text_part(artifact_text))

            if parts:

                artifact = Artifact(
                    artifact_id=artifact_id,
                    name="task_result",
                    description="任务最终成果",
                    parts=parts,
                    metadata={
                        "task_id": task_id_value,
                        "kind": "task_result",
                    },
                )
                artifact_dict = MessageToDict(artifact)

                artifacts = item.get("artifacts")

                if not isinstance(artifacts, list):
                    artifacts = []

                # 同一 task 重试时替换同一 artifact，不堆多份。
                artifacts = [
                    existing for existing in artifacts
                    if str(existing.get("artifactId") or "")
                    != artifact_id
                ]
                artifacts.append(artifact_dict)
                item["artifacts"] = artifacts

            # history 只留一条“完成”状态，不重复放完整结果 ——
            # 完整结果在 artifacts 里。
            history = item.get("history")

            if not isinstance(history, list):
                history = []

            history_text = (
                f"任务完成，结果见 artifact {artifact_id}。"
                if parts else "任务完成。"
            )

            if summary.strip():
                # history 放摘要（对话可读），完整成果仍只在 artifact。
                history_text += f"\n\n摘要：{summary.strip()}"

            history.append({
                "messageId": f"{task_id_value}:status:{uuid4()}",
                "taskId": task_id_value,
                "contextId": context_id or item.get("contextId", ""),
                "role": "ROLE_AGENT",
                "parts": [{"text": history_text}],
            })
            item["history"] = history
            item["metadata"] = metadata

        updated.append(item)

    return updated


def _claim(
        task: Mapping[str, Any],
        *,
        owner: str,
        lease_seconds: int,
        now: datetime | None = None,
) -> dict[str, Any]:
    """给任务写 WORKING + claim；返回新 dict，不改原对象。"""

    claimed = copy.deepcopy(dict(task))
    created = now or datetime.now(timezone.utc)
    expires = created + timedelta(seconds=lease_seconds)

    status = claimed.get("status")

    if not isinstance(status, dict):
        status = {}

    status["state"] = "TASK_STATE_WORKING"
    status["timestamp"] = _iso(created)
    claimed["status"] = status

    metadata = claimed.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    previous = metadata.get("claim")
    attempt = 1

    if isinstance(previous, Mapping):
        try:
            attempt = int(previous.get("attempt") or 0) + 1
        except (TypeError, ValueError):
            attempt = 1

    metadata["claim"] = {
        "owner": owner,
        "token": str(uuid4()),
        "claimed_at": _iso(created),
        "lease_expires_at": _iso(expires),
        "attempt": attempt,
    }
    claimed["metadata"] = metadata

    return claimed


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ============================================================
# State source
# ============================================================


class StateTaskSubscriber:
    """从 graph state 的 `tasks` 读任务；claim 结果回写 state。"""

    def __init__(
            self,
            state: Mapping[str, Any],
            *,
            lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self._tasks = _coerce_tasks(state.get("tasks"))
        self._lease_seconds = lease_seconds

    async def claim_next(self, *, owner: str) -> ClaimedTask | None:
        ready = select_ready(self._tasks)

        if ready is None:
            return None

        claimed = _claim(
            ready, owner=owner, lease_seconds=self._lease_seconds,
        )

        tid = task_id(claimed)
        updated = [
            claimed if task_id(task) == tid else task
            for task in self._tasks
        ]

        # 连续 claim 时下一次必须看到「上一个已 WORKING」；不更新内部副本
        # 会把同一个任务再 claim 一遍（dispatch 一次要 claim N 个）。
        self._tasks = updated

        return ClaimedTask(task=claimed, tasks=updated, source="state")

    async def snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._tasks)


# ============================================================
# Redis source
# ============================================================


_CLAIM_LUA = """
local raw = redis.call('HGET', KEYS[1], 'task')
if not raw then return nil end
local task = cjson.decode(raw)
task['status'] = task['status'] or {}
local state = task['status']['state'] or ''
if state ~= 'TASK_STATE_SUBMITTED' and state ~= 'submitted' then
  return nil
end
task['status']['state'] = 'TASK_STATE_WORKING'
task['status']['timestamp'] = ARGV[2]
task['metadata'] = task['metadata'] or {}
local prev = task['metadata']['claim']
local attempt = 1
if prev and prev['attempt'] then attempt = (prev['attempt'] or 0) + 1 end
local claim = cjson.decode(ARGV[1])
claim['attempt'] = attempt
task['metadata']['claim'] = claim
redis.call('HSET', KEYS[1], 'task', cjson.encode(task), 'state', 'TASK_STATE_WORKING')
return cjson.encode(task)
"""


class RedisTaskSubscriber:
    """从 Redis 任务板读任务；claim 用 Lua CAS。"""

    def __init__(
            self,
            client: Any,
            *,
            context_id: str,
            plan_id: str | None = None,
            lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> None:
        if not context_id:
            raise ValueError("context_id 不能为空")

        self._client = client
        self._context_id = context_id
        self._plan_id = plan_id
        self._lease_seconds = lease_seconds

    async def claim_next(self, *, owner: str) -> ClaimedTask | None:
        try:
            tasks = await self._load()
        except Exception as exc:  # noqa: BLE001 - Redis 不可用时退回“没有 task”
            logger.warning(
                "Redis 任务板读取失败（%s: %s），本轮不领取任务。",
                type(exc).__name__,
                exc,
            )
            return None

        # 可能有多个 ready；逐个 CAS，抢输的换下一个。
        while True:

            ready = select_ready(tasks)

            if ready is None:
                return None

            claimed = await self._cas_claim(
                task_id(ready), owner=owner,
            )

            if claimed is not None:
                # 顺手把本地快照里的那一条也更新掉：父图拿到的
                # `tasks` 至少是“claim 之后”的一致投影；完整同步
                # 由下一次 listener 重新读板完成。
                updated = [
                    claimed if task_id(task) == task_id(ready) else task
                    for task in tasks
                ]

                return ClaimedTask(
                    task=claimed,
                    tasks=updated,
                    source="redis",
                )

            # 这个任务被别的 worker 抢走了：从本地快照移除，重选。
            tasks = [
                task for task in tasks
                if task_id(task) != task_id(ready)
            ]

    async def snapshot(self) -> list[dict[str, Any]]:
        """
        读任务板最新快照。**不吞异常**：调用方（dispatch）要靠它区分
        「真的没有 ready task」与「板读不到」—— 后者必须 fail-closed。
        """

        return await self._load()

    async def _load(self) -> list[dict[str, Any]]:
        key = f"{TASK_INDEX_PREFIX}:{self._context_id}"
        ids = await self._client.smembers(key)

        tasks: list[dict[str, Any]] = []

        for raw_id in ids or ():
            tid = _text(raw_id)
            task = await self._read_task(tid)

            if task is None:
                continue

            if self._plan_id:
                meta = metadata_of(task)
                if str(meta.get("plan_id") or "") != self._plan_id:
                    continue

            tasks.append(task)

        # 拓扑序由发布端保证；这里只按 id 排序兜底。
        tasks.sort(key=task_id)

        return tasks

    async def _read_task(self, tid: str) -> dict[str, Any] | None:
        key = f"{TASK_KEY_PREFIX}:{tid}"

        try:
            raw = await self._client.hget(key, "task")
        except Exception:  # noqa: BLE001 - 不是 hash，继续按 string 读
            raw = None

        if raw is None:
            try:
                raw = await self._client.get(key)
            except Exception:  # noqa: BLE001 - key 可能是 hash
                raw = None

        if raw is None:
            return None

        try:
            value = json.loads(_text(raw))
        except (TypeError, ValueError):
            logger.warning("任务板里的 task 不是合法 JSON：%s", key)
            return None

        return value if isinstance(value, dict) else None

    async def _cas_claim(
            self,
            tid: str,
            *,
            owner: str,
    ) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        claim = {
            "owner": owner,
            "token": str(uuid4()),
            "claimed_at": _iso(now),
            "lease_expires_at": _iso(
                now + timedelta(seconds=self._lease_seconds)
            ),
            "attempt": 1,
        }

        key = f"{TASK_KEY_PREFIX}:{tid}"

        try:
            raw = await self._client.eval(
                _CLAIM_LUA,
                1,
                key,
                json.dumps(claim, ensure_ascii=False),
                _iso(now),
            )
        except Exception as exc:  # noqa: BLE001 - Redis 不可用时不 claim
            logger.warning(
                "Redis 任务板 claim 失败（%s: %s），本轮不领取任务。",
                type(exc).__name__,
                exc,
            )
            return None

        if raw is None:
            return None

        try:
            value = json.loads(_text(raw))
        except (TypeError, ValueError):
            return None

        return value if isinstance(value, dict) else None


# ============================================================
# 工厂
# ============================================================


def build_task_subscriber(
        state: Mapping[str, Any],
        *,
        context_id: str,
        plan_id: str | None = None,
) -> TaskSubscriber:
    """
    选数据源。

    当前默认 state；`subagent_mode == "on"` 时优先 Redis（子 agent / worker
    以任务板为准）。Redis 不可用则退回 state，让主图仍能顺序执行。
    """

    if str(state.get("subagent_mode") or "") == "on":

        try:
            from app.common.factories import get_redis_factory

            client = get_redis_factory().optional_client()

        except Exception:  # noqa: BLE001 - 没装配 Redis 是正常状态
            client = None

        if client is not None:
            return RedisTaskSubscriber(
                client,
                context_id=context_id,
                plan_id=plan_id,
            )

    return StateTaskSubscriber(state)


# ============================================================
# internals
# ============================================================


def _coerce_tasks(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []

    return [
        copy.deepcopy(dict(task))
        for task in raw
        if isinstance(task, Mapping)
    ]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)
