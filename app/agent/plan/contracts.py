"""Planner 的 TaskDraft 契约与 DAG 校验。

模型只写它真正知道的东西：

    id        计划内 local id（t1 / t2）
    message   任务说明 + 参数，整块原样保留
    metadata  skill_key / blocked_by / block_policy / capabilities

它**不写** A2A 的 messageId / contextId / status / blocked / claim ——
那些由 `a2a.py` 补齐。这样模型不需要记 A2A 线格式，转换器也只做一次。

为什么 metadata 里允许额外键
----------------------------
任务将来会长出新的调度字段（优先级、超时、预算……）。契约只校验自己
认识的键，其余原样带到 A2A metadata；不认识就丢弃会让"加一个字段"变成
"改两处模型输出与转换器"。
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_BLOCK_POLICY",
    "DEFAULT_SKILL_KEY",
    "PlanDraft",
    "PlanValidationError",
    "TaskDraft",
    "ValidatedPlan",
    "normalize_payload",
    "validate_plan",
]


#: 第一版只支持这一种入边策略。
DEFAULT_BLOCK_POLICY = "all_succeeded"

#: 没有 skill 目录 / 模型没给 skill_key 时的兜底。它只是让链路能跑，
#: 不代表这个 task 真能被一个已注册的 skill 执行 —— 调用方应把
#: 默认值写进 notes。
DEFAULT_SKILL_KEY = "general"

_SUPPORTED_BLOCK_POLICIES = frozenset({DEFAULT_BLOCK_POLICY})

#: local id 只允许稳定、可读、可拼进 A2A id 的字符。
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: 单任务 message + metadata 的软上限。超过就拒绝，不静默截断。
_MAX_TASK_BYTES = 64 * 1024


class PlanValidationError(ValueError):
    """planner 输出不满足契约。"""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(str(p) for p in problems)
        super().__init__("；".join(self.problems))


class TaskDraft(BaseModel):
    """一个任务的草案（planner 输出）。"""

    model_config = ConfigDict(extra="ignore")

    id: str
    message: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def skill_key(self) -> str:
        key = str(self.metadata.get("skill_key") or "").strip()
        return key or DEFAULT_SKILL_KEY

    @property
    def blocked_by(self) -> list[str]:
        raw = self.metadata.get("blocked_by")

        if raw is None:
            return []

        if isinstance(raw, str):
            return [raw] if raw.strip() else []

        if isinstance(raw, Sequence):
            return [
                str(item).strip()
                for item in raw
                if isinstance(item, str) and item.strip()
            ]

        return []

    @property
    def block_policy(self) -> str:
        return str(
            self.metadata.get("block_policy") or DEFAULT_BLOCK_POLICY
        ).strip() or DEFAULT_BLOCK_POLICY

    @property
    def capabilities(self) -> list[str]:
        raw = self.metadata.get("capabilities")

        if raw is None:
            return []

        if isinstance(raw, str):
            return [raw] if raw.strip() else []

        if isinstance(raw, Sequence):
            return [
                str(item).strip()
                for item in raw
                if isinstance(item, str) and item.strip()
            ]

        return []


class PlanDraft(BaseModel):
    """planner 的一次完整输出。"""

    model_config = ConfigDict(extra="ignore")

    revision: int = 1
    goal: str = ""
    tasks: list[TaskDraft] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ValidatedPlan:
    """校验后的结果：拓扑序任务 + 可观测的 warning。"""

    tasks: list[TaskDraft]
    warnings: list[str] = field(default_factory=list)


# ============================================================
# 归一化
# ============================================================


def normalize_payload(payload: Any) -> dict[str, Any]:
    """
    把模型输出收成 PlanDraft 能吃的形状。

    这三条是**兼容旧提示词 / 模型手滑**的，不是新契约：

        instruction     -> message.description
        depends_on      -> metadata.blocked_by
        顶层调度字段    -> metadata.*

    真正的契约仍以 §2 的 `message` + `metadata` 为准。
    """

    if not isinstance(payload, Mapping):
        raise PlanValidationError(
            [f"planner 输出必须是 JSON object，收到 {type(payload).__name__}"]
        )

    raw = dict(payload)
    raw_tasks = raw.get("tasks")

    if raw_tasks is None:
        raw["tasks"] = []
        return raw

    if not isinstance(raw_tasks, Sequence) or isinstance(
        raw_tasks, (str, bytes)
    ):
        raise PlanValidationError(["tasks 必须是数组"])

    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(raw_tasks):

        if not isinstance(item, Mapping):
            raise PlanValidationError(
                [f"tasks[{index}] 必须是 object，收到 {type(item).__name__}"]
            )

        task = dict(item)
        metadata = task.get("metadata")

        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, Mapping):
            raise PlanValidationError(
                [f"tasks[{index}].metadata 必须是 object"]
            )
        else:
            metadata = dict(metadata)

        # 顶层调度字段 -> metadata（只搬契约认识的，其余留给 extra）
        for key in (
            "skill_key",
            "blocked_by",
            "depends_on",
            "block_policy",
            "capabilities",
            "claim",
            "blocked",
        ):
            if key in task and key not in metadata:
                metadata[key] = task.pop(key)

        if "depends_on" in metadata and "blocked_by" not in metadata:
            metadata["blocked_by"] = metadata.pop("depends_on")

        if "message" not in task and "instruction" in task:
            task["message"] = {
                "description": str(task.pop("instruction")),
                "params": {},
            }

        # 模型偶尔把 message 写成一段字符串（而不是 object）。
        if isinstance(task.get("message"), str):
            task["message"] = {
                "description": task["message"],
                "params": {},
            }

        task["metadata"] = metadata
        normalized.append(task)

    raw["tasks"] = normalized

    return raw


# ============================================================
# 校验
# ============================================================


def validate_plan(
        plan: PlanDraft,
        *,
        known_skill_keys: set[str] | None = None,
        known_capabilities: set[str] | None = None,
) -> ValidatedPlan:
    """
    校验 DAG、字段形状与能力 code，返回拓扑序任务。

    known_* 为 None 表示"本轮没有目录可查"，只做形状校验；
    传空集合表示"目录存在但为空"，此时所有未知值都会被记 warning。
    """

    problems: list[str] = []
    warnings: list[str] = []
    tasks = list(plan.tasks)

    if not tasks:
        problems.append("tasks 为空")

    ids: list[str] = []
    by_id: dict[str, TaskDraft] = {}

    for index, task in enumerate(tasks):

        tid = str(task.id or "").strip()
        task.id = tid

        if not tid:
            problems.append(f"tasks[{index}].id 为空")
            continue

        if not _ID_RE.match(tid):
            problems.append(
                f"tasks[{index}].id={tid!r} 不匹配 {_ID_RE.pattern}"
            )

        if tid in by_id:
            problems.append(f"任务 id 重复：{tid!r}")
            continue

        ids.append(tid)
        by_id[tid] = task

        _validate_message(task, index, problems)
        _validate_metadata(
            task, index, problems, warnings,
            known_skill_keys=known_skill_keys,
            known_capabilities=known_capabilities,
        )

    # 依赖存在性 / 自依赖 / 去重
    known_ids = set(ids)

    for tid, task in by_id.items():

        deps = list(dict.fromkeys(task.blocked_by))
        task.metadata["blocked_by"] = deps

        for dep in deps:

            if dep == tid:
                problems.append(f"任务 {tid!r} 不允许自依赖")

            elif dep not in known_ids:
                problems.append(
                    f"任务 {tid!r} 的 blocked_by 引用了不存在的 id {dep!r}"
                )

    if problems:
        raise PlanValidationError(problems)

    # Kahn 拓扑排序（保持输入顺序作为同层顺序）
    indegree = {tid: len(by_id[tid].blocked_by) for tid in ids}
    dependents: dict[str, list[str]] = {tid: [] for tid in ids}

    for tid in ids:
        for dep in by_id[tid].blocked_by:
            dependents[dep].append(tid)

    queue = deque(tid for tid in ids if indegree[tid] == 0)
    ordered: list[str] = []

    while queue:

        tid = queue.popleft()
        ordered.append(tid)

        for child in dependents[tid]:
            indegree[child] -= 1

            if indegree[child] == 0:
                queue.append(child)

    if len(ordered) != len(ids):
        raise PlanValidationError(["blocked_by 存在环"])

    for index, task in enumerate(tasks):
        if task.metadata.get("skill_key") in (None, ""):
            task.metadata["skill_key"] = DEFAULT_SKILL_KEY
            warnings.append(
                f"tasks[{index}] 没有 skill_key，回落 {DEFAULT_SKILL_KEY!r}"
            )

    return ValidatedPlan(
        tasks=[by_id[tid] for tid in ordered],
        warnings=warnings,
    )


# ============================================================
# internals
# ============================================================


def _validate_message(
        task: TaskDraft,
        index: int,
        problems: list[str],
) -> None:

    if not isinstance(task.message, dict) or not task.message:
        problems.append(f"tasks[{index}].message 必须是非空 object")
        return

    try:
        encoded = json.dumps(
            task.message, ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        problems.append(
            f"tasks[{index}].message 不是合法 JSON：{exc}"
        )
        return

    if len(encoded.encode("utf-8")) > _MAX_TASK_BYTES:
        problems.append(
            f"tasks[{index}].message 超过 {_MAX_TASK_BYTES} 字节"
        )


def _validate_metadata(
        task: TaskDraft,
        index: int,
        problems: list[str],
        warnings: list[str],
        *,
        known_skill_keys: set[str] | None,
        known_capabilities: set[str] | None,
) -> None:

    metadata = task.metadata

    if not isinstance(metadata, dict):
        problems.append(f"tasks[{index}].metadata 必须是 object")
        return

    try:
        encoded = json.dumps(
            metadata, ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        problems.append(
            f"tasks[{index}].metadata 不是合法 JSON：{exc}"
        )
        return

    if len(encoded.encode("utf-8")) > _MAX_TASK_BYTES:
        problems.append(
            f"tasks[{index}].metadata 超过 {_MAX_TASK_BYTES} 字节"
        )

    skill_key = task.skill_key

    if not skill_key:
        problems.append(f"tasks[{index}].skill_key 为空")

    elif (
        known_skill_keys is not None
        and known_skill_keys
        and skill_key not in known_skill_keys
    ):
        warnings.append(
            f"tasks[{index}].skill_key={skill_key!r} 不在技能目录里"
        )

    policy = task.block_policy

    if policy not in _SUPPORTED_BLOCK_POLICIES:
        problems.append(
            f"tasks[{index}].block_policy={policy!r} 不受支持；"
            f"第一版只支持 {sorted(_SUPPORTED_BLOCK_POLICIES)}"
        )

    capabilities = task.capabilities
    task.metadata["capabilities"] = capabilities

    if known_capabilities is not None:

        kept: list[str] = []

        for code in capabilities:
            if code in known_capabilities:
                kept.append(code)
            else:
                warnings.append(
                    f"tasks[{index}].capabilities 未知 code {code!r} 已丢弃"
                )

        task.metadata["capabilities"] = kept
