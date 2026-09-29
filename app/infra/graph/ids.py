"""
标识与指纹的生成规则。

这里集中三件容易各写各的事：

    thread_id            一次运行的幂等标识（断点续跑靠它）
    topology_fingerprint 图拓扑的指纹（防误改，也给前端做兼容判断）
    node_slug / branch_key  观测表的稳定键（历史行永远能对上）
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from uuid import UUID

from app.common.utils import to_snake

__all__ = [
    "branch_key",
    "new_thread_id",
    "normalize_edges",
    "to_node_slug",
    "topology_fingerprint",
]


# ============================================================
# Thread
# ============================================================

THREAD_ID_MAX_LENGTH = 255


def new_thread_id(
        namespace: str,
        subject_id: UUID,
        pipeline_run_id: UUID,
) -> str:
    """
    构造一次运行的 LangGraph thread_id。

    形如：

        <namespace>:<subject_uuid>:run:<run_uuid>

    例如 document 摄入流水线传 namespace="docver"：

        docver:550e8400-...:run:7c9e6679-...

    约 85 字符，在 LangGraph 的 255 上限内，且可读 ——
    排查时能直接从 thread_id 看出是哪个主体的哪次运行。

    namespace 由业务侧给，infra 不认识具体含义。

    用 pipeline_run_id 而非 subject_id 单独做主键，是因为同一个
    主体会被 reprocess 多次，每次都必须是一个全新的 thread
    （否则第二次会从第一次的 checkpoint 续跑，而不是重新处理）。

    真正的「续跑」用同一个 thread_id 再调一次 ainvoke(None, ...)。
    """

    thread_id = (
        f"{namespace}:{subject_id}:run:{pipeline_run_id}"
    )

    if len(thread_id) > THREAD_ID_MAX_LENGTH:
        # UUID 定长，构造上不可能超；真超了说明格式或 namespace 被改坏了。
        raise ValueError(
            "thread_id exceeds "
            f"{THREAD_ID_MAX_LENGTH} chars: {thread_id!r}"
        )

    return thread_id


# ============================================================
# Topology
# ============================================================


def topology_fingerprint(
        nodes: Iterable[tuple[str, int]],
        edges: Iterable[tuple[str, str]],
) -> str:
    """
    对图拓扑取 sha256 指纹。

    nodes 传 (node_id, sequence) 对，edges 传 (from, to) 对。

    规范化方式：先排序再序列化，保证与入参顺序无关；
    用 sort_keys + 紧凑分隔符，保证跨版本 Python 结果一致。

    CI 拿它与签入的 golden 值比对，任何拓扑改动都会被逮住。
    前端也可用它判断一次历史运行是否属于当前图版本。
    """

    payload = {
        "nodes": sorted(
            (str(node_id), int(sequence))
            for node_id, sequence in nodes
        ),
        "edges": sorted(
            (str(src), str(dst))
            for src, dst in edges
        ),
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


# ============================================================
# Observability keys
# ============================================================


def to_node_slug(name: str) -> str:
    """
    把任意节点名规范成稳定的 node_id。

    节点名在代码里可能是 CamelCase 或带空格，
    但 node_id 一旦落库就是前端契约的一部分，
    因此统一转 snake_case 并去掉首尾分隔符。

    规则本身复用 app.common.utils.to_snake，
    避免与项目其他地方的命名转换不一致。
    """

    slug = to_snake(name.strip()).strip("_")

    if not slug:
        raise ValueError(
            f"node name yields an empty slug: {name!r}"
        )

    return slug


def branch_key(
        branch_scope_id: UUID | None,
) -> str:
    """
    扇出分支在观测表里的键。

    线性阶段与控制节点传 None，得到空串；
    扇出阶段传该分支的主体 ID（如某个 retrieval definition），
    得到其字符串形式。

    scope 的**含义由业务侧定义**，infra 只负责把它变成稳定的
    字符串键。

    空串而非 NULL 是刻意的：它是
    UNIQUE(pipeline_run_id, node_id, branch_key, attempt)
    的组成部分，而 Postgres 的唯一约束不约束 NULL
    （多个 NULL 互不冲突），用 NULL 会让"线性节点重复写入"
    这条幂等保证失效。
    """

    return (
        ""
        if branch_scope_id is None
        else str(branch_scope_id)
    )


def normalize_edges(
        edges: Iterable[Sequence[str]],
) -> list[tuple[str, str]]:
    """
    把边的各种写法（2 元组 / 2 元列表）统一成 (str, str)。

    topology_fingerprint 的入参容错，避免调用方为了
    传参形式多写一层转换。
    """

    normalized: list[tuple[str, str]] = []

    for edge in edges:
        src, dst = edge[0], edge[1]
        normalized.append((str(src), str(dst)))

    return normalized
