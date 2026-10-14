"""
标识与指纹的生成规则。

这里集中三件容易各写各的事：

    thread_id            一次运行的幂等标识（断点续跑靠它）
    topology_fingerprint 图拓扑的指纹（防误改，也给前端做兼容判断）
    node_slug / branch_key  观测表的稳定键（历史行永远能对上）
    branch_scope_id      扇出分支的身份：本来没有 UUID 的地方由它
                         从稳定字符串派生一个（必须是确定性的，
                         否则 resume 会给同一分支多写一行）
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from uuid import UUID, uuid5

from app.common.utils import canonical_json, sha256_text, to_snake

__all__ = [
    "branch_key",
    "branch_scope_id",
    "new_session_thread_id",
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


def new_session_thread_id(
        run_level: str,
        agent_type: str,
        thread_uuid: UUID,
) -> str:
    """
    构造**按会话稳定**的 LangGraph thread_id。

    形如：

        <run_level>:<agent_type>:<thread_uuid>

    例如 dev 环境的 react 会话：

        dev:react:550e8400-e29b-41d4-a716-446655440000

    ⚠️ 为什么不复用 `new_thread_id`（**两者的语义正好相反**）
    ----------------------------------------------------
    `new_thread_id` 刻意把 `pipeline_run_id` 放进唯一分量，理由写在它
    自己的 docstring 里：document 被 reprocess 时**必须**得到全新的
    thread，否则第二次会从第一次的 checkpoint 续跑而不是重新处理。

    而这一张图要的恰恰是反面：同一会话的多轮**必须**落在同一个 thread
    上，否则每轮都是全新 checkpoint、上一轮的消息不在。这正是"同一会话
    多次 invoke，messages 里只有一条 HumanMessage"的成因。

    一个函数不可能同时满足这两种要求 —— 共用一个必然把另一边写错，
    而且错法是**静默**的（thread_id 照样是个合法字符串，只是语义不对，
    要等下一次跑才发现历史没了）。因此两个函数并存，各自带自己的理由。

    run_level / agent_type 由调用方给，infra 不认识它们的含义
    ------------------------------------------------------
    与 `new_thread_id` 的 namespace 同一条规矩：本模块只负责拼与校验，
    不负责解释。"什么算一个环境"是 app.common.environment 的事，
    "这张图是哪一种 Agent"是 react catalog 的事。

    ⚠️ thread_uuid 必须是**跨轮不变**的那个
    --------------------------------------
    调用方要把它持久化（会话行上），不能每次调用都新生成 —— 那等于
    退化成"每轮换 thread"。传进来而不是在这里生成，就是为了让这个
    事实摆在调用点看得见。
    """

    thread_id = f"{run_level}:{agent_type}:{thread_uuid}"

    if len(thread_id) > THREAD_ID_MAX_LENGTH:
        # UUID 定长、前后缀都很短，构造上不可能超；真超了说明有人
        # 往 run_level / agent_type 里塞了别的东西。
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

    # ⚠️ default_str=False：拓扑指纹**不做** str() 兜底。
    # 它是对外契约（有 golden 校验），输入本该全是 str/int；真混进
    # 别的对象时抛 TypeError，比按 repr 摘要要好 —— repr 会随对象
    # 状态变，摘要跟着变，而没人知道为什么。详见
    # app/common/utils/digest.py 的说明。
    return sha256_text(
        canonical_json(payload, default_str=False)
    )


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


def branch_scope_id(
        namespace: UUID,
        key: str,
) -> UUID:
    """
    从一个**稳定字符串**派生出扇出分支的 scope ID。

    与 `branch_key` 的分工：那个把已有 UUID 变成观测键，这个负责
    "本来没有 UUID 的地方怎么造一个稳定的"。

    ⚠️ 必须是**确定性**的（因此是 uuid5 而不是 uuid4）
    -------------------------------------------------
    扇出分支第一次执行时 `node_span` 会 upsert 一行
    `branch_key=<这个值>`。而 `interrupt()` 会让那个分支在 resume 时
    **重新执行**一次节点函数 —— 如果这次算出一个不同的 scope，
    观测表里就会多出一行（唯一键是
    `(pipeline_run_id, node_id, branch_key, attempt)`），
    一次中断换来一行幽灵数据。

    所以调用方给的 `key` 必须是"同一次运行里同一个分支永远是同一个
    字符串"。真正撑住这件事的通常不是这个函数，而是
    **Send payload 会被 checkpoint 原样恢复**（路由不会在 resume 时
    重算）—— 本函数只负责把那个字符串变成一个 UUID。

    `key` 的含义由业务侧定义（与 `branch_key` 同一约定）：infra 不
    关心它是工具调用 id、分支名还是别的，只要求它稳定且非空。

    空 key 直接报错而不是照算：uuid5 对空串是合法的，算得出一个
    "所有忘记传 key 的地方都长一样"的值 —— 那种碰撞是静默的，
    而唯一约束要到写库时才炸，离原因很远。
    """

    if not key:
        raise ValueError(
            "branch_scope_id 的 key 不能为空 —— 空 key 会让所有忘了"
            "传 key 的分支拿到同一个 scope，观测行互相覆盖。"
        )

    return uuid5(namespace, key)


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
