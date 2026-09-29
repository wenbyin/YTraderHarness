"""
扇出节点的共享小工具。

四个扇出阶段要做同样的两件小事：

    1. 把 node_span 的参数凑齐，其中 retrieval_definition_id
       必须从 Send payload 的 branch 里取（分支身份只有节点自己知道，
       callback 拿不到 —— 见 infra/graph/callbacks.py 的说明）
    2. 把 ArtifactRef 压成观测摘要（只留 checksum 与计数，绝不放正文）

放这里避免四个节点各写一遍、然后各自漂移。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..state import BranchInput

__all__ = ["branch_span_kwargs", "ref_summary"]


def branch_span_kwargs(
        spec: Any,
        state: BranchInput,
) -> dict[str, Any]:
    """
    组装 node_span 的关键字参数。

    branch_scope_id 是**分支身份的唯一来源**：
    Send 扇出的 N 个分支在 LangGraph 的 callback metadata 里
    共享同一个节点名，无法区分属于哪个 retrieval。
    这里从 payload 显式取出来交给 node_span，它才能把
    branch_key 写对 —— 这是扇出观测能成立的前提。

    对 document 摄入流水线，scope 就是 retrieval definition id；
    infra 侧只认它是个可索引的主体 ID（见 infra/graph/ids.py）。
    """

    branch = (
        state.get("branch")
        if isinstance(state, dict)
        else None
    )

    if not branch:
        # 这条路径不常见，但一旦走到就很费解，所以给一个能直接
        # 定位的提示。典型原因是图被装配错了 —— 例如给扇出节点
        # 加了普通边，导致它拿到父状态而不是 Send payload。
        #
        # ⚠️ 注意：本函数在 node_span(...) 的**入参求值**阶段调用，
        # 在上下文管理器之外，因此这里抛的错**不会**在节点行里留记录，
        # 只会记到 run 行的 error_message 上。
        raise ValueError(
            f"{spec.node_id}: 扇出节点没有收到 Send payload。"
            "它拿到的 state 里没有 'branch' 键，说明上游是用"
            "普通边（而非条件边 + Send）进来的。"
            "检查 graph.py 的边装配。"
        )

    return {
        "node_id": spec.node_id,
        "label": spec.label,
        "group": spec.group,
        "sequence": spec.sequence,
        "branch_scope_id": UUID(
            branch["document_retrieval_definition_id"]
        ),
    }


def ref_summary(
        ref: Any,
) -> dict[str, Any] | None:
    """
    把 ArtifactRef 压成只含 checksum 与计数的摘要。

    preview 一并带上 —— 它有 512 字符上限，是唯一允许进观测
    内容的字段，前端用它做「这一步产出的是不是我以为的东西」
    的快速确认。
    """

    if not isinstance(ref, dict):
        return None

    return {
        "checksum": ref.get("checksum"),
        "kind": ref.get("kind"),
        "byte_size": ref.get("byte_size"),
        "char_count": ref.get("char_count"),
        "preview": ref.get("preview"),
    }
