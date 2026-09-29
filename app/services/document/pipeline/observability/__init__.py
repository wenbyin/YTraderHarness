"""
Document 流水线的可观测性。

    artifacts.py      中间产物读写（基于 FileStorage，不落库表）
    recorder_impl.py  观测写入的 SQLAlchemy 实现
    queries.py        读模型，供 API 层组装响应

可观测性的两条写入路径：

    instrumentation.node_span     主写入方（节点内，带真实 attempt）
    callbacks.PersistingCallback  兜底网（只能捕获节点体之外的失败）

两者都 upsert 到
(pipeline_run_id, node_id, branch_key, attempt)，
因此重复事件天然幂等。
"""

from .artifacts import (
    ArtifactStore,
    FileArtifactStore,
    sha256_of_bytes,
    sha256_of_text,
)
from .recorder_impl import SqlAlchemyRunRecorder

__all__ = [
    "ArtifactStore",
    "FileArtifactStore",
    "SqlAlchemyRunRecorder",
    "sha256_of_bytes",
    "sha256_of_text",
]
