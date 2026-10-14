"""
Memory 模块的对外契约。

全部结构逐字段对应各自 spec 的 components/schemas；字段名与枚举取值
都以那几份为准：

    memory.yaml        L0 —— 原始对话的检索
    memory_atom.yaml   L1 —— 提炼出来的记忆的检索
    memory_api_v2.yaml 管理端 —— 提取进度水位 + 跨层溯源（query.py）

⚠️ **模型输出**的解析形状都**不在**这里
---------------------------------
    atom.py           L1 抽取 / 冲突检测的 JSON
    scene.py          L2 整合的 JSON（写入清单）
    persona.py        L3 生成的 JSON（no_change + 整篇 content）
    _model_output.py  上面三份共用的剥围栏 / 解析 / 逐项校验

它们不对任何调用方负责，只对各自那一份提示词负责，所以不进这个对外契约
的包。**这不是疏漏**：这份包会被照着渲染成 OpenAPI，而模型输出的形状
随时会跟着提示词改。

⚠️ META 块的解析器也不在这里，它住在提示词模块里
---------------------------------------
`parse_meta`（从存量 scene content 里读 created/updated/summary/heat）
被搬到了 `app/agent/prompt/prompts/memory_scene_consolidate.py` —— 那个块
的**格式**由那份模板定义，解析器跟模板住在一起才不会漂。
"""

from .enum import (
    MemoryRecallRoute,
    MemorySearchMode,
    MemorySearchScope,
)
from .memory import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    DegradedRoute,
    MemorySearchHit,
    MemorySearchRequest,
    MemorySearchResponse,
)
from .query import (
    ConversationHit,
    ConversationHitSection,
    ConversationRecord,
    DegradedHitRoute,
    DegradedLayer,
    FactHit,
    FactHitSection,
    FactRecord,
    L0AtomsRequest,
    L0AtomsResponse,
    L0QueryOrder,
    L0QueryRequest,
    L0QueryResponse,
    L1CursorRecord,
    L1ExtractionRequest,
    L1ExtractionResponse,
    L1QueryOrder,
    L1QueryRequest,
    L1QueryResponse,
    L1SourcesRequest,
    L1SourcesResponse,
    L2CursorRecord,
    L2DetailRequest,
    L2DetailResponse,
    L2ExtractionRequest,
    L2ExtractionResponse,
    L2QueryOrder,
    L2QueryRequest,
    L2QueryResponse,
    L3CursorRecord,
    L3ExtractionRequest,
    L3ExtractionResponse,
    L3QueryResponse,
    RetrievalTarget,
    Pagination,
    PersonaRecord,
    SceneDetail,
    SceneSummary,
    ScopeKind,
    TargetRequest,
)

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "ConversationHit",
    "ConversationHitSection",
    "ConversationRecord",
    "DegradedHitRoute",
    "DegradedLayer",
    "DegradedRoute",
    "FactHit",
    "FactHitSection",
    "FactRecord",
    "L0AtomsRequest",
    "L0AtomsResponse",
    "L0QueryOrder",
    "L0QueryRequest",
    "L0QueryResponse",
    "L1CursorRecord",
    "L1ExtractionRequest",
    "L1ExtractionResponse",
    "L1QueryOrder",
    "L1QueryRequest",
    "L1QueryResponse",
    "L1SourcesRequest",
    "L1SourcesResponse",
    "L2CursorRecord",
    "L2DetailRequest",
    "L2DetailResponse",
    "L2ExtractionRequest",
    "L2ExtractionResponse",
    "L2QueryOrder",
    "L2QueryRequest",
    "L2QueryResponse",
    "L3CursorRecord",
    "L3ExtractionRequest",
    "L3ExtractionResponse",
    "L3QueryResponse",
    "MemoryRecallRoute",
    "MemorySearchHit",
    "MemorySearchMode",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemorySearchScope",
    "Pagination",
    "PersonaRecord",
    "RecallRoute",
    "RetrievalTarget",
    "SceneDetail",
    "SceneSummary",
    "ScopeKind",
    "TargetRequest",
]
