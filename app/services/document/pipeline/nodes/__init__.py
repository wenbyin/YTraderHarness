"""
流水线节点。

10 个阶段 + 1 个控制节点，与阶段表 1:1 对应：

    线性    ingestion / format_detection / parsing
            / normalization / cleaning / metadata_extraction

    扇出    chunking / chunk_enrichment / embedding / indexing
            （每个 DocumentRetrievalDefinition 各跑一遍）

    控制    finalize_version（扇入，写回业务状态）

共享逻辑在 _processor.py（三个 Processor 阶段）与
_branch.py（四个扇出阶段的 span 参数与摘要压缩）。

写新节点时的两个坑
------------------
1. **runtime 参数必须叫 `runtime` 且有类型注解。**

       async def my_node(state, runtime: Runtime[Ctx]) -> dict:   ✅
       async def my_node(state, rt: Runtime[Ctx]) -> dict:       ❌
       async def my_node(state, runtime) -> dict:                ❌

   LangGraph 靠「参数名 + 类型注解」两者同时匹配来决定是否注入
   runtime（见 langgraph/_internal/_runnable.py 的 kwarg 表）。
   不匹配时它会把 runtime 当成普通位置参数，报
   "missing 1 required positional argument" —— 报错信息完全
   指不到真正的原因。

2. **不要在 `async with node_span(...)` 的入参里做会抛错的计算。**
   入参在上下文管理器**之外**求值，那里抛的错不会在节点行里
   留下失败记录，只会记到 run 行的 error_message 上。
"""

from .chunk_enrichment import chunk_enrichment_node
from .chunking import chunking_node
from .cleaning import cleaning_node
from .embedding import embedding_node
from .finalize import finalize_version_node
from .format_detection import format_detection_node
from .indexing import indexing_node
from .ingestion import ingestion_node
from .metadata_extraction import metadata_extraction_node
from .normalization import normalization_node
from .parsing import parsing_node

__all__ = [
    "chunk_enrichment_node",
    "chunking_node",
    "cleaning_node",
    "embedding_node",
    "finalize_version_node",
    "format_detection_node",
    "indexing_node",
    "ingestion_node",
    "metadata_extraction_node",
    "normalization_node",
    "parsing_node",
]
