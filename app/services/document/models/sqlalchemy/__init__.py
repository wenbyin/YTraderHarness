"""
Document 模块的全部 SQLAlchemy Model。

集中导出有两个作用：
    1. 让 initializer 的整包扫描能一次性注册所有表
    2. 避免 relationship 的字符串前向引用因为模块未导入而解析失败

注意第 2 点是**真的会炸**的：document_chunk 里有
`Mapped["DocumentRetrievalDefinition"]` 这样的前向引用，
若该模型模块没被导入，mapper configuration 阶段会直接抛错。
initializer 的 walk_packages 会兜住运行时，但只 import 本包
（例如写测试）时就需要这份完整列表。
"""

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentIndexBinding",
    "DocumentPipelineNodeRun",
    "DocumentPipelineRun",
    "DocumentProcessorDefinition",
    "DocumentProcessorRetrievalBinding",
    "DocumentRetrievalDefinition",
    "DocumentTypeDefinition",
    "DocumentVersion",
    "DocumentVersionRetrievalBinding",
]

from .document import Document
from .document_chunk import DocumentChunk
from .document_index_binding import DocumentIndexBinding
from .document_pipeline_node_run import DocumentPipelineNodeRun
from .document_pipeline_run import DocumentPipelineRun
from .document_processor_definition import (
    DocumentProcessorDefinition,
)
from .document_processor_retrieval_binding import (
    DocumentProcessorRetrievalBinding,
)
from .document_retrieval_definition import (
    DocumentRetrievalDefinition,
)
from .document_type_definition import DocumentTypeDefinition
from .document_version import DocumentVersion
from .document_version_retrieval_binding import (
    DocumentVersionRetrievalBinding,
)
