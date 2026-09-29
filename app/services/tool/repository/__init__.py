"""
Tool 模块的持久化契约与实现。

契约 -> 实现的对应**没有任何登记表**：
工厂靠 common/introspect.py 的 resolve_impl 反射发现
（契约的子类，或同包同名的 X_impl.py）。
"""

from .tool_classification import ToolClassificationRepository
from .tool_classification_impl import (
    ToolClassificationRepositoryImpl,
)
from .tool_definition import ToolDefinitionRepository
from .tool_definition_impl import ToolDefinitionRepositoryImpl
from .tool_documentation import ToolDocumentationRepository
from .tool_documentation_impl import (
    ToolDocumentationRepositoryImpl,
)
from .tool_processor import ToolProcessorRepository
from .tool_processor_impl import ToolProcessorRepositoryImpl

__all__ = [
    "ToolClassificationRepository",
    "ToolClassificationRepositoryImpl",
    "ToolDefinitionRepository",
    "ToolDefinitionRepositoryImpl",
    "ToolDocumentationRepository",
    "ToolDocumentationRepositoryImpl",
    "ToolProcessorRepository",
    "ToolProcessorRepositoryImpl",
]
