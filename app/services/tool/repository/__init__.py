"""
Tool 模块的持久化契约与实现。

契约 -> 实现的对应**没有任何登记表**：
工厂靠 common/introspect.py 的 resolve_impl 反射发现
（契约的子类，或同包同名的 X_impl.py）。

⚠️ 能力树（taxonomy）的仓储不在这里 —— 它是
app/services/capability/repository/。本模块只有工具自己的四张表
加上 tool ↔ capability 的关联表。
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
from .tool_execution import ToolExecutionRepository
from .tool_execution_impl import ToolExecutionRepositoryImpl
from .tool_processor import ToolProcessorRepository
from .tool_processor_impl import ToolProcessorRepositoryImpl

__all__ = [
    "ToolClassificationRepository",
    "ToolClassificationRepositoryImpl",
    "ToolDefinitionRepository",
    "ToolDefinitionRepositoryImpl",
    "ToolDocumentationRepository",
    "ToolDocumentationRepositoryImpl",
    "ToolExecutionRepository",
    "ToolExecutionRepositoryImpl",
    "ToolProcessorRepository",
    "ToolProcessorRepositoryImpl",
]
