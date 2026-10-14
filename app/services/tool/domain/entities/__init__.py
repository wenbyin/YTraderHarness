"""
Tool 模块的领域实体。

⚠️ 能力（Capability）的实体不在这里 —— 它跟着能力树去了
app/services/capability/domain/entities/。本模块只在需要嵌一份
能力进 ToolSummary / ToolDefinitionDetail 时 import 它。
"""

from .tool_definition import ToolDefinition
from .tool_documentation import (
    ParameterDocumentation,
    ToolDocumentation,
)
from .tool_execution import ToolExecution
from .tool_processor import ToolProcessor

__all__ = [
    "ParameterDocumentation",
    "ToolDefinition",
    "ToolDocumentation",
    "ToolExecution",
    "ToolProcessor",
]
