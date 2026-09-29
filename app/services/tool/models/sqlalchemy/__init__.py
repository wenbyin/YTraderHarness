"""
Tool 模块的 SQLAlchemy 模型。

这里逐个 import 全部模型，而不是只靠 initializer 的 walk_packages ——
模型之间是**字符串**形式的 relationship 目标（例如
ToolDefinition.documentation -> "ToolDocumentation"），SQLAlchemy
要到 mapper 配置时才去注册表里解析。只要有哪一个模块没被加载，
配置就会以 "expression 'ToolDocumentation' failed to locate a name"
失败 —— 而且报错离原因很远。

import 本包 = 6 张表全部注册完毕。
"""

from .tool_classification import (
    ToolClassification,
    ToolDefinitionClassification,
)
from .tool_definition import ToolDefinition
from .tool_documentation import ToolDocumentation
from .tool_parameter_documentation import (
    ToolParameterDocumentation,
)
from .tool_processor import ToolProcessor

__all__ = [
    "ToolClassification",
    "ToolDefinition",
    "ToolDefinitionClassification",
    "ToolDocumentation",
    "ToolParameterDocumentation",
    "ToolProcessor",
]
