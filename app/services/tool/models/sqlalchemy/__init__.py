"""
Tool 模块的 SQLAlchemy 模型。

这里逐个 import 全部模型，而不是只靠 initializer 的 walk_packages ——
模型之间是**字符串**形式的 relationship 目标（例如
ToolDefinition.documentation -> "ToolDocumentation"），SQLAlchemy
要到 mapper 配置时才去注册表里解析。只要有哪一个模块没被加载，
配置就会以 "expression 'ToolDocumentation' failed to locate a name"
失败 —— 而且报错离原因很远。

import 本包 = 本模块的表全部注册完毕。

⚠️ 能力树（capability 表）**不在本模块**，它搬去了
app/services/capability/。这里只剩关联表，它跨模块地指向
Capability，因此 app/infra/datasource/initializer.py 必须把两个
模块的模型包都 import 一遍，mapper 才能解析那个字符串目标。
"""

from .tool_classification import ToolDefinitionClassification
from .tool_definition import ToolDefinition
from .tool_definition_state import ToolDefinitionState
from .tool_documentation import ToolDocumentation
from .tool_execution import ToolExecution
from .tool_parameter_documentation import (
    ToolParameterDocumentation,
)
from .tool_processor import ToolProcessor

__all__ = [
    "ToolDefinition",
    "ToolDefinitionClassification",
    "ToolDefinitionState",
    "ToolDocumentation",
    "ToolExecution",
    "ToolParameterDocumentation",
    "ToolProcessor",
]
