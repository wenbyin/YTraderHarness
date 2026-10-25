"""
Tool 模块的对外契约。

domain/schemas/
│
├── enum.py
│   └── SourceType / SourceState
│
├── tool.py
│   └── ToolManifest / ToolSummary / ToolDefinitionDetail /
│       ToolEnabledUpdate / ToolEnabledState /
│       PagedToolSummaryResponse
│
├── documentation.py
│   └── ToolDocumentation / ParameterDocumentation 及其 Update 版本
│
├── classification.py
│   └── ToolClassificationUpdate（工具 ↔ 能力的关联）
│
├── discovery.py
│   └── ToolSyncRequest / ToolSyncResult
│
└── processor.py
    └── ProcessorDefinition / ToolProcessor / ToolProcessorChain
        及其 Create / Update / Replace / Reorder 四个请求

全部结构逐字段对应 tool.yaml 与 tool_processor.yaml 的
components/schemas；字段名与枚举取值都以那两份 spec 为准。

⚠️ 能力（Capability）本体的契约**不在本模块** —— 它跟着能力树去了
app/services/capability/domain/schemas/，对应 capability.yaml。
本模块只在嵌一份能力进 ToolSummary / ToolDefinitionDetail 时 import
它（见 tool.py）。
"""

from .classification import ToolClassificationUpdate
from .discovery import (
    ToolSyncRequest,
    ToolSyncResult,
)
from .documentation import (
    ParameterDocumentation,
    ParameterDocumentationUpdate,
    ToolDocumentation,
    ToolDocumentationUpdate,
)
from .enum import (
    ExecutionDecision,
    ExecutionMatch,
    ExecutionMode,
    ExecutionScope,
    GuardAction,
    ProcessorHook,
    ProcessorType,
    SourceState,
    SourceType,
)
from .execution import (
    ExecutionGuardRequest,
    ExecutionReservation,
    ExecutionStatusLimit,
)
from .mcp import (
    MCPProviderListResponse,
    MCPProviderSummary,
)
from .processor import (
    CreateToolProcessorRequest,
    PagedProcessorDefinitionResponse,
    ProcessorDefinition,
    ReorderToolProcessorsRequest,
    ReplaceToolProcessorChainRequest,
    ReplaceToolProcessorItem,
    ToolProcessor,
    ToolProcessorChain,
    UpdateToolProcessorRequest,
)
from .tool import (
    PagedToolSummaryResponse,
    ToolDefinitionDetail,
    ToolEnabledState,
    ToolEnabledUpdate,
    ToolManifest,
    ToolSummary,
)

__all__ = [
    "CreateToolProcessorRequest",
    "ExecutionDecision",
    "ExecutionGuardRequest",
    "ExecutionMatch",
    "ExecutionMode",
    "ExecutionReservation",
    "ExecutionScope",
    "ExecutionStatusLimit",
    "GuardAction",
    "MCPProviderListResponse",
    "MCPProviderSummary",
    "PagedProcessorDefinitionResponse",
    "PagedToolSummaryResponse",
    "ParameterDocumentation",
    "ParameterDocumentationUpdate",
    "ProcessorDefinition",
    "ProcessorHook",
    "ProcessorType",
    "ReorderToolProcessorsRequest",
    "ReplaceToolProcessorChainRequest",
    "ReplaceToolProcessorItem",
    "SourceState",
    "SourceType",
    "ToolClassificationUpdate",
    "ToolDefinitionDetail",
    "ToolDocumentation",
    "ToolEnabledState",
    "ToolEnabledUpdate",
    "ToolDocumentationUpdate",
    "ToolManifest",
    "ToolProcessor",
    "ToolProcessorChain",
    "ToolSummary",
    "ToolSyncRequest",
    "ToolSyncResult",
    "UpdateToolProcessorRequest",
]
