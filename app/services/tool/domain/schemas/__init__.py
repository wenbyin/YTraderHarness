"""
Tool 模块的对外契约。

domain/schemas/
│
├── enum.py
│   └── SourceType / SourceState
│
├── tool.py
│   └── ToolManifest / ToolSummary / ToolDefinitionDetail /
│       PagedToolSummaryResponse
│
├── documentation.py
│   └── ToolDocumentation / ParameterDocumentation 及其 Update 版本
│
├── classification.py
│   └── Classification 及其 Create / Update，以及 ToolClassificationUpdate
│
├── discovery.py
│   └── ToolSyncRequest / ToolSyncResult
│
└── processor.py
    └── ProcessorDefinition / ToolProcessor / ToolProcessorChain
        及其 Create / Update / Replace / Reorder 四个请求

全部结构逐字段对应 tool.yaml 与 tool_processor.yaml 的
components/schemas；字段名与枚举取值都以那两份 spec 为准。
"""

from .classification import (
    Classification,
    ClassificationCreate,
    ClassificationUpdate,
    ToolClassificationUpdate,
)
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
    ExecutionMode,
    ProcessorHook,
    ProcessorType,
    SourceState,
    SourceType,
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
    ToolManifest,
    ToolSummary,
)

__all__ = [
    "Classification",
    "ClassificationCreate",
    "ClassificationUpdate",
    "CreateToolProcessorRequest",
    "ExecutionMode",
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
    "ToolDocumentationUpdate",
    "ToolManifest",
    "ToolProcessor",
    "ToolProcessorChain",
    "ToolSummary",
    "ToolSyncRequest",
    "ToolSyncResult",
    "UpdateToolProcessorRequest",
]
