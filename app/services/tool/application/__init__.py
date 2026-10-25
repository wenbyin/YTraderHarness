"""
Tool 模块的应用服务。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现
（同包同名的 X_impl.py）。服务的 __init__ 参数用仓储契约注解，
由 ServiceFactory 自动注入。

_assembly.py 不是服务，是两个读服务共用的纯函数组装逻辑 ——
放在这里是因为它只被本层用，且不该被更外层直接 import。

⚠️ 能力树本体的服务**不在本模块** —— 它是
app/services/capability 的 CapabilityService。这里只有工具 ↔ 能力的
关联服务（ToolClassificationService）。
"""

from .mcp_provider_service import MCPProviderService
from .mcp_provider_service_impl import MCPProviderServiceImpl
from .processor_definition_service import (
    ProcessorDefinitionService,
)
from .processor_definition_service_impl import (
    ProcessorDefinitionServiceImpl,
)
from .tool_classification_service import ToolClassificationService
from .tool_classification_service_impl import (
    ToolClassificationServiceImpl,
)
from .tool_definition_state_service import (
    ToolDefinitionStateService,
)
from .tool_definition_state_service_impl import (
    ToolDefinitionStateServiceImpl,
)
from .tool_discovery_service import ToolDiscoveryService
from .tool_discovery_service_impl import ToolDiscoveryServiceImpl
from .tool_documentation_service import ToolDocumentationService
from .tool_documentation_service_impl import (
    ToolDocumentationServiceImpl,
)
from .tool_execution_service import ToolExecutionService
from .tool_execution_service_impl import ToolExecutionServiceImpl
from .tool_processor_service import ToolProcessorService
from .tool_processor_service_impl import ToolProcessorServiceImpl
from .tool_query_service import ToolQueryService
from .tool_query_service_impl import ToolQueryServiceImpl

__all__ = [
    "MCPProviderService",
    "MCPProviderServiceImpl",
    "ProcessorDefinitionService",
    "ProcessorDefinitionServiceImpl",
    "ToolClassificationService",
    "ToolClassificationServiceImpl",
    "ToolDefinitionStateService",
    "ToolDefinitionStateServiceImpl",
    "ToolDiscoveryService",
    "ToolDiscoveryServiceImpl",
    "ToolDocumentationService",
    "ToolDocumentationServiceImpl",
    "ToolExecutionService",
    "ToolExecutionServiceImpl",
    "ToolProcessorService",
    "ToolProcessorServiceImpl",
    "ToolQueryService",
    "ToolQueryServiceImpl",
]
