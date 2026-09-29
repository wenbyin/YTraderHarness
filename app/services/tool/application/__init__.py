"""
Tool 模块的应用服务。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现
（同包同名的 X_impl.py）。服务的 __init__ 参数用仓储契约注解，
由 ServiceFactory 自动注入。

_assembly.py 不是服务，是两个读服务共用的纯函数组装逻辑 ——
放在这里是因为它只被本层用，且不该被更外层直接 import。
"""

from .classification_service import ClassificationService
from .classification_service_impl import ClassificationServiceImpl
from .processor_definition_service import (
    ProcessorDefinitionService,
)
from .processor_definition_service_impl import (
    ProcessorDefinitionServiceImpl,
)
from .tool_discovery_service import ToolDiscoveryService
from .tool_discovery_service_impl import ToolDiscoveryServiceImpl
from .tool_documentation_service import ToolDocumentationService
from .tool_documentation_service_impl import (
    ToolDocumentationServiceImpl,
)
from .tool_processor_service import ToolProcessorService
from .tool_processor_service_impl import ToolProcessorServiceImpl
from .tool_query_service import ToolQueryService
from .tool_query_service_impl import ToolQueryServiceImpl

__all__ = [
    "ClassificationService",
    "ClassificationServiceImpl",
    "ProcessorDefinitionService",
    "ProcessorDefinitionServiceImpl",
    "ToolDiscoveryService",
    "ToolDiscoveryServiceImpl",
    "ToolDocumentationService",
    "ToolDocumentationServiceImpl",
    "ToolProcessorService",
    "ToolProcessorServiceImpl",
    "ToolQueryService",
    "ToolQueryServiceImpl",
]
