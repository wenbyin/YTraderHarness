"""
Tool 自注册与发现。

    decorator.py    @tool 描述符 —— 声明即注册
    registry.py     进程级工具表（含包遍历兜底）
    scanner.py      扫描器：一条来源一个
    discoverer.py   编排扫描器，合并去重
    schema.py       schema 推导（纯函数，两条来源共用）
    docstring.py    docstring 切段（纯函数，两条来源共用）
    spec.py         两个数据结构
    selection.py    按 key 选工具（纯函数）—— "哪些工具能用"的唯一判定

    processor_registry.py
                    处理器实现的注册表（反射发现，与上面几个无关 ——
                    那是"工具怎么被发现"，这是"处理器怎么被挂载"）

分层理由：装饰器要能被**任何** service 模块 import，而它只依赖
注册表；扫描与 schema 推导只在同步时用得到。拆开可以让业务模块的
import 链上少拖进 langchain 与 schema 生成。
"""

from .decorator import ToolDescriptor, tool
from .discoverer import ToolDiscoverer
from .docstring import parse_docstring, resolve_returns
from .processor_registry import (
    PROCESSOR_PACKAGES,
    ToolProcessorRegistry,
    get_processor_registry,
)
from .registry import (
    TOOL_SOURCE_PACKAGES,
    ToolRegistry,
    get_tool_registry,
    iter_package_modules,
)
from .scanner import (
    LangchainToolScanner,
    ServiceMethodScanner,
    ToolScanner,
)
from .schema import (
    extract_parameter_descriptions,
    output_description,
    output_schema_of,
    resolve_request_model,
    synthesize_schema,
)
from .selection import select_specs, tool_key_of, wire_name
from .spec import ToolManifestData, ToolSpec

__all__ = [
    "PROCESSOR_PACKAGES",
    "TOOL_SOURCE_PACKAGES",
    "LangchainToolScanner",
    "ServiceMethodScanner",
    "ToolDescriptor",
    "ToolDiscoverer",
    "ToolManifestData",
    "ToolProcessorRegistry",
    "ToolRegistry",
    "ToolScanner",
    "ToolSpec",
    "extract_parameter_descriptions",
    "get_processor_registry",
    "get_tool_registry",
    "iter_package_modules",
    "output_description",
    "output_schema_of",
    "parse_docstring",
    "resolve_request_model",
    "resolve_returns",
    "select_specs",
    "synthesize_schema",
    "tool",
    "tool_key_of",
    "wire_name",
]
